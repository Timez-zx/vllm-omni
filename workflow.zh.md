# MiniCPM-o 原生全双工 Serving 工作流

## 目标与边界

目标是在持续音视频交互中，以实时性为约束测量单机多用户容量，并定位容量失效时的 engine 瓶颈。研究对象是调度、batching、KV cache 和流水线延迟，不是模型质量。

本分支使用 `openbmb/MiniCPM-o-4_5` 的原生 duplex 路径。与 Qwen3-Omni 的 duplex-like 模拟不同，MiniCPM 会持续接收音视频，并由模型自行决定 listen 或 speak，因此更接近目标业务负载。

## 阶段一：应用与 engine 接口

```text
每个用户建立长期 WebSocket
  → 每 200 ms 上传一段 PCM16 音频
  → 每 1 秒附带一帧视频
  → 应用将 5 段音频组成一个 1 秒原生 model unit
  → Thinker 增量处理该 unit，并决定 listen 或 speak
  → speak 时进入 Talker → Code2Wav
  → 输入流继续，不等待音频播放完成
```

- 每个 session 使用一条可续接的 Thinker request/KV lineage；新 unit 只追加新 token，不重复 prefill 全部历史。
- `auto_response` 保持开启。客户端不提交合成 query，也不强制模型回答。
- 不同 session 直接并发进入 engine；应用层没有全局 gate，也不做跨用户 batching。
- 音频上传频率是 5 Hz，但模型计算单位是 1 Hz。零散 PCM 只在应用输入缓冲区中聚合，不会形成五个 Thinker prefill。
- 用户输入在 assistant 输出期间仍可进入已有 session；播放状态不控制模型 admission。
- KV lineage 是可丢弃的执行状态。session、输入缓冲、重连和输出状态由应用层维护。

## 阶段二：固定部署

正式基线只使用 `benchmarks/minicpmo/deploy_capacity_3gpu.yaml`：

| Stage | GPU | 配置 |
|---|---:|---|
| Thinker | 0 | BF16，单个非 P/D vLLM engine |
| Talker | 1 | BF16 |
| Code2Wav | 2 | BF16 |

- 硬件：3 × RTX PRO 6000 Blackwell 96 GB。
- 三个 stage 的 `max_num_seqs` 均为 64，使用同步调度。
- `active_stream_window: 0`，不在应用层限制同时说话的用户数。
- 本阶段不做 P/D 分离；容量瓶颈必须先在这一固定基线上确认。

启动：

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_3gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

## 阶段三：正式 workload 与指标

- 每个用户连续播放同一真实 MP4 中对齐的音频和视频。
- 音频：16 kHz mono PCM16，每 200 ms 上传。
- 视频：1 FPS，保留源分辨率 960×540，`max_slice_nums=4`。
- 官方切图算法实际生成一张全局图和两个局部 crop。每个稳态 unit 含 198 个视觉 scheduler tokens，共 211 个 scheduler tokens。
- 用户连接分批建立；全部 session 就绪后从同一 barrier 开始，并在 `[0, 1 s)` 内按固定 seed 随机错相。
- 每个 cell 运行 30 秒。该长度已经包含短 context 和逐渐增长的 context；只有跨窗口测试才循环媒体。

Thinker 和 Talker 每处理一个 1 秒 model unit，定义：

```text
RTF = 1000 ms / stage service time
```

实时要求为 RTF `> 1`。严格容量要求所有观测到的 Thinker 和 Talker unit 都低于 1000 ms；任何一次超时即判定该并发数失败。Code2Wav 使用覆盖整个 session、包含静默期的 persistent request，因此其 request wall time 不作为 unit RTF。

正式运行示例：

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 8 --duration-s 30 --phase-window-s 1 --seed 20260829 \
  --connect-stagger-s 0.5 --post-stream-s 4 --gpus 0 1 2 \
  --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --out /tmp/minicpm-hd4-u8.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-server.log \
  --run-json /tmp/minicpm-hd4-u8.json \
  --out /tmp/minicpm-hd4-u8-rtf.json
```

## 阶段四：长 session context 管理

模型最大 context 为 40,960 tokens。估计长度达到 36,000 tokens 后，下一个 unit 重开 KV lineage，只保留：

- system prompt 和 reference audio；
- 上一个完整 AV unit 及其已确认 Thinker 输出；
- 当前 AV unit。

旧 lineage 的 KV blocks 在新 prompt admission 前释放。36k 阈值为最大输出、在途 unit 和估计误差保留约 5k tokens，不生成 summary，也不会阻塞在线路径等待额外模型调用。

180 秒单用户 HD4 测试在 unit 157 发生一次 rollover：

| 指标 | 结果 |
|---|---:|
| 新 prompt | 494 tokens |
| Thinker p50/p95/p99 | 175/407/567 ms |
| Talker p50/p95/p99 | 231/591/621 ms |
| RTF ≤ 1 | 0 |
| session/server error | 0 |

rollover unit 的 Thinker service time 为 175 ms；此前 20 个 unit 均值为 394 ms，此后 20 个为 199 ms。该策略已验证能在线越过 context 上限并清除长历史执行成本。归档结果：`benchmarks/minicpmo/results/long_context_hd4_3gpu_20260829.json`。

## 阶段五：当前容量

当前代码、预热 server、30 秒 HD4 workload 的结果：

| Users | Seed | Thinker p50/p95/p99/max | Thinker misses | Talker p50/p95/p99/max | Talker misses | 结论 |
|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 20260829 | 284/613/706/731 ms | 0/220 | 168/474/687/701 ms | 0/189 | 通过 |
| 8 | 20260829 | 464/888/959/991 ms | 0/237 | 219/460/609/668 ms | 0/208 | 通过 |
| 8 | 20260828 | 407/923/981/994 ms | 0/242 | 200/575/737/838 ms | 0/207 | 通过 |
| 9 | 20260829 | 566/1007/1851/2071 ms | 13/228 | 199/626/1224/1450 ms | 5/230 | 失败 |

测得的严格容量为 8 个 session，但两次 8 用户测试的最大延迟余量都不足 10 ms，因此需要安全余量时应使用 7 个 session。9 用户是第一个明确失效点。

## 阶段六：9 用户失效的根因

9 用户最慢 5% Thinker unit 的平均 service time 为 1493 ms：

| 部分 | 平均时间 | 占比 |
|---|---:|---:|
| 应用提交到 engine admission | 133 ms | 8.9% |
| scheduler 等待 | 0.3 ms | <0.1% |
| runner 执行 | 1302 ms | 87.2% |
| └ 首次 AV prefill forward | 280 ms | 18.7% |
| └ 后续 decode forwards | 1023 ms | 68.5% |
| forward 间控制间隔 | 3 ms | 0.2% |
| 结果暴露 | 54 ms | 3.6% |

关键证据：

1. 同一 9 用户运行中，与其他 session 的 AV prefill 混合执行的 decode forward 为 159/414 ms（p50/p95）；decode-only forward 仅为 24/42 ms，分别相差 6.7× 和 9.8×。
2. 最慢 unit 的 1023 ms decode 中，981 ms 位于 mixed prefill/decode forward。
3. 保留相同 9 用户、270 个 HD4 AV prefill，但强制每个 unit 在 listen decision 结束后，Thinker p50/p95/p99/max 为 215/385/405/436 ms，零 miss。
4. scheduler queue 只有 0.3 ms，因此不是应用串行或 scheduler admission 堵塞。

结论：9 用户失效的直接原因是 Thinker runner 将多模态 prefill 与其他 session 的多步 decode 放入同一 iteration，重复拉长 decode forward。AV prefill 单独可以满足 1 秒预算；prefill 与持续 decode 混合后才越过实时边界。Talker 的超时是次要且主要受上游 Thinker 延迟影响。

GPU 0 的 NVML busy p95 为 83%，memory-I/O busy p95 为 57%。这些是设备忙碌时间，不是 SM occupancy，不能据此声称 GPU 算力或显存带宽已完全饱和。当前证据支持的是 mixed prefill/decode runner 效率与调度问题。

归档结果：`benchmarks/minicpmo/results/capacity_hd4_3gpu_20260829.json`。

## 阶段七：研究基线结论

当前应用设计保留：原生 1 秒 duplex unit、session 内增量 KV、跨 session 无全局 gate、模型自行 listen/speak、36k context rollover。它能够真实暴露持续 AV prefill 与 decode 的并发负载，不应通过应用层串行化隐藏竞争。

下一步 engine 研究应在固定输入 trace 下优化 mixed multimodal-prefill/decode batching 或 deadline/QoS 调度，并比较相同实时 SLO 下的用户容量。

## 快速恢复入口

| 内容 | 路径 |
|---|---|
| 固定部署 | `benchmarks/minicpmo/deploy_capacity_3gpu.yaml` |
| 多用户 workload | `benchmarks/minicpmo/continuous_av.py` |
| RTF 与 tail 分析 | `benchmarks/minicpmo/analyze_rtf.py` |
| 当前容量归档 | `benchmarks/minicpmo/results/capacity_hd4_3gpu_20260829.json` |
| 长 context 归档 | `benchmarks/minicpmo/results/long_context_hd4_3gpu_20260829.json` |
| MiniCPM 输入聚合 | `vllm_omni/experimental/fullduplex/minicpmo45/input.py` |
| Context rollover | `vllm_omni/experimental/fullduplex/minicpmo45/runtime.py` |
| Thinker 多模态输入 | `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py` |
| Realtime orchestration | `vllm_omni/experimental/fullduplex/openai/runtime_bridge.py` |
