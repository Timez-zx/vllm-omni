# MiniCPM-o 非 P/D 原生全双工工作流

## 阶段一：目标

测量单机四 GPU 的非 P/D 部署能实时维持多少个持续音视频 session，并用相同 workload 与 MiniCPM P/D 部署对照。模型质量不在本实验范围内。

本分支使用 MiniCPM-o 4.5 的原生 duplex 路径：模型持续消费一秒 AV unit，并自行决定 listen 或 speak；应用不插入合成 query，也不强制回答。

## 阶段二：应用设计

```text
每个用户建立一个 WebSocket
  -> 每 200 ms 上传 PCM16 音频，每秒上传一帧视频
  -> 五段音频和一帧视频组成一个原生一秒 unit
  -> 将 unit 追加到该 session 常驻的 Thinker request/KV lineage
  -> Thinker 决定 listen 或 speak
  -> speak 时运行 Talker -> Code2Wav
```

- 不同 session 独立进入 engine；应用层没有全局 admission gate，也不做跨用户 batch。
- 新输入不等待 Talker、Code2Wav 或音频播放。
- 常驻 request 只追加新输入。已有 lineage 直接保存增量 KV，因此关闭通用 prefix cache。
- 较新的累计 session snapshot 可以替代仍在排队的旧 snapshot。新 snapshot 已包含旧输入；analyzer 只有在该累计 generation 真正执行后才计入完成量。
- 估计 context 达到 36,000 tokens 时，下一个 unit 重开 lineage，只保留固定 system/reference context、最近一个完整 AV unit 及其确认后的 Thinker 输出、当前 unit。模型上限为 40,960 tokens。

## 阶段三：四 GPU 非 P/D 部署

| GPU | 作用 |
|---:|---|
| 0 | Thinker LLM |
| 1 | Talker |
| 2 | Code2Wav |
| 3 | Vision Encoder 和 resampler |

配置：`benchmarks/minicpmo/deploy_capacity_4gpu.yaml`。

- Thinker 使用 `max_model_len=40960`、`max_num_batched_tokens=32768`、`max_num_seqs=64` 和同步调度。
- 辅助 GPU 只提供给视觉塔，不改变 Thinker 的 TP 或 world size。
- 视频到达后先做 CPU prepare，再按精确 tensor shape 跨 session 聚合，最多八个样本做一次 vision microbatch。
- 正式 Thinker append 优先消费已完成的视觉 embedding；未命中时会废弃该 cache key，并在 GPU 3 正确补算。迟到的投机结果不能覆盖 session 状态。
- Encoder RPC 绕开繁忙的 Thinker Core 主循环，在进入 sidecar queue 时即确认；API 不等待共享结果队列。GPU 3 sidecar 保持单线程，避免并发调用模型。
- session/reference context cache、CPU 音频准备、异步 stage 输出消费、SDPA 视觉 attention 和 resampler 等适用优化与 P/D 分支一致。NIXL、KV handoff 和 P/D feedback 不属于非 P/D 路径，未迁移。

## 阶段四：容量 workload 与判据

- 输入为真实 960×540 `omni_duplex1.mp4`、对齐的 16 kHz mono 音频和 `HT_ref_audio.wav`。
- 音频每 200 ms 到达，视频为 1 FPS；每个用户每秒产生一个模型 unit。
- `frame_max_side=0`、`max_slice_nums=4`。该视频每帧产生一张全局图和两个局部 crop，即 198 个视觉 scheduler rows；稳态 unit 共 211 rows。
- 用户按 seed `20260839` 在 `[0, 1 s)` 内随机错相。
- 正式测试持续 360 秒并循环媒体；每个 session 都两次越过 36k context 阈值。

容量看长期处理速度，不用单个 unit 延迟判定：

```text
stream RTF = 360 秒输入预算 /
             从第一个客户端 AV admission
             到最后一个 unit 的 Thinker runner 完成之间的 wall time
```

只有每个 session 的 `stream RTF >= 1`、所有预期输入完成且没有用户失败时，该并发数才通过。单轮 Thinker/Talker 延迟只用于诊断 jitter。模型可能自行 listen 而不进入 Talker，Code2Wav request 又覆盖 session 静默期，因此二者不能作为周期性容量时钟。

## 阶段五：最终非 P/D 结果

下表每行都是 360 秒长 session 测试。每个 session 接收 360 个一秒 AV unit，并两次越过 36k context rollover 阈值。

| 用户数 | 完成 unit | Stream RTF min/p50/max | 末端 backlog p99 | 结论 |
|---:|---:|---:|---:|:---:|
| 4 | 1440/1440 | 1.001/1.001/1.002 | -296 ms | 通过 |
| 5 | 1800/1800 | 0.999/0.999/1.000 | 522 ms | 失败 |
| 7 | 2520/2520 | 0.985/0.986/0.987 | 5666 ms | 失败 |
| 8 | 2880/2880 | 0.947/0.948/0.949 | 19981 ms | 失败 |

严格长程容量为 4 个用户。5 用户是按既定 `RTF >= 1` 规则的第一个失败点；它只差约 0.1%，但不能向上取整。

4 用户运行中有 7 个旧输入 snapshot 被后续累计 snapshot 合并，1440 个输入全部完成，因此这是合法的 request replacement，不是数据丢失。trace 中有 8 次 490-row rollover admission，即每个 session 两次。

旧的 30 秒容量表不可直接比较，已由本结果替代：短测试可能在长期队列漂移或多次 context cycle 暴露前结束。

## 阶段六：瓶颈

独立 Encoder 不是容量边界：

- Vision Encoder p50/p95/p99：`27/44/55 ms`；
- GPU 3 utilization mean/p95/p99：`6.7%/32%/41%`；
- 1440 帧中有 1417 帧进入投机 embedding cache，其余 23 帧由正式路径正确补算，没有丢帧；
- 正式路径始终使用已完成的投机结果或正确补算结果。

4 用户通过点中，Thinker、Talker 和 Code2Wav 的 NVML device-busy mean/p95/p99 分别为 `25.3%/72%/79%`、`2.5%/18%/27%` 和 `10.1%/58%/64%`。这些是 GPU 忙碌时间采样，不是 SM occupancy；下面的瓶颈结论来自 runner trace，而不是把 NVML utilization 当作算力饱和度。

Mixed step 指同一次 Thinker forward 同时包含已有请求的 decode token 和新到达 AV 的 prefill rows。同一次运行的 GPU 0 trace 给出直接对照：

| 用户数 | Mixed decode step 占比 | Decode-only p50 | Mixed p50 | 倍数 | Stream RTF | 末端 backlog p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7.4% | 18.8 ms | 86.2 ms | 4.6x | 1.001 | -296 ms |
| 5 | 13.0% | 19.4 ms | 81.3 ms | 4.2x | 0.999 | 522 ms |
| 8 | 27.8% | 21.3 ms | 102.2 ms | 4.8x | 0.947 | 19981 ms |

该 trace 直接证明：在共享 Thinker GPU 上，AV prefill 会把 decode iteration 拉长约 4-5 倍。并发升高后 mixed-step 占比增加，Thinker 长期进度低于输入速度，backlog 持续积累。反向的“decode 拖慢 prefill”符合共享执行资源的预期，但本实验没有对它做独立隔离测量。

因此，当前非 P/D 的容量限制是多模态 prefill 与多步 decode 共享 GPU 0；不是视觉预处理、Talker、Code2Wav 或应用层全局串行。

## 阶段七：与 P/D 对照

两组实验使用相同模型、逐字节一致的 MP4、HD4 输入、一秒 cadence、seed、360 秒长度和 36k rollover 策略。

| 部署 | 四 GPU 分配 | 已验证实时点 |
|---|---|---:|
| 非 P/D | Thinker / Talker / Code2Wav / Encoder | 4 用户 |
| P/D | Thinker-P / Thinker-D / Talker / Code2Wav+Encoder | 15 用户 |

P/D 运行完成 5400/5400 个 unit，每个 session 的 stream RTF 均为 `1.002`。P 和 D service p50/p95/p99 分别为 `310/849/1060 ms` 与 `299/790/1096 ms`。

这是部署级对照，不是相同 Thinker GPU 资源下的效率结论：P/D 给 Thinker 分配两个 GPU；非 P/D 只有一个 Thinker GPU，并把第四张卡独占给 Encoder。它仍直接回答了四 GPU setup 的容量问题，并说明对该 workload 分离 prefill/decode 可以显著增加可持续 session 数。

## 阶段八：复现

启动非 P/D server：

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 MINICPMO45_LOG_PREP_DIAG=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

运行一个容量 cell 并分析：

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 4 --duration-s 360 --phase-window-s 1 --seed 20260839 \
  --connect-stagger-s 0.5 --post-stream-s 30 --close-timeout-s 480 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --gpus 0 1 2 3 \
  --out /tmp/minicpm-nonpd-u4x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-nonpd-server.log \
  --run-json /tmp/minicpm-nonpd-u4x360.json \
  --out /tmp/minicpm-nonpd-u4x360-analysis.json
```

关键文件：

- `benchmarks/minicpmo/deploy_capacity_4gpu.yaml`
- `vllm_omni/deploy/minicpmo_4_5_4gpu.yaml`
- `benchmarks/minicpmo/continuous_av.py`
- `benchmarks/minicpmo/analyze_rtf.py`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u4_360s_20260831.*.json`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u5_360s_20260831.*.json`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u8_360s_20260831.*.json`
- `benchmarks/minicpmo/results/pd_4gpu_hd4_u15_360s_20260830.*.json`
