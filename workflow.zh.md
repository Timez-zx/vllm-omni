# MiniCPM-o 原生全双工 P/D 工作流

## 目标

测量单机在实时约束下可持续服务的音视频 session 数量，并定位容量瓶颈。模型质量不在本实验范围内。

## 阶段一：最终 serving 设计

```text
每个用户建立一个 WebSocket
  -> 每 200 ms 上传 PCM16 音频，每秒上传一帧视频
  -> 聚合成一个原生 1 秒 model unit
  -> Thinker-P 只把新增 AV unit 追加到该 session 的 KV lineage
  -> 将新增 KV delta 交给 Thinker-D
  -> Thinker-D 完成有限长度的自回归 decode，并决定 listen/speak
  -> D 输出回写到下一轮 P lineage
  -> speak 时运行 Talker -> Code2Wav
```

- 应用层维护 session、输入缓冲、重连和输出状态；engine KV 是可丢弃的执行状态。
- 不同 session 独立进入 engine；应用层没有全局 gate，也不做跨用户 batch。
- 媒体预处理可以提前，但 `D(i-1)` 必须先于 `P(i)` 完成，因为其 Thinker 输出属于下一轮 lineage。
- 下一轮 Thinker 不等待 Talker 或 Code2Wav。
- 估计 context 达到 36,000 tokens 时重开 lineage，只保留 system/reference input、上一完整 AV unit 及其确认后的 Thinker 输出、当前 unit。模型上限为 40,960 tokens。

部署：

| Stage | GPU | 作用 |
|---|---:|---|
| Thinker-P | 0 | 增量多模态 prefill |
| Thinker-D | 1 | 有限长度自回归 decode |
| Talker | 2 | 生成语音 code |
| Code2Wav + Vision Encoder | 3 | 生成波形和无状态视频编码 |

P/D 使用 `NixlDeltaPushConnector`。D 保留 prefix KV，每个 unit 只传输新增的 block-aligned KV suffix。

## 阶段二：正式 workload 与容量判据

- 循环真实 MP4：960×540 视频、对齐的 16 kHz mono 音频和 reference audio。
- 音频每 200 ms 到达，视频为 1 FPS；模型以 1 Hz 消费 1 秒 unit。
- `max_slice_nums=4`，每个视频 unit 使用 HD4 路径。
- 15 用户、360 秒、`[0, 1 s)` 随机相位、seed `20260839`。
- 共 5,400 个输入 unit；每个 session 两次越过 context 阈值。

单轮 RTF 用于诊断 jitter：

```text
unit RTF = 1000 ms / stage service time
```

容量使用长期处理速度：

```text
stream RTF = 已完成的 1 秒输入总量 /
             从首个 input-ready 到最后一个 D 完成的 wall time
```

只有所有 session 的 `stream RTF >= 1`、全部输入 unit 完成且没有用户失败时，容量才通过。单轮超过 1 秒只是 tail miss；如果后续 unit 能追回 backlog，就不属于容量失效。

## 阶段三：最终测量

P/D-only 控制实验设置 `VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC=1`。它保留完整 P 计算、KV-delta handoff、完整有限 D decode 和 D 到下一轮 P 的反馈，只跳过 Talker/Code2Wav；两个下游 stage 的请求数为 0。D 输出长度与完整 pipeline 等价：mean/p95/p99 分别为 3.038/8/8 和 2.982/8/8 tokens。Vision 正式等待 p99 为 1 ms，因此 GPU 3 不阻塞该控制实验。

| 指标 | 完整 pipeline | P/D-only 控制实验 |
|---|---:|---:|
| P service p50/p95/p99 | 310/849/1060 ms | 126/405/686 ms |
| D service p50/p95/p99 | 299/790/1096 ms | 166/833/1110 ms |
| Input-ready 到 D 完成 p50/p95/p99 | 1080/2631/3031 ms | 306/1420/2441 ms |
| 等待前一轮 D p50/p95/p99 | 223/1352/1593 ms | 0.2/643/1223 ms |
| 超过 1 秒的 unit | 2818/5400 | 605/5400 |
| 每 session stream RTF min/p50/p95 | 1.002/1.002/1.002 | 1.002/1.002/1.003 |

两次运行均完成 5,400 个 D unit，无失败，并可长期维持 15 用户。约 0.2% 的 RTF 余量说明该点已接近实测边界。下游会显著放大延迟，但去掉下游后 P/D tail 仍然存在。

## 阶段四：最终根因

同一 session 存在无法消除的模型依赖：

```text
D(i-1) feedback -> P(i) -> KV handoff -> D(i) -> feedback -> P(i+1)
```

一个 session 的 P 可以和另一个 session 的 D 重叠，但同一 session 的 `P(i)`、`D(i)` 和 `P(i+1)` 不能重叠。

P/D-only 最慢 1% unit 的 input-ready-to-D 平均延迟为 2,895 ms：

| 串行部分 | 均值 | 占比 |
|---|---:|---:|
| 等待前一轮 D feedback | 1,319 ms | 45% |
| 当前 P service | 541 ms | 19% |
| 当前 P 完成到 D 完成 | 1,048 ms | 36% |

D 不会让正在执行的 decode 停下来等待独立的 prefill-only request。新 KV 就绪的请求会加入 active decode，形成 mixed activation batch。Decode-only runner step 的 p50/p95/p99 为 20/49/75 ms，mixed step 为 75/165/249 ms。

最终结论：并发增长会提高 P batch 和 mixed D activation/decode 的开销；同一 session 的递归依赖又把前一轮 D 等待、当前 P、handoff 和当前 D 串在一条关键路径上。偶发长 unit 只产生可恢复 jitter；只有 backlog 无法清空、某个 session 的长期 `stream RTF` 低于 1 时，才是容量失效。这来自模型依赖与 engine service time，不是应用层错误串行化。

## 阶段五：复现

启动完整 pipeline：

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

P/D-only 控制实验在 server 环境中增加 `VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC=1`。

运行并分析：

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 15 --duration-s 360 --phase-window-s 1 --seed 20260839 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --close-timeout-s 120 \
  --gpus 0 1 2 3 --out /tmp/minicpm-pd-u15x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-u15-server.log \
  --run-json /tmp/minicpm-pd-u15x360.json \
  --out /tmp/minicpm-pd-u15x360-rtf.json
```

关键文件：

- `benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml`
- `benchmarks/minicpmo/continuous_av.py`
- `benchmarks/minicpmo/analyze_rtf.py`
- `vllm_omni/engine/orchestrator.py`
