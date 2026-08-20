# vLLM-Omni 实时多用户 Serving：研发流程与当前结论

本文记录 `thinker-talker-vllm` 的研究路径、可复现实验和当前结论。目标不是改进模型质量，而是在满足语音交互体验的前提下，用更少的 GPU 服务更多并发会话。

由于没有可本地部署的开源 realtime 模型，本项目用 Qwen3-Omni 的 thinker → talker → code2wav 流水线模拟 realtime 服务。它不是 Seed Realtime 或 Gemini Live 的等价实现，但足以研究长期会话、KV 占用、流水线调度和多租户尾延迟。

## 先统一口径

- **有状态增量**：一个 WebSocket session 对应一个持续的 engine request；新一轮只追加增量，历史保留在 KV cache。
- **无状态全量重放**：每轮创建新 request，并重新 prefill 全部历史。
- **有状态但释放 KV**：应用保留 session 和历史；每轮结束释放 engine request，下轮从有界历史重建。

核心体验目标是 TTFA p99 < 1 s、可闻卡顿 p99 < 50 ms。当前 TTFA 从发送 query 开始，不包含产品中的 VAD/endpointing 时间。

当前主实现选择有状态增量，因为它最接近 realtime 产品的服务语义，也最适合研究长期 KV 和多租户调度。另两种模式作为计算量与显存占用的对照，不预设谁在所有负载下都更优。

所有容量数字必须同时绑定 commit、部署 YAML、硬件、workload、随机种子和指标口径。不同分支或不同预缓冲设置的结果不能直接比较。

## 阶段一：确定实现路线

上游 vLLM-Omni 已提供 Qwen3-Omni 的 thinker → talker → code2wav 流水线和视频 WebSocket 入口，但推理单位仍是逐轮 request：收到 query 后组装 prompt，生成一次回复，然后结束。它不提供面向 realtime 产品的长期 session 语义。

缺少的能力包括：

- 一个连接内跨多轮持续存在的 engine request 和 KV；
- 音频、视频到达时增量 prefill，而不是等 query 后一次性处理；
- session、turn、request epoch 和输出 segment 的明确身份；
- 长上下文 roll/compression，以及断连、重建和迟到输出隔离；
- 面向长期在线用户的 KV/slot 准入与多租户保护；
- 以 TTFA 和播放卡顿为目标的 realtime workload 与指标。

本分支选择“应用层长期 session + vLLM engine 增量 request”这条路线，原因是：

1. 每轮只 prefill 新增内容，避免无状态全历史重放的重复计算。
2. KV 生命周期、空闲显存占用和多用户竞争都真实暴露给 engine，适合研究 serving 优化。
3. 到达式 prefill 可以把媒体编码移出 TTFA 关键路径。
4. WebSocket session、turn fence 和生命周期事件更接近 Seed Realtime、Gemini Live 一类产品接口。
5. 不依赖模型原生 realtime 能力，可以在现有开源 Qwen3-Omni 上完成可控实验。

这是一种服务系统近似，不是原生 duplex/realtime 模型：交互仍按 turn 结束，模型级全双工生成、语义级 barge-in 和模型内部连续状态不在当前范围内。

有状态增量是主实现；无状态全历史重放和“应用有状态、每轮释放 KV”只作为对照，用来分别量化重复 prefill 成本和空闲 KV 成本。

## 阶段二：把逐轮请求改成长会话

原生逐请求模式适合 API 基线，但不能代表 realtime session：它没有长期 request、持续 KV、到达式输入和跨轮生命周期。

当前分支增加了：

- 一个 session 对应一个可续的 engine request；
- turn 只提交新文本、音频和视频，不重复 prefill 已有 KV；
- 视频和麦克风音频可在到达时提前 prefill，移出 TTFA 关键路径；
- 达到 talker token 上限后 roll request，并用最近文本恢复会话；
- thinker 上下文通过 shadow request 预热、压缩和切换；
- 按共享 KV pool、活跃 session 和 `max_num_seqs` 做准入，过载时拒绝新会话。

这是应用层 session 语义，不等于模型原生支持 realtime。代价是 KV 在用户静默时仍驻留，单个 request 故障影响整个 session，barge-in 不能简单 abort，roll/compression 也只能保留有界历史。

## 阶段三：控制多模态与传输成本

- 帧下采样到不超过 640×352；相对 1280×720，视觉 token 约减少 4 倍。
- `max_frames=8`，缓冲满时丢最旧帧。
- 相似帧过滤同时设置 `min_gap` 和 `max_gap`，分别限制高运动成本和静止画面失明。
- query 末尾追加最新帧，保证“现在”靠近问题，而不是单纯提高帧率。
- JPEG 解码和缩放移到预热子进程池，避免阻塞 WebSocket event loop。
- thinker→talker 只传 talker 需要的文本和 decode 字段，不携带视觉载荷与完整输出 ID。
- stage 间使用常驻双槽共享内存 mailbox，并在调度线程内联发送，减少逐 chunk 生命周期和线程跳转。

这些改动降低 application 和 connector 的固定开销，不改变 vLLM 的调度策略。

## 阶段四：修正 pipeline 调度节奏

目标是让每个活跃语音流每 80 ms 获得一个 codec frame。

最关键的修正是 scheduler 内联接收 chunk。原路径会把等待 chunk 的 request 移出 running，导致它至少错过一个调度轮次；修正后，56 会话下旧口径 deadline miss 从 20.1% 降到 0.9%。

其他已验证改动：

- windowed vocoder：输出保持一致，减少重复卷积；
- lean decode payload：不再传接收端不读取的隐藏状态和完整输出 ID；
- benchmark 客户端关闭 WebSocket deflate：200 会话 TTFA p99 1654 → 728 ms。服务端保持原行为。

WebSocket 结果说明：共享 event loop 上的串行固定成本会先推高 p99，而不是先打满整机 CPU。

## 阶段五：优化 talker 与 code2wav

### 锁页内存传输

talker 每步把一行 CPU payload 搬到 GPU。由普通 host memory 的同步 `.to(device)` 改为锁页缓冲和异步 H2D 后，176 会话的旧口径卡顿比例从 27.2% 降到 2.03%。

### Code predictor 固定大小 KV cache

原实现生成剩余 15 个码本时，每步重复前向整个 17 位置窗口。改为固定 17 槽 KV、每步只前向一个位置后：code predictor 每轮约 35.4 → 8.6 ms，talker 每轮约 76.0 → 34.7 ms，当时的纯音频容量约从 180 提高到 200 会话。该路径数值等价，不保证 bf16 逐位一致。

### 融合 code2wav Snake 激活

把 28 个多 kernel Snake 激活替换为 Triton 融合实现。离线 B=80 时 code2wav 108.7 → 72.0 ms；300 会话线上 rtf 0.84 → 1.01。

它改善中途卡顿，但不能解决 TTFA：即使短路 code2wav，300 会话第一声 p99 仍超过 3 s。第一声发生在首段音频生成之前，不能只靠优化声码器解决。

### 历史容量结论

以下数字来自提交 `f91091a6` 附近的已归档两卡实验，使用旧版 0 ms 预缓冲口径；对应旧 runner、配置和 analyzer 已从当前工作树删除：

| Workload | 最大已测通过容量 | 首个失败点 | 主要限制 |
|---|---:|---:|---|
| 纯音频 | 230 | 240 | TTFA 先于卡顿失败 |
| AV，合成视频 480 ms/帧 | 32 | 36 | thinker 多模态 prefill step 阻塞新请求接纳 |

AV 慢请求自己的输入 token 与快请求接近；差别是它到达时撞上 175–324 ms 的大 prefill step。vLLM 只能在 step 边界接纳请求。调小 `max_num_batched_tokens` 虽缩短单步，却因重复支付 MoE 固定成本而降低总吞吐，TTFA 反而更差。

因此当前 engine 研究重点是限制 step 时长、允许更细粒度接纳、按 deadline 调度，以及减少多模态 prefill 固定成本。仅调 batch token 上限不能同时获得低延迟和高吞吐。

## 阶段六：校准 session baseline 与 benchmark

提交：`b726effe`。这一阶段不修改 scheduler、KV manager 或模型执行，只修正 application 语义和测量可信度。

### Session 正确性

- 新增 `session_id`、server-generated `incarnation`、request `epoch`、`turn_id` 和 `segment_id`。
- 用 typed segment ledger 区分正常 turn 与 shadow seed。
- 有状态路径的 response 和所有 session 生命周期事件都携带身份；客户端拒绝旧 epoch、错误 turn 或跨 segment 输出。
- roll、compression 和断连时清理 ledger，避免迟到输出污染新一轮。

### 三种正式 baseline

| 模式 | 行为 | 用途 |
|---|---|---|
| `stateless_full_replay` | 每轮新 request，`history_max_turns=null` | 测完整历史重复 prefill 的成本 |
| `persistent_incremental` | request/KV 跨轮保留 | 当前主实现 |
| `stateful_evict_rebuild` | 每轮释放 request/KV，下轮从有界文本历史重建 | 测空闲显存收益与重建成本 |

此前 stateless handler 固定使用 `message_history[-2:]`，只包含上一轮，不是完整历史。现在 `history_max_turns` 定义为：`null` 全历史、`0` 无历史、正整数为最近 N 轮；默认仍为兼容性的 1 轮。

### 唯一容量 workload

容量测试只保留一种目标场景：持续 AV session。每个用户使用一条长期 WebSocket，行为与浏览器一致：

- 视频在整个 session 内每 500 ms 上传一帧；
- 麦克风在用户聆听、思考和说话期间每 200 ms 上传 PCM；
- assistant 第一段音频到达后暂停麦克风，等实际播放结束，再保留 300 ms echo guard；
- 每轮使用真实 16 kHz mono PCM16 语音；同一 session 固定 speaker，录音不循环；
- 语音末尾追加 700 ms endpoint silence，然后发送空文本 `video.query`，问题语义只来自音频；
- 回复实际播放完后再进入 think time；think time 为确定性长尾分布，中位数约 3 秒，范围 1–12 秒；
- 使用固定视频序列，但每个用户从不同 offset 开始。

三种 session 策略使用同一个由 seed 生成的 `workload_plan.json`。计划、语音 corpus、视频帧集、source commit 和 deploy YAML 都写入哈希，确保策略之间只改变 session/KV policy。

客户端按 60 ms prebuffer 重放音频。容量通过条件为：预热轮之后全部 turn 完成、TTFA p99 < 1 s、每轮最大卡顿 p99 < 50 ms、无 protocol mismatch、client error 和 bad engine probe。

旧的按视频内容拆分 matrix、audio-only p99 ladder 和 synthetic AV cell runner 已删除；`probe.py` 中的 synthetic media 只用于协议 smoke test，不参与容量结论。

CPU 测试覆盖真实 WAV manifest、speaker/turn 计划、媒体 cadence、session identity 和 playback timeline。

### 当前缺口

- 正式运行必须提供有足够 speaker/turn 的真实语音 manifest 和一个固定的真实视频帧序列。
- 新 workload 尚未产生 GPU 容量结果；旧 0 ms prebuffer 和其他 workload 的数字只保留为历史诊断，不能与新结果直接比较。

## 当前标准实验流程

1. 固定 source commit、deploy YAML 和硬件；结果必须记录 commit、dirty 状态和 YAML SHA256。
2. 固定用户数、轮数、语音 corpus 与 turn plan、视频序列与 cadence、think time、stagger 和 seed。
3. 每个 cell 重启 engine，避免残留 KV、request 或故障污染下一组。
4. 三种 session baseline 只改变 session policy。
5. 拐点附近至少使用多个 seed；历史运行间波动为 8–17%，单次小于 20% 的变化不直接宣称有效。
6. 同时报 TTFA、播放卡顿、RTF、timeout、实际接入数、身份错误、GPU 指标和 engine probes。
7. 先用日志证明 workload 与开关确实生效，再讨论容量。

新的正式容量结论只使用 `origin_deploy_3gpu.yaml`、本节定义的 workload、60 ms prebuffer 和 p99 口径；历史结果不并入容量曲线。

## 关键代码

| 内容 | 路径 |
|---|---|
| Session 生命周期与配置 | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Stateless 历史拼接 | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Session/turn/segment identity | `vllm_omni/entrypoints/openai/video_stream_state.py` |
| 多用户 workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| AV workload 计划与媒体加载 | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| AV 容量阶梯 | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| 三种 session baseline | `benchmarks/live_agent/web_client/run_session_baselines.sh` |
| 播放时间线 | `benchmarks/live_agent/playback_metrics.py` |
| 正式三卡部署 | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| 运行机制验证 | `benchmarks/live_agent/analysis/verify_run.py` |
| Scheduler/connector 开关 | `vllm_omni/core/sched/runtime_flags.py` |
| Chunk transport | `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` |
| Code predictor KV | `vllm_omni/model_executor/models/common/qwen3_code_predictor.py` |
| Code2wav/Snake | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py` |
