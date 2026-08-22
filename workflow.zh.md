# vLLM-Omni 实时多用户 Serving 工作流

## 目标与当前决定

目标是在 TTFA 和语音流畅度达标的前提下，用更少 GPU 服务更多持续音视频会话。研究对象是 engine 的容量、调度和尾延迟，不是模型质量。

没有可本地部署、接口和 Seed Realtime 或 Gemini Live 等价的开源 realtime 模型。因此本分支用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似目标产品：客户端持续上传音视频，但模型仍按 turn 回答。该近似足以研究多模态 prefill、语音 decode、KV cache 和多租户竞争，但不代表模型原生全双工或语义级 barge-in。

当前架构决定：

- WebSocket 应用维护 session、完整多模态对话和媒体接收状态。
- 每个媒体 warm-up 和最终回答都创建新的、有限生命周期的普通 engine request。
- Engine 在请求内维护 KV；请求结束后只允许保留可淘汰的 prefix/KV cache。
- 每轮仍提交完整 canonical history；prefix cache 命中只减少计算，不影响语义正确性。
- 不再把跨轮生命周期塞进同一个 resumable engine request。

这比 engine 长期持有 session 更适合作为研究基线：它符合通用 engine 的 request 抽象，cache miss 只影响性能，并能直接接入 routing、replication 和 P/D 分离。应用缓存已处理的 canonical message blocks，每轮只 render 新消息，再拼出完整 prompt；prefix cache 淘汰后 engine 仍会重新 prefill，但不会迫使应用重复解码和处理全部历史媒体。

## 阶段一：应用层基线

当前实现位于 `vllm_omni/entrypoints/openai/video_stream_base.py` 和 `serving_video_stream.py`。

- 每个 `video.query` 生成唯一的 `video-<uuid>` 回答 request ID。
- 应用保存已完成的 user/assistant turn；user history 保留原始文本、音频和选中视频，不退化为纯文本摘要。
- 应用同时保存 renderer 已处理的 canonical blocks。每轮只处理当前 user block 和生成后的短 assistant block，再合并 token、媒体特征、hash 和修正后的 placeholder offset；engine 仍收到完整历史，而不是 delta request。
- 本轮已接收的音视频只消费一次；生成期间到达的数据进入下一轮。
- similarity/freshness filter 是唯一的视频选择策略；接受后的帧在当前轮全部按到达顺序保留，不再做最近 8 帧滑动或二次采样。
- 每次接受新帧，应用触发或合并进 `video-warm-<uuid>`：完整历史加当前轮累计帧，`output_modalities=["text"]`、Thinker `max_tokens=1`。输出 token 被丢弃，Talker/Code2Wav 不运行，也不向客户端发送 response 事件。
- 同一 session 的 warm-up 串行执行并合并积压快照。后一次请求通过 Thinker prefix cache 复用前一次的完整块，只计算新增帧和未满 block 尾部；cache miss 只增加计算。
- warm-up 是可丢弃的后台优化；query 到达时立即取消本 session 尚未完成的 warm-up，不等待缓存填充。最终请求始终带完整 prompt，取消只影响命中率。
- `video.query` 提交同一媒体前缀加完整 WAV，并正常运行 Thinker → Talker → Code2Wav。音频不做增量切块，保持当前 Qwen 输入语义。
- 实验开关 `enable_audio_arrival_prefill_approximation` 默认关闭。开启后，应用按实际到达顺序保存本轮音视频，把音频封成不可变的 1 秒块；每个块只触发静默 Thinker finite request，query 补上尾块后才允许 decode 和 Talker。该路径只模拟 duplex engine 负载，不与 Qwen 整段音频推理等价。
- 正常回答的 Thinker 上限为 256 tokens，防止模型忽略简短回答提示后生成分钟级语音；arrival warm-up 仍为 1 token。
- 49,152 tokens 是硬压缩阈值；当 prompt 达到 32,768 tokens 时，应用会在回答播放后或 arrival warm-up 阶段提前按完整 turn 压到不超过 16,384 tokens，避免把压缩留到下一次 query 的关键路径。压缩后形成新的 cache lineage。
- 视频分辨率不超过 640×352；通过 filter 的帧没有第二个数量上限或采样步骤。
- similarity filter 阈值 0.95，freshness gap 为 `[0,4]`；相似帧可丢弃，但连续过滤 4 帧后强制保留一帧。
- JPEG 解码、缩放和 thumbnail 生成在子进程池完成，避免阻塞 WebSocket event loop。
- 流式音频 DELTA 逐块转发；客户端用播放时间线判断启动和卡顿。

已删除的旧路径包括：跨轮 persistent request、向同一 resumable request 做 arrival append、Talker 45k rolling、Thinker shadow compression、session epoch/segment ledger，以及依赖这些机制的 benchmark 和诊断脚本。当前 arrival prefill 使用独立 finite request 和可淘汰 prefix cache，不使用 streaming/resumable engine state。

## 阶段二：固定部署

正式实验只使用 `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`：

| Stage | GPU | 关键配置 |
|---|---:|---|
| Thinker | 0 | FP8 weight/KV，prefix caching 开启，priority scheduler |
| Talker | 1 | FP8 weight/KV，按 session conditioning lineage 开启 prefix caching |
| Code2Wav | 2 | 独立进程 |

三阶段独立进程，部署 YAML 在不同用户数之间保持不变。Talker request 仍是有限生命周期，只复用可淘汰的 conditioning prefix cache。当前树中与 request 生命周期无关的 mailbox、vocoder、Snake 和 code-predictor 优化继续保留。

最终回答使用 priority 0，静默 warm-up 使用 priority 10。`run_qwen_server.sh` 自动发现环境内 CUDA toolkit，并拒绝 Thinker 未选择 FlashInfer 的正式运行。多模态 processor cache 使用 API-side `processor_only` 兼容模式，避免 vLLM mirrored LRU 在并发请求下发生 sender/receiver 淘汰顺序分叉。

启动：

```bash
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

## 阶段三：唯一正式 workload

容量测试只模拟持续 AV session。每个用户拥有一条长期 WebSocket：

- 视频在整个 session 内以 2 FPS 上传。
- 每个通过 filter 的帧立即触发或合并进静默 Thinker warm-up；当前轮累计帧保持 append-only。
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条真实 16 kHz mono 录音，末尾追加 700 ms endpoint silence；query 文本为空，问题语义来自音频。
- 音频在 query 时作为一个完整 WAV 加到已 warm 的视频 prefix 后；只有该最终请求会触发 Talker 发声。
- 同一 session 固定 speaker，录音不重复；视频使用固定序列和不同起点。
- 下一轮在回复按 1× 速度播放完成后开始，属于 playback-paced closed loop。
- 用户启动时间在 0–40 秒内确定性错开。

素材由 SLURP 语音和 DAVIS 视频构造。正式 cell 为 30 轮/用户，前 2 轮只预热；用户数按 8、16、32……增长，每个 seed 在首个 SLO 失败点停止。每个 cell 重启 engine。

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
RESULT_PREFIX=finite_request USERS="8 16 32" SEEDS="7 17" \
TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_av_session_ladder.sh
```

`probe.py` 的合成媒体只验证协议，不能产生容量结论。

## 阶段四：指标与判定

- **TTFA**：发送 query 到第一块语音包到达。它受首包切块大小影响，只用于定位传输路径，不作为跨配置体验指标。
- **Audio-ready-500**：客户端累计得到 500 ms 可播放音频的时间；短回复在 `audio.done` 时释放。这是固定的启动 SLO，不能由环境变量修改。
- **Stall max**：按 1× 播放时，某块到达晚于已缓存音频耗尽时间所造成的最大单次断流。
- **RTF deliver**：生成的音频时长 / 交付耗时；用于判断持续供给能力。

容量通过必须同时满足：预热后所有 turn 完成、audio-ready-500 p99 < 1 s、stall-max p99 < 50 ms、无 client/protocol/fatal engine error。TTFA、GPU 利用率、显存和 engine step 数据用于 root-cause，不替代体验 SLO。新结果使用 workload schema 4；验证脚本拒绝混入旧的可变 prebuffer 口径。

`analysis/verify_run.py` 还会检查：每个 turn 对应一个唯一回答 request、warm-up request 唯一且无失败、engine 最终处理帧数与客户端 consumed ledger 一致、Thinker prefix cache 已开启且出现实际命中、部署确为三阶段独立进程。

## 阶段五：8 用户 finite-request 定位结果

2026-08-21 使用 FlashInfer、seed 7、30 轮/用户、前 2 轮预热定位当前 finite-request 路径。旧 persistent 结果只读取归档，不部署、不重跑。

### 默认策略

结果：`/home/ubuntu/data/results/finite_request_foreground_priority_cap256_cbf2226a_flashinfer_20260821/finite_foreground_priority_cap256_seed7_u8`

- 224/224 个测量 turn 成功，无 timeout、client error、preemption 或 recompute。
- TTFA p50/p95/p99：1.84/4.83/6.70 秒；playback-start：2.89/8.65/11.51 秒；不通过容量 SLO。
- 2,958 个 Thinker-only warm-up、240 个回答 request；3,168/3,169 次 prefix-cache 观测命中。
- query→warm-up 交接 p99 16 ms，render p99 767 ms，engine→首文字 p99 2.47 秒，engine→首音频 p99 6.16 秒。
- warm-up 逻辑输入 7,496 万 tokens，但实际 prefix miss 111 万，命中率 98.5%。没有重新计算全部历史；长 context 仍增加 attention、KV 绑定和调度成本。

取消 query 前的 warm-up barrier 后，TTFA p99 从 7.29 秒降至 6.41 秒；增加 foreground/background priority 后为 6.70 秒。单 seed live closed-loop 不能证明回退，但可以证明 priority 不能消除已经运行的 prefill 干扰。

### 与旧归档的公平性

两边 workload plan SHA256 都是 `99dc083388...`，但 engine 输入并不等价：

- 旧 persistent 在约 27.0k–36.7k tokens 滚动，默认只携带 281–858 tokens 的文本 seed；最终 context p50/p95/p99/max 为 14.8k/30.9k/34.5k/36.8k。
- 当前默认在 49,152 tokens 压缩到约 16,384，并保留最近完整多模态 turn；最终 context 为 26.6k/46.3k/48.1k/49.0k。
- live closed-loop 会放大差异：默认当前运行发送 9,027 帧、消费 3,979 帧；旧归档发送 7,367 帧。相同 plan hash 不代表相同媒体轨迹。

当前 TTFA 与逻辑 context 的相关系数为 0.66；`>=40k` context 的 TTFA p50/p95 为 3.15/6.69 秒。不能把默认结果与旧归档直接解释为 request 生命周期差异。

### 当前方案的计算预算控制组

只重跑当前 finite-request 方案，将压缩设为 `32k -> 0`，使 context 包络接近旧归档。该配置用于隔离计算量，不是生产语义策略，因为压缩时会丢弃旧 turn。

修复后的结果：`/home/ubuntu/data/results/finite_request_old_budget_abort_cleanup_cbf2226a_20260821/finite_old_budget_abort_cleanup_seed7_u8`

- 224/224 成功；TTFA p50/p95/p99：0.984/2.50/3.23 秒；playback-start：1.58/3.79/4.87 秒。
- context p50/p95/p99/max：15.0k/29.4k/31.1k/32.0k，已接近旧归档；GPU 0/1/2 平均 SM 活跃度约 20.5%/6.8%/0.4%，也与旧归档 21.4%/7.0%/0.4% 接近。差距不是 GPU 饱和。
- 旧归档 TTFA p50/p95/p99 为 0.244/0.392/0.479 秒，max 0.532 秒；预算对齐后仍有真实差距。

剩余差距主要在 Talker：旧 persistent 跨轮保留 Talker KV；当前每轮创建新的 Talker request，stage 1 未启用 prefix cache，并从本轮完整 Thinker prompt 重建 placeholder/prefill。当前 `history_messages` 与 TTFT→TTFA gap 的相关系数为 0.81；history 从 0–4 条增到 16+ 条时，该 gap p50 从 226 ms 增至 1,111 ms。Thinker 侧也有普通 warm-up request 的 admission、KV 绑定和 chunked-prefill 开销，但不是全部差距。

另修复一处 finite pipeline bug：最终 stage 完成后先 abort 残余上游工作，再清理 request。对齐组的 Talker late output 从 464 条降到 0，TTFA p50/p99 从 1.036/3.540 秒降到 0.984/3.225 秒。它是资源浪费和部分 tail 来源，不是主根因。

### Disposable Thinker lineage handle

实现：应用继续提交完整 canonical prompt，并携带 session lineage、父 revision 和 token LCP。KV cache manager 只保存已完成有限 request 的 block-hash snapshot，不持有活 request 或固定 GPU block；cache miss 自动回退完整 prefill。历史压缩会更换 lineage，Thinker handle 不传入 Talker。

正式结果：`/home/ubuntu/data/results/finite_cache_handle_formal_20260821/cache_handle_formal_seed7_u8`

- 224/224 成功，frame ledger 一致，无 timeout 或 stall。
- TTFA p50/p95/p99：521/944/1265 ms；严格 finite v6 为 518/878/1051 ms，旧 persistent 为 244/392/479 ms。
- 2,532 次 request 复用 hash snapshot，累计跳过约 4,919 万个 prefix token 的重复 hash；render p50 仍为 72.6 ms，Thinker→首文字 p50 仍为 224 ms。
- 结论：hash handle 正确且减少控制面工作，但不是 TTFA 主解；普通 prefix cache 本来已命中绝大部分 GPU KV。

### 消除重复 render 和测量混杂

2026-08-22 将已完成历史保存为应用侧 processed canonical blocks。新一轮只 render 当前 user message；回答完成后只 render assistant message。应用随后拼出完整 token/media prompt，engine request 生命周期和 cache-miss 语义均未改变。Qwen ChatML 的完整 message boundary 是 append-only；其他模板默认回退完整 render。

短验证：`/home/ubuntu/data/results/canonical_render_short_20260822`

- u1×4：4/4 完成；prompt 从 1,261 增至 11,184 tokens，render 仅从 8.1 增至 10.2 ms，无 full-render fallback。
- u8×4：预热后 24/24 完成；audio-ready-500 p50/p99 为 345/442 ms，stall p99 为 0；32 个回答 request 均提交并 commit canonical turn。
- u8 各轮 render p50 为 9.7/11.6/10.7/16.6 ms；没有 arrival failure、multimodal cache error、query failure 或 late output。
- 测量口径固定为 500 ms 可播放音频；当前首包约 537 ms，因此本部署的 TTFA 与 audio-ready-500 数值接近，但以后改变 codec 切块也不能改变 SLO 含义。

结论：此前约 73 ms 的 full-history render 中位数是应用实现开销，现已基本消除。剩余 tail 才适合归因于 engine 的 prefill/decode 竞争、KV 行为和多阶段启动；以上只是正确性短测，不替代 30 轮正式容量实验。

### 8 用户正式容量结果

2026-08-22 使用 seed 7、30 轮/用户、前 2 轮预热运行 schema 4：`/home/ubuntu/data/results/canonical_render_formal_cbf2226a_20260822/canonical_render_formal_seed7_u8`。

- 224/224 个测量 turn 完成，无 timeout、stall、client/protocol error；frame ledger 为 3,317，验证通过。
- audio-ready-500 p50/p95/p99 为 456/854/1,107 ms。p99 超过 1 秒门槛 107 ms，因此 8 用户是当前容量边界；没有继续跑 16 用户。
- TTFT p50/p99 为 239/707 ms；首文字到首音频 gap p50/p99 为 213/673 ms。
- render p50/p95/p99 为 15/46/104 ms；无 full-render fallback。最慢三个 p99 turn 的 render 仅为 46/14/18 ms，应用 render 已不是主因。
- p99 tail 相对中位数的额外延迟中，Thinker 占 72.4%，后续语音路径占 27.6%；p95 tail 分别占 57.2% 和 42.8%。GPU0 SM active p95 在整体窗口为 45%，在 tail 窗口升至 94%；GPU1/2 未同步饱和。
- 2,929 个 finite arrival warm-up 全部成功；3,381/3,400 次 prefix-cache 观测命中，最大命中 32,704 tokens；发生 32 次完整 turn compaction。
- 当前首音频包约 537 ms，已经超过固定 500 ms 阈值，因此本部署中 audio-ready-500 等于首包 TTFA；这不改变跨 packetization 使用固定阈值的测量定义。

结论：8 用户已出现轻微启动 tail 超标，直接 root cause 是并发 Thinker 工作造成的首 token tail；语音流水线贡献部分延迟，但 Talker/Code2Wav 资源没有饱和。下一步应讨论 Thinker prefill/decode 隔离或调度，而不是继续调应用 render 参数。

### 音频 arrival-prefill 近似实验

2026-08-22 实现了显式实验路径：PCM16 每满 1 秒形成稳定媒体 item，音视频按实际到达顺序 append-only；相邻音频块删除内部 `<audio_end><audio_start>`，保留独立媒体 hash 和连续 MRoPE。arrival request 仍是低优先级、Thinker-only、有限生命周期；query 补未满 1 秒的尾块并正常发声。Qwen audio encoder 在约 8 秒窗口内使用双向 attention，因此每块独立编码会改变模型语义；这是 workload approximation，不是等价推理。

单用户 8 轮验证：`/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u1_t8`

- 7/7 个计量 turn 成功，无 processor error 或 stall；audio-ready-500 p50/p99 为 280/344 ms。
- 旧实现的第 6 轮失败没有复现；processed canonical blocks 可以保存累计分块历史。

8 用户冷启动 A/B 均使用 schema 4、6 轮/用户、前 1 轮预热、seed 7、0–8 秒 stagger，workload plan SHA256 均为 `b6160ec3...`：

- query-time 完整 WAV：`/home/ubuntu/data/results/audio_arrival_ab_baseline_20260822/u8_t6`，40/40；audio-ready-500 p50/p95/p99 为 368/475/654 ms。
- audio arrival：`/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u8_t6`，40/40；audio-ready-500 为 343/554/692 ms。
- arrival 将 foreground prefix residual 的 p50 从 109 降至 23 tokens，TTFT p50 从 166 降至 156 ms；但 warm-up 从 614 增至 893，query 撞上并取消 warm-up 的次数从 6 增至 15。stage Thinker p99 从 294 增至 333 ms，stage audio TTFA p99 从 393 增至 429 ms。
- 近似改变了回答分布：输出音频 p95 从 11.5 增至 18.2 秒，闭环 wall time 从 114 增至 137 秒。因此两组输入 plan 相同，但运行中的媒体轨迹不是严格 engine-only replay。

结论：该路径功能正确，并把一部分 query-time 音频工作移到 arrival，但在 8 用户下只有约 25 ms 中位数收益，tail 没有改善；新增 Thinker warm-up contention 抵消了 foreground 节省。它保留为默认关闭的研究 arm，不能替代正式 query-time 完整 WAV baseline。若要求语义等价且真正受益，需要原生 causal/streaming audio encoder 或模型提供的 streaming cache；当前 Qwen 最多只能在完整约 8 秒窗口后安全缓存，对本 workload 的短语音帮助很小。

这也修正了此前的归因：短基线即使在 query 时处理完整 WAV，p99 仍只有 654 ms；30 轮正式结果的 1,107 ms tail 不能主要归因于 query-time 音频，而是长 context 下多用户 Thinker prefill/decode 与 arrival warm-up 竞争。

## 快速恢复入口

| 内容 | 路径 |
|---|---|
| Session 与 finite-request 生命周期 | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Canonical 多模态 history | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Prefix-cache 观测 | `vllm_omni/worker/gpu_model_runner.py` |
| 多用户 workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload 计划与媒体加载 | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| 容量阶梯 | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| 正式部署 | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| 运行验证 | `benchmarks/live_agent/analysis/verify_run.py` |
| GPU 采样 | `benchmarks/live_agent/harness/gpu_sampler.py` |
