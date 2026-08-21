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

这比 engine 长期持有 session 更适合作为研究基线：它符合通用 engine 的 request 抽象，cache miss 只影响性能，并能直接接入 routing、replication 和 P/D 分离。代价是应用每轮都要组装和预处理历史，prefix cache 淘汰后会重新 prefill；这些成本应被明确测量，而不是靠私有 engine 状态隐藏。

## 阶段一：应用层基线

当前实现位于 `vllm_omni/entrypoints/openai/video_stream_base.py` 和 `serving_video_stream.py`。

- 每个 `video.query` 生成唯一的 `video-<uuid>` 回答 request ID。
- 应用保存已完成的 user/assistant turn；user history 保留原始文本、音频和选中视频，不退化为纯文本摘要。
- 本轮已接收的音视频只消费一次；生成期间到达的数据进入下一轮。
- similarity/freshness filter 是唯一的视频选择策略；接受后的帧在当前轮全部按到达顺序保留，不再做最近 8 帧滑动或二次采样。
- 每次接受新帧，应用触发或合并进 `video-warm-<uuid>`：完整历史加当前轮累计帧，`output_modalities=["text"]`、Thinker `max_tokens=1`。输出 token 被丢弃，Talker/Code2Wav 不运行，也不向客户端发送 response 事件。
- 同一 session 的 warm-up 串行执行并合并积压快照。后一次请求通过 Thinker prefix cache 复用前一次的完整块，只计算新增帧和未满 block 尾部；cache miss 只增加计算。
- `video.query` 提交同一媒体前缀加完整 WAV，并正常运行 Thinker → Talker → Code2Wav。音频不做增量切块，保持当前 Qwen 输入语义。
- prompt 达到 49,152 tokens 时，只按完整 turn 从最旧处删除，直到不超过 16,384 tokens。压缩后形成新的 cache lineage。
- 视频分辨率不超过 640×352；通过 filter 的帧没有第二个数量上限或采样步骤。
- similarity filter 阈值 0.95，freshness gap 为 `[0,4]`；相似帧可丢弃，但连续过滤 4 帧后强制保留一帧。
- JPEG 解码、缩放和 thumbnail 生成在子进程池完成，避免阻塞 WebSocket event loop。
- 流式音频 DELTA 逐块转发；客户端用播放时间线判断启动和卡顿。

已删除的旧路径包括：跨轮 persistent request、向同一 resumable request 做 arrival append、Talker 45k rolling、Thinker shadow compression、session epoch/segment ledger，以及依赖这些机制的 benchmark 和诊断脚本。当前 arrival prefill 使用独立 finite request 和可淘汰 prefix cache，不使用 streaming/resumable engine state。

## 阶段二：固定部署

正式实验只使用 `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`：

| Stage | GPU | 关键配置 |
|---|---:|---|
| Thinker | 0 | FP8 weight/KV，prefix caching 开启 |
| Talker | 1 | FP8 weight/KV，prefix caching 关闭 |
| Code2Wav | 2 | 独立进程 |

三阶段独立进程，部署 YAML 在不同用户数之间保持不变。Talker 不用跨轮 prefix cache；它处理每个有限请求的语音生成。当前树中与 request 生命周期无关的 mailbox、vocoder、Snake 和 code-predictor 优化继续保留。

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

- **TTFA**：发送 query 到第一块语音到达。用于定位服务路径，不直接等于用户听到声音。
- **Playback start**：客户端累计到 1.4 秒音频后开始播放；短回复在 `audio.done` 时释放。当前首块约 217 ms，通常要等后续块。
- **Stall max**：按 1× 播放时，某块到达晚于已缓存音频耗尽时间所造成的最大单次断流。
- **RTF deliver**：生成的音频时长 / 交付耗时；用于判断持续供给能力。

容量通过必须同时满足：预热后所有 turn 完成、playback-start p99 < 1 s、stall-max p99 < 50 ms、无 client/protocol/fatal engine error。TTFA、GPU 利用率、显存和 engine step 数据用于 root-cause，不替代体验 SLO。

`analysis/verify_run.py` 还会检查：每个 turn 对应一个唯一回答 request、warm-up request 唯一且无失败、engine 最终处理帧数与客户端 consumed ledger 一致、Thinker prefix cache 已开启且出现实际命中、部署确为三阶段独立进程。

## 阶段五：8 用户基线结果与结论

2026-08-21 在 clean commit `cbb3556c` 上完成正式 8 用户 cell。部署为固定的 `origin_deploy_3gpu.yaml`，backend 为 `TRITON_ATTN`，seed 7，30 轮/用户，前 2 轮预热。结果位于 `/home/ubuntu/data/results/finite_request_capacity_cbb3556c_triton_20260821/finite_request_seed7_u8`。

运行前修复了一个服务层错误：WebSocket handler 显式复制 deploy sampling params 后绕过了 AsyncOmni 的 `DELTA` 转换，导致 Code2Wav 每次发送累计 waveform，而客户端按协议将其当作新增音频播放。旧运行中的 345.8 秒音频、超长 session 和后续 context overflow 均受此错误污染，不能用于容量结论。修复后首块恒为 5,205 samples，其余 1,141 个非首块均不超过 48,000 samples；三卡 smoke 和 38 个相关测试通过。

8 用户结果：

- 224/224 个测量 turn 成功，无 timeout、client error、warm-up failure、preemption 或 recompute。
- TTFA p50/p95/p99：2.42/7.65/9.07 秒。
- Playback-start p50/p95/p99：4.37/12.77/17.48 秒。
- Stall-max p50/p95/p99：0/9.8/1,658 ms。因此该 cell 不通过容量 SLO，停止 16 用户测试。
- 共 2,830 个静默 warm-up 和 240 个回答 request；prefix cache 观测 3,068/3,069 次命中。
- Thinker GPU 的 SM active 总体 p50/p95 为 39%/98%，playback-tail 窗口为 43%/99%。Talker 总体为 1.1%/19.8%、tail 为 0%/14%；Code2Wav 总体为 0%/1.9%。显存占用不能替代这些 active 指标。

结论：8 用户下的主要瓶颈仍是 Thinker。arrival prefill 和回答 decode 共享 GPU0；等待首音频的并发回答从 0 增至 4–5 个时，TTFA p50 从约 1.37 秒升至 7.1–7.6 秒。第一块音频之后的等待也不能解释为 Talker/Code2Wav 饱和：后两级大多空闲，它们在等待 Thinker 继续提供 hidden/text。下一项直接实验应固定本 workload、媒体、YAML 和 SLO，在 8 用户比较 Thinker P/D 分离；若需要精确报告非 P/D 容量，再补 1/2/4 用户 cell。拐点附近至少跑两个 seed。

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
