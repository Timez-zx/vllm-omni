# vLLM-Omni 实时多用户 Serving 工作流

## 目标与当前决定

目标是在 TTFA 和语音流畅度达标的前提下，用更少 GPU 服务更多持续音视频会话。研究对象是 engine 的容量、调度和尾延迟，不是模型质量。

没有可本地部署、接口和 Seed Realtime 或 Gemini Live 等价的开源 realtime 模型。因此本分支用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似目标产品：客户端持续上传音视频，但模型仍按 turn 回答。该近似足以研究多模态 prefill、语音 decode、KV cache 和多租户竞争，但不代表模型原生全双工或语义级 barge-in。

当前架构决定：

- WebSocket 应用维护 session、完整多模态对话和媒体接收状态。
- 每一轮创建一个新的、有限生命周期的普通 engine request。
- Engine 在请求内维护 KV；请求结束后只允许保留可淘汰的 prefix/KV cache。
- 每轮仍提交完整 canonical history；prefix cache 命中只减少计算，不影响语义正确性。
- 不再把跨轮生命周期塞进同一个 resumable engine request。

这比 engine 长期持有 session 更适合作为研究基线：它符合通用 engine 的 request 抽象，cache miss 只影响性能，并能直接接入 routing、replication 和 P/D 分离。代价是应用每轮都要组装和预处理历史，prefix cache 淘汰后会重新 prefill；这些成本应被明确测量，而不是靠私有 engine 状态隐藏。

## 阶段一：应用层基线

当前实现位于 `vllm_omni/entrypoints/openai/video_stream_base.py` 和 `serving_video_stream.py`。

- 每个 `video.query` 生成唯一的 `video-<uuid>` request ID。
- 应用保存已完成的 user/assistant turn；user history 保留原始文本、音频和选中视频，不退化为纯文本摘要。
- 本轮已接收的音视频只消费一次；生成期间到达的数据进入下一轮。
- 每轮重新渲染完整 canonical prompt。Thinker 开启 prefix caching，复用与上一轮完全一致的前缀。
- prompt 达到 49,152 tokens 时，只按完整 turn 从最旧处删除，直到不超过 16,384 tokens。压缩后形成新的 cache lineage。
- 视频最多保留并提交最近 8 帧，分辨率不超过 640×352。
- similarity filter 阈值 0.95，freshness gap 为 `[0,4]`；相似帧可丢弃，但连续过滤 4 帧后强制保留一帧。
- JPEG 解码、缩放和 thumbnail 生成在子进程池完成，避免阻塞 WebSocket event loop。
- 流式音频 DELTA 逐块转发；客户端用播放时间线判断启动和卡顿。

已删除的旧路径包括：跨轮 persistent request、arrival-time prefill-only append、Talker 45k rolling、Thinker shadow compression、session epoch/segment ledger，以及依赖这些机制的 benchmark 和诊断脚本。上游通用 streaming/resumable 能力仍保留，但当前应用不使用。

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
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条真实 16 kHz mono 录音，末尾追加 700 ms endpoint silence；query 文本为空，问题语义来自音频。
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

`analysis/verify_run.py` 还会检查：每个 turn 对应一个唯一 finite request、Thinker prefix cache 已开启、后续轮次出现实际 prefix hit、部署确为三阶段独立进程。

## 阶段五：旧架构实验归档

以下结果只描述已删除的 persistent-request + arrival-prefill 实现，不能并入新架构容量曲线。

30 轮 continuous-AV 容量实验中，两个 seed 均为 8 用户通过、16 用户失败：8 用户 TTFA p99 为 479–541 ms、playback-start p99 为 824–972 ms；16 用户 TTFA p99 为 1005–1178 ms、playback-start p99 为 2062–2117 ms。失败来自播放启动 tail，不是 timeout 或播放中断。

16 用户公平 prefill-timing RCA 固定了相同客户端 trace 和逐轮 media ledger：三组均提交 2,358 个视频 occurrence、36,884,706 audio bytes，0 丢帧、0 replay slip。

| 旧实现提交方式 | Prefill chunks | TTFA p99 | Playback-start p99 | 第二块 gap p99 | Stall p99 |
|---|---:|---:|---:|---:|---:|
| Arrival incremental | 2783 | 1283 ms | 2337 ms | 1240 ms | 0 |
| Video query-time | 967 | 2353 ms | 3172 ms | 1371 ms | 0 |
| All query-time | 160 | 2416 ms | 2946 ms | 2011 ms | 223 ms |

结论仅限旧实现：把同量媒体集中到 query 边界会制造更强的 Thinker prefill burst；之前看似改善的结果由 8-frame buffer 丢帧混杂，不能成立。日志同时表明 Talker 的长等待大多与 Thinker step 重叠，Code2Wav 不是主要 tail 来源。这支持继续研究 Thinker prefill/decode 干扰和 P/D 分离，但不能预言新 finite-request 架构的容量。

归档：

- `/home/ubuntu/data/results/archive/2026-08-20_continuous_av_capacity_rca/`
- `/home/ubuntu/data/results/archive/2026-08-20_prefill_timing_fair_rca/`

## 阶段六：下一轮实验

2026-08-21 的两轮 direct smoke 已通过：prompt 分别为 280/314 tokens，两个 request ID 不同，Thinker prefix hit 为 0/288 tokens，两轮均生成完整语音。此机缺少 `nvcc`，smoke 临时使用 `--attention-backend TRITON_ATTN`；该结果只验证功能，不是容量数据。正式实验必须先固定 FlashInfer 环境或明确采用 Triton，二者结果不能混用。

当前代码尚无新的正式容量数字。下一步必须在 clean commit 上重新建立 8 → 16 → 32 容量曲线：

1. 先确认每轮 request ID 唯一，第二轮以后 prefix hit 非零，实际处理帧数符合配置。
2. 在首个失败点同时记录 TTFA/playback tail、Thinker prefill/decode step、Talker 等待、GPU SM active、显存和 KV cache 使用。
3. 区分“GPU 真正在计算”与“显存占满但 GPU 空闲”。
4. 再以同一 workload 和媒体 trace 比较 P/D 分离；不因实现难度降低实验优先级。

只有 commit、YAML、输入哈希、seed、用户数、轮数、预缓冲和 SLO 口径全部一致的结果才可直接比较。拐点附近至少跑两个 seed；小于历史运行波动的差异不作结论。

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
