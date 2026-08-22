# vLLM-Omni 实时多用户 Serving 工作流

## 目标与边界

目标是在 TTFA 和语音连续性达标的前提下，用更少 GPU 服务更多持续音视频会话。研究重点是 engine 的容量、调度、KV cache 和尾延迟，不是模型质量。

目前没有可本地部署、交互方式与 Seed Realtime 或 Gemini Live 等价的开源 realtime 模型。本分支使用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似目标场景：客户端持续上传音视频，模型按 turn 回答。它可以产生合理的多模态 prefill、语音 decode 和多用户竞争负载，但不代表原生全双工或语义级 barge-in。

## 阶段一：应用基线

当前采用“应用有状态、engine request 有限生命周期”的设计：

```text
持续音视频到达
  → 视频帧触发静默 Thinker finite request，预热 prefix KV
  → 用户说完，提交完整 canonical history + 完整 WAV
  → Thinker → Talker → Code2Wav
  → 回答结束，request 销毁
  → 应用保存本轮，下一轮创建新 request ID
```

关键约束：

- WebSocket 应用维护 session、媒体接收状态和完整 canonical 多模态历史。
- 每个静默 warm-up 和最终回答都是新的普通 finite request；engine 不持有跨轮活 request。
- 每轮向 engine 提交完整 canonical prompt。prefix/KV cache 可淘汰；cache miss 只增加 prefill，不影响正确性。
- 应用缓存已处理的 canonical message blocks，每轮只 render 新 user/assistant block，再拼接完整 prompt，避免重复处理全部历史媒体。
- 视频以 append-only 方式进入本轮；similarity/freshness filter 决定是否接受，接受后不再做 8 帧滑动淘汰或二次采样。
- 新视频帧触发或合并进低优先级 `video-warm-<uuid>`。它只运行 Thinker、`max_tokens=1`、不返回文字、不进入 Talker。query 到达时立即取消未完成 warm-up，不等待 cache。
- 用户音频默认在 query 时作为一个完整 WAV 输入，保持 Qwen 的整段音频语义。只有最终 query 会触发语音回答。
- 回答期间新到达媒体归入下一轮；本轮媒体只消费一次。
- 正常回答最多生成 256 个 Thinker tokens；视频不超过 640×352，JPEG 处理放在子进程池中。
- history 硬阈值为 49,152 tokens；16,384-token headroom 使应用通常在约 32,768 tokens 提前按完整 turn 压缩到不超过 16,384 tokens，并更换 cache lineage。

`enable_audio_arrival_prefill_approximation` 默认关闭。它把音频封成 1 秒块做静默 arrival prefill，只用于模拟 duplex engine 负载，不保证与 Qwen 整段音频推理语义等价。

旧的跨轮 persistent request、resumable append、Talker 45k rolling、Thinker shadow compression 和 session ledger 已删除。当前基线与通用 engine request 抽象兼容，也更容易接入 routing、replication 和 P/D 分离。

## 阶段二：固定部署

正式实验只使用 `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`：

| Stage | GPU | 配置 |
|---|---:|---|
| Thinker | 0 | FP8 weight/KV、prefix cache、priority scheduler |
| Talker | 1 | FP8 weight/KV、session-isolated conditioning prefix cache |
| Code2Wav | 2 | 独立进程 |

最终回答优先级为 0，静默 warm-up 为 10。部署 YAML 在不同用户数和 session-policy 对照之间保持不变。`run_qwen_server.sh` 自动定位 CUDA toolkit，并检查 Thinker 使用 FlashInfer。多模态 processor cache 使用 API-side `processor_only` 模式。

## 阶段三：正式 workload

唯一正式 workload 是持续 AV session：

- 每个用户维持一条长期 WebSocket。
- 视频全程以 2 FPS 上传；通过 filter 的帧立即触发或合并进静默 Thinker warm-up。
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条不重复的真实 16 kHz mono SLURP 录音，末尾追加 700 ms endpoint silence；query 文本为空。
- 音频在 query 时作为完整 WAV 加到已 warm 的视频 prefix 后，只有该请求触发 Talker。
- 每个 session 固定 speaker；DAVIS 视频使用固定序列和不同起点。
- 下一轮在上一轮语音按 1× 播放完成后开始，形成 playback-paced closed loop。
- 用户启动时间在 0–40 秒内确定性错开。

正式 cell 为 30 轮/用户，前 2 轮预热。用户数按 8、16、32……增长，每个 seed 在首个 SLO 失败点停止；每个 cell 重启 engine。`probe.py` 的合成媒体只验证协议，不能用于容量结论。

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

## 阶段四：指标与通过条件

- **TTFA**：query 到第一块音频包。它受首包大小影响，只用于单次部署内定位。
- **Audio-ready-500**：累计收到 500 ms 可播放音频的时间，是跨 packetization 的启动指标。
- **Stall max**：按 1× 播放时最大的单次断流。
- **RTF deliver**：生成音频时长 / 交付耗时，用于判断持续供给能力。

容量通过必须同时满足：

- 预热后的所有 turn 完成；
- Audio-ready-500 p99 < 1 s；
- Stall-max p99 < 50 ms；
- 无 client、protocol 或 fatal engine error。

结果必须使用 workload schema 4，并运行 `benchmarks/live_agent/analysis/verify_run.py`。验证器检查 finite request 唯一性、arrival warm-up、frame ledger、实际 prefix-cache 命中和三阶段独立进程。

## 阶段五：当前基线与关键结论

最新正式对照的精确 setup：

- source：clean commit `854535bb85789a882ff5995362e5528a5f52f83d`；
- deploy：`origin_deploy_3gpu.yaml`，SHA256 `ae7cbeb615b24ee8654cf6c867887b2920324fc81ec93be6aaaa8995c55e4e24`；
- workload：schema 4、seed 7、8 用户×30 轮、前 2 轮预热，plan SHA256 `99dc083388931ffcbf9479a7f4344613229350546824371fec3744b548db0ed4`；
- 默认组：不设置 `MU_SESSION_CFG_JSON`；
- 对齐组：`MU_SESSION_CFG_JSON='{"context_window_trigger_tokens":32000,"context_window_target_tokens":0,"context_window_compaction_headroom_tokens":0}'`。

| 路径 | context p50/p95/p99/max | Audio-ready-500 p50/p95/p99/max |
|---|---:|---:|
| 当前默认 history | 22.5k/32.0k/32.4k/32.8k | 478/723/1186/1215 ms |
| 当前 `32k -> 0` 对齐控制 | 16.8k/30.6k/31.8k/31.9k | 410/635/796/883 ms |
| 旧 persistent 归档 | 14.8k/30.9k/34.5k/36.8k | 515/740/824/900 ms |

结果路径：

- 默认：`/home/ubuntu/data/results/current_default_854535bb_20260822/current_default_seed7_u8`
- 对齐：`/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`
- persistent 归档：`/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`

三组比较得到以下结论：

1. **不能比较原始 TTFA。** persistent 首包只有约 217 ms 音频，其原始 TTFA p50/p95/p99 为 244/392/479 ms；表中按音频 delta 累计到固定 500 ms 后重算为 515/740/824 ms。
2. **history policy 会显著影响 tail。** 同一提交中，对齐控制将 p99 从 1186 ms 降到 796 ms；默认组 p99-tail 的 GPU0 SM-active p95 为 94.7%，对齐组为 52.4%。差异来自 context 长度、保留的多模态内容、compaction/cache-lineage churn 和闭环轨迹，不能只解释成 token 数量。
3. **finite-request 生命周期不是性能问题。** context 对齐后，当前 p99 为 796 ms，persistent 为 824 ms，已处于同一水平。应用维护 session、engine 每轮处理 finite request、依靠可淘汰 prefix/KV cache 的架构成立。
4. **应用层重复工作已不再是主要 tail。** processed canonical blocks 消除了完整历史重复 render；warm-up 不阻塞 query，残余上游工作会在回答完成后清理。默认组 tail 同时包含 Thinker 延迟和首文字后的流水线等待，但 Talker/Code2Wav GPU 没有饱和。
5. **`32k -> 0` 不是生产策略。** 它会丢失已完成 turn，只用于证明 request 生命周期和计算包络。单 seed live closed-loop 也不是 bit-identical replay。

当前应用架构可以作为 engine research 的基线；尚未确定的是生产级 history compression，而不是 request 生命周期。

## 阶段六：音频 arrival-prefill 实验臂

Qwen audio encoder 在约 8 秒窗口内使用双向 attention，因此把音频独立切成 1 秒块会改变语义。该路径默认关闭，只用于研究 arrival prefill 负载。

该短 A/B 使用 schema 4、seed 7、8 用户×6 轮、前 1 轮预热、0–8 秒 stagger，plan SHA256 为 `b6160ec3d78a2126fb07830e5c1b75f9319ba579543555c8474852ad9f9ddf9b`。baseline 不设置 override；arrival 组使用 `MU_SESSION_CFG_JSON='{"enable_audio_arrival_prefill_approximation":true}'`。结果 metadata 标记 source 为 dirty `cbf2226a`，因此它只能支持方向性结论，不能作为可由单个 commit 精确复现的正式结果。

8 用户×6 轮短 A/B：

| 输入方式 | Audio-ready-500 p50/p95/p99 |
|---|---:|
| query-time 完整 WAV | 368/475/654 ms |
| 1 秒 audio arrival approximation | 343/554/692 ms |

arrival 模式只改善约 25 ms 中位数，p95/p99 反而上升，因为新增 warm-up 与 foreground query 竞争。正式 workload 因此继续使用 query-time 完整 WAV；只有原生 causal/streaming audio encoder 才适合做语义等价的音频 arrival prefill。

结果：`/home/ubuntu/data/results/audio_arrival_ab_baseline_20260822/u8_t6`、`/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u8_t6`。

## 阶段七：下一步

1. 在应用层实现有语义的有界压缩：短文本 summary/seed，加少量最近完整 turn；不要使用 `32k -> 0` 作为正式配置。
2. 固定压缩策略和 context 包络后，重新从 8 用户开始跑 8、16、32……容量阶梯。
3. 在首次 SLO 失败点拆分 Thinker prefill、Thinker decode、首文字后流水线等待和 GPU 活跃度。
4. 若 tail 仍由并发 Thinker 工作主导，再对比调度隔离和 P/D 分离；不要继续用应用参数掩盖 engine 问题。

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
