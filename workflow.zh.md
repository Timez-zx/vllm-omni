# vLLM-Omni 实时多用户 Serving 工作流

## 目标与边界

目标是在 TTFA 和语音连续性达标时，用更少 GPU 服务更多持续音视频会话。研究对象是 engine 的容量、调度、KV cache 和尾延迟，不是模型质量。

目前没有可本地部署、交互方式与 Seed Realtime 或 Gemini Live 等价的开源模型。本分支用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似持续 AV 会话。它重点模拟“持续视频增量计算与回答 decode 并发”，不代表原生全双工或语义级 barge-in。

## 阶段一：应用基线

应用维护 session；engine 每次只处理普通、有限生命周期的 request：

```text
视频到达 → 静默 Thinker finite request 预热 prefix KV
用户说完 → 完整 canonical history + 当前完整 WAV
          → Thinker → Talker → Code2Wav
回答结束 → request 销毁，应用保存本轮
```

- 每轮和每个 arrival warm-up 都使用新 request ID；engine 不保留跨轮 live request。
- 每次提交完整 canonical prompt。prefix/KV cache 可淘汰；cache miss 只增加计算，不改变语义。
- 应用缓存已处理的 canonical message block，只 render 新 block，再组装完整 prompt。
- 视频通过 similarity/freshness filter 后在本轮 append-only；没有 8 帧滑动窗口或二次采样。
- 每个 session 同时最多运行一个 arrival request。它运行 Thinker、`max_tokens=1`、不返回文字、不进入 Talker。
- arrival 运行期间接受的新帧保留为独立图片，并一起进入下一次累计 prompt snapshot。“合并”只减少 request 次数，不合并或丢弃图片内容。
- 不同 session 的 arrival 直接并发提交给原生 FCFS；没有全局 gate 或应用 priority。
- final query 等待本 session 已提交的 arrival 完成，再提交完整 prompt，以保持 prefix lineage 有序。
- 用户音频在 query 时作为一个完整 WAV 输入，以保持 Qwen 的整段音频语义。只有 final query 触发 Talker。
- 回答期间新媒体属于下一轮；每个接受的媒体项只被一个 final query 消费。
- Thinker 回答最多 256 tokens；视频不超过 640×352；JPEG 处理在子进程池执行。
- prompt 达到 49,152 tokens 时才压缩：保留最近两个已完成 turn 的用户音频/文本和 assistant 文本，删除历史图片，当前 turn 只保留最新图片；如果仍超限，继续按完整旧 turn 丢弃。不会生成 summary request。

旧的跨轮 persistent request、resumable append、Talker rolling、shadow compression 和 session ledger 已删除。

## 阶段二：固定部署

正式实验只使用 `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`：

| Stage | GPU | 配置 |
|---|---:|---|
| Thinker | 0 | FP8 weight/KV、prefix cache、同步调度、等优先级请求 |
| Talker | 1 | FP8 weight/KV、conditioning prefix cache |
| Code2Wav | 2 | 独立进程 |

`run_qwen_server.sh` 检查 Thinker 使用 FlashInfer，并在 worker 启动前创建 stage-0 SHM 多模态缓存。API renderer 与 input processor 共享 sender；worker 复用已 materialize 的媒体，避免有限请求反复恢复完整媒体历史。

## 阶段三：正式 workload

- 每个用户维持一条长期 WebSocket。
- 视频全程以 2 FPS 上传；通过 filter 的帧触发或排入下一次静默 Thinker arrival prefill。
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条不重复的真实 16 kHz mono SLURP 录音，末尾追加 700 ms endpoint silence；query 文本为空。
- 音频在 query 时作为完整 WAV 接到已 warm 的视频 prefix 后。
- 每个 session 固定 speaker；DAVIS 视频使用固定序列和不同起点。
- 下一轮在上一轮语音按 1× 播放结束后开始，形成 playback-paced closed loop。
- 用户在 0–8 秒内确定性错开。

这是 duplex-like 而非完整 AV duplex：视频在用户说话和 assistant 播放期间持续 arrival prefill；音频虽以 5 Hz 上传，但 Qwen 只在 query 时处理完整 WAV，播放期间麦克风暂停。它适合研究持续视觉增量负载，不用于声称复现 Gemini Live 或 Seed Realtime 的模型结构与绝对容量。

正式 cell 为 30 轮/用户，前 2 轮不计分；每个 cell 重启 engine。`probe.py` 的合成媒体只验证协议，不能用于容量结论。

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
RESULT_PREFIX=finite_request USERS=16 SEEDS=7 TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_av_session_ladder.sh
```

## 阶段四：指标与通过条件

- **TTFT**：query 到第一段文字。
- **TTFA / Audio-ready-500**：query 到累计 500 ms 可播放音频。当前首包超过 500 ms，两者相同。
- **Stall max**：按 1× 播放时最大的单次断流。
- **RTF deliver**：生成音频时长 / 交付耗时。

通过条件：所有计分 turn 完成、Audio-ready-500 p99 < 1 s、Stall-max p99 < 50 ms，并且没有 client、protocol 或 fatal engine error。

结果必须使用 workload schema 4，并运行 `benchmarks/live_agent/analysis/verify_run.py`。验证器检查 finite request、arrival request、frame ledger、prefix-cache 命中和三个独立 stage 进程。

## 阶段五：当前 16 用户结果

Setup：

- source：`726ebd90fe1120d8d0ce1d8c4d99dd54e6703912` 上的 dirty working tree；
- deploy：`origin_deploy_3gpu.yaml`，测量时 SHA256 `8cfcd31af59031ba20dc822632510a2de721dca9ede8a80070c8cf5647a7787f`；
- workload：schema 4、seed 7、16 用户×12 轮、前 2 轮预热、0–8 秒 stagger；
- plan SHA256：`1dfd08e50710f188e5d83526358b9c3aecf4ad0e8616a5b0bb44d6b45d3b0a6f`；
- 结果：`/home/ubuntu/data/results/nonpd_shm_sync_u16_t12_20260826/nonpd_shm_sync_seed7_u16`。

| 指标 | 结果 |
|---|---:|
| 完成 | 160/160，0 timeout，0 client error |
| TTFT p50/p99 | 341/2550 ms |
| TTFA p50/p95/p99 | 1371/5371/8212 ms |
| Stall-max p99 | 1972 ms |
| same-session arrival wait p50/p95/p99 | 0/662/1153 ms |
| prompt render p50/p95/p99 | 19/81/149 ms |
| Thinker TTFT p50/p95/p99 | 268/1022/1392 ms |
| Thinker ITL p50/p95/p99 | 87/345/544 ms |
| engine→first audio p50/p95/p99 | 1292/4664/7601 ms |

验证结果：192 个 final request、2,196 个 arrival request、2,712 个已消费 frame occurrence、2,559 次 prefix-cache hit；0 preemption、0 recompute、0 arrival failure。

Tail 结论：

1. 将 stagger 从 0–8 秒扩大到 0–40 秒后，TTFA p99 仍为 7,794 ms；错峰不是根因。对照结果：`/home/ubuntu/data/results/nonpd_shm_sync_stagger40_u16_t12_20260826/nonpd_shm_sync_stagger40_seed7_u16`。
2. prompt render p99 为 149 ms；same-session arrival wait p99 为 1,153 ms。两者会放大 tail，但都解释不了 8.2 秒 TTFA。
3. 主要延迟在 Thinker。正式请求的 decode ITL p99 达到 544 ms，说明生成每个 token 都反复等待包含 arrival prefill 的重 batch；Talker 和 Code2Wav 主要在等上游 token。
4. 当前每个 session 最多一个 arrival 在途，但 16 个 session 可同时形成 16 个 arrival。vLLM 每步先调度 `running` 请求，再从 `waiting` 队列接纳新请求；priority 不会抢占已有 `running` prefill。因此已进入 engine 的 arrival 会与正式 decode 共享 token budget 和 model forward，新 arrival 持续补充后形成 decode 饥饿。

结论：当前应用/session 实现可以作为 engine research 基线。16 用户 tail 的主要原因不是错峰、prompt render、summary、Talker 或 Code2Wav 饱和，而是多用户小粒度 arrival prefill 在 Thinker scheduler 中持续拖慢 decode。

## 阶段六：已验证的架构结论

保留“应用有状态、engine request 有限、依靠可淘汰 prefix/KV cache”的设计。正式 workload 不加入跨用户全局 gate；否则应用会改变 duplex-like engine 负载并隐藏竞争。下一步应在固定输入 trace 下研究 engine 的 deadline/QoS-aware 增量 prefill 调度，并在实现冻结后补跑 16×30 正式长测。

## 快速恢复入口

| 内容 | 路径 |
|---|---|
| Session 与 finite-request 生命周期 | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| 多用户 workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload 计划与媒体加载 | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| 容量脚本 | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| 固定部署 | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| 结果验证 | `benchmarks/live_agent/analysis/verify_run.py` |
| Tail 分解 | `benchmarks/live_agent/analysis/p99_attribution.py`、`benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU 采样 | `benchmarks/live_agent/harness/gpu_sampler.py` |
