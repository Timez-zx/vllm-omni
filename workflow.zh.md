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
- 新视频帧触发或合并进低优先级 `video-warm-<uuid>`。它只运行 Thinker、`max_tokens=1`、不返回文字、不进入 Talker。全局最多一个 arrival warm-up；query 到达时关闭后台 admission、立即取消已登记和本 session 待执行的 warm-up，不等待 cache。
- 前台 gate 在首个 engine output 后重开。此后新媒体仍可做 arrival prefill，并与正在进行的 Thinker decode 竞争；这是目标 realtime workload 的 engine 压力，不是跨轮 persistent request。
- 用户音频默认在 query 时作为一个完整 WAV 输入，保持 Qwen 的整段音频语义。只有最终 query 会触发语音回答。
- 回答期间新到达媒体归入下一轮；本轮媒体只消费一次。
- 正常回答最多生成 256 个 Thinker tokens；视频不超过 640×352，JPEG 处理放在子进程池中。
- history 硬阈值为 49,152 tokens；16,384-token headroom 使应用通常在约 32,768 tokens 提前压缩。压缩生成一段最多 512 tokens 的 durable text summary，保留最近 2 个完整 turn，并将新 prompt 控制在约 16,384 tokens 内。summary 生成或重写失败时，主动压缩不修改 history；到硬阈值后才按完整 turn 丢弃作为安全 fallback。压缩与 turn commit 使用同一把 session lock，并更换 cache lineage。

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

### 架构对照归档

8 用户的 context 对齐实验已证明 finite-request 生命周期不是性能问题：当前 finite request 的 Audio-ready-500 p99 为 796 ms，旧 persistent 归档按固定 500 ms 音频重算后为 824 ms。应用维护 session、engine 每轮接收完整 finite request、依靠可淘汰 prefix/KV cache 的路线成立。结果分别位于：

- `/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`
- `/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`

### 16 用户 summary + recent 诊断

Setup：

- source：`3b661ed19ae343b90576eec39575baf1e1c27a5e` 上的 dirty working tree；这是提交前诊断，不是 clean-commit 归档；
- deploy：`origin_deploy_3gpu.yaml`，SHA256 `ae7cbeb615b24ee8654cf6c867887b2920324fc81ec93be6aaaa8995c55e4e24`；
- workload：schema 4、seed 7、16 用户×30 轮、前 2 轮预热、不设置 `MU_SESSION_CFG_JSON`；
- plan SHA256：`bdaa528c57ba688b3c0d6889d0877029a595a5f249ef81b631e0b8e13b597aa7`；
- input trace SHA256：`f217432fa3fdc7362f4926f9a7c2c4c9219bc0dd453901d0d93de8f2245fe561`；
- 结果：`/home/ubuntu/data/results/nonpd_summary_recent_u16_t30_diag_20260823_v2/nonpd_summary_recent_diag_seed7_u16`。

结果：

| 指标 | 结果 |
|---|---:|
| 完成 | 448/448，0 timeout，0 client error |
| TTFT p50/p99 | 297/914 ms |
| TTFA p50/p95/p99 | 554/1111/1412 ms |
| Audio-ready-500 p99 | 1412 ms |
| Stall-max p99 | 0 ms |
| Prompt tokens p50/p95/p99/max | 20.4k/33.5k/35.4k/37.1k |

该 cell 因 Audio-ready-500 p99 超过 1 秒而失败，但语音一旦启动没有卡顿。`verify_run.py` 确认 480 个唯一 finite request、2613 个 arrival-prefill、6228 个已消费 frame occurrence、3508/3538 次 prefix-cache 命中和独立的三个 stage 进程。

一次无效的 v1 测量发现 P/D admission 握手被错误移植到非 P/D 路径，导致前台最多等待 13.843 秒。非 P/D 现改为直接取消后台 task，并由本地 AsyncOmni 清理 request。v2 的 query gate p50/p95/p99/max 为 0.2/1.3/2.4/2.9 ms；该应用层混杂因素已消除。

Tail 分解：

1. **主要来源是同一张 Thinker GPU 上的并发 prefill/decode 竞争。** tail95 超额中 Thinker 侧约占 63%，首文字后约占 37%。Thinker TTFT p99 为 914 ms；Thinker token 间隔中位数从全局 18.8 ms 增至 tail95 的 37.8 ms。
2. **并发量比单请求 prompt 长度更能解释 tail。** 单请求新增 prefill token 平均值只从全局 850 增至 tail95 的 1025；等待首文字期间到达的前台 prefill token 则从 1135 增至 2612，约 2.3 倍。250 ms 内同时到达 1/2/3/4 个 query 时，TTFT p50 为 254/369/418/642 ms。
3. **GPU0 在 tail 窗口确实更忙。** Thinker SM-active p50/p95 从全局 38%/66% 升至 tail95 的 53%/87%；Talker 为 14%/24%，Code2Wav 为 3%/7%，后两者不是容量瓶颈。显存占用不等于计算饱和。
4. **排队与 tail 同步增长。** query 到达时等待首音频的其他 session 平均数从全局 0.54 增至 tail95 的 1.13；该数为 0/1/2/3 时，TTFA p50 为 505/595/696/867 ms。
5. **summary 是次要放大因素。** 本次有 42 次 summary；按秒级 server 时间近似，summary 与 72/448 个计分轮次、9/23 个 tail95 轮次重叠。无 summary 重叠的 TTFA p95 仍约 1018 ms，因此它不能解释主要 tail。45 次 SHM mailbox fallback 仅覆盖 2/23 个 tail95，也不是主因。

结论：当前应用实现已足够作为 engine research 基线。16 用户的主要延迟不是 history 重复 render、前台 admission、Talker 或 Code2Wav 饱和，而是多用户前台 prefill 与 arrival prefill 在 GPU0 上干扰 Thinker decode；summary maintenance 进一步放大少数 tail。

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

1. 提交当前应用基线后，用已记录的 input trace 在 clean commit 上复跑 16 用户，形成正式归档。
2. 若需要严格声明首次容量边界，再补同一提交和 seed 的 8 用户 cell；本轮按要求只测了 16 用户。
3. engine 实验优先隔离 foreground decode 与 arrival/foreground prefill，再在独立分支比较 P/D 分离；不继续通过应用参数隐藏竞争。

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
| Tail 分解 | `benchmarks/live_agent/analysis/p99_attribution.py`、`benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU 采样 | `benchmarks/live_agent/harness/gpu_sampler.py` |
