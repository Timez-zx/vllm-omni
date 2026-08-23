# vLLM-Omni 实时多用户 Serving 工作流

本文只记录当前可复现路径、支撑设计决策的对照实验和已确认的结论。研究分支为 `thinker-talker-pd`。

## 阶段一：目标与模型边界

目标是在语音启动延迟和连续性达标的前提下，用更少 GPU 服务更多长期音视频会话。研究对象是 engine 的调度、KV cache、数据传输、容量和尾延迟，不是模型质量。

当前没有可本地部署、交互方式与 Seed Realtime 或 Gemini Live 等价的开源模型。本项目用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似目标负载：客户端持续上传音视频，模型按 turn 回答。它能产生真实的多模态 prefill、语音 decode 和多用户竞争，但不是原生全双工模型，也不研究语义级 barge-in。

## 阶段二：应用与 engine 边界

当前设计是“应用维护 session，engine 处理有限生命周期 request”：

```text
媒体持续到达
  → 通过 filter 的视频帧触发或合并进静默 Thinker warm-up
  → 用户说完，提交完整 canonical history + 完整 WAV
  → Thinker P → Thinker D → Talker → Code2Wav
  → 回答结束，request 销毁；应用保存本轮
```

关键约束：

- WebSocket 应用维护 session、媒体状态和 canonical 多模态历史；engine 不持有跨轮活 request。
- 每个 warm-up 和最终回答都是新的 finite request。每轮提交完整 canonical prompt，engine 的 prefix/KV cache 只是可淘汰的加速层；cache miss 只增加计算，不影响正确性。
- 应用只 render 新增 message block，再拼出完整 prompt，避免重复处理历史媒体。
- 视频经 similarity/freshness filter 后 append-only 地进入当前 turn；不再使用 8 帧滑动淘汰。
- 视频 warm-up 优先级为 10，只运行 Thinker，`max_tokens=1`，不返回文字、不进入 Talker。最终 query 优先级为 0，并会取消尚未完成的 warm-up。
- 音频在 query 时作为一个完整 WAV 输入。Qwen audio encoder 使用双向 attention，切成独立小段不能保证与整段推理语义等价。
- 回答期间到达的媒体归入下一轮。history 通常在约 32k tokens 时按完整 turn 压缩到不超过 16k；硬阈值为 49,152 tokens。

这一边界与常见的“无状态 API + prefix cache”计算语义接近，但 session 状态明确留在应用层。旧的跨轮 persistent request、resumable append、Talker rolling 和 shadow request 路径已经删除。当前应用可以独立于 engine lifecycle 接入 routing、replication 和 P/D 分离。

## 阶段三：固定部署与 workload

### 部署

上游 vLLM-Omni 不能把同一个 Thinker 拆成独立 P/D stage，同时继续向 Talker 提供 conditioning states。本分支补齐了这条路径；应用协议和 finite-request 生命周期没有改变。

正式 P/D 部署固定为 `benchmarks/thinker_talker/pd_deploy_4gpu.yaml`，当前 SHA256 为 `7bc4502494c19d07046a6ce5e2c272ebf4536108d1336831ea4294d78437ba24`。

| Stage | GPU | 主要配置 |
|---|---:|---|
| Thinker P | 0 | FP8 weight/KV、prefix cache、priority、32k batched tokens |
| Thinker D | 1 | FP8 weight/KV、prefix cache、priority、Delta-KV consumer |
| Talker | 2 | FP8 weight/KV、prefix cache、流式 codec 输出 |
| Code2Wav | 3 | 独立 stage，流式生成音频 |

P→D 使用 `NixlDeltaPushConnector`；D→Talker→Code2Wav 使用 shared-memory connector。正式实验只改变用户数、seed 或明确标注的实验变量，不改部署参数。

### 持续 AV session workload

- 每个用户维持一条长期 WebSocket。
- 视频全程以 2 FPS 上传；通过 filter 的帧触发或合并进 Thinker warm-up。
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条不重复的真实 16 kHz mono SLURP 录音，末尾追加 700 ms endpoint silence；query 文本为空，完整 WAV 在 query 时提交。
- 每个 session 固定 speaker；视频来自 DAVIS 固定序列，但起点不同。
- 下一轮在上一轮音频按 1× 播放完成后开始，形成 playback-paced closed loop。
- 用户启动时间确定性分布在 0–8 秒内。

协议 smoke test 使用 1–2 用户；固定性能 replay 使用 8 用户×12 轮、首轮预热；容量实验使用 30 轮/用户、前 2 轮预热，并按 8、16、32……增长。每个 capacity cell 重启 engine，在首个 SLO 失败点停止。

固定 8 用户×12 轮 trace：

- 文件：`/home/ubuntu/data/results/pd_deferred_free_fix_20260823_v1/u8_t12_seed7_u8/input_trace.jsonl.gz`
- SHA256：`03591906c1dd356df55eaac3081dc2efc4f6357f900f3411919c97b55fecf33b`
- workload plan SHA256：`7f08495fdc2b9a2ff565deb8f1923347124ae5f24527d2153167332749d5b1bd`
- 固定账本：3005 帧发送、1323 帧接受、1278 帧消费、3606 个音频 chunk。

复现命令：

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
MU_INPUT_TRACE_MODE=replay \
MU_REPLAY_INPUT_TRACE=/home/ubuntu/data/results/pd_deferred_free_fix_20260823_v1/u8_t12_seed7_u8/input_trace.jsonl.gz \
RESULTS_DIR=/home/ubuntu/data/results/pd_replay_<commit> \
RESULT_PREFIX=pd_replay USERS=8 SEEDS=7 TURNS=12 WARMUP_TURNS=1 \
bash benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh
```

做容量阶梯时移除 replay 变量，改为 `USERS="8 16 32" TURNS=30 WARMUP_TURNS=2`。`probe.py` 的合成媒体只验证协议，不能产生容量结论。

## 阶段四：指标与通过条件

- **TTFA**：query 到第一块音频包；受 packetization 影响，只在同一部署内比较。
- **Audio-ready-500**：query 到累计获得 500 ms 可播放音频；作为正式启动延迟。
- **Stall max**：按 1× 播放时最大的单次断流。
- **RTF deliver**：输出音频时长 / 交付耗时；判断能否持续供给。

一个 capacity cell 必须同时满足：所有计分 turn 完成、Audio-ready-500 p99 < 1 s、Stall-max p99 < 50 ms，且无 client、protocol 或 fatal engine error。

结果必须使用 workload schema 4，并通过：

```bash
MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml \
MU_EXPECTED_STAGE_IDS=0,1,2,3 \
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

验证器检查 finite request 唯一性、arrival warm-up、frame ledger、实际 prefix-cache 命中和四个独立 stage。

## 阶段五：已验证的设计决策

以下结论只来自每一行内部的同配置或对齐对照，不能跨行直接比较绝对延迟。

| 问题 | 证据 | 决策 |
|---|---|---|
| finite request 是否天然更慢 | context 对齐后，当前 Audio-ready-500 p99 为 796 ms，旧 persistent 为 824 ms | session 放在应用层、engine 每轮 finite request 是合理基线 |
| 音频是否应 arrival prefill | 8 用户短测：完整 WAV p99 654 ms；1 秒分段近似为 692 ms，且语义不等价 | 正式 workload 使用 query-time 完整 WAV |
| P→D 是否天然需要秒级传输 | 旧 NIXL pull 最差约 19.7 s；异步 packed push 后约 1.2 s，链路实测约 35 GiB/s | 秒级等待来自 connector 实现，不是 PCIe/P-D 的必然代价 |
| P 输入为何排队 | 完整历史媒体使 stage-0 wire p99 达 253 MiB；有序镜像 cache 后为 6.3 MiB，TTFA p99 977→714 ms | 保留完整 token 历史，只省略 receiver 已缓存的媒体 tensor |
| P snapshot 为何变慢 | 每次重建并传完整 conditioning states；delta chunk chain 将同 trace p99 1326→988 ms | 只传 lineage delta，最终 query 前合并一次 |
| 是否需要 Delta-KV | 88 turn 的理论 P→D payload 从约 72.5 GiB 降到 16.0 GiB，但 p99 为 1118 vs 1165/1085 ms | Delta-KV 解决重复传输；8 用户 tail 不是带宽问题，容量收益留到更高并发评估 |

当前 P/D 数据路径累计保留以下实现：

1. stage-0 有序镜像多模态 cache，避免重复发送历史媒体 tensor；
2. Talker conditioning snapshot 使用可丢弃的 lineage-delta chunk chain；
3. D 保留可丢弃 prefix KV，P 只 push block-aligned missing suffix；
4. P runner 将 layer 0/layer 24 delta 直接写入 request-owned shared storage，控制面只传 handle；
5. foreground gate 保护媒体 cache 顺序，但不被当作 GPU 抢占机制。

## 阶段六：当前 8 用户 P/D 基线与根因

最新同 trace 对比保持 88 个计分 turn 和全部媒体账本不变，无 timeout、stall 或 replay slip：

| 实现 | TTFA p50/p95/p99 | P p50/p95/p99 | Core-ready→API p50/p95/p99 |
|---|---:|---:|---:|
| 直接共享前，带诊断 | 443/681/1002 ms | 141/365/633 ms | 29/98/113 ms |
| P 输出直接共享，带诊断 | 381/563/770 ms | 87/172/399 ms | 5/15/24 ms |
| P 输出直接共享，关闭诊断 | **392/546/815 ms** | — | — |

关闭诊断组的 TTFT p50/p99 为 189/512 ms；Audio-ready-500 p99 为 815 ms，Stall p99 为 0，因此通过当前 8 用户 SLO。输出编码 p99 为 1.8 ms，普通 fallback payload 约 0.012 MiB。直接共享已经消除 Core→API 的主要 tail。

剩余最慢 P 请求为 399 ms。foreground 自身 runner/Core→API 只有 77/5 ms，之前约 317 ms 在等待一个已经 abort、但已进入 GPU 的 arrival warm-up。该 warm-up 一次处理 13.3k tokens：runner 535 ms、CUDA event 441 ms，并写出约 105 MiB snapshot。

确切结论是：

- 当前第一大头是 **Thinker P 上不可抢占的大 warm-up prefill**。低优先级请求一旦进入最大 32k-token 的 scheduler step，之后到达的 foreground query 只能等待。
- Talker 和 Code2Wav 没有饱和；P→D Delta-KV 传输与 Core→API 共享内存也不是当前 p99 主因。
- foreground gate 能保证 request/cache 顺序，但 asyncio abort 不能撤销已经启动的 GPU kernel。
- 应用生命周期和 workload 已足够合理，可以把该问题作为 engine 调度研究对象，而不是继续修改应用语义。

最新结果路径：

- 诊断：`/home/ubuntu/data/results/pd_direct_shared_long_diag_20260823_v1/u8_t12_direct_shared_seed7_u8`
- 关闭诊断：`/home/ubuntu/data/results/pd_direct_shared_long_clean_20260823_v1/u8_t12_direct_shared_clean_seed7_u8`

该运行记录 `source_commit=2a90a9e2` 且 `source_dirty=true`；这里的 “关闭诊断” 只表示关闭额外日志，不表示 git worktree clean。当前数据是开发基线，提交后应使用同一 trace 再回放一次，形成可由单个 commit 精确复现的正式基线。

## 阶段七：下一步

1. 提交当前实现后，用固定 8 用户×12 轮 trace 做一次 diagnostics-off replay，固定正式基线。
2. 在 stage 0 A/B 更小的 chunked-prefill 调度粒度、可抢占 warm-up，或在 compaction 后推迟 cold warm-up；目标是 foreground 最多等待一个小 chunk。
3. 确认被 abort 的 warm-up 不再执行完整大 prefill，再运行 8、16、32……用户、30 轮/用户的容量阶梯。
4. 到首个 SLO 失败点后重新分解 tail 和 GPU 有效利用率；在更高并发或跨节点条件下复测 Delta-KV 的容量收益。
5. RCA 时开启控制面诊断；正式容量结果关闭诊断。

## 快速恢复入口

| 内容 | 路径 |
|---|---|
| Session 与 finite-request 生命周期 | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Canonical 多模态 history | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| P/D orchestrator 与 snapshot | `vllm_omni/engine/orchestrator.py` |
| P EngineCore IPC | `vllm_omni/engine/stage_engine_core_proc.py` |
| Delta-KV connector | `vllm_omni/engine/nixl_delta_push_connector.py` |
| Shared tensor storage | `vllm_omni/utils/mm_outputs.py` |
| P/D 部署 | `benchmarks/thinker_talker/pd_deploy_4gpu.yaml` |
| 多用户 workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload 计划与媒体加载 | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| P/D 容量入口 | `benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh` |
| 运行验证 | `benchmarks/live_agent/analysis/verify_run.py` |
| P/D 分阶段延迟 | `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU 采样 | `benchmarks/live_agent/harness/gpu_sampler.py` |
