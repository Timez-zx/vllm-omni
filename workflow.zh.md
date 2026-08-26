# vLLM-Omni 实时多用户 Serving 工作流

本文只记录当前可复现实现、固定实验方法和已确认结论。研究分支为 `thinker-talker-pd`。

## 阶段一：目标与模型边界

目标是在首音延迟和语音连续性达标的前提下，用更少 GPU 服务更多长期音视频会话。研究对象是 engine 的调度、KV cache、P/D 传输、容量和尾延迟，不是模型质量。

目前没有可本地部署、交互方式与 Seed Realtime 或 Gemini Live 等价的开源模型。本项目用 Qwen3-Omni 的 Thinker → Talker → Code2Wav 流水线近似目标负载：客户端持续上传音视频，模型按 turn 回答。它能产生真实的多模态 prefill、文本 decode 和语音生成竞争，但不是原生全双工模型，也不研究语义级 barge-in。

## 阶段二：应用与 engine 边界

应用维护 session，engine 只处理可销毁的 finite request：

```text
视频持续到达
  → similarity/freshness filter
  → 同一 session 最多提交一个 Thinker-only arrival request
  → 处理期间的新帧合并为最新累计快照，不逐帧排队
  → P 完成并保存 snapshot 后 ACK，P lineage 线性推进
  → P/D 模式下，D cache-sync 按 lineage 在后台有序执行
用户说完
  → 停止提交待处理快照，只等待当前 arrival 的 P-ready ACK
  → 提交完整 canonical context + 一个完整 WAV
  → Thinker P → Thinker D → Talker → Code2Wav
回答结束
  → request 销毁，应用保存本轮；下一轮使用新 request ID
```

当前约束：

- WebSocket 应用维护 canonical 多模态历史和媒体归属；engine 不持有跨轮 live request。
- 每个 warm-up 和最终回答都是独立 finite request。请求携带完整 canonical prompt；engine 的 prefix/KV cache 是可淘汰加速层，cache miss 只影响延迟。
- 视频经 filter 后在本轮 append-only 保留；不使用 8 帧滑动淘汰。
- 同一 session 最多有一个 arrival 在 P 中执行；期间到达的媒体只标记最新累计快照，P-ready 后最多再提交一次。query 停止该 coalescing loop，并等待当前 arrival 的 P-ready；不同 session 仍可并发。
- P 的 KV/snapshot lineage 线性推进，每个 lineage 只保留最新 revision；没有多个同 session P 请求共享旧 parent。D cache-sync 可落后但按 lineage 有序执行。正式 query 以 D 实际命中的 prefix 为准，必要时一次接收累计缺失 suffix；cache miss 或 revision 不匹配只影响延迟，不影响正确性。
- warm-up 使用 `max_tokens=1`、`output_modalities=["text"]`，只维护 Thinker KV，不进入 Talker。P/D 模式下 P-ready 是应用完成边界，D cache-sync 是后台加速；非 P/D 模式只经过 Thinker。arrival 与正式 query 都使用原生 FCFS scheduler，不由应用分配 priority。
- 音频在 query 时作为一个完整 WAV 输入。Qwen audio encoder 使用双向 attention，独立音频分段不保证与整段推理语义等价。
- rendered prompt 达到 49,152 tokens 时，最近 2 个完成轮次只保留完整用户语音、用户文本和 Assistant 文本，删除历史图片；当前轮保留完整语音、文本和最新一张通过 filter 的图片。之后历史继续正常增长，到达上限后再次压缩。不生成 summary request。

这与生产中的“应用维护 session、engine 用 prefix cache 加速普通请求”一致，也能直接研究 engine 调度和 P/D 分离。旧的跨轮 persistent request、resumable append、Talker rolling 和 shadow request 路径不再使用。

## 阶段三：固定部署与 workload

### P/D 部署：4 GPU

上游 vLLM-Omni 不能把同一个 Thinker 拆为 P/D 两个 stage 后继续向 Talker 提供 conditioning states；本分支补齐了这条路径。正式 P/D 测试只能使用 `benchmarks/thinker_talker/pd_deploy_4gpu.yaml`：

| GPU | Stage |
|---:|---|
| 0 | Thinker P |
| 1 | Thinker D |
| 2 | Talker |
| 3 | Code2Wav |

`origin_deploy_3gpu.yaml` 只用于非 P/D 对照：Thinker、Talker、Code2Wav 各一张卡。它不能产生 P/D 结论。

P→D 使用 Delta-KV push；D→Talker→Code2Wav 使用共享内存。当前 P/D YAML SHA256 为 `d90a5a2c34ba365be4d8a400e50d6c3a28c5e535fb35ea67a25392ed74006461`。

### 持续 AV session workload

- 每个用户维持一条长期 WebSocket。
- 视频全程 2 FPS；输入阶段每个通过 filter 的帧立即触发累计快照 arrival request。回答期间的帧先缓存，回答完成后作为下一轮的首个累计快照提交。
- 麦克风以 5 Hz 上传 PCM16；assistant 播放期间暂停，并保留 300 ms echo guard。
- 每轮使用一条不重复的真实 16 kHz mono SLURP 录音，末尾追加 700 ms endpoint silence；query 文本为空，完整 WAV 在 query 时提交。
- 每个 session 固定 speaker；视频来自 DAVIS 固定序列，不同用户使用不同起点。
- 下一轮在上一轮音频按 1× 播放完成后开始；用户启动时间确定性分布在 0–8 秒。

协议 smoke 使用 1–2 用户；容量测试使用 30 轮/用户、前 2 轮不计分，从 8、16、32……递增。每个 cell 重启 engine；首个 SLO 失败点停止，不继续提高用户数。

16 用户 P/D 复现入口：

P/D launcher 默认设置 `VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES=34359738368`（32 GiB 主机内存）；A/B 复现 8 GiB 时显式覆盖为 `8589934592`。

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/pd_capacity_<commit> \
RESULT_PREFIX=pd_capacity USERS=16 SEEDS=7 TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh
```

不要用通用 `run_av_session_ladder.sh` 启动 P/D；它默认是 3 GPU 非 P/D 基线。

## 阶段四：指标与结果校验

- **TTFA / Audio-ready-500**：query 到第一块可播放的 500 ms 音频；当前首包已超过 500 ms，两者数值相同。
- **TTFT**：query 到第一段文字，用于把 Thinker 与语音启动分开。
- **Stall max**：按 1× 播放时最大的单次断流。
- **RTF deliver**：输出音频时长 / 交付耗时；大于 1 表示供给快于播放。

一个 capacity cell 必须满足：全部计分 turn 完成、Audio-ready-500 p99 < 1 s、Stall-max p99 < 50 ms，且无 client、protocol 或 fatal engine error。

```bash
MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml \
MU_EXPECTED_STAGE_IDS=0,1,2,3 \
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

验证器检查部署、四个 stage、finite request 唯一性、arrival warm-up、frame ledger 和实际 prefix-cache 命中。

## 阶段五：已确认的实现决策

1. context 对齐后，finite request 与旧 persistent request 没有固有延迟劣势；session 状态留在应用层是当前研究基线。
2. 视频 arrival warm-up 能模拟持续多模态 prefill；音频正式使用 query-time 完整 WAV，保证 Qwen 语义正确。
3. P→D 秒级等待曾来自 connector 实现，不是 PCIe 或 P/D 的固有代价。每个 arrival 的 ACK 只表示 P snapshot 已保存；应用据此线性推进 P lineage，D cache-sync 在后台按 lineage 有序完成。
4. P conditioning states 支持 lineage delta 和共享 tensor handle。每个 lineage 只保留最新 snapshot；找不到声明的 parent 时为正确性回退 full snapshot。
5. 同一 session 的 P arrival 严格串行并合并中间更新；query 只等待当前 arrival 的 P-ready，不等待 D-ready。不同 session 仍直接进入原生 FCFS scheduler，不使用全局 arrival gate 或应用 priority。
6. context 只在 49,152-token 上限处压缩：最近 2 个完成轮次保留语音和文本但删除图片，当前轮只保留最新一张图片及完整语音/文本，然后继续增长。无提前压缩和 summary request。

本轮重构还修复了 P chunked prefill 初始 hit 判断、Talker 极短输出的 3-row 前缀，以及 P snapshot 的 payload/ack 协议。线性 lineage、arrival coalescing、多模态原子提交和 P/D cache-sync 均有定向测试。

旧 D-ready 严格串行实现的短测（P/D，8 用户 × 6 轮，2 轮 warm-up）：32/32 计分轮成功，timeout/stall 为 0，TTFA p50/p95/p99 为 386/521/557 ms。该数据是历史基线，不代表当前 P-ready 实现。

未保留的对照实验曾允许同一 session 的 arrival 乱序并发。相同 16 用户 × 12 轮计划下，TTFA 从线性实现的 770/1,473/2,261 ms 变为 693/4,675/6,323 ms。并发请求共享旧 parent，snapshot suffix 总量从 634k 增至 1.738M tokens，full snapshot 从 16 增至 59，P→D delta 从 634k 增至 938k。该设计虽能用 seq/fence 保证状态不回退，但破坏线性增量复用，已撤回。

## 阶段六：16 用户 × 30 轮 P/D 诊断

本阶段前半部分数字来自旧的同 session D-ready 严格串行边界，保留为可复现实验背景；P-ready 边界的同计划对照见本阶段末尾。

历史 8/32 GiB A/B 使用完全相同的 workload，TTFA p99 分别为 5,054/5,285 ms；32 GiB 组 snapshot cache 峰值仅 18,747 MiB。由此确认主机 snapshot cache 容量不是当时的根因。

随后定位并修复两处协议错误：

1. `prefill_only/final_stage_id=0` 不进入 D/Talker，但仍必须向 orchestrator 返回 layer-0/layer-24 snapshot；runner 现在将两种路由分开，并在异步请求清理前冻结 payload 决策。
2. full 与 delta 同 batch 时，merged source 和 raw delta tail 必须同时构造；旧实现会丢掉同 batch 的 delta payload。

Orchestrator 现在只在 snapshot 实际缓存成功后返回成功；应用也只在成功后推进 lineage。114 个定向测试和 4 用户 smoke 均通过。

正式结果：`/home/ubuntu/data/results/pd_snapshot_ack_u16_t30_20260824_v2/pd_snapshot_ack_seed7_u16`

该结果与修复前 32 GiB 组使用完全相同的 `workload_plan.json`（SHA256 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`）。480/480 回合完成，448/448 计分，0 timeout、0 skip、0 client error、0 stall；6,178 个 arrival request 均为 finite request，frame ledger 和 prefix-cache 校验通过。

| 指标 | 修复前 32 GiB | 修复后 |
|---|---:|---:|
| TTFA p50/p95/p99 | 1,267/4,400/5,285 ms | 780/1,924/3,006 ms |
| Session serial wait p50/p95/p99 | 52/1,604/2,305 ms | 0/501/855 ms |
| Thinker-P p50/p95/p99 | 532/2,096/2,396 ms | 267/790/1,120 ms |
| P cache miss tokens p50/p95/p99 | 129/449/808 | 110/325/411 |
| 计分 query full snapshot | 83/448 | 0/448 |

本次准备 6,704 个 snapshot，6,701 个成功缓存；其余 3 个由应用在 session 收尾时显式 abort，不是 snapshot 失败。62 个 full warm-up 等于 16 条初始 lineage 加 46 次 history compaction 后的新 lineage；所有计分 query 均为 delta。snapshot cache 峰值 15,358 MiB。

### 剩余 tail

| 阶段 | p50/p95/p99 |
|---|---:|
| 应用：等待本 session 已提交 arrival | 0/501/855 ms |
| 应用：prompt render | 22/101/200 ms |
| Engine audio TTFA | 709/1,481/2,219 ms |
| Thinker-P | 267/790/1,120 ms |
| Thinker-D 增量 | 179/507/1,238 ms |
| Talker 增量 | 60/172/247 ms |
| Code2Wav 增量 | 189/253/442 ms |

最慢 5 个回合均同时包含 765–1,120 ms 的 session wait 和 2,060–2,404 ms 的 engine 延迟。client TTFA 与 engine TTFA 的相关系数为 0.934，与 session wait 为 0.746；engine TTFA 与 P/D 的相关系数分别为 0.883/0.789。因此当前 3 秒 p99 的主因是并发 arrival 下的 **Thinker P/D service tail**，严格 session 串行等待进一步放大 client 延迟；Talker 和 Code2Wav 不是主要瓶颈。

GPU0（P）busy p95/p99 为 93%/100%，SM active p95/p99 为 57%/81%；GPU1（D）busy p50/p95 为 0%/73%。这是突发 P/D 压力，不是四张 GPU 持续饱和，也不是 snapshot 容量、payload 丢失或 full fallback。16 用户已违反 1 秒 p99 SLO，因此按 capacity 规则不继续测 32 用户。

### D-side lineage 与 cache-only sync

旧长测只更新 P cache，D 到正式 query 才接收 KV；因此 P 只 miss 110–411 tokens 时，P→D suffix p99 仍达 13,436 tokens。这是实现缺口，不是 P/D 的固有代价。

arrival 在 P 计算 delta 后，通过 cache-only control operation 导入 D 的普通 prefix cache。该操作不进入 D inference scheduler、不执行模型，也不进入 Talker。旧实现等待 D 导入成功后才 ACK 应用；当前实现改为 P snapshot 保存成功后立即 ACK，D cache-sync 在后台继续。

第一版 cache-only 实现仍在唯一的 stage 输出循环中 `await` 每个 sync，错误地把不同 session 全局串行化。D/Code2Wav 已生成的输出因此在 API 侧积压 1.5–1.8 秒。随后公共循环改为只派发 sync 并继续取输出，但应用仍等待本 session 的 D-ready ACK。当前版本进一步把完成边界前移到 P-ready：同 session P 请求仍线性串行，D sync 则按 lineage 后台有序执行。

正式 query 不依赖所有中间 D sync 都已完成。D 以自己实际持有的 prefix 为准；若后台 sync 落后，现有 Delta-KV connector 会从 P 一次传输累计缺失 suffix。这样只改变加速是否及时，不改变 prompt 或输出语义。

修复前后使用完全相同的 16 用户 × 6 轮计划（SHA256 `8c04386fa673fd05ad77db3212846056d1d98b7d3b7cbaff2524eadac85ba7d4`），各有 80 个计分 turn：

| 指标 p50/p95/p99 | 全局等待 | 请求级并发 |
|---|---:|---:|
| TTFA | 743/1,851/3,075 ms | 594/806/1,150 ms |
| 本 session serial wait | 0/267/350 ms | 0/198/453 ms |
| arrival D cache sync | 23/83/131 ms | 24/89/121 ms |
| Thinker-P | 236/506/1,051 ms | 167/315/390 ms |
| Thinker-D 增量 | 213/1,484/1,970 ms | 113/211/333 ms |

修复后 96/96 finite query、1,264 个 arrival request 和 frame ledger 全部通过；1,532/1,551 次 prefix-cache 观测命中，无 timeout、client error 或 stall。慢请求的 core-output 到 orchestrator 接收恢复为低个位数毫秒。

剩余 engine audio TTFA 为 535/733/844 ms；Talker、Code2Wav 增量 p99 为 325/266 ms。最慢回合还可能等待本 session 前一个 arrival，但已无跨 session 的应用层 head-of-line blocking。短测 p99 仍略高于 1 秒；完成 30 轮正式容量测试前，不外推容量结论。

### 16 用户 × 30 轮正式长测

结果：`/home/ubuntu/data/results/pd_direct_cache_sync_u16_t30_concurrent_20260824/pd_direct_sync_concurrent_seed7_u16`

480/480 query、448/448 计分回合完成；5,790 个 arrival request、6,585 个消费帧和 prefix-cache 校验全部通过。无 timeout、client error 或 stall；所有计分 query 都使用 delta snapshot。

| 阶段 p50/p95/p99 | 延迟 |
|---|---:|
| Client TTFA | 784/2,100/2,610 ms |
| 本 session 等待前一个 arrival | 0/650/1,176 ms |
| Prompt render | 23/102/157 ms |
| Engine audio TTFA | 695/1,392/1,835 ms |
| Thinker-P | 271/773/1,128 ms |
| Thinker-D 增量 | 172/401/579 ms |
| Talker 增量 | 60/203/318 ms |
| Code2Wav 增量 | 183/268/328 ms |

Thinker-P query 实际 cache miss 仅 113/346/403 tokens，scheduler admission 仅 2/9/14 ms。P tail 的主要内部成分是：core preprocess 后到 scheduler 执行 31/183/371 ms（最大 1,259 ms）、query batch output build p99 248 ms，以及 runner 完成到 core 暴露输出 p99 448 ms；实际 GPU forward wall p99 仅 93 ms。payload 与 output build 的相关系数为 0.898。最慢请求的 1,259 ms ingress gap 与上一批 arrival NIXL sync burst 同时发生。

P 的 tail-95 窗口中 GPU busy/SM active/tensor active p95 为 100%/78%/49%，D 为 74%/44%/4%。因此根因不是 Talker、Code2Wav 或全局 API await，也不是 query 重新 prefill 大量历史；而是 **并发 arrival 下 P engine/connector 的 burst 使 input、snapshot/output 和结果暴露链路积压**。正确的 session 串行把该积压传播为正式 query 的等待，D 的 579 ms p99 是次要来源。

长测各 turn bucket 的 p99 未随轮数单调上升（1–10/11–20/21–30 轮为 2,877/2,600/2,138 ms），所以 2.61 秒不是 context 越长越慢，而是短测较难采到的稀有并发 burst。16 用户未满足 1 秒 p99 SLO，不继续测 32 用户。

### 当前音频/文本历史 + 最新图片策略：16 用户 × 30 轮

结果：`/home/ubuntu/data/results/pd_audio_text_latest1_u16_t30_20260824/pd_audio_text_latest1_seed7_u16`

该结果使用当前压缩策略：达到 49,152 tokens 后，最近两个完成轮次保留用户语音、用户文本和 Assistant 文本但删除图片；当前轮只保留最新图片。它与下一节旧策略使用完全相同的 `workload_plan.json`（SHA256 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`），因此可直接 A/B。

480/480 query、448/448 计分回合完成；6,322 个 arrival finite request、7,442 个消费帧和 prefix-cache 校验通过。无 timeout、skip、client error 或 stall。

| 阶段 p50/p95/p99 | 延迟 |
|---|---:|
| Client TTFA | 836/2,775/3,853 ms |
| 本 session 等待前一个 arrival | 0/946/1,715 ms |
| Prompt render | 27/124/203 ms |
| Engine audio TTFA | 715/1,830/2,541 ms |
| Thinker-P | 279/1,125/1,705 ms |
| Thinker-D 增量 | 175/494/705 ms |
| Talker 增量 | 57/200/422 ms |
| Code2Wav 增量 | 186/254/348 ms |

共发生 27 次压缩，压缩后的完整 prompt 仅 600–951 tokens；全部 query 的 P cache miss p50/p95/p99 为 117/629/739 tokens。压缩后的 27 个 full query 的 Thinker-P 为 136/677/1,207 ms，反而低于 421 个普通 delta query 的 299/1,125/1,705 ms。旧策略的 full-query cache miss 曾达 12,663/25,959/36,011 tokens；该应用层开销已经消除。

在相同 workload 下，TTFA 从旧策略的 1,135/3,248/4,459 ms 降至 836/2,775/3,853 ms，但仍未满足 1 秒 p99 SLO。正式长测表明 P event queue p99 仅 0.045 ms，而 scheduled-to-output p99 为 1,334 ms；但 `vllm_prefill_ms` 包含模型执行和 runner 输出准备，不能直接等同于 CUDA prefill 计算。

为区分两者，使用当前实现补跑了开启 runner/handoff/scheduler 计时的 16 用户 × 12 轮诊断：`/home/ubuntu/data/results/pd_latest1_u16_diag_t12_20260825/pd_latest1_diag_seed7_u16`。192/192 query、160/160 计分回合完成，无 timeout 或 stall，TTFA p50/p95/p99 为 741/2,029/2,433 ms。正式 query 所在 P batch 的 GPU forward p50/p95/p99 为 51/154/228 ms，runner output build 为 17/111/184 ms，runner 完成到 core 暴露输出为 70/338/515 ms，core ingress 到 scheduler 为 36/158/233 ms（最大 660 ms）。最慢两个 P 请求约 1.25–1.32 秒，其中 GPU forward 为 242 ms、output build 为 200 ms、runner→core 暴露为 670 ms；另一个 1.16 秒请求有 660 ms 消耗在 core ingress。

因此确切结论不是“纯 prefill 计算竞争”，而是 **持续 arrival 使 P 形成 arrival/query 混合 batch；GPU prefill 计算是次要组成，主要 tail 来自 P 的 conditioning snapshot/output build、异步结果队列和 core 暴露流水线积压**。正式 query 等待本 session 前一个 arrival 的 1,715 ms p99 是同一 P 流水线压力经严格 session 串行传播后的结果，不是独立的应用锁问题。D 是次要来源；Talker、Code2Wav 和 PCIe 不是瓶颈。

这次测试已去除历史图片重建和压缩 full prefill 的混杂因素。下一步先消除 Thinker-P snapshot/output/handoff 的自定义工程开销，再判断剩余部分是否属于 engine 调度。

### P snapshot 同步输出开销修复：严格 16 用户 × 12 轮 A/B

修复前，exact-parent delta 虽然已有可复用 lineage，但 P runner 仍在主路径同步把 layer-0/layer-24 tensor 复制到共享 CPU 内存并构造输出。现在该路径复用已有 pinned-memory 异步 snapshot：GPU→CPU copy 与输出构造不再阻塞 runner；full snapshot 或 prefix gap 仍使用同步路径保证正确性。请求语义、tensor 内容、lineage、priority 和 scheduler 均未改变。

修复后结果：`/home/ubuntu/data/results/pd_async_delta_u16_t12_20260825/pd_async_delta_seed7_u16`

它与修复前诊断使用完全相同的 `workload_plan.json`（SHA256 `983b6a9f0111ddb669fc36d77d4e528f2b30c6970ade740f379c98e6f7c73fa8`）。192/192 query、160/160 计分回合、2,495 个 arrival request 和 2,718 个消费帧校验通过；3,047/3,067 次 prefix-cache 观测命中，无 timeout 或 stall。

| p50/p95/p99 | 修复前 | 修复后 |
|---|---:|---:|
| Client TTFA | 741/2,029/2,433 ms | 770/1,473/2,261 ms |
| 本 session 等待前一个 arrival | 0/652/1,102 ms | 0/440/708 ms |
| Thinker-P | 252/911/1,252 ms | 243/490/907 ms |
| P scheduled→output | 154/674/1,157 ms | 158/364/528 ms |
| P query-batch output build | 17/111/184 ms | 7/39/66 ms |
| P runner→core output 暴露 | 70/338/515 ms | 57/143/268 ms |

GPU-forward p99 基本不变（228→239 ms），而 output build、结果暴露和同 session 等待明显下降，证明被消除的是自定义 P/D 输出路径开销，不是通过减少 workload 或计算量换来的结果。剩余 TTFA p99 主要由本 session 串行等待 708 ms、Thinker-P 907 ms 和 Thinker-D 555 ms 叠加；P 内部还包括 core ingress p99 414 ms，以及 async scheduler 为先提交下一 batch 而延后暴露上一 batch 的一批等待。大 batch 的异步 `output_wait` 与 GPU event 同步增长，属于并发 prefill 的实际完成时间。至此，原 snapshot 同步构造问题已修复；剩余问题是 engine 的 batch/compute 排队与 P/D 执行路径，不再是应用层历史或同步 tensor 构造问题。

### Arrival prefill 计算量与请求碎片化对照

为判断剩余 tail 是否是“prefill token 总量超过 GPU 算力”，使用同一个 16 用户 × 6 轮 `workload_plan.json`（SHA256 `3c55f3fd30bd3668723d3330c4c87ddfbfe2cb7d85f712d0c3ecfe89a93eb00a`）做诊断 A/B：正式组保留视频 arrival prefill；对照组仅关闭 arrival prefill，使相同视频内容在正式 query 集中 prefill。后者只用于归因，不是生产路线建议。两组消费 1,314/1,322 帧，差异 0.6%，均为 64/64 成功且无 timeout/stall。

结果：`/home/ubuntu/data/results/pd_connector_diag_u16_t6_20260825/pd_connector_diag_seed7_u16` 与 `/home/ubuntu/data/results/pd_query_video_diag_u16_t6_20260825/pd_query_video_diag_seed7_u16`

| 指标 | Arrival prefill | Query-time 视频 |
|---|---:|---:|
| Client TTFA p50/p95/p99 | 555/1,065/1,492 ms | 528/983/1,078 ms |
| Session 等待 p50/p95/p99 | 0/335/763 ms | 0/0/0 ms |
| Thinker-P p50/p95/p99 | 177/399/678 ms | 235/411/661 ms |
| 正式 query P miss p50/p95/p99 | 112/346/362 | 2,801/5,958/8,014 tokens |
| 全程 P miss tokens | 354,455 | 306,810 |
| P runner batches | 996 | 84 |
| P GPU-forward 累计 | 50.9 s | 14.4 s |

Query-time 组的正式 query 最多需要 prefill 约 8k 新 tokens，但 Thinker-P p99 仍未高于 arrival 组；arrival 组只多计算 15.5% tokens，却产生 11.9 倍 runner batches 和 3.5 倍累计 GPU-forward 时间。另对 1,387 次 P→D NIXL push 单独计时，单次总耗时 p50/p95/p99 为 0.281/0.969/1.877 ms，全程累计 575 ms，不能解释 200–500 ms 的请求等待。

因此剩余问题不是 GPU 被 prefill 总 token 持续压满。**Arrival traffic 是触发因素，但主要放大来自大量小 finite prefill 的碎片化：小 batch GPU 效率低、每个 arrival 都经过 async result/snapshot 与 D cache-sync，burst 时又经同 session 串行传播到 query。** 平均 GPU 容量仍有余量，tail 是 burst 和执行粒度问题。后续 engine 研究应保持 arrival 语义，同时优化跨 session batch accumulation、增量 prefill 粒度和 cache-sync 流水线，而不是改回 query-time 主实现。

### Scheduler microbatch 验证

为直接验证“小 batch 低效”而不改变 workload，在同步调度下使用同一份 16 用户 × 12 轮录制 trace（SHA256 `992716466dff3464f654e24baad90216d2b41d262eeb8382defefbca33f6a9cb`）做 A/B。两组均完成 160/160 个计分回合，无 timeout/stall；都发送 6,202 帧、接受 2,764 帧、消费 2,643 帧和发送 7,047 个音频 chunk。实验组只让 P scheduler 最多等待 100 ms，使 waiting queue 内不同 session 的请求形成真实 batch；应用、prompt、KV 和 P/D 路径不变。

结果：`/home/ubuntu/data/results/pd_microbatch_ab_20260825/window0/pd_mb0_seed7_u16` 与 `/home/ubuntu/data/results/pd_microbatch_ab_20260825/scheduler100/pd_sched_mb100_seed7_u16`

| 指标 | 原生 work-conserving | Scheduler 100 ms |
|---|---:|---:|
| TTFA p50/p95/p99 | 542/906/1,211 ms | 667/1,188/1,545 ms |
| P runner batches | 2,055 | 927 |
| 平均 batch size / singleton | 1.326 / 78.1% | 2.901 / 19.5% |
| Tokens per batch | 326 | 721 |
| P GPU-forward 累计 | 99.95 s | 46.73 s |
| 有效 scheduled tokens/s | 6,693 | 14,293 |
| P scheduler admission p50/p95/p99 | 1.6/4.5/7.1 ms | 101.7/109.7/122.1 ms |

两组累计 scheduled tokens 仅差 0.15%，但真实聚合使 batch 数和 GPU-forward 时间均减少约 53%，有效吞吐提高 2.14 倍，直接证明碎片化是主要效率损失。TTFA 反而变差，说明固定等待窗口不是解决方案：它用显式等待和 burst/HOL 换取吞吐，并把 session wait p95 从 177 ms 推高到 329 ms。当前 vLLM 的立即调度倾向低 admission latency，却产生大量 singleton；固定 accumulation 提高吞吐，却违反交互延迟。研究问题因此是 **SLO/deadline-aware 的跨 session 增量 batching**，不是继续清理低个位数 IPC 开销，也不是简单设置固定窗口。应用侧提前对齐 100 ms 仅把平均 batch 从 1.326 提到 1.366，证明聚合必须发生在 engine waiting queue，而不是 API 层。

补充 16 用户 × 4 轮应用/orchestrator 诊断：`/home/ubuntu/data/results/pd_app_diag_u16_t4_20260825/pd_app_diag_seed7_u16`。稳态 arrival render p50/p95/p99 为 8/22/35 ms；API submit、core decode、core preprocess 和 output encode/IPC 的 p99 分别约 10/11/4/8 ms，均不是主瓶颈。旧实现每个 arrival 同步等待 D cache-sync，812 次观测为 34/97/139 ms；该 D-ready ACK 同时串住下一 arrival 和正式 query。一个具体 p99 样本中，前一 arrival 总耗时 604 ms：P runner/GPU 约 60/49 ms，P 结果暴露到 orchestrator 约 233 ms，D transfer/load 与 cache-sync 约 122/149 ms；query 在其执行 164 ms 后到达，剩余等待 440 ms。这证明等待包含自定义 P/D 后处理，不应全部归因于 prefill 计算。

当前版本已经拆分 P-ready 与 D-ready：P snapshot 保存后向应用返回一次 terminal ACK，D sync 继续在 request-scoped 后台任务中执行；同 lineage 的 D sync 有序，不产生第二个 terminal。该改动不消除 P core-output 到 orchestrator 的暴露延迟，只移除前台对 D transfer/sync 的等待。相关 orchestrator、connector、Core 和 API 回归共 133 项通过，39 项依赖运行环境的测试跳过。

P-ready 正式短测：`/home/ubuntu/data/results/pd_pready_u16_t12_20260825/pd_pready_seed7_u16`。配置为 P/D 4 GPU、16 用户 × 12 轮、前 2 轮 warm-up、seed 7；deploy SHA256 为 `d90a5a2c34ba365be4d8a400e50d6c3a28c5e535fb35ea67a25392ed74006461`，原始 `workload_plan.json` SHA256 为 `983b6a9f0111ddb669fc36d77d4e528f2b30c6970ade740f379c98e6f7c73fa8`，canonical plan hash 为 `1dfd08e50710f188e5d83526358b9c3aecf4ad0e8616a5b0bb44d6b45d3b0a6f`；三者与旧 D-ready native 对照完全相同。192/192 query、160/160 计分回合和 2,608 个 finite arrival request 通过；消费 2,674 帧，3,104/3,125 次 prefix-cache 观测命中，无 timeout、skip、stall、client error 或 engine warning。2,549 个实际提交的后台 D sync 全部完成，无失败。

| p50/p95/p99 | 旧 D-ready | 当前 P-ready |
|---|---:|---:|
| Client TTFA | 542/906/1,211 ms | 529/939/1,073 ms |
| 等待本 session 前一 arrival | 0/177/430 ms | 0/101/238 ms |
| Prompt render | 17/60/128 ms | 17/48/166 ms |
| Engine audio TTFA | 510/746/979 ms | 497/820/946 ms |
| Thinker-P | 114/297/404 ms | 128/299/385 ms |
| Thinker-D 增量 | 139/261/479 ms | 136/278/399 ms |

P-ready 将 session-wait p99 降低 192 ms，Client TTFA p99 降低 139 ms，验证旧等待确实包含不应暴露给应用的 D transfer/sync。P、D 执行 tail 量级基本不变；当前最慢请求主要由 883–1,018 ms 的 engine path 或 106–174 ms 的 prompt render 组成，而不是等待 D-ready。TTFA p99 仍比 1 秒 SLO 高 73 ms，因此 16 用户仍判失败。P GPU busy p50/p95/p99 为 33%/75%/94%，SM active 为 27%/57%/65%；这是 burst tail，不是持续满载。该结论来自一次 live run，p95 的小幅波动不作容量提升结论。

#### P-ready 16 用户 × 30 轮长测

结果：`/home/ubuntu/data/results/pd_pready_u16_t30_20260825/pd_pready_seed7_u16`。原始 `workload_plan.json` SHA256 为 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`，canonical plan hash 为 `9f807d41b63a9b76fe7d2e0e30b7a1ae2dd350a02fba2998a32ff6786d194514`。480/480 query、448/448 计分回合和 6,572 个 finite arrival request 通过；消费 6,984 帧，7,666/7,701 次 prefix-cache 观测命中，无 timeout、skip、stall、client error 或 engine warning。6,259 个实际提交的后台 D sync 全部完成，最大短时 backlog 为 52；最慢请求到达时 backlog 仅 0–9，因此后台 sync 堆积不是主因。

| p50/p95/p99 | 16×30 |
|---|---:|
| Client TTFA | 558/1,463/2,209 ms |
| 等待本 session 前一 arrival P-ready | 0/191/524 ms |
| Prompt render | 20/80/124 ms |
| Engine audio TTFA | 519/1,234/1,696 ms |
| Thinker-P | 132/513/714 ms |
| Thinker-D 增量 | 145/474/678 ms |
| P 实际 cache miss | 107/571/662 tokens |
| P→D transfer delta | 107/482/574 tokens |

长测的主因是 **40–50k 长 prefix 下的 Thinker-P/Thinker-D burst**，不是 full-history 重算。按正式 query 的 prompt 长度分桶，中位数如下：

| Prompt tokens | 请求数 | Thinker-P | Thinker-D | Engine audio TTFA |
|---:|---:|---:|---:|---:|
| 0–8k | 68 | 75 ms | 73 ms | 351 ms |
| 8–24k | 161 | 110 ms | 112 ms | 446 ms |
| 24–40k | 142 | 185 ms | 179 ms | 618 ms |
| 40–50k | 77 | 342 ms | 285 ms | 918 ms |

prompt 长度与 P、D、engine TTFA 的相关系数分别为 0.636/0.660/0.691。prefix cache 只避免重算旧 token 的 KV；新增 token 的 P attention 和 D 的首轮 decode 仍需读取整段 40–50k KV。多个 session 同时接近 49,152-token 上限时形成 burst：最慢窗口中 P GPU busy/SM active p95 为 100%/77%，但 PCIe TX/RX p99 仅约 0.70/1.11 GiB/s。27 个压缩后的 full query 与 421 个 delta query 的 engine TTFA p99 都约 1.70 秒，说明压缩 full prefill 本身不是唯一根因；长 prefix burst 同时拖慢周围的 delta request。session wait p99 524 ms 是同一 P 压力经 P-ready 串行传播的次要部分，render p99 124 ms 更小。Talker/Code2Wav scheduled-to-output p99 仅 59/16 ms，不是瓶颈。

另有 46/876 个 snapshot 在超过 16 chunks 后于 orchestrator 事件循环同步合并完整 layer-0/layer-24，单次中位数/p95/max 为 26/214/270 ms，会阻塞其他 session 的输出处理。该问题与 P-ready/D-ready 边界独立，仍属于待消除的自定义 P/D 工程开销。

### 此前两轮原始 AV 硬上限策略：16 用户 × 30 轮

结果：`/home/ubuntu/data/results/pd_recent2_long_u16_t30_20260824/pd_recent2_long_t30_seed7_u16`

该历史结果使用“不提前压缩、不生成 summary；达到 49,152 tokens 后保留最近 2 个完整原始 AV turn 与当前轮”的旧策略。480/480 query、448/448 计分回合完成，6,189 个 arrival request、frame ledger 和 prefix-cache 校验通过，无 timeout、skip 或 stall；TTFA p50/p95/p99 为 1,135/3,248/4,459 ms，16 用户未满足 1 秒 p99 SLO。该结果仅保留为上节当前策略的 A/B 基线。

33 次压缩对应 33 个计分 query 的 full snapshot。full query 的 P cache miss p50/p95/p99 为 12,663/25,959/36,011 tokens，Thinker-P 为 1,242/2,183/3,537 ms。其余 415 个 delta query 只 miss 121/395/597 tokens，但 Thinker-P 仍达 384/1,412/1,823 ms；tail-95 的 23 个回合中 21 个是 delta。由此确认主因是压缩后的 full prefill 占用 P 执行窗口，并将 tail 扩散到其他 session 的 delta query 和 arrival。正式 query 等待本 session 前一个 arrival 的 p99 为 1,667 ms，是同一 P 压力的传播结果。

D transfer/load p99 为 323 ms；Talker、Code2Wav 实际 scheduled-to-output p99 为 88/15 ms。tail 窗口中 P 的 SM/tensor active p95 为 94%/65%，PCIe TX p95 约 1.16 GiB/s，因此不是 PCIe、Talker 或 Code2Wav 瓶颈。当前 4.46 秒 p99 同时包含应用策略因素：arrival 到达上限时跳过 warm-up，压缩和 full prefill 被推迟到正式 query；在调整该位置前，不能把全部 tail 归因于 engine 的稳态容量。

## Recovery map

| 内容 | 路径 |
|---|---|
| Session 与 finite-request 生命周期 | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| P/D snapshot 与 stage routing | `vllm_omni/engine/orchestrator.py` |
| P runner delta output | `vllm_omni/worker/gpu_ar_model_runner.py` |
| P/D 部署 | `benchmarks/thinker_talker/pd_deploy_4gpu.yaml` |
| 多用户 workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| P/D capacity 入口 | `benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh` |
| 运行验证 | `benchmarks/live_agent/analysis/verify_run.py` |
| P/D 分阶段延迟 | `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| Thinker P tail 诊断 | `benchmarks/live_agent/analysis/pd_tail_diagnosis.py` |
| Tail 与 GPU 归因 | `benchmarks/live_agent/analysis/p99_attribution.py` |
