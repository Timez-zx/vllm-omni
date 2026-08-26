# 持续 AV 多用户 P/D Workload

本文只记录当前实现、正式 workload 和最新实验结论。研究分支为 `thinker-talker-pd`。

## 当前实现

应用维护 session 状态，engine 只处理有限生命周期的 request：

```text
视频帧持续到达
  → similarity/freshness filter
  → 构造当前轮累计媒体快照
  → 提交 Thinker-only arrival request
  → P-ready 后推进 P lineage
  → D 在后台按 lineage 同步 KV delta

用户说完
  → 等待本 session 当前 arrival 到达 P-ready
  → 提交完整 canonical context 和一个完整 WAV
  → Thinker P → Thinker D → Talker → Code2Wav
  → request 销毁，应用保存本轮历史
```

关键约束：

- 每个 arrival 和正式 query 都是新的 finite request；engine 不保存跨轮 live request。
- 请求携带完整 canonical prompt，engine 使用可淘汰的 prefix/KV cache，只计算未命中的 suffix。
- 同一 session 最多有一个 arrival 在 P 中执行；期间的新帧合并为最新累计快照，不逐帧排队。
- arrival 使用 `max_tokens=1` 和 `output_modalities=["text"]`，只维护 Thinker KV，不进入 Talker。
- query 只等待 P-ready，不等待后台 D cache-sync；正式 query 必要时一次传输 D 缺失的累计 KV delta。
- arrival 和 query 都使用原生 FCFS scheduler，不设置应用层 priority 或全局 gate。
- 音频以 200 ms PCM chunk 到达应用，但 Qwen engine 在 query 时接收一个完整 WAV；当前正式实验没有测量碎片化音频 prefill。
- context 达到 49,152 tokens 时，最近两个完成轮次保留用户语音、用户文本和 Assistant 文本，删除历史图片；当前轮只保留最新图片。之后继续增长，到达上限后再次压缩，不生成 summary request。

P/D 固定使用 `benchmarks/thinker_talker/pd_deploy_4gpu.yaml`：

| GPU | Stage |
|---:|---|
| 0 | Thinker P |
| 1 | Thinker D |
| 2 | Talker |
| 3 | Code2Wav |

P→D 使用 Delta-KV push，D→Talker→Code2Wav 使用共享内存。

## 正式 workload

- 16 个长期 WebSocket session，每个 session 30 轮，前 2 轮不计分。
- 视频来自 DAVIS，2 FPS；每个通过 filter 的帧进入本轮累计 prefix。
- 语音来自 SLURP，16 kHz mono PCM16，每轮不重复，末尾增加 700 ms endpoint silence。
- query 文本为空，语义输入来自完整语音。
- Assistant 播放期间暂停麦克风，并保留 300 ms echo guard。
- 下一轮在上一轮语音按 1× 播放完成后开始；用户启动时间确定性分布在 0–8 秒。
- TTFA/Audio-ready-500 从 query 到至少 500 ms 可播放音频；容量 SLO 为 p99 < 1 s，Stall-max p99 < 50 ms。

复现入口：

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/pd_capacity \
RESULT_PREFIX=pd_capacity USERS=16 SEEDS=7 TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh
```

## 最新结果

16 用户结果：

`/home/ubuntu/data/results/pd_pready_diag_p_d_u16_t30_20260826/pd_pready_diag_pd_seed7_u16`

- 480/480 轮完成，448/448 计分轮成功，无 timeout、skip、stall、client error 或 engine warning。
- 16,307 帧发送，7,322 帧通过 filter，7,167 帧被正式 query 消费。
- 6,760 个 finite arrival request；prefix cache 观测 7,952/7,987 次命中。
- 原始 `workload_plan.json` SHA256：`5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`；canonical plan SHA256：`9f807d41b63a9b76fe7d2e0e30b7a1ae2dd350a02fba2998a32ff6786d194514`。
- deploy SHA256：`d90a5a2c34ba365be4d8a400e50d6c3a28c5e535fb35ea67a25392ed74006461`。

| p50/p95/p99 | 结果 |
|---|---:|
| Client TTFA | 549/1,305/1,807 ms |
| 等待本 session 前一 arrival | 0/209/516 ms |
| Prompt render | 19/83/154 ms |
| Engine audio TTFA | 512/1,113/1,497 ms |
| Thinker-P time-to-output | 130/479/578 ms |
| D submit→首个输出 | 135/388/549 ms |

该 cell 未满足 1 秒 TTFA p99 SLO。

单用户长 context 对照：

`/home/ubuntu/data/results/pd_d_cause_u1_t30_20260826/pd_d_cause_seed7_u1`

- 两个独立 30 轮 session，56/56 计分轮成功，无 timeout 或 stall。
- TTFA p50/p95/p99 为 320/448/481 ms。
- prefix 最大达到 48,982 tokens，可与 16 用户的长 context 路径比较。

| D 路径 p99 | 1 用户 | 16 用户 |
|---|---:|---:|
| 完整请求反序列化 | 15 ms | 99 ms |
| KV transfer/load completion | 6 ms | 235 ms |
| 首步 D GPU | 7.5 ms | 12 ms |
| Talker conditioning 构造 | 77 ms | 122 ms |
| D Core 输出→API 收到 | 16 ms | 210 ms |
| D submit→首个输出 | 120 ms | 549 ms |

## 实验结论

当前 tail 的逻辑链条是：

```text
持续视频输入
  → 多用户聚合后产生大量高频、小粒度 incremental prefill
  → P 侧计算被切成短生命周期的小 request，难以形成稳定大 batch，固定处理成本占比上升
  → 每个实际 arrival 又触发一次独立 P→D KV cache-sync
  → connector、scheduler 和 orchestrator 的固定成本被反复支付
  → 并发时形成 transfer/progress queue 和输出处理 head-of-line blocking
  → arrival 与正式 query 延迟同时增长
  → TTFA tail 超过 1 秒
```

证据：

1. 16 用户最慢 D 请求只传输 8 个 KV delta block，P 提交 NIXL 仅用 0.3 ms，但 D 在 375 ms 后才观察到完成；期间另有 6 个小 push 提交，且没有 D model batch 占用 GPU。
2. 相同长 context 下，KV completion p99 从单用户 6 ms 增至 16 用户 235 ms，D Core→API 从 16 ms 增至 210 ms，而 D 首步 GPU 仅从 7.5 ms 增至 12 ms。主要增量来自控制与进度流水线，不是 D decode 算力或 PCIe 带宽。
3. 当前 D request 仍反序列化完整 prompt 元数据；首次进入 Talker 前还会拼接完整历史 conditioning。两者随 context 增长，是当前 P/D glue 的工程开销，不是 P/D 的固有要求。
4. Orchestrator 使用单个协程依次轮询和处理各 stage。处理 P arrival snapshot/cache-sync 时，D 已完成的输出可能等待 100–323 ms 才被 API 取走，这是可消除的工程 head-of-line blocking。

因此，主要矛盾不是 arrival prefill 的应用语义，而是当前 request-granularity engine/P-D 接口处理高频小增量的效率：

- P 侧需要面向跨 session 小增量的 batching/scheduling，减少计算碎片化。
- P→D 不应为每个小 arrival 重复支付完整同步生命周期；应支持合并、覆盖或累计 delta，并把 arrival prefill 与 D 同步解耦。
- 完整 prompt 控制面、Talker 全历史拼接和串行 Orchestrator 属于独立工程问题，应先消除，避免污染 engine research 结论。

当前数据直接证明的是视频 arrival prefill；不能把结论直接表述为已经验证了高频碎片化音频 prefill。
