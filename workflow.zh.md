# DuplexOmni 实时多用户 Serving 工作流

## 目标与边界

目标是在 480 ms 实时时钟下测量 DuplexOmni 的多用户容量，并定位容量失稳时的 engine 根因。研究对象是延迟、调度、batching、prefix/KV cache 和 stage pipeline，不是模型质量。

DuplexOmni 与 `thinker-talker-vllm` 的 workload 不同。后者主要是持续视频 prefill、query 时集中生成；DuplexOmni 每个 480 ms slot 都执行完整预测：

```text
480 ms PCM + 当前图像
  → Thinker 增量 prefill + 约 24 个文本/控制 token decode
  → Talker 生成 6 帧、16 codebook 的 codec，并验证 EOS
  → Code2Wav 生成约 457 ms 波形
```

因此每用户约产生 `24 / 0.48 ≈ 50` Thinker decode tokens/s，并持续运行 Talker。两个 branch 的用户数不能直接比较。

## 阶段一：接入模型协议

新增独立 `duplexomni` pipeline，复用 Qwen3-Omni 的 Thinker、Talker 和 Code2Wav 权重结构，但按 DuplexOmni 协议连接三个 stage：

- 捕获 Thinker embedding 和最终归一化 hidden state，并按训练位置对齐；最后一个无后继 hidden 的 token 不传给 Talker。
- Talker prompt 按 `assistant conditioning → codec BOS → 6 RVQ frames → codec EOS` 组织。
- 每帧包含 16 个 codebook；Talker 自回归生成 layer-0，MTP 生成其余 15 个。
- 只把完成 6 帧且产生 EOS 的 turn 写入 Talker codec history。
- API 返回结构化 Thinker 控制字段、`16×6` codec、EOS/valid 标记、波形和各 stage 指标。
- Thinker/Talker 支持在线 W8A8 FP8 和 per-token/per-head FP8 KV；Code2Wav 保持 BF16。

这不是把普通 Qwen3-Omni 请求改名。checkpoint、每 480 ms 的控制输出和持续 codec 生成共同定义了原生 Duplex workload。

## 阶段二：应用与 engine 边界

应用维护 WebSocket session 和对话状态；engine 每个 slot 只处理一个普通、有限生命周期的 request：

```text
媒体持续到达 → 应用组成 480 ms slot 和 canonical prompt
             → 新 finite request → engine prefix/KV cache
             → request 完成并销毁
```

- 每个 session 每 480 ms 封装一段 24 kHz mono PCM；当前 slot 只携带最近一张通过 filter 的图片。
- 图片按 similarity/freshness filter 处理，默认阈值 0.95，最多连续 4 帧后强制保留；manifest 分别记录 sent、acked 和 accepted。
- 应用缓存已处理的 canonical message block，只 render 新 user/assistant block，再拼接完整 prompt。
- 每个 epoch 使用独立 `cache_salt`。engine cache 可随时淘汰；cache miss 只增加计算，不改变语义。
- prompt 达到 6,144 Thinker tokens 时，先 drain 当前 epoch，再只保留最近一个完整 slot，建立新 cache lineage。该阈值来自单用户延迟拐点，不是模型 context 上限。
- 默认每个 session 最多 4 个 slot 在途。Thinker 只等待前一个 Thinker 完成，不等待其 Talker/Code2Wav：

```text
Thinker(t) ─────────→ Thinker(t+1)
    └→ ordered Talker(t) → Code2Wav(t)
```

- Talker history 必须有序。orchestrator 在前一 Talker 完成后提交下一 Talker，并注入已验证的 codec history；不同 session 互不串行。
- Talker 的动态 conditioning embedding 使用由 Thinker prefix 和媒体 hash 生成的 cache identity，避免错误 prefix 命中。

结果是：应用有状态，engine request 有限，跨 slot 只依赖可丢弃的 prefix/KV cache；Duplex 语义不依赖 persistent engine request。

## 阶段三：固定部署

硬件为 3 张 NVIDIA RTX PRO 6000 Blackwell Server Edition，每张 96 GB：

| Stage | GPU | 配置 |
|---|---:|---|
| Thinker | 0 | FP8 weight、FP8 KV、prefix cache、`max_num_seqs=16` |
| Talker + MTP | 1 | FP8 weight、FP8 KV、conditioning prefix cache、`max_num_seqs=32` |
| Code2Wav | 2 | BF16、无 prefix cache |

三个 stage 使用 shared-memory connector；调度均为同步模式。正式配置为 `benchmarks/duplexomni/deploy_fp8_3gpu.yaml`，BF16 配置只用于回归。

```bash
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-server \
bash benchmarks/duplexomni/run_server.sh fp8
```

## 阶段四：正式 workload 与指标

- 每用户一条 server-owned WebSocket session。
- 每 480 ms 发送一段 PCM 和一张图，即每用户 2.08 requests/s、2.08 FPS。
- 用户相位独立采样自 `Uniform[0, 480 ms)`，不制造同步 burst。
- 不同用户的 PCM 加不可听 LSB dither，JPEG 加极小角标，避免不真实的跨用户多模态 cache 命中。
- 默认语音来自 `tests/assets/minicpmo_4_5/response_required_16k.wav` 并重采样到 24 kHz；未指定 `--image` 时使用确定性生成帧。该 workload 测量 serving 负载，不用于评估语义质量。
- 容量点使用 60 slots/user。每次使用新 session ID 和 cache lineage；测量前先 warm kernel。

核心指标：

- **E2E slot latency**：计划到达时间到完整 slot 结果返回。
- **Request latency**：服务端提交 finite request 到完整结果返回。
- **Thinker latency**：服务端提交到 Thinker 控制文本完成。
- **Application queue**：slot 输入就绪到服务端提交。
- **Deadline miss**：E2E slot latency 超过 480 ms。

Strict realtime 要求 E2E p99 ≤ 480 ms 且 miss rate ≤ 1%。若最后三分之一的 application-queue p50 比最初三分之一增加超过 480 ms，则判定 throughput collapse。

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/multi_user.py \
  --users 2 --slots 60 --seed 8001 \
  --session-prefix capacity-current-u2-long \
  --output /tmp/duplexomni-capacity-current-2x60

/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/analyze_capacity.py \
  1x60=/tmp/duplexomni-capacity-current-1x60 \
  2x60=/tmp/duplexomni-capacity-current-2x60 \
  3x60=/tmp/duplexomni-capacity-current-3x60
```

## 阶段五：功能与长 session 验证

FP8 是当前默认部署。它保持结构化控制、`16×6` codec、EOS 和波形协议正确，但不声称与 BF16 文本或波形语义等价。

单用户 300-slot AV 长测结果：

| 指标 | 结果 |
|---|---:|
| 完成与有效 codec/EOS | 300/300 |
| E2E p50/p95/p99/max | 376/442/462/463 ms |
| Request p99/max | 456/459 ms |
| Application queue p99 | 0.50 ms |
| Prompt render p99 | 16.3 ms |
| Deadline miss | 0 |

测试发生 8 次 context compaction，prompt 最大 6,217 tokens；压缩没有造成可见延迟尖峰。结果保存在 `/tmp/duplexomni-fixed-long-300/manifest.json`。

## 阶段六：当前容量

同一 warmed FP8 server、seed 8001、每用户 60 slots：

| Users | E2E p50/p99 | Request p50/p99 | Thinker p50/p99 | Queue p99 | Miss | Queue growth | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 373/462 ms | 368/457 ms | 282/354 ms | 0.46 ms | 0% | 0.00 ms | strict realtime |
| 2 | 444/570 ms | 437/555 ms | 343/446 ms | 26.9 ms | 22.5% | 0.00 ms | throughput 稳定，SLO 失败 |
| 3 | 1067/1758 ms | 639/836 ms | 466/622 ms | 1202 ms | 100% | 569 ms | throughput collapse |

当前 strict capacity 为 1 用户；不要求每个 slot 都在 480 ms 内时，2 用户仍能持续跟上输入，3 用户是容量拐点。

## 阶段七：Prefill 对 Thinker decode 的因果验证

实验固定 8 个完整 Duplex probe session，每个 12 slots；另加入 0/4/8/16 个相同 AV cadence 的 prefill-only session。后者只建立 Thinker KV，不生成任何 token、不进入 Talker，trace 中 `decode_entries=0`。实验重复两次。

| Prefill-only users | Thinker p99，run 1 | Thinker p99，run 2 |
|---:|---:|---:|
| 0 | 879 ms | 851 ms |
| 4 | 1123 ms | 1059 ms |
| 8 | 1393 ms | 1435 ms |
| 16 | 2168 ms | 2720 ms |

在 16 个 prefill-only session 下：

- 不含任何 prefill 的 clean decode gap p99 为 16–17 ms。
- 暴露于后台 prefill 的 decode gap p99 为 315–316 ms。
- decode-only GPU batch p50 为约 8.4 ms；prefill 与 probe decode 混合 batch p50 为约 52.7 ms。
- 将同样的 16×12 prefill 全部先完成，再运行 probes，Thinker p99 为 820 ms；只有并发时升至 2.17–2.72 s。

因此增加的 tail 来自并发 prefill 拉长 decode token cadence，不是后台 session 偷做了 decode，也不是单纯增加了相同总工作量。

复现入口：

```bash
VLLM_OMNI_LOG_PD_ITER=1 VLLM_OMNI_PD_ITER_STAGE=0 \
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-prefill-causal-server \
bash benchmarks/duplexomni/run_server.sh fp8

/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/multi_user.py \
  --users 8 --prefill-only-users 16 --slots 12 \
  --output /tmp/duplexomni-prefill-causal-check
```

## 阶段八：当前结论

1. DuplexOmni 容量低于 query-driven Qwen workload 的第一原因是持续 decode：每用户约 50 Thinker tokens/s，并且每 slot 都运行 Talker codec decode。
2. 持续 AV prefill 是明确的 tail 放大器。它与持续 Thinker decode 共享 GPU batch，使 decode gap 从十几毫秒扩大到数百毫秒。
3. 高频 finite request 的固定调度成本、小 batch 和 context 增长进一步降低效率，但不是完整历史重复 prefill；prefix cache 已按 append-only lineage 命中。
4. 6144-token compaction 已解决单用户长 session 的 context 延迟增长。当前容量首先受 Thinker 限制，Talker/Code2Wav 不是第一瓶颈。
5. 应用/session/finite-request 边界合理。下一步 engine research 应针对“持续短 decode + 小增量多模态 prefill”做 deadline/QoS-aware batching 和调度，而不是把跨 slot 生命周期重新塞回一个 persistent request。

## 阶段九：DuplexOmni P/D 部署

`duplexomni-pd` 保持阶段二的应用/session/finite-request 设计不变，只拆分 Thinker engine：

| Stage | GPU | 配置与职责 |
|---|---:|---|
| Thinker-P | 0 | FP8 weight/KV；多模态 prompt prefill；保存 layer 0 和 layer 48 hidden snapshot |
| Thinker-D | 1 | FP8 weight/KV；导入 P 的 KV 后生成约 24 个文本/控制 token |
| Talker + MTP | 2 | FP8 weight/KV；生成 `16×6` codec |
| Code2Wav | 3 | BF16；生成波形 |

P/D 使用 `NixlDeltaPushConnector`。首个 slot 建立完整 lineage；后续 slot 依赖应用提供的精确 prefix lineage，P 只向 D 传新增 KV block。D 同时输出客户端可见文本，并把完整 decode hidden rows 与 P 的 prompt snapshot 组合后交给 Talker。为此，scheduler→runner 数据契约显式保留 `pd_prefill_payload`；否则 D 虽能生成文本，Talker 会缺少 prompt hidden state。

正式配置为 `benchmarks/duplexomni/deploy_pd_fp8_4gpu.yaml`；BF16 回归配置为 `deploy_pd_bf16_4gpu.yaml`。

```bash
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-pd-server \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
bash benchmarks/duplexomni/run_server.sh pd

/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/single_user.py \
  --label fp8 --slots 12 --output /tmp/duplexomni-pd-1x12
```

warmed 单用户 12-slot AV 结果：

| 指标 | 结果 |
|---|---:|
| 有效 codec/EOS | 12/12 |
| E2E p50/p95/p99/max | 361/387/390/391 ms |
| Request p50/p95/p99/max | 357/383/387/387 ms |
| Thinker p50/p95/p99/max | 273/298/301/301 ms |
| Application queue p99 | 0.16 ms |
| Deadline miss | 0 |
| P→D delta load | 54–81 ms |

所有 slot 都经过 P、D、Talker 和 Code2Wav；每轮返回 `[16,6]` codec 和 10,965 个 24 kHz audio samples。首请求包含 NIXL 握手、hidden cache 初始化和 shape JIT，不能计入稳态延迟。

### P/D 优化继承

`duplexomni-pd` 基于 `thinker-talker-pd` 的最新提交，已直接继承以下通用优化：delta KV、跨 layer block packing、P 计算期间提前注册 D、独立 P 输出 worker、shared-tensor IPC、event-loop 外 snapshot 压缩，以及非阻塞 stage 输出消费。没有移植 Qwen 专用的 arrival admission、`async_chunk` 和 summary 压缩；DuplexOmni 使用固定 480 ms slot 和 6144-token epoch compaction。

容量分析器已适配四阶段 P/D 指标，分别报告 Thinker-P、Thinker-D、Talker 和 Code2Wav，避免把 D 误标为 Talker。

### P/D 容量

正式测量使用 60 slots/user、连续 audio + 每 slot 一张图、随机用户相位、不同用户媒体和 warmed engine。frame filter 每用户接收 32/60 张图；所有用户采用相同规则。SLO 和 collapse 判据沿用阶段四。

| 用户 | E2E p50/p99 | Request p50/p99 | App queue p99 | Miss | Queue p50 增长 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 386/484 ms | 382/479 ms | 0.47 ms | 3.3% | 0 ms | 严格 SLO 边界外 |
| 2 | 397/515 ms | 393/511 ms | 0.82 ms | 10.8% | 0 ms | 吞吐稳定，SLO 失败 |
| 3 | 519/942 ms | 505/696 ms | 413 ms | 67.8% | 0 ms | 服务时间拐点 |
| 4 | 748/1463 ms | 583/840 ms | 823 ms | 94.6% | 342 ms | 明显积压 |
| 5 | 2017/3367 ms | 644/856 ms | 2751 ms | 98.0% | 2153 ms | throughput collapse |

按预先定义的 `p99 <= 480 ms 且 miss <= 1%`，60-slot 长测没有通过严格实时标准的容量点；1 用户仅超出约 4 ms。吞吐崩溃发生在 5 用户，4 用户是未触发 collapse 判据的最高点，但已不具备实时可用性。

| Engine stage | 1 user p50/p99 | 5 users p50/p99 |
|---|---:|---:|
| Thinker-P | 69/111 ms | 95/183 ms |
| Thinker-D | 205/242 ms | 374/544 ms |
| Talker | 73/82 ms | 104/309 ms |
| Code2Wav | 9/10 ms | 9/28 ms |

容量首先受 Thinker-D 限制：每用户每 480 ms 都生成约 24 个 Thinker token，约等于 50 tokens/s/user 的持续 decode。5 用户时 D GPU busy p50/p95 为 70%/80%，P 为 39%/74%；两者 SM-active p95 仅约 46%/49%，因此不是单纯的算力或显存带宽打满。当前 tail 是持续 decode、短 finite request 的固定调度成本和 P/D pipeline 等待共同形成的服务率上限；P→D delta handoff 和 Code2Wav 不是第一瓶颈。5 用户的队列在 context compaction 前已经持续增长，压缩只会改变局部 tail，不是 collapse 根因。

## 快速恢复入口

| 内容 | 路径 |
|---|---|
| WebSocket session、slot、filter、压缩 | `vllm_omni/entrypoints/openai/serving_duplexomni_stream.py` |
| Thinker/Talker 跨 slot 顺序 | `vllm_omni/engine/duplexomni_pipeline.py` |
| Thinker→Talker 协议与 cache identity | `vllm_omni/model_executor/stage_input_processors/duplexomni.py` |
| 三/四阶段 pipeline | `vllm_omni/model_executor/models/duplexomni/pipeline.py` |
| 非 P/D / P/D 部署 | `benchmarks/duplexomni/deploy_fp8_3gpu.yaml`、`benchmarks/duplexomni/deploy_pd_fp8_4gpu.yaml` |
| 单/多用户 workload | `benchmarks/duplexomni/single_user.py`、`benchmarks/duplexomni/multi_user.py` |
| 容量分析 | `benchmarks/duplexomni/analyze_capacity.py` |
| Prefill 因果分析 | `benchmarks/duplexomni/analyze_prefill_causal.py` |
| 回归测试 | `tests/model_executor/stage_input_processors/test_duplexomni.py`、`tests/engine/test_duplexomni_pipeline.py`、`tests/benchmarks/test_duplexomni_harness.py` |
