# DuplexOmni Realtime Multi-User Serving Workflow

## Goal and scope

The goal is to measure DuplexOmni multi-user capacity on its 480 ms realtime clock and identify the engine cause when capacity becomes unstable. The research target is latency, scheduling, batching, prefix/KV cache, and the stage pipeline rather than model quality.

DuplexOmni does not have the same workload as `thinker-talker-vllm`. The latter mainly performs continuous video prefill and generates at query time. DuplexOmni performs a complete prediction every 480 ms slot:

```text
480 ms PCM + current image
  → incremental Thinker prefill + about 24 text/control decode tokens
  → Talker generates six 16-codebook codec frames and validates EOS
  → Code2Wav generates about 457 ms of waveform
```

Each user therefore produces about `24 / 0.48 ≈ 50` Thinker decode tokens/s and continuously runs Talker. User capacity from the two branches is not directly comparable.

## Phase 1: model-protocol integration

A dedicated `duplexomni` pipeline reuses the Qwen3-Omni Thinker, Talker, and Code2Wav module structure while connecting the stages with the DuplexOmni protocol:

- Capture the Thinker embedding and final normalized hidden state with training-position alignment. The last token, which has no successor hidden row, is not sent to Talker.
- Build Talker prompts as `assistant conditioning → codec BOS → six RVQ frames → codec EOS`.
- Each frame has 16 codebooks. Talker autoregressively generates layer 0, and MTP produces the other 15.
- Commit a Talker turn to codec history only after six frames and EOS are both present.
- Return structured Thinker controls, `16×6` codec data, EOS/valid flags, waveform, and per-stage metrics through the API.
- Support online W8A8 FP8 and per-token/per-head FP8 KV for Thinker and Talker. Code2Wav remains BF16.

This is not an ordinary Qwen3-Omni request under another name. The checkpoint, per-480 ms control output, and continuous codec generation jointly define the native Duplex workload.

## Phase 2: application and engine boundary

The application owns the WebSocket session and dialogue state. The engine handles one ordinary finite request per slot:

```text
continuous media → application assembles a 480 ms slot and canonical prompt
                 → new finite request → engine prefix/KV cache
                 → destroy the request after completion
```

- Each session seals one 24 kHz mono PCM slice every 480 ms. A slot carries only the newest image accepted by the filter.
- Images pass a similarity/freshness filter. The default similarity threshold is 0.95, with forced retention after at most four rejected frames. The manifest records sent, acknowledged, and accepted frames separately.
- The application caches processed canonical message blocks, renders only each new user/assistant block, and then assembles the complete prompt.
- Every epoch uses a distinct `cache_salt`. Engine cache is disposable; a miss changes compute cost but not semantics.
- When the prompt reaches 6,144 Thinker tokens, the application drains the epoch, retains only the newest complete slot, and starts a new cache lineage. This threshold comes from the measured one-user latency knee, not the model context limit.
- A session permits at most four in-flight slots by default. Thinker waits only for the preceding Thinker, not for its Talker or Code2Wav:

```text
Thinker(t) ─────────→ Thinker(t+1)
    └→ ordered Talker(t) → Code2Wav(t)
```

- Talker history remains ordered. After the predecessor Talker completes, the orchestrator submits the successor Talker with verified codec history. Different sessions are not globally serialized.
- Dynamic Talker conditioning uses cache identities derived from the Thinker prefix and media hashes, preventing false prefix hits.

The result is a stateful application with finite engine requests and only disposable prefix/KV reuse across slots. Duplex semantics do not depend on a persistent engine request.

## Phase 3: fixed deployment

The hardware is three NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 96 GB each:

| Stage | GPU | Configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights, FP8 KV, prefix cache, `max_num_seqs=16` |
| Talker + MTP | 1 | FP8 weights, FP8 KV, conditioning prefix cache, `max_num_seqs=32` |
| Code2Wav | 2 | BF16, no prefix cache |

The stages use a shared-memory connector and synchronous scheduling. Formal runs use `benchmarks/duplexomni/deploy_fp8_3gpu.yaml`; BF16 is retained only for regression.

```bash
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-server \
bash benchmarks/duplexomni/run_server.sh fp8
```

## Phase 4: formal workload and metrics

- Each user owns one server-managed WebSocket session.
- Every 480 ms, each user sends one PCM slice and one image: 2.08 requests/s and 2.08 FPS per user.
- Session phases are sampled independently from `Uniform[0, 480 ms)`, avoiding an artificial synchronized burst.
- Each user's PCM receives inaudible LSB dither, and each JPEG receives a tiny corner mark, preventing unrealistic cross-user multimodal cache hits.
- The default speech input is `tests/assets/minicpmo_4_5/response_required_16k.wav`, resampled to 24 kHz. When `--image` is omitted, deterministic generated frames are used. The workload measures serving load, not semantic quality.
- Capacity points use 60 slots/user. Each point uses new session IDs and cache lineages after kernel warm-up.

Primary metrics:

- **E2E slot latency**: scheduled arrival to the complete slot response.
- **Request latency**: server submission of the finite request to complete response.
- **Thinker latency**: server submission to completed Thinker control text.
- **Application queue**: slot input readiness to server submission.
- **Deadline miss**: E2E slot latency above 480 ms.

Strict realtime requires E2E p99 at or below 480 ms and a miss rate at or below 1%. Throughput collapse is declared when application-queue p50 grows by more than 480 ms from the first third to the final third.

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

## Phase 5: functional and long-session validation

FP8 is the default deployment. It preserves the structured-control, `16×6` codec, EOS, and waveform contracts, but no claim is made that its text or waveform is semantically equivalent to BF16.

One-user, 300-slot AV result:

| Metric | Result |
|---|---:|
| Completed and valid codec/EOS | 300/300 |
| E2E p50/p95/p99/max | 376/442/462/463 ms |
| Request p99/max | 456/459 ms |
| Application queue p99 | 0.50 ms |
| Prompt render p99 | 16.3 ms |
| Deadline misses | 0 |

The run performed eight context compactions, with a maximum prompt of 6,217 tokens. Compaction produced no visible latency spike. The result is stored at `/tmp/duplexomni-fixed-long-300/manifest.json`.

## Phase 6: current capacity

Same warmed FP8 server, seed 8001, and 60 slots/user:

| Users | E2E p50/p99 | Request p50/p99 | Thinker p50/p99 | Queue p99 | Miss | Queue growth | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 373/462 ms | 368/457 ms | 282/354 ms | 0.46 ms | 0% | 0.00 ms | strict realtime |
| 2 | 444/570 ms | 437/555 ms | 343/446 ms | 26.9 ms | 22.5% | 0.00 ms | throughput-stable, SLO failure |
| 3 | 1067/1758 ms | 639/836 ms | 466/622 ms | 1202 ms | 100% | 569 ms | throughput collapse |

Strict capacity is one user. If every slot is not required to finish within 480 ms, two users still keep pace with the input clock; three users are the capacity knee.

## Phase 7: causal validation of prefill interference

The experiment fixes eight full Duplex probe sessions at 12 slots each and adds 0/4/8/16 prefill-only sessions with the same AV cadence. A prefill-only session materializes Thinker KV, generates no token, and never reaches Talker; its trace has `decode_entries=0`. The experiment was repeated twice.

| Prefill-only users | Thinker p99, run 1 | Thinker p99, run 2 |
|---:|---:|---:|
| 0 | 879 ms | 851 ms |
| 4 | 1123 ms | 1059 ms |
| 8 | 1393 ms | 1435 ms |
| 16 | 2168 ms | 2720 ms |

With 16 prefill-only sessions:

- Clean decode gaps with no prefill have a 16–17 ms p99.
- Decode gaps exposed to background prefill have a 315–316 ms p99.
- Decode-only GPU batches have an approximately 8.4 ms p50; mixed prefill/probe-decode batches have an approximately 52.7 ms p50.
- Running the same 16×12 prefills to completion before the probes produces an 820 ms Thinker p99. Only concurrent execution raises it to 2.17–2.72 s.

The added tail therefore comes from concurrent prefill stretching decode-token cadence. It is not hidden background decode and is not explained merely by adding the same amount of total work.

Reproduction entry point:

```bash
VLLM_OMNI_LOG_PD_ITER=1 VLLM_OMNI_PD_ITER_STAGE=0 \
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-prefill-causal-server \
bash benchmarks/duplexomni/run_server.sh fp8

/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/multi_user.py \
  --users 8 --prefill-only-users 16 --slots 12 \
  --output /tmp/duplexomni-prefill-causal-check
```

## Phase 8: current conclusions

1. The first-order reason DuplexOmni capacity is lower than a query-driven Qwen workload is continuous decode: about 50 Thinker tokens/s/user plus Talker codec decode for every slot.
2. Continuous AV prefill is a causal tail amplifier. It shares GPU batches with continuous Thinker decode and expands decode gaps from tens to hundreds of milliseconds.
3. Per-request fixed costs, small batches, and context growth further reduce efficiency, but the system is not repeatedly prefilling the complete history; prefix cache follows the append-only lineage.
4. The 6,144-token compaction policy removes one-user long-session context growth. Thinker is the first capacity limit; Talker and Code2Wav are not the primary bottleneck.
5. The application/session/finite-request boundary is sound. The next engine research target is deadline/QoS-aware batching and scheduling for continuous short decode plus small incremental multimodal prefill, not another cross-slot persistent request.

## Phase 9: DuplexOmni P/D deployment

`duplexomni-pd` preserves the Phase 2 application/session/finite-request design and splits only the Thinker engine:

| Stage | GPU | Configuration and role |
|---|---:|---|
| Thinker-P | 0 | FP8 weights/KV; multimodal prompt prefill; layer-0 and layer-48 hidden snapshot |
| Thinker-D | 1 | FP8 weights/KV; imports P KV and generates about 24 text/control tokens |
| Talker + MTP | 2 | FP8 weights/KV; generates the `16×6` codec |
| Code2Wav | 3 | BF16; generates the waveform |

P/D uses `NixlDeltaPushConnector`. The first slot establishes the complete lineage. Later slots use the exact application-provided prefix lineage, so P transfers only new KV blocks to D. D is both the client-visible text stage and the Talker producer: its complete decode hidden rows are combined with P's prompt snapshot before Talker starts. The scheduler-to-runner contract therefore preserves `pd_prefill_payload`; without it, D can generate text but Talker lacks prompt hidden states.

The formal config is `benchmarks/duplexomni/deploy_pd_fp8_4gpu.yaml`; `deploy_pd_bf16_4gpu.yaml` is the BF16 regression config.

```bash
DUPLEXOMNI_RESULTS_DIR=/tmp/duplexomni-pd-server \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
bash benchmarks/duplexomni/run_server.sh pd

/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/duplexomni/single_user.py \
  --label fp8 --slots 12 --output /tmp/duplexomni-pd-1x12
```

Warmed one-user, 12-slot AV result:

| Metric | Result |
|---|---:|
| Valid codec/EOS | 12/12 |
| E2E p50/p95/p99/max | 361/387/390/391 ms |
| Request p50/p95/p99/max | 357/383/387/387 ms |
| Thinker p50/p95/p99/max | 273/298/301/301 ms |
| Application queue p99 | 0.16 ms |
| Deadline misses | 0 |
| P-to-D delta load | 54–81 ms |

Every slot traversed P, D, Talker, and Code2Wav and returned a `[16,6]` codec tensor plus 10,965 24 kHz audio samples. The first request pays NIXL handshake, hidden-cache initialization, and shape JIT, so it is not a steady-state latency sample.

### Inherited P/D optimizations

`duplexomni-pd` is based on the latest `thinker-talker-pd` commit and already inherits the generic delta-KV, cross-layer block packing, early D registration during P compute, dedicated P-output worker, shared-tensor IPC, off-event-loop snapshot compaction, and nonblocking stage-output consumption optimizations. Qwen-specific arrival admission, `async_chunk`, and summary compaction were not copied; DuplexOmni uses fixed 480 ms slots and 6144-token epoch compaction.

The capacity analyzer now understands the four-stage P/D layout and reports Thinker-P, Thinker-D, Talker, and Code2Wav separately instead of mislabeling D as Talker.

### P/D capacity

The formal run used 60 slots/user, continuous audio plus one image per slot, independent random phases, distinct cross-user media, and a warmed engine. The frame filter accepted 32/60 frames per user under the same policy. The SLO and collapse definitions are unchanged from Phase 4.

| Users | E2E p50/p99 | Request p50/p99 | App queue p99 | Miss | Queue-p50 growth | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 386/484 ms | 382/479 ms | 0.47 ms | 3.3% | 0 ms | Just outside strict SLO |
| 2 | 397/515 ms | 393/511 ms | 0.82 ms | 10.8% | 0 ms | Stable throughput, SLO failure |
| 3 | 519/942 ms | 505/696 ms | 413 ms | 67.8% | 0 ms | Service-time knee |
| 4 | 748/1463 ms | 583/840 ms | 823 ms | 94.6% | 342 ms | Material backlog |
| 5 | 2017/3367 ms | 644/856 ms | 2751 ms | 98.0% | 2153 ms | Throughput collapse |

Under the predefined `p99 <= 480 ms and miss <= 1%` rule, no 60-slot point passes strict realtime; one user misses the p99 bound by about 4 ms. Throughput collapse begins at five users. Four is the highest point below the formal collapse threshold, but is not realtime-usable.

| Engine stage | 1 user p50/p99 | 5 users p50/p99 |
|---|---:|---:|
| Thinker-P | 69/111 ms | 95/183 ms |
| Thinker-D | 205/242 ms | 374/544 ms |
| Talker | 73/82 ms | 104/309 ms |
| Code2Wav | 9/10 ms | 9/28 ms |

Thinker-D is the first capacity limit. Every user generates about 24 Thinker tokens per 480 ms slot, or roughly 50 continuously decoded tokens/s/user. At five users, D GPU busy p50/p95 is 70%/80% versus 39%/74% on P; SM-active p95 is only about 46%/49%, so neither GPU is simply compute- or memory-bandwidth-saturated. The tail reflects the service-rate limit formed by continuous decode, fixed per-request scheduling costs, and P/D pipeline waiting. Delta-KV handoff and Code2Wav are not the primary bottlenecks. The five-user queue starts growing before context compaction, so compaction changes local tails but does not cause the collapse.

## Recovery map

| Purpose | Path |
|---|---|
| WebSocket session, slots, filtering, compaction | `vllm_omni/entrypoints/openai/serving_duplexomni_stream.py` |
| Cross-slot Thinker/Talker ordering | `vllm_omni/engine/duplexomni_pipeline.py` |
| Thinker-to-Talker protocol and cache identity | `vllm_omni/model_executor/stage_input_processors/duplexomni.py` |
| Three/four-stage pipelines | `vllm_omni/model_executor/models/duplexomni/pipeline.py` |
| Non-P/D and P/D deployment | `benchmarks/duplexomni/deploy_fp8_3gpu.yaml`, `benchmarks/duplexomni/deploy_pd_fp8_4gpu.yaml` |
| Single/multi-user workload | `benchmarks/duplexomni/single_user.py`, `benchmarks/duplexomni/multi_user.py` |
| Capacity analysis | `benchmarks/duplexomni/analyze_capacity.py` |
| Prefill causal analysis | `benchmarks/duplexomni/analyze_prefill_causal.py` |
| Regression tests | `tests/model_executor/stage_input_processors/test_duplexomni.py`, `tests/engine/test_duplexomni_pipeline.py`, `tests/benchmarks/test_duplexomni_harness.py` |
