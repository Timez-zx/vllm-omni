# vLLM-Omni real-time multi-user serving workflow

This document records only the current reproducible implementation, fixed experiment method, and confirmed results. The research branch is `thinker-talker-pd`.

## Phase 1: objective and model boundary

The objective is to serve more long-running audio/video sessions with fewer GPUs while meeting first-audio latency and playback-continuity targets. The research target is engine scheduling, KV caching, P/D transfer, capacity, and tail latency—not model quality.

No locally deployable open model currently matches Seed Realtime or Gemini Live interaction semantics. This project approximates that load with Qwen3-Omni's Thinker → Talker → Code2Wav pipeline: clients upload continuous audio/video and the model answers by turn. It creates real multimodal prefill, text decode, and speech-generation contention, but it is not a native full-duplex model and does not study semantic barge-in.

## Phase 2: application/engine boundary

The application owns the session; the engine handles disposable finite requests:

```text
video arrives continuously
  → similarity/freshness filter
  → submit at most one Thinker-only arrival request per session
  → coalesce newer frames into the latest cumulative snapshot instead of queueing every frame
  → ACK after P computes KV and stores its snapshot; the P lineage advances linearly
  → in P/D mode, D Core imports revisions in lineage order in the background
user finishes speaking
  → stop pending snapshot submission and wait only for the admitted arrival's P-ready ACK
  → submit the full canonical context plus one complete WAV
  → Thinker P → Thinker D → Talker → Code2Wav
answer completes
  → destroy the request, persist the turn in the application, use a new request ID next turn
```

Current invariants:

- The WebSocket application owns canonical multimodal history and media attribution. The engine has no cross-turn live request.
- Every warm-up and final answer is an independent finite request carrying the complete canonical prompt. Engine prefix/KV caches are disposable accelerators; a miss changes latency only.
- Accepted video is append-only within the turn. The old latest-eight sliding eviction is gone.
- At most one arrival from a session executes on P. New media marks one latest cumulative snapshot, which is submitted after P-ready. A query stops that coalescing loop and waits for the admitted arrival's P-ready ACK; different sessions still overlap.
- The P KV/snapshot lineage advances linearly and retains only the latest revision; same-session P requests never share a stale parent. D cache-sync may lag, but D Core orders revisions locally without an API result barrier. A final query uses D's actual local prefix and can receive the cumulative missing suffix in one transfer. A miss or parent mismatch changes latency, not correctness.
- Warm-ups use `max_tokens=1` and `output_modalities=["text"]`. They maintain Thinker KV only and never enter Talker. In P/D mode P-ready is the application completion boundary and D cache-sync is a background accelerator; in non-P/D mode they visit only the Thinker. Arrival and final requests use the native FCFS scheduler.
- Audio is submitted as one complete WAV at query time. Qwen's audio encoder is bidirectional, so independently encoded audio chunks are not guaranteed to preserve whole-audio semantics.
- When the rendered prompt reaches 49,152 tokens, the newest two completed turns retain the full user audio, user text, and assistant text but discard historical images. The in-flight turn retains its full audio/text and only the newest image accepted by the filter. History then grows normally until the next threshold. No summary request is generated.

This matches the production pattern of application-owned sessions plus ordinary engine requests accelerated by prefix caching, and cleanly exposes engine scheduling and P/D separation. The old cross-turn persistent request, resumable append, Talker rolling, and shadow-request paths are no longer used.

## Phase 3: fixed deployment and workload

### P/D deployment: four GPUs

Upstream vLLM-Omni could not split one Thinker into separate P and D stages while continuing to supply Talker conditioning states; this branch implements that path. Formal P/D runs must use `benchmarks/thinker_talker/pd_deploy_4gpu.yaml`:

| GPU | Stage |
|---:|---|
| 0 | Thinker P |
| 1 | Thinker D |
| 2 | Talker |
| 3 | Code2Wav |

`origin_deploy_3gpu.yaml` is only the non-P/D control: one GPU each for Thinker, Talker, and Code2Wav. It cannot support a P/D conclusion.

P→D uses Delta-KV push; D→Talker→Code2Wav uses shared memory. The current P/D YAML SHA256 is `844cec83bf53ae26f9636054ba2ca2bee65b7bdf3deedbb90ed14124715d5244`.

### Continuous-AV session workload

- Each user owns one long-lived WebSocket.
- Video runs continuously at 2 FPS. During an input turn every accepted frame immediately triggers a cumulative-snapshot arrival request; frames received during an answer are buffered and form the next turn's first cumulative snapshot after that answer completes.
- PCM16 microphone audio is sent at 5 Hz, paused during assistant playback, with a 300 ms echo guard.
- Each turn uses a unique real mono 16 kHz SLURP recording plus 700 ms endpoint silence. Query text is empty and the complete WAV is submitted at query time.
- Each session keeps one speaker. Video comes from fixed DAVIS sequences with different user offsets.
- The next turn begins after 1× playback of the previous answer. User starts are deterministically staggered over 0–8 seconds.

Use one or two users for protocol smoke tests. Capacity runs use 30 turns/user, exclude the first two turns, and increase through 8, 16, 32, ... users. Restart the engine for every cell and stop at the first SLO failure.

Sixteen-user P/D entry point:

The P/D launcher defaults `VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES=34359738368` (32 GiB of host memory). Override it with `8589934592` to reproduce the 8-GiB A/B arm.

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/pd_capacity_<commit> \
RESULT_PREFIX=pd_capacity USERS=16 SEEDS=7 TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh
```

Do not invoke the generic `run_av_session_ladder.sh` for P/D; it defaults to the three-GPU non-P/D baseline.

## Phase 4: metrics and validation

- **TTFA / Audio-ready-500**: query to the first playable 500 ms of audio. The current first packet exceeds 500 ms, so the two values are identical.
- **TTFT**: query to first text, separating the Thinker path from speech startup.
- **Stall max**: largest single underflow during 1× playback.
- **RTF deliver**: output-audio duration divided by delivery time; above one means delivery is faster than playback.

A capacity cell passes only if every scored turn completes, Audio-ready-500 p99 is below 1 s, Stall-max p99 is below 50 ms, and client, protocol, and fatal engine checks are clean.

```bash
MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml \
MU_EXPECTED_STAGE_IDS=0,1,2,3 \
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

The verifier checks deployment, all four stages, finite-request uniqueness, arrival warm-ups, the frame ledger, and observed prefix-cache hits.

## Phase 5: confirmed implementation decisions

1. After context alignment, finite requests have no inherent latency disadvantage versus the old persistent request. Application-owned sessions remain the research baseline.
2. Video arrival warm-ups approximate continuous multimodal prefill. Formal audio stays as one complete query-time WAV to preserve Qwen semantics.
3. Earlier multi-second P→D waits were connector defects, not an inherent PCIe or P/D cost. Each arrival ACK now means only that its P snapshot is stored; the application advances the linear P lineage while D Core imports revisions in order without waiting for utility results.
4. P conditioning states support lineage deltas and shared tensor handles. Each lineage retains only its latest snapshot; a missing declared parent falls back to a full snapshot for correctness.
5. Same-session P arrivals are strictly serialized and intermediate updates are coalesced; a query waits only for the admitted arrival's P-ready ACK, not D-ready. Different sessions still enter native FCFS concurrently, with no global arrival gate or application priority.
6. Context is compacted only at the 49,152-token limit: the newest two completed turns retain audio/text but discard images, while the in-flight turn retains full audio/text and only its newest image. Normal growth then resumes. There is no proactive compaction or summary request.

This refactor also fixed P chunked-prefill hit detection, Talker's fixed three-row prefix for very short output, and the P snapshot payload/ack protocol. Linear lineage, arrival coalescing, atomic multimodal submission, and P/D cache sync have focused coverage.

The old D-ready strictly serialized smoke test (P/D, 8 users × 6 turns, 2 warm-up turns) completed 32/32 measured turns with no timeout or stall and TTFA p50/p95/p99 of 386/521/557 ms. It is a historical baseline, not a measurement of the current P-ready implementation.

A rejected experiment allowed same-session arrivals to overlap and finish out of order. On the identical 16-user × 12-turn plan, TTFA changed from the linear implementation's 770/1,473/2,261 ms to 693/4,675/6,323 ms. Concurrent requests shared stale parents: total snapshot suffix grew from 634k to 1.738M tokens, full snapshots from 16 to 59, and P→D delta from 634k to 938k. Sequence fences protected state correctness but destroyed linear incremental reuse, so that design was removed.

## Phase 6: 16-user × 30-turn P/D diagnosis

The first part of this phase uses the old same-session D-ready serialization boundary as reproducible experimental background. A same-plan P-ready comparison appears at the end of the phase.

An identical-workload 8/32-GiB A/B had TTFA p99 values of 5,054/5,285 ms, while the 32-GiB snapshot cache peaked at only 18,747 MiB. Host snapshot capacity was therefore not the cause.

Two protocol defects were then fixed:

1. `prefill_only/final_stage_id=0` does not enter D/Talker but must still return layer-0/layer-24 snapshots to the orchestrator. The runner now separates those routing decisions and freezes payload routing before async request cleanup.
2. A mixed full/delta batch needs both the merged source and raw delta tails. The old builder dropped delta payloads whenever the same batch contained a full request.

The orchestrator now reports success only after storing a usable snapshot, and the application advances lineage only after that acknowledgment. All 114 focused tests and a four-user smoke pass.

Formal result: `/home/ubuntu/data/results/pd_snapshot_ack_u16_t30_20260824_v2/pd_snapshot_ack_seed7_u16`

It uses the same `workload_plan.json` as the pre-fix 32-GiB run (SHA256 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`). All 480 turns and 448/448 scored turns completed with zero timeout, skip, client error, or stall. All 6,178 arrival requests were finite; the frame ledger and prefix-cache checks pass.

| Metric | Pre-fix 32 GiB | Fixed |
|---|---:|---:|
| TTFA p50/p95/p99 | 1,267/4,400/5,285 ms | 780/1,924/3,006 ms |
| Session serial wait p50/p95/p99 | 52/1,604/2,305 ms | 0/501/855 ms |
| Thinker-P p50/p95/p99 | 532/2,096/2,396 ms | 267/790/1,120 ms |
| P cache-miss tokens p50/p95/p99 | 129/449/808 | 110/325/411 |
| Scored queries using full snapshots | 83/448 | 0/448 |

The run prepared 6,704 snapshots and stored 6,701. The remaining three were explicitly aborted by the application during session teardown, not snapshot failures. The 62 full warm-ups are exactly 16 initial lineages plus 46 new lineages after history compaction; every scored query is delta. Snapshot cache usage peaked at 15,358 MiB.

### Remaining tail

| Stage | p50/p95/p99 |
|---|---:|
| Application: wait for this session's submitted arrival | 0/501/855 ms |
| Application: prompt render | 22/101/200 ms |
| Engine audio TTFA | 709/1,481/2,219 ms |
| Thinker-P | 267/790/1,120 ms |
| Thinker-D added | 179/507/1,238 ms |
| Talker added | 60/172/247 ms |
| Code2Wav added | 189/253/442 ms |

Each of the five slowest turns combines 765–1,120 ms of session wait with 2,060–2,404 ms of engine latency. Client TTFA correlates 0.934 with engine TTFA and 0.746 with session wait; engine TTFA correlates 0.883/0.789 with P/D. The remaining 3-second p99 is therefore a **Thinker P/D service tail** under concurrent arrivals, amplified by strict per-session serialization. Talker and Code2Wav are not the primary bottleneck.

GPU0 (P) busy p95/p99 is 93%/100%, with SM-active p95/p99 of 57%/81%. GPU1 (D) busy p50/p95 is 0%/73%. This is bursty P/D pressure, not sustained saturation of all four GPUs, snapshot capacity, payload loss, or full fallback. Sixteen users already violate the one-second p99 SLO, so the capacity procedure stops before 32.

### D-side lineage and cache-only sync

The old long run updated only P's cache, so D received KV at the final query. P missed only 110–411 tokens while the P→D suffix still reached 13,436 tokens at p99. This was an implementation gap, not an inherent P/D cost.

An arrival computes its delta on P and imports it into D's ordinary prefix cache through a cache-only control operation. It does not enter D's inference scheduler, execute the model, or reach Talker. The old implementation acknowledged the application only after D imported the revision. The current implementation acknowledges as soon as P stores the snapshot and lets D cache-sync continue in the background.

The first cache-only implementation awaited every sync inside the single stage-output loop. That accidentally serialized unrelated sessions and left already-generated D/Code2Wav outputs queued at the API for 1.5–1.8 seconds. The next version dispatched sync work without blocking the shared loop but still made each application session wait for D-ready. The current version moves the completion boundary to P-ready: same-session P requests remain linear, while D Core orders the actual imports locally and no longer gates query submission on a utility result.

A final query does not require every intermediate D sync to have completed. D uses the prefix it actually owns; if background sync is behind, the existing Delta-KV connector transfers the cumulative missing suffix from P. This affects acceleration availability, not prompt or output correctness.

The before/after runs use the identical 16-user × 6-turn plan (SHA256 `8c04386fa673fd05ad77db3212846056d1d98b7d3b7cbaff2524eadac85ba7d4`) and contain 80 scored turns each:

| Metric p50/p95/p99 | Global await | Request-scoped concurrency |
|---|---:|---:|
| TTFA | 743/1,851/3,075 ms | 594/806/1,150 ms |
| Per-session serial wait | 0/267/350 ms | 0/198/453 ms |
| Arrival D cache sync | 23/83/131 ms | 24/89/121 ms |
| Thinker-P | 236/506/1,051 ms | 167/315/390 ms |
| Thinker-D added | 213/1,484/1,970 ms | 113/211/333 ms |

After the fix, all 96 finite queries, 1,264 arrival requests, and the frame ledger verify successfully. Prefix-cache observations hit in 1,532/1,551 cases, with no timeout, client error, or stall. Core-output-to-orchestrator delivery for slow requests is back to low single-digit milliseconds.

The remaining engine audio TTFA is 535/733/844 ms; Talker and Code2Wav added p99 are 325/266 ms. A slow turn may still wait for its own preceding arrival, but there is no longer cross-session application head-of-line blocking. The short-run p99 remains slightly above one second; do not extrapolate a capacity result until the 30-turn run is complete.

### Formal 16-user × 30-turn long run

Result: `/home/ubuntu/data/results/pd_direct_cache_sync_u16_t30_concurrent_20260824/pd_direct_sync_concurrent_seed7_u16`

All 480 queries and 448 scored turns completed. The verifier accepts all 5,790 arrival requests, 6,585 consumed frames, and prefix-cache observations. There are no timeouts, client errors, or stalls; every scored query uses a delta snapshot.

| Stage p50/p95/p99 | Latency |
|---|---:|
| Client TTFA | 784/2,100/2,610 ms |
| Wait for this session's preceding arrival | 0/650/1,176 ms |
| Prompt render | 23/102/157 ms |
| Engine audio TTFA | 695/1,392/1,835 ms |
| Thinker-P | 271/773/1,128 ms |
| Thinker-D added | 172/401/579 ms |
| Talker added | 60/203/318 ms |
| Code2Wav added | 183/268/328 ms |

The Thinker-P query cache miss is only 113/346/403 tokens and scheduler admission is only 2/9/14 ms. Its internal tail instead comes from 31/183/371 ms between core preprocessing and scheduler execution (1,259 ms maximum), query-batch output-build p99 of 248 ms, and runner-to-core output exposure p99 of 448 ms. Actual GPU forward wall p99 is only 93 ms. Payload size correlates 0.898 with output-build time. The worst 1,259 ms ingress gap coincides with the preceding batch's arrival NIXL-sync burst.

Within tail-95 windows, P GPU busy/SM active/tensor active p95 are 100%/78%/49%; D reaches 74%/44%/4%. The root cause is therefore not Talker, Code2Wav, the former global API await, or large query re-prefill. It is **burst accumulation inside the P engine/connector input, snapshot/output, and result-exposure path under concurrent arrivals**. Correct per-session serialization propagates that delay to the final query; D's 579 ms p99 is secondary.

TTFA p99 does not increase monotonically by turn bucket: turns 1–10/11–20/21–30 measure 2,877/2,600/2,138 ms. The 2.61-second long-run p99 is therefore a rare concurrency burst that the short run undersamples, not context-length degradation. Sixteen users fail the one-second p99 SLO, so the ladder stops before 32.

### Current audio/text history plus latest-image policy: 16 users × 30 turns

Result: `/home/ubuntu/data/results/pd_audio_text_latest1_u16_t30_20260824/pd_audio_text_latest1_seed7_u16`

This run uses the current compaction policy: at 49,152 tokens, the newest two completed turns retain user audio, user text, and assistant text but discard images, while the in-flight turn retains only its newest image. It uses exactly the same `workload_plan.json` as the old-policy run below (SHA256 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`), making this a direct A/B comparison.

All 480 queries and 448/448 scored turns completed. Verification passes for 6,322 finite arrival requests, 7,442 consumed frames, and prefix-cache behavior. There was no timeout, skip, client error, or stall.

| Stage p50/p95/p99 | Latency |
|---|---:|
| Client TTFA | 836/2,775/3,853 ms |
| Wait for this session's preceding arrival | 0/946/1,715 ms |
| Prompt render | 27/124/203 ms |
| Engine audio TTFA | 715/1,830/2,541 ms |
| Thinker-P | 279/1,125/1,705 ms |
| Thinker-D increment | 175/494/705 ms |
| Talker increment | 57/200/422 ms |
| Code2Wav increment | 186/254/348 ms |

The run performed 27 compactions, each producing a complete prompt of only 600–951 tokens. P cache-miss p50/p95/p99 across all queries is 117/629/739 tokens. The 27 full queries after compaction have Thinker-P latency of 136/677/1,207 ms, lower than the 299/1,125/1,705 ms of the 421 ordinary delta queries. The old policy's full-query cache miss was 12,663/25,959/36,011 tokens; that application-layer cost has been eliminated.

On the same workload, TTFA improves from the old policy's 1,135/3,248/4,459 ms to 836/2,775/3,853 ms, but still fails the one-second p99 SLO. The formal long run shows P event-queue p99 of only 0.045 ms and scheduled-to-output p99 of 1,334 ms. However, `vllm_prefill_ms` includes both model execution and runner-side output preparation and must not be treated as a CUDA-only prefill timer.

To separate them, a 16-user × 12-turn diagnostic run of the current implementation enabled runner, handoff, and scheduler timing: `/home/ubuntu/data/results/pd_latest1_u16_diag_t12_20260825/pd_latest1_diag_seed7_u16`. All 192 queries and 160/160 scored turns completed with no timeout or stall; TTFA p50/p95/p99 is 741/2,029/2,433 ms. For P batches containing a scored query, GPU-forward p50/p95/p99 is 51/154/228 ms, runner output-build is 17/111/184 ms, runner completion to core output exposure is 70/338/515 ms, and core ingress to scheduler is 36/158/233 ms with a 660-ms maximum. The two slowest P requests take about 1.25–1.32 seconds, of which GPU forward is 242 ms, output build is 200 ms, and runner-to-core exposure is 670 ms. Another 1.16-second request spends 660 ms in core ingress.

The exact conclusion is therefore not “pure prefill-compute contention.” **Continuous arrival forms mixed arrival/query batches on P. GPU prefill is a secondary component; most tail comes from conditioning-snapshot/output build, the asynchronous result queue, and core output exposure in the P software pipeline.** The 1,715-ms p99 wait for the same session's preceding arrival propagates this same P-pipeline pressure through correct session serialization; it is not a separate application-lock problem. D is secondary; Talker, Code2Wav, and PCIe are not bottlenecks.

This run removes historical-image rebuild and compaction full-prefill as confounders. The next step is to remove custom engineering overhead from the Thinker-P snapshot/output/handoff path before attributing the remainder to engine scheduling.

### Removing synchronous P-snapshot output: strict 16-user × 12-turn A/B

Before the fix, an exact-parent delta had a reusable lineage but the P runner still copied layer-0/layer-24 tensors into shared CPU memory and built the result synchronously on its critical path. This path now reuses the existing pinned-memory asynchronous snapshot mechanism, so GPU-to-CPU copy and output construction no longer block the runner. Full snapshots and prefix gaps remain synchronous for correctness. Request semantics, tensor contents, lineage, priorities, and scheduler behavior are unchanged.

Post-fix result: `/home/ubuntu/data/results/pd_async_delta_u16_t12_20260825/pd_async_delta_seed7_u16`

It uses the exact same `workload_plan.json` as the pre-fix diagnostic (SHA256 `983b6a9f0111ddb669fc36d77d4e528f2b30c6970ade740f379c98e6f7c73fa8`). Verification passes for all 192 queries, 160/160 scored turns, 2,495 arrival requests, and 2,718 consumed frames. Prefix-cache observations hit in 3,047/3,067 cases, with no timeout or stall.

| p50/p95/p99 | Before | After |
|---|---:|---:|
| Client TTFA | 741/2,029/2,433 ms | 770/1,473/2,261 ms |
| Wait for this session's preceding arrival | 0/652/1,102 ms | 0/440/708 ms |
| Thinker-P | 252/911/1,252 ms | 243/490/907 ms |
| P scheduled→output | 154/674/1,157 ms | 158/364/528 ms |
| P query-batch output build | 17/111/184 ms | 7/39/66 ms |
| P runner→core output exposure | 70/338/515 ms | 57/143/268 ms |

GPU-forward p99 is effectively unchanged (228→239 ms), while output build, result exposure, and same-session wait all fall substantially. The removed latency was therefore custom P/D output-path overhead, not less workload or less model computation. The remaining TTFA p99 combines same-session serialization at 708 ms, Thinker-P at 907 ms, and Thinker-D at 555 ms. P also contains a 414-ms p99 core-ingress delay and one batch of exposure delay because the async scheduler submits the next batch before consuming the previous result. For large batches, asynchronous `output_wait` rises with the GPU event and is real concurrent-prefill completion time. The original synchronous snapshot-construction problem is fixed; the remainder is engine batch/compute queueing and the P/D execution path, not application history handling or synchronous tensor construction.

### Arrival-prefill compute versus request fragmentation

To determine whether the remaining tail means that total prefill-token demand exceeds GPU capacity, a diagnostic A/B uses the same 16-user × 6-turn `workload_plan.json` (SHA256 `3c55f3fd30bd3668723d3330c4c87ddfbfe2cb7d85f712d0c3ecfe89a93eb00a`). The production arm keeps video arrival prefill. The control disables only arrival prefill, concentrating the same video content in each final query. The control is for attribution, not a production recommendation. The arms consume 1,314/1,322 frames, a 0.6% difference; both complete 64/64 scored turns with no timeout or stall.

Results: `/home/ubuntu/data/results/pd_connector_diag_u16_t6_20260825/pd_connector_diag_seed7_u16` and `/home/ubuntu/data/results/pd_query_video_diag_u16_t6_20260825/pd_query_video_diag_seed7_u16`

| Metric | Arrival prefill | Query-time video |
|---|---:|---:|
| Client TTFA p50/p95/p99 | 555/1,065/1,492 ms | 528/983/1,078 ms |
| Session wait p50/p95/p99 | 0/335/763 ms | 0/0/0 ms |
| Thinker-P p50/p95/p99 | 177/399/678 ms | 235/411/661 ms |
| Final-query P miss p50/p95/p99 | 112/346/362 | 2,801/5,958/8,014 tokens |
| Total P miss tokens | 354,455 | 306,810 |
| P runner batches | 996 | 84 |
| Accumulated P GPU-forward | 50.9 s | 14.4 s |

The query-time arm prefills up to about 8k new tokens in a final query, yet its Thinker-P p99 does not exceed the arrival arm. Arrival computes only 15.5% more tokens, but creates 11.9× as many runner batches and 3.5× as much accumulated GPU-forward time. Separate timing of 1,387 P-to-D NIXL pushes gives per-push p50/p95/p99 of 0.281/0.969/1.877 ms and only 575 ms in aggregate, so connector submission cannot explain 200–500 ms request waits.

The remainder is therefore not sustained GPU saturation from total prefill tokens. **Arrival traffic is the trigger, but the main amplification is fragmentation into many small finite prefills: small batches have poor GPU efficiency, every arrival traverses asynchronous result/snapshot and D cache-sync, and bursts propagate to the query through correct same-session serialization.** Average GPU capacity remains available; the tail is a burst and execution-granularity problem. Engine research should preserve arrival semantics while improving cross-session batch accumulation, incremental-prefill granularity, and the cache-sync pipeline rather than changing the benchmark's main implementation to query-time video.

### Scheduler microbatch validation

To test small-batch inefficiency without changing the workload, a synchronous-scheduler A/B replays the same 16-user × 12-turn input trace (SHA256 `992716466dff3464f654e24baad90216d2b41d262eeb8382defefbca33f6a9cb`). Both arms complete 160/160 scored turns with no timeout or stall, send 6,202 frames, accept 2,764, consume 2,643, and send 7,047 audio chunks. The experimental arm changes only P scheduler admission: it waits at most 100 ms so independent sessions already in the waiting queue form a real batch. The application, prompts, KV lineage, and P/D path are unchanged.

Results: `/home/ubuntu/data/results/pd_microbatch_ab_20260825/window0/pd_mb0_seed7_u16` and `/home/ubuntu/data/results/pd_microbatch_ab_20260825/scheduler100/pd_sched_mb100_seed7_u16`

| Metric | Native work-conserving | Scheduler 100 ms |
|---|---:|---:|
| TTFA p50/p95/p99 | 542/906/1,211 ms | 667/1,188/1,545 ms |
| P runner batches | 2,055 | 927 |
| Mean batch size / singleton share | 1.326 / 78.1% | 2.901 / 19.5% |
| Tokens per batch | 326 | 721 |
| Accumulated P GPU-forward | 99.95 s | 46.73 s |
| Effective scheduled tokens/s | 6,693 | 14,293 |
| P scheduler admission p50/p95/p99 | 1.6/4.5/7.1 ms | 101.7/109.7/122.1 ms |

Total scheduled tokens differ by only 0.15%, while real aggregation cuts both runner batches and GPU-forward time by about 53% and raises effective throughput by 2.14×. This directly confirms fragmentation as the primary efficiency loss. TTFA becomes worse, so a fixed waiting window is not the solution: it trades explicit delay and burst/HOL effects for throughput, raising session-wait p95 from 177 to 329 ms. Native vLLM admission favors low immediate latency but produces mostly singleton prefills; fixed accumulation improves throughput but violates the interactive SLO. The engine research problem is therefore **SLO/deadline-aware cross-session incremental batching**, not more low-single-digit IPC cleanup or a fixed timer. A separate 100-ms application-side alignment changes mean batch size only from 1.326 to 1.366, showing that aggregation must operate on the engine waiting queue rather than at the API layer.

A further 16-user × 4-turn application/orchestrator diagnostic is stored at `/home/ubuntu/data/results/pd_app_diag_u16_t4_20260825/pd_app_diag_seed7_u16`. Steady-state arrival render p50/p95/p99 is 8/22/35 ms. API submit, core decode, core preprocess, and output encode/IPC have p99 values of about 10/11/4/8 ms, so none is a primary bottleneck. In the old implementation, every arrival waited synchronously for D cache-sync: 812 observations are 34/97/139 ms, and this D-ready ACK serialized both the next arrival and the final query. In one concrete p99 sample, the preceding arrival took 604 ms: about 60/49 ms for P runner/GPU, 233 ms from P result exposure to orchestrator receipt, and 122/149 ms for D transfer/load and cache-sync. The query arrived 164 ms into it and waited the remaining 440 ms. This proves that custom P/D post-processing, not only prefill compute, was exposed as session wait.

The current version splits P-ready from D-ready. It emits one terminal ACK after the P snapshot is stored, while request-scoped D sync continues without a second terminal. Same-lineage imports are ordered inside D Core, not by awaiting API utility results. This removes foreground waiting for D transfer/sync.

Formal P-ready short run: `/home/ubuntu/data/results/pd_pready_u16_t12_20260825/pd_pready_seed7_u16`. It uses four-GPU P/D, 16 users × 12 turns, two warm-up turns, and seed 7. Deploy SHA256 is `d90a5a2c34ba365be4d8a400e50d6c3a28c5e535fb35ea67a25392ed74006461`; raw `workload_plan.json` SHA256 is `983b6a9f0111ddb669fc36d77d4e528f2b30c6970ade740f379c98e6f7c73fa8`, and the canonical plan hash is `1dfd08e50710f188e5d83526358b9c3aecf4ad0e8616a5b0bb44d6b45d3b0a6f`. All three exactly match the old native D-ready control. Verification passes for all 192 queries, 160/160 scored turns, and 2,608 finite arrival requests. It consumes 2,674 frames and records 3,104/3,125 prefix-cache hits, with no timeout, skip, stall, client error, or engine warning. All 2,549 background D syncs actually submitted complete without failure.

| p50/p95/p99 | Old D-ready | Current P-ready |
|---|---:|---:|
| Client TTFA | 542/906/1,211 ms | 529/939/1,073 ms |
| Wait for this session's preceding arrival | 0/177/430 ms | 0/101/238 ms |
| Prompt render | 17/60/128 ms | 17/48/166 ms |
| Engine audio TTFA | 510/746/979 ms | 497/820/946 ms |
| Thinker-P | 114/297/404 ms | 128/299/385 ms |
| Thinker-D added | 139/261/479 ms | 136/278/399 ms |

P-ready cuts session-wait p99 by 192 ms and client-TTFA p99 by 139 ms, confirming that the old foreground wait exposed D transfer/sync to the application. P and D execution tails remain in the same range. The current slowest requests are dominated by an 883–1,018 ms engine path or 106–174 ms of prompt rendering, not D-ready wait. TTFA p99 remains 73 ms above the one-second SLO, so 16 users still fail the capacity cell. P GPU busy p50/p95/p99 is 33%/75%/94%, and SM-active is 27%/57%/65%; this is a burst tail rather than sustained saturation. Because this is one live run, the small p95 movement is not treated as a capacity-improvement claim.

#### P-ready 16-user × 30-turn long run

Result: `/home/ubuntu/data/results/pd_pready_u16_t30_20260825/pd_pready_seed7_u16`. Raw `workload_plan.json` SHA256 is `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`; the canonical plan hash is `9f807d41b63a9b76fe7d2e0e30b7a1ae2dd350a02fba2998a32ff6786d194514`. Verification passes for all 480 queries, 448/448 scored turns, and 6,572 finite arrival requests. It consumes 6,984 frames and records 7,666/7,701 prefix-cache hits, with no timeout, skip, stall, client error, or engine warning. All 6,259 submitted background D syncs complete; the maximum transient backlog is 52, and the slowest requests arrive with only 0–9 pending. Background-sync accumulation is therefore not the primary cause.

| p50/p95/p99 | 16×30 |
|---|---:|
| Client TTFA | 558/1,463/2,209 ms |
| Wait for this session's preceding arrival P-ready | 0/191/524 ms |
| Prompt render | 20/80/124 ms |
| Engine audio TTFA | 519/1,234/1,696 ms |
| Thinker-P | 132/513/714 ms |
| Thinker-D added | 145/474/678 ms |
| Actual P cache miss | 107/571/662 tokens |
| P→D transfer delta | 107/482/574 tokens |

The long-run primary cause is a **Thinker-P/Thinker-D burst with 40–50k-token prefixes**, not full-history recomputation. Medians by final-query prompt length are:

| Prompt tokens | Requests | Thinker-P | Thinker-D | Engine audio TTFA |
|---:|---:|---:|---:|---:|
| 0–8k | 68 | 75 ms | 73 ms | 351 ms |
| 8–24k | 161 | 110 ms | 112 ms | 446 ms |
| 24–40k | 142 | 185 ms | 179 ms | 618 ms |
| 40–50k | 77 | 342 ms | 285 ms | 918 ms |

Prompt length correlates 0.636/0.660/0.691 with P, D, and engine TTFA. Prefix caching avoids recomputing old-token KV, but P attention for new tokens and D's first decode step still read the complete 40–50k KV prefix. Multiple sessions approach the 49,152-token limit together and form a burst. In the slowest window, P GPU-busy/SM-active p95 reaches 100%/77%, while PCIe TX/RX p99 is only about 0.70/1.11 GiB/s. The 27 post-compaction full queries and 421 delta queries both have engine-TTFA p99 near 1.70 seconds, so compaction full prefill alone is not the cause; the long-prefix burst also delays surrounding delta requests. Session-wait p99 of 524 ms is a secondary propagation of the same P pressure through P-ready serialization, and render p99 of 124 ms is smaller. Talker/Code2Wav scheduled-to-output p99 is only 59/16 ms and is not the bottleneck.

Separately, 46/876 snapshots exceed 16 chunks and synchronously coalesce complete layer-0/layer-24 tensors on the orchestrator event loop; per-compaction median/p95/max is 26/214/270 ms and can block output handling for other sessions. That issue is independent of the P-ready/D-ready boundary and remains custom P/D engineering overhead to remove.

### D-local cache handoff: 16 users × 12 turns

D pre-registers destination blocks while P computes. The formal query is submitted immediately and keeps ordinary remote-prefill parameters as a failure fallback; it never waits for the cache-sync utility result on D's output socket. If KV is pending, D Core holds the ADD by request ID and activates it locally when the import completes. Imports from different sessions remain concurrent, while revisions of one lineage are ordered inside D Core so each cumulative snapshot can reuse its predecessor. The formal Talker snapshot no longer duplicates the complete prompt-token list. Data still moves through NIXL/PCIe.

Removing lineage order entirely was invalid: `/home/ubuntu/data/results/pd_local_handoff_u16_t12_20260826/pd_local_handoff_seed7_u16` inflated formal P→D suffix p99 to 10,757 tokens and TTFA p99 to 1,009 ms. D-local revision order restores suffix p99 to 574 tokens without restoring an application/API barrier.

Final result: `/home/ubuntu/data/results/pd_local_ordered_u16_t12_20260826/pd_local_ordered_seed7_u16`. The setup and workload plan are identical to the preceding 16×12 run (canonical hash `1dfd08e50710f188e5d83526358b9c3aecf4ad0e8616a5b0bb44d6b45d3b0a6f`). All 192 queries, 160/160 scored turns, 2,462 finite arrival requests, and 2,455 consumed frame occurrences pass verification; prefix-cache observations hit in 3,014/3,035 cases, with no timeout, skip, or stall.

| p50/p95/p99 | Earlier early-D | D-local ordered |
|---|---:|---:|
| Client TTFA / Audio-ready-500 | 548/794/978 ms | 475/658/706 ms |
| Wait for this session's preceding arrival | 0/66/193 ms | 0/22/93 ms |
| Thinker-P | 121/268/314 ms | 110/198/219 ms |
| Engine audio TTFA | 502/757/956 ms | 441/619/652 ms |
| Thinker-D added | 129/286/340 ms | 110/183/226 ms |

Of all 192 formal requests, 112 reached D before KV was ready. Their KV-ready-to-local-activation latency is 0.002/0.004/0.223 ms, proving that the utility-result/API synchronization is no longer on the critical path. Remaining D latency is real data and execution work: P-ready→D-ready is 56/97/155 ms, write-submit→D-ready is 45/66/98 ms, D scheduler queue is 1/3/6 ms, and D scheduled→output is 57/84/102 ms. Formal D wire decode falls from 12.6/43.6/86.7 to 9.5/36.3/65.7 ms after removing the duplicate prompt list. All 160 scored activations preserve `imported_tokens = prompt_tokens - 1`.

The formal 16-user × 30-turn run is `/home/ubuntu/data/results/pd_local_ordered_u16_t30_20260826/pd_local_ordered_seed7_u16`, with canonical workload hash `9f807d41b63a9b76fe7d2e0e30b7a1ae2dd350a02fba2998a32ff6786d194514`. All 480 queries, 448/448 scored turns, 6,311 finite arrivals, 6,591 consumed frame occurrences, and 6,604/6,604 D registrations/completions finish without timeout or stall. TTFA is 551/1,266/1,781 ms, so 16 users still fail the one-second long-run p99 SLO.

Long-run p50/p95/p99 is 0/247/534 ms for same-session wait, 141/466/708 ms for Thinker-P, and 131/333/546 ms for Thinker-D. For D, P-ready→KV-ready is 65/169/263 ms, scheduler→output is 61/117/178 ms, and local activation after a pending import is only 0/0/1 ms. Thus the remaining tail is not utility/API synchronization: it is a combination of P bursts propagated through session order, actual ordered KV availability/transfer, and D execution. In tail-95 windows, P SM/Tensor active p95 is 74%/54%, while D is only 35%/4%; D is waiting on the handoff rather than compute-saturated. Formal D request deserialization still reaches 126 ms at p99 and is removable engineering overhead, but it is not the dominant source of the 1.78-second tail.

### D completion wake-race fix

The push writer normally sleeps when it has no unmatched P blocks. A D Core `get_finished()` call wakes it but can drain the forwarded-notification queue before the writer publishes the NIXL notification, deferring completion by another engine step. The retained fix waits at most 1 ms for the notification event and retries inside the same Core poll. It does not continuously poll or change lineage admission.

Two aggressive variants were removed after equal-plan 16×12 tests. Same-step P flush advanced finished metadata by only 0–2 ms because the slow formal requests were actually waiting for D registration; it inflated suffix p99 to 9,523 tokens. Active transfer polling shortened notification delay only slightly: 1 ms and 5 ms polling inflated suffix p99 to 3,928 and 10,590 tokens, respectively. These paths accelerated D cache churn and are not part of the final implementation.

First long run: `/home/ubuntu/data/results/pd_event_handoff_final_u16_t30_20260826/pd_event_handoff_final_seed7_u16`. Its raw workload plan is byte-identical to the earlier run, SHA256 `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855`. All 480/480 queries, 448/448 scored turns, 6,976 arrival requests, and 7,424 consumed frame occurrences completed without timeout or stall; 8,207/8,242 prefix-cache observations hit.

| p50/p95/p99 | Before fix, 16×30 | First fixed run | Final-code repeat |
|---|---:|---:|---:|
| Client TTFA | 551/1,266/1,781 ms | 555/1,180/1,593 ms | 520/1,300/1,902 ms |
| Session wait | 0/247/534 ms | 0/140/406 ms | 0/159/444 ms |
| Thinker-P | 141/466/708 ms | 132/499/699 ms | 129/445/745 ms |
| Thinker-D | 131/333/546 ms | 121/340/515 ms | 120/361/518 ms |
| P→D suffix tokens | 107/497/3,724 | 106/514/665 | 105/476/2,916 |
| P-ready→D-ready | 65/169/263 ms | 62/156/213 ms | 58/140/216 ms |

The new markers report P-finished→worker-metadata at 8/55/87 ms and worker→write at 1/61/145 ms. Slow samples in the latter almost entirely wait for D registration; after registration arrives, WRITE is normally submitted in about 1 ms. Write→NIXL-notification is 46/77/102 ms, notification→D-Core is 0/8/24 ms, and complete write→D-ready is 47/84/113 ms. Percentiles are not additive. Write→notification is a lifecycle interval covering writer processing, transfer, and completion notification, not pure DMA time. The fix removes the extra Core-step race and prevents cumulative D-lineage lag, but it does not remove this handoff lifecycle. The 16-user p99 remains above one second; the remaining dominant terms are P bursts, their propagation through per-session order, and D's first execution over long contexts.

In tail-95 windows, P SM/Tensor active p95 is 77%/51% while D is only 32%/3%. D remains compute-unsaturated; the long tail is primarily P pressure propagated through handoff waits rather than insufficient D GPU compute capacity.

The exact final code, including the queue-before-wake ordering hardening, was rerun end to end at `/home/ubuntu/data/results/pd_event_handoff_queuefix_u16_t30_20260826/pd_event_handoff_queuefix_seed7_u16`. The workload-plan SHA256 is unchanged. All 448/448 scored turns completed with no timeout or stall; 6,297 finite arrivals and 6,726 frame occurrences passed verification, and 7,518/7,553 prefix-cache observations hit. Write→notification is 45/69/87 ms, notification→D-Core is 0/7/13 ms, and write→D-ready is 46/71/95 ms. The targeted wake race is therefore removed, but end-to-end TTFA p99 does not improve consistently.

The two post-fix long runs contain only four and six formal requests, respectively, with D-prefix eviction/large-suffix refill. With 448 scored samples, the p99 boundary falls between roughly the fourth and fifth tail samples. Consequently suffix p99 jumps from 665 to 2,916 tokens and TTFA p99 varies from 1,593 to 1,902 ms. This is not slower ordinary notification transfer. The completion race is removable engineering overhead, but 16-user overall p99 remains jointly controlled by P bursts, per-session propagation, D-prefix eviction, and D's first execution over a long context. In the final repeat's tail-95 windows, P/D SM-active p95 is 74%/35% and Tensor-active p95 is 51%/3%.

### Current Thinker-P tail attribution

The final-code 16-user × 30-turn repeat above is the current reference. Client TTFA is 520/1,300/1,902 ms and Thinker-P time-to-output is 129/445/745 ms. The following attribution supersedes the earlier, pre-fix conclusion that synchronous snapshot construction or result exposure dominates P.

| Thinker-P interval, p50/p95/p99 | Time | Attribution |
|---|---:|---|
| Core ingress → scheduler | 20/105/213 ms | Mostly waiting for the already-running, non-preemptible P batch to finish; not pure IPC time |
| Scheduler admission | 2/10/17 ms | Small engine bookkeeping cost |
| Runner input preparation | 16/45/69 ms | Removable engineering cost; reaches 110 ms in the concrete tail batch below |
| Actual CUDA forward | 45/151/285 ms | Real model work |
| Runner output build | 1/6/11 ms | Small after the asynchronous delta-snapshot fix |
| Runner → Core output exposure | 1/3/5 ms | No longer a tail source |
| P-side NIXL push | 0.4/1.0/1.9 ms | Not a P-tail source |

Percentiles are not additive because they need not select the same request or batch. One concrete tail query, `video-e4499b63e578-bc903ac1`, makes the lifecycle explicit. After about 31 ms of StagePool delivery, decode, and request construction, it reached P Core while a two-request batch was already running. That batch took 262 ms in the runner, including 194 ms of CUDA work, leaving this query waiting 226 ms for the next batch boundary. It then joined five other requests: the batch processed 1,591 new tokens over 46–48k-token prefixes and spent 110 ms preparing inputs, 324 ms in actual CUDA execution, and 9 ms building output. Its 78-ms host-side `forward_wall` is only CUDA enqueue time; the following 246-ms sampling/bookkeeping interval waits for the same asynchronous GPU work and is not an additional 246-ms snapshot computation.

The complete run shows why both long context and fragmented arrivals matter. Formal queries miss only 105/523/660 P-cache tokens, but every new token still attends to the retained prefix:

| Formal-query prompt length | Requests | Thinker-P median/p95 |
|---:|---:|---:|
| 0–8k | 73 | 83/153 ms |
| 8–24k | 170 | 115/216 ms |
| 24–40k | 138 | 162/323 ms |
| 40–50k | 66 | 322/762 ms |

Prompt length correlates 0.603 with Thinker-P latency. Across all 4,683 P runner batches, actual CUDA time correlates 0.939 with the approximate attention work `sum(delta_tokens × retained_context)`. Prefix caching avoids rebuilding old-token KV but does not remove attention from the new suffix to the old keys and values.

The same run has a mean P batch size of 1.41 and 76.1% singleton batches. Summed CUDA-forward time occupies only about 46.7% of the 602-second run, so P is not continuously compute-saturated; arrivals create inefficient small batches during quiet periods and expensive long-context mixed batches during bursts. The equal-token scheduler A/B above confirms the trade-off: a fixed 100-ms aggregation window reduces batch count and CUDA time by about 53% and raises throughput 2.14×, but worsens TTFA p99 from 1,211 to 1,545 ms. A fixed timer is therefore not the solution.

The final distinction is:

- **Model/workload cost:** a new AV suffix still performs attention over a long cached prefix; burst batches perform genuine CUDA work.
- **Engine-design mismatch:** work-conserving immediate admission fragments steady arrival traffic, while a running batch cannot admit or yield to a newly arrived formal query. This batching-efficiency versus deadline trade-off is structural under the current execution abstraction, though the exact latency is not fundamental.
- **Remaining engineering cost:** input preparation and request ingress can still be reduced. Synchronous snapshot copying, output exposure, and P-side NIXL submission have already been reduced below the dominant scale.

The current P-tail root cause is therefore **bursty, long-context incremental prefill interacting with immediate and non-preemptible batch execution**: a formal query commonly waits for one active P batch, then executes in another costly mixed batch. The relevant engine direction is deadline-aware cross-session incremental batching plus a bounded prefill execution quantum/yield point, with input-preparation subphases instrumented separately. It is not another fixed accumulation delay or further NIXL/snapshot micro-optimization.

### Removing full-prefix output reconstruction on Thinker-D

Thinker-D already receives its attention prefix through the native KV cache, while Talker's complete prompt conditioning comes from the P snapshot. The old runner still stored and reconstructed full-prefix layer-0/layer-24 outputs in a separate CPU tensor side-cache for every formal D request before appending the current decode row. That prompt-length-dependent work was redundant. D now emits only rows produced by the current scheduler step, and `thinker2talker_async_chunk` appends them to the P snapshot. Native attention KV caching, request tokens, the P snapshot, and Talker input semantics are unchanged.

Formal long-run result: `/home/ubuntu/data/results/pd_decode_tailonly_u16_t30_20260826/pd_decode_tailonly_t30_seed7_u16`; pre-fix baseline: `/home/ubuntu/data/results/pd_event_handoff_queuefix_u16_t30_20260826/pd_event_handoff_queuefix_seed7_u16`. Both use four-GPU P/D, 16 users × 30 turns, two warm-up turns, seed 7, and the same deployment and workload plan. Raw/canonical plan hashes are `5e62bb6915cced1918465f94c0bc690634730e722c05768a585d1df489dc8855` / `9f807d41b63a9b76fe7d2e0e30b7a1ae2dd350a02fba2998a32ff6786d194514`. Both finish 480/480 queries and 448/448 scored turns without timeout or stall. Because this is a live closed-loop workload, the fixed run actually processes more finite arrivals/frame occurrences than the baseline (6,826/7,215 versus 6,297/6,726), so the gain is not caused by reducing input load.

| p50/p95/p99 | Pre-fix | Fixed |
|---|---:|---:|
| Client TTFA | 520/1,300/1,902 ms | 491/1,097/1,487 ms |
| Same-session wait | 0/159/444 ms | 0/124/341 ms |
| Thinker-P | 129/445/745 ms | 130/452/678 ms |
| P-ready→D-ready | 58/140/216 ms | 60/138/196 ms |
| Thinker-D added | 120/361/518 ms | 82/249/507 ms |
| D scheduled→output | 61/131/174 ms | 23/35/46 ms |
| First D batch GPU forward | 6.7/9.7/12.2 ms | 6.7/9.5/12.4 ms |
| First D batch output build | 41.9/73.0/117.4 ms | 6.2/9.2/12.1 ms |
| First D batch runner total | 47.8/81.9/137.2 ms | 12.2/17.3/20.6 ms |

Across all 448 scored requests, first-D-batch GPU-forward time is unchanged while output-build p99 falls 89.7% and D scheduled-to-output p99 falls 73.6%; this is the direct causal validation of the change. End-to-end TTFA p95/p99 also falls 15.6%/21.8%, so the gain survives a 30-turn long session, though a single-seed live closed-loop A/B is not by itself a capacity claim.

Before the output-consumer follow-up below, the remaining tail was no longer D model computation. Thinker-P p99 was 678 ms, including 250 ms from Core ingress to scheduler and 389 ms from scheduling to output; formal-query runner prepare/GPU-forward/output-build p99 was 63/294/13 ms. A request-aligned, non-overlapping P-to-D timeline was: P output received by StagePool→D scheduler selection at 53/134/196 ms, D scheduler→D Core output at 23/35/45 ms, and D Core output→API receive at 5/134/303 ms. This isolated shared-output consumption as a removable engineering component. When the query arrived first and waited for KV, local activation after KV ready was already 0/0/0 ms; the earlier 216-ms activation-hold p99 mostly represented the reverse case—KV ready while the formal D request had not arrived.

### Shared-output consumption and snapshot suffix packing

The orchestrator previously polled stages serially with a separate 1-ms timeout for each empty stage, consumed at most one message per stage per pass, and synchronously routed P outputs. P snapshot chains were also compacted by copying the complete layer-0/layer-24 history whenever the chain exceeded 16 chunks. At 16 users this full-history copy had p50/p95/p99/max of 76/136/166/272 ms and a 0.89 correlation with prompt length; while it ran, D/Talker outputs accumulated in the shared event loop.

The current implementation keeps P routing FIFO in an independent worker, scans decoded stage queues with non-blocking round-robin, and stores snapshots as immutable packed-prefix slabs plus a recent suffix. Crossing the 16-chunk threshold packs only the new suffix; older slabs retain the same shared storage. This preserves request order, tensor contents, prefix/KV lineage, and arrival frequency.

Final validation: `/home/ubuntu/data/results/pd_output_suffixpack_u16_t30_20260826/pd_output_suffixpack_t30_seed7_u16`. Setup is four-GPU P/D, 16 users × 30 turns, two warm-up turns, seed 7, with the same continuous-AV workload. All 480 finite queries and 448 scored turns complete; 6,878 arrival requests, 7,297 accepted frame occurrences, and 7,980/8,015 prefix-cache observations validate, with no timeout or stall.

| p50/p95/p99 | Full-copy + blocking poll | Full-copy + non-blocking poll | Suffix-pack + non-blocking poll |
|---|---:|---:|---:|
| Client TTFA | 501/1,047/1,382 ms | 519/1,122/1,670 ms | 497/1,015/1,430 ms |
| Full/suffix packing time | 76/136/166 ms | 74/145/221 ms | 32/57/85 ms |
| All D output-queue wait | 2/69/196 ms | 1/38/132 ms | 1/14/71 ms |
| First formal D output-queue wait | 2/18/78 ms | 1/12/47 ms | 1/10/49 ms |
| First formal Talker output-queue wait | 2/146/305 ms | 1/17/84 ms | 1/13/88 ms |
| First formal Code2Wav output-queue wait | 2/27/136 ms | 1/20/63 ms | 1/13/35 ms |

Suffix packing reduces compaction p99 by 49% and its prompt-length correlation to 0.26. Non-blocking consumption removes the steady per-stage polling tax; the formal D first-output queue component is now 1/10/49 ms. On the broader, directly comparable boundary, D Core encoded→API receive is 2/52/198 ms versus 5/134/303 ms before this follow-up. The remaining p99 before queue insertion is 133 ms, so socket/decode exposure is reduced but not eliminated. End-to-end p99 does not improve monotonically across single-seed closed-loop repeats, so this is not presented as a capacity gain. In the final run, same-session wait, Thinker-P, P-ready→D-ready, and D scheduled→output are respectively 0/113/419, 129/407/567, 59/145/209, and 24/38/50 ms. The remaining 1.43-second TTFA p99 is dominated by P bursts propagated through session order plus the P/D rendezvous; shared-output exposure is now secondary rather than the largest component, and D GPU execution remains small.

### Previous raw-AV two-turn hard-limit policy: 16 users × 30 turns

Result: `/home/ubuntu/data/results/pd_recent2_long_u16_t30_20260824/pd_recent2_long_t30_seed7_u16`

This historical run used the old policy: no proactive compaction or summary, and at 49,152 tokens the final request retained the newest two complete raw AV turns plus the current turn. All 480 queries, 448/448 scored turns, 6,189 arrival requests, the frame ledger, and prefix-cache checks pass, with no timeout, skip, or stall. TTFA p50/p95/p99 is 1,135/3,248/4,459 ms, so 16 users fail the one-second p99 SLO. It is retained only as the A/B baseline for the current-policy result above.

The 33 compactions correspond exactly to 33 scored queries using full snapshots. Their P cache-miss p50/p95/p99 is 12,663/25,959/36,011 tokens and Thinker-P latency is 1,242/2,183/3,537 ms. The other 415 delta queries miss only 121/395/597 tokens, yet their Thinker-P latency still reaches 384/1,412/1,823 ms; 21 of the 23 tail-95 turns are delta. Full prefills after compaction therefore occupy P execution windows and spread tail to other sessions' delta queries and arrivals. The final query's wait for its own preceding arrival reaches 1,667 ms at p99 and propagates the same P pressure.

D transfer/load p99 is 323 ms; Talker and Code2Wav scheduled-to-output p99 is 88/15 ms. In tail windows, P reaches SM/tensor-active p95 of 94%/65%, while PCIe TX p95 is only about 1.16 GiB/s. PCIe, Talker, and Code2Wav are not the bottleneck. The 4.46-second p99 also contains an application-policy effect: an arrival that reaches the limit skips its warm-up, deferring compaction and the full prefill to the final query. Until that placement is changed, the entire tail must not be attributed to steady-state engine capacity.

## Recovery map

| Area | Path |
|---|---|
| Session and finite-request lifecycle | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| P/D snapshots and stage routing | `vllm_omni/engine/orchestrator.py` |
| P/D runner prefix output | `vllm_omni/worker/gpu_ar_model_runner.py` |
| P/D deployment | `benchmarks/thinker_talker/pd_deploy_4gpu.yaml` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| P/D capacity entry point | `benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh` |
| Run verification | `benchmarks/live_agent/analysis/verify_run.py` |
| Per-stage P/D latency | `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| Thinker-P tail diagnosis | `benchmarks/live_agent/analysis/pd_tail_diagnosis.py` |
| Tail and GPU attribution | `benchmarks/live_agent/analysis/p99_attribution.py` |
