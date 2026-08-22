# vLLM-Omni Realtime Multi-User Serving Workflow

## Goal and current decision

The goal is to serve more continuous audio/video sessions with fewer GPUs while meeting TTFA and speech-smoothness targets. The research target is engine capacity, scheduling, and tail latency, not model quality.

No locally deployable open-source realtime model exposes semantics equivalent to Seed Realtime or Gemini Live. This branch therefore approximates that product workload with the Qwen3-Omni Thinker → Talker → Code2Wav pipeline: clients upload audio and video continuously, while the model still answers in turns. This is sufficient for studying multimodal prefill, speech decode, KV cache, and multi-tenant contention, but it is not model-native full duplex or semantic barge-in.

The current architecture is:

- the WebSocket application owns the session, full multimodal conversation, and media-receive state;
- every media warm-up and final response creates a new, finite, ordinary engine request;
- the engine owns KV within a request and may retain only disposable prefix/KV cache after it ends;
- every turn submits the complete canonical history; a prefix-cache hit reduces work but never determines correctness;
- a cross-turn lifecycle is no longer embedded in one resumable engine request.

This is a better research baseline than an engine-resident session: it matches the general engine request abstraction, makes cache misses performance-only events, and composes with routing, replication, and P/D disaggregation. The application retains processed canonical message blocks, renders only new messages, and assembles the complete prompt from those blocks. An evicted engine prefix must still be prefilled, but the application does not decode and process all historical media again.

## Phase 1: application baseline

The implementation lives in `vllm_omni/entrypoints/openai/video_stream_base.py` and `serving_video_stream.py`.

- Each `video.query` receives a unique `video-<uuid>` response request ID.
- The application stores completed user/assistant turns. User history preserves accepted text, audio, and selected video rather than degrading to text-only summaries.
- The application also retains renderer-processed canonical blocks. Each turn processes only the current user block and the short completed assistant block, then merges tokens, media features, hashes, and rebased placeholder offsets. The engine still receives the complete history, not a delta request.
- Audio and video received for the current turn are consumed exactly once. Data arriving during generation belongs to the next turn.
- The similarity/freshness filter is the only video-selection policy. Every accepted frame remains in the current turn in arrival order; there is no latest-eight sliding window or second sampling pass.
- Each accepted frame triggers or coalesces a `video-warm-<uuid>` request containing full history plus the cumulative current-turn frames. It uses `output_modalities=["text"]` and Thinker `max_tokens=1`; the token is discarded, Talker/Code2Wav do not run, and the client receives no response event.
- Warm-ups are serial per session. A later request uses Thinker prefix caching to reuse complete blocks from its predecessor and computes only new frames plus the partial-block tail. A miss changes cost only.
- A warm-up is disposable background work. A query cancels the session's unfinished warm-up instead of waiting for cache population; the complete final prompt remains correct on a miss.
- `video.query` appends one complete WAV to that media prefix and runs Thinker → Talker → Code2Wav. Audio is not incrementally split, preserving the current Qwen input semantics.
- The experimental `enable_audio_arrival_prefill_approximation` switch is off by default. When enabled, the application preserves the current turn's true media-arrival order and seals audio into immutable one-second items. Each item can trigger only a silent Thinker finite request; the query appends the tail before decode and Talker may run. This emulates duplex engine load and is not equivalent to Qwen full-utterance inference.
- Normal Thinker responses are capped at 256 tokens so a model that ignores the short-answer prompt cannot create minutes of speech; arrival warm-ups remain one token.
- 49,152 tokens is the hard compaction threshold. At 32,768 tokens, the application proactively removes complete oldest turns toward 16,384 during post-response playback or an arrival warm-up, keeping compaction off the next query's critical path. The compacted prompt starts a new cache lineage.
- Frames are at most 640×352. Filter-accepted frames have no second count bound or sampling pass.
- The similarity threshold is 0.95 and the freshness gap is `[0,4]`: redundant frames may be dropped, but one frame is forced after four consecutive drops.
- JPEG decode, resize, and thumbnail generation run in a subprocess pool so they do not block the WebSocket event loop.
- Streaming audio DELTAs are forwarded one by one; the client evaluates startup and stalls on a playback timeline.

Removed paths include the cross-turn persistent request, arrival append into the same resumable request, Talker 45k rolling, Thinker shadow compression, the session epoch/segment ledger, and benchmark/diagnostic scripts that required them. Current arrival prefill uses independent finite requests plus disposable prefix cache, not streaming/resumable engine state.

## Phase 2: fixed deployment

Formal experiments use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`:

| Stage | GPU | Key configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights/KV, prefix caching enabled, priority scheduler |
| Talker | 1 | FP8 weights/KV, prefix caching keyed by session conditioning lineage |
| Code2Wav | 2 | separate process |

All stages use separate processes and the YAML remains fixed across user counts. Talker requests remain finite and reuse only disposable conditioning-prefix cache entries. Request-lifecycle-independent mailbox, vocoder, Snake, and code-predictor optimizations remain in the tree.

Final responses use priority 0 and silent warm-ups use priority 10. `run_qwen_server.sh` discovers the environment's CUDA toolkit and rejects formal runs where Thinker does not select FlashInfer. The multimodal processor cache uses API-side `processor_only` compatibility mode to avoid concurrent sender/receiver eviction divergence in vLLM's mirrored LRU.

Start the deployment with:

```bash
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

## Phase 3: the only formal workload

Capacity testing models continuous AV sessions. Each user owns one long-lived WebSocket:

- video is uploaded throughout the session at 2 FPS;
- each filter-accepted frame immediately triggers or coalesces into a silent Thinker warm-up, while the current-turn media prefix remains append-only;
- the microphone uploads PCM16 at 5 Hz, pausing during assistant playback plus a 300 ms echo guard;
- each turn uses one real 16 kHz mono recording followed by 700 ms endpoint silence; query text is empty, so semantics come from speech;
- audio is appended as one complete WAV at query time; only this final request invokes Talker;
- a session keeps one speaker and does not reuse recordings; video uses a fixed sequence with different starting offsets;
- the next turn starts after the response has played at 1× speed, making this a playback-paced closed loop;
- user starts are deterministically staggered over 0–40 seconds.

SLURP speech and DAVIS video provide the media. A formal cell has 30 turns per user with two warm-up turns. The ladder increases through 8, 16, 32, ... users and stops each seed at its first SLO failure. The engine restarts for every cell.

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

Synthetic media in `probe.py` is for protocol validation only and cannot support capacity claims.

## Phase 4: metrics and pass rule

- **TTFA**: query submission to the first audio packet. It depends on initial packetization and is a transport diagnostic, not a cross-configuration experience metric.
- **Audio-ready-500**: time until 500 ms of playable audio is available; a short reply is released at `audio.done`. This fixed startup SLO cannot be changed by an environment variable.
- **Stall max**: the largest single underflow when a granule arrives after all previously buffered audio would have played at 1×.
- **RTF deliver**: generated audio duration divided by delivery duration; this measures sustained supply.

A cell passes only when every post-warm-up turn completes, audio-ready-500 p99 is below 1 second, stall-max p99 is below 50 ms, and no client, protocol, or fatal engine error occurs. TTFA, GPU utilization, memory, and engine-step data are root-cause signals, not substitutes for the experience SLO. New results use workload schema 4; the verifier rejects the old variable-prebuffer contract.

`analysis/verify_run.py` additionally checks unique response requests, unique successful warm-ups, agreement between engine frame counts and the client consumed ledger, real Thinker prefix hits, and separate processes for all three stages.

## Phase 5: eight-user finite-request diagnosis

On 2026-08-21, the current finite-request path was diagnosed with FlashInfer, seed 7, 30 turns per user, and two warm-up turns. The archived persistent result was read only; it was neither deployed nor rerun.

### Default policy

Result: `/home/ubuntu/data/results/finite_request_foreground_priority_cap256_cbf2226a_flashinfer_20260821/finite_foreground_priority_cap256_seed7_u8`

- 224/224 measured turns completed with no timeout, client error, preemption, or recompute.
- TTFA p50/p95/p99 was 1.84/4.83/6.70 seconds; playback-start was 2.89/8.65/11.51 seconds. The cell failed the capacity SLO.
- 2,958 Thinker-only warm-ups and 240 response requests ran; 3,168/3,169 observed prefix-cache lookups hit.
- Query-to-warm-up handoff p99 was 16 ms, render p99 767 ms, engine-to-first-text p99 2.47 seconds, and engine-to-first-audio p99 6.16 seconds.
- Warm-ups submitted 74.96 million logical tokens, but only 1.11 million missed prefix cache: a 98.5% hit rate. Full history was not recomputed, although long contexts still increase attention, KV binding, and scheduling cost.

Removing the query-time warm-up barrier reduced TTFA p99 from 7.29 to 6.41 seconds. Foreground/background priority then produced 6.70 seconds. A single-seed live closed-loop result cannot establish a regression, but it does show that priority cannot remove interference from prefill already in flight.

### Fairness against the archive

Both runs have workload-plan SHA256 `99dc083388...`, but the rendered engine workloads differ:

- The old persistent path rolled at roughly 27.0k–36.7k tokens and carried a text-only seed of 281–858 tokens. Final context p50/p95/p99/max was 14.8k/30.9k/34.5k/36.8k.
- The current default compacts at 49,152 tokens to roughly 16,384 and retains recent complete multimodal turns. Final context was 26.6k/46.3k/48.1k/49.0k.
- The live closed loop amplifies the difference: the current default sent 9,027 frames and consumed 3,979, while the archive sent 7,367. Equal plan hashes do not imply equal media trajectories.

Current TTFA correlates with logical context at 0.66. For contexts at or above 40k, TTFA p50/p95 was 3.15/6.69 seconds. The default result therefore cannot be interpreted as a request-lifecycle-only comparison.

### Current-path compute-budget control

Only the current finite-request path was rerun with compaction set to `32k -> 0`, bringing its context envelope near the archive. This is a compute-isolation control, not a production semantic policy, because old turns are discarded at compaction.

Fixed result: `/home/ubuntu/data/results/finite_request_old_budget_abort_cleanup_cbf2226a_20260821/finite_old_budget_abort_cleanup_seed7_u8`

- 224/224 completed. TTFA p50/p95/p99 was 0.984/2.50/3.23 seconds; playback-start was 1.58/3.79/4.87 seconds.
- Context p50/p95/p99/max was 15.0k/29.4k/31.1k/32.0k. Mean GPU 0/1/2 SM activity was about 20.5%/6.8%/0.4%, close to the archive's 21.4%/7.0%/0.4%. The gap is not GPU saturation.
- Archived TTFA p50/p95/p99 was 0.244/0.392/0.479 seconds, with a 0.532-second maximum. A real gap remains after compute-budget alignment.

Most of the remaining gap is Talker startup. The persistent path retained Talker KV across turns. The current path creates a new Talker request every turn, stage-1 prefix caching is disabled, and its placeholder/prefill is rebuilt from the turn's complete Thinker prompt. In the current control, `history_messages` correlates with the TTFT-to-TTFA gap at 0.81; as history grows from 0–4 to 16+ messages, median gap rises from 226 to 1,111 ms. Ordinary warm-up admission, KV binding, and chunked prefill still add Thinker-side cost, but do not explain the whole gap.

One finite-pipeline bug was also fixed: after the final stage completes, residual upstream work is aborted before request cleanup. Talker late outputs fell from 464 to zero; TTFA p50/p99 changed from 1.036/3.540 to 0.984/3.225 seconds in the aligned control. This was wasted work and part of the tail, not the primary root cause.

### Disposable Thinker lineage handle

Implementation: the application still submits the complete canonical prompt plus a session lineage, parent revision, and token LCP. The KV cache manager stores only block-hash snapshots from completed finite requests; it owns no live request and pins no GPU block. A cache miss falls back to full prefill. History compaction rotates the lineage, and Thinker handles never enter Talker.

Formal result: `/home/ubuntu/data/results/finite_cache_handle_formal_20260821/cache_handle_formal_seed7_u8`

- 224/224 turns completed with an exact frame ledger, no timeout, and no stall.
- TTFA p50/p95/p99 was 521/944/1265 ms, versus 518/878/1051 ms for strict finite v6 and 244/392/479 ms for the persistent archive.
- 2,532 requests reused a hash snapshot, avoiding repeated hashing of about 49.19 million prefix tokens. Render p50 remained 72.6 ms and Thinker-to-first-text p50 remained 224 ms.
- The hash handle is correct and reduces control-plane work, but is not the TTFA solution because ordinary prefix caching already hits most GPU KV.

### Removing repeated rendering and measurement confounding

On 2026-08-22, completed history was retained as application-side processed canonical blocks. A new turn renders only its user message, and completion renders only the assistant message. The application then assembles a complete token/media prompt; engine request lifetime and cache-miss semantics are unchanged. Complete message boundaries in Qwen ChatML are append-only. Other templates fall back to full rendering.

Short validation: `/home/ubuntu/data/results/canonical_render_short_20260822`

- u1×4 completed 4/4 turns. Prompt length grew from 1,261 to 11,184 tokens while render time grew only from 8.1 to 10.2 ms, with no full-render fallback.
- u8×4 completed all 24 post-warm-up turns. Audio-ready-500 p50/p99 was 345/442 ms, stall p99 was zero, and all 32 response requests committed canonical turns.
- Per-turn u8 render medians were 9.7/11.6/10.7/16.6 ms. There was no arrival failure, multimodal-cache error, query failure, or late output.
- Startup is now measured at a fixed 500 ms of playable audio. The current first packet is about 537 ms, so TTFA and audio-ready-500 are close for this deployment, but changing codec packetization can no longer change the SLO definition.

Conclusion: the former roughly 73 ms median full-history render was application overhead and is now largely removed. Remaining tail is suitable for studying engine prefill/decode contention, KV behavior, and multi-stage startup. These runs validate correctness only; they do not replace a formal 30-turn capacity cell.

### Formal eight-user capacity result

On 2026-08-22, schema 4 was run with seed 7, 30 turns per user, and two warm-up turns: `/home/ubuntu/data/results/canonical_render_formal_cbf2226a_20260822/canonical_render_formal_seed7_u8`.

- All 224 measured turns completed without timeout, stall, client error, or protocol error. The verified consumed-frame ledger was 3,317.
- Audio-ready-500 p50/p95/p99 was 456/854/1,107 ms. P99 exceeded the one-second limit by 107 ms, so eight users are the current capacity boundary; 16 users were not run.
- TTFT p50/p99 was 239/707 ms. The first-text-to-first-audio gap p50/p99 was 213/673 ms.
- Render p50/p95/p99 was 15/46/104 ms with no full-render fallback. The three p99-tail turns spent only 46/14/18 ms rendering, so application rendering is no longer the primary cause.
- Of excess p99-tail latency above the median, Thinker accounted for 72.4% and the downstream speech path for 27.6%; at p95 the shares were 57.2% and 42.8%. GPU 0 SM-active p95 rose from 45% overall to 94% in tail windows, while GPUs 1 and 2 did not saturate with it.
- All 2,929 finite arrival warm-ups succeeded. Prefix-cache observations hit 3,381/3,400 times with a 32,704-token maximum hit, and 32 complete-turn compactions occurred.
- The current first audio packet is about 537 ms, already above the fixed 500 ms threshold. Audio-ready-500 therefore equals first-packet TTFA in this deployment, while retaining a packetization-independent metric definition.

Conclusion: eight users now show a small startup-tail violation whose direct source is concurrent Thinker work delaying first token. The speech pipeline contributes part of the delay, but Talker and Code2Wav are not resource-saturated. The next discussion should focus on Thinker prefill/decode isolation or scheduling, not further application-render tuning.

### Same-commit context-alignment control

On 2026-08-22, commit `854535bb` was tested with the same schema-4 workload, seed 7, eight users × 30 turns, and two warm-up turns. Both arms have workload-plan SHA256 `99dc083388...`. The default arm has no override; the aligned arm changes only history compaction to `32k -> 0`. Because that arm discards completed turns, it is a compute-envelope control rather than a production semantic policy.

Results: `/home/ubuntu/data/results/current_default_854535bb_20260822/current_default_seed7_u8` and `/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`; persistent archive: `/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`.

| Path | Context p50/p95/p99/max | Audio-ready-500 p50/p95/p99/max |
|---|---:|---:|
| Current default `49k -> 16k`, 16k headroom | 22.5k/32.0k/32.4k/32.8k | 478/723/1186/1215 ms |
| Current `32k -> 0` control | 16.8k/30.6k/31.8k/31.9k | 410/635/796/883 ms |
| Archived persistent path | 14.8k/30.9k/34.5k/36.8k | 515/740/824/900 ms |

Both current arms completed 224/224 measured turns without timeout, skip, or stall and passed finite-request, arrival-request, frame-ledger, prefix-cache, and three-process validation. The persistent archive's first packet held only about 217 ms of audio, so its raw TTFA p50/p95/p99 of 244/392/479 ms is not comparable with the current roughly 537 ms packet. The archived values in the table were recomputed from `turns.jsonl` audio deltas at a cumulative 500 ms of PCM.

At the same commit, shortening context reduced p99 by 389 ms. GPU0 SM-active p95 in p99-tail windows fell from 94.7% in the default arm to 52.4% in the aligned arm, which returned inside the SLO. More importantly, after alignment the finite-request p99 is only 28 ms from the persistent archive and is slightly lower. Destroying each engine request after a turn is therefore not the fundamental source of the old gap. Application-owned canonical session state, finite engine requests, and disposable prefix/KV reuse can deliver the same class of tail latency.

Limitations: this is a single-seed live closed-loop control, not a bit-identical replay; different response lengths alter later media-arrival trajectories. `32k -> 0` also loses conversational semantics. A production policy should retain semantics through an application-generated short text summary/seed plus a few recent complete turns, then hold that context envelope fixed while studying Thinker scheduling or P/D isolation at higher concurrency.

### Audio arrival-prefill approximation

On 2026-08-22 an explicit experimental path was implemented. Each full second of PCM16 becomes an immutable media item, and audio/video items remain append-only in actual arrival order. Adjacent audio items lose only their internal `<audio_end><audio_start>` pair while retaining separate media hashes and continuous MRoPE. Arrival requests remain low-priority, Thinker-only, and finite; the query appends the sub-second tail and is the only request allowed to speak. Qwen's audio encoder uses bidirectional attention within an approximately eight-second window, so independent item encoding changes model semantics. This is a workload approximation, not equivalent inference.

Single-user eight-turn validation: `/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u1_t8`

- All 7/7 measured turns completed without processor errors or stalls; audio-ready-500 p50/p99 was 280/344 ms.
- The old turn-six failure did not recur; processed canonical blocks can retain accumulated chunked history.

The cold-start eight-user A/B used schema 4, six turns per user, one warm-up turn, seed 7, a 0–8 second stagger, and the same workload-plan SHA256 `b6160ec3...`:

- Query-time complete WAV: `/home/ubuntu/data/results/audio_arrival_ab_baseline_20260822/u8_t6`; 40/40 completed and audio-ready-500 p50/p95/p99 was 368/475/654 ms.
- Audio arrival: `/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u8_t6`; 40/40 completed and audio-ready-500 was 343/554/692 ms.
- Arrival reduced foreground prefix residual p50 from 109 to 23 tokens and TTFT p50 from 166 to 156 ms. However, warm-ups increased from 614 to 893, and queries that collided with and cancelled a warm-up increased from 6 to 15. Stage Thinker p99 rose from 294 to 333 ms and stage audio TTFA p99 from 393 to 429 ms.
- The approximation changed the response distribution: output-audio p95 rose from 11.5 to 18.2 seconds and closed-loop wall time from 114 to 137 seconds. The input plan is identical, but the resulting live media trajectory is therefore not a strict engine-only replay.

Conclusion: this path is functional and moves some audio work out of the foreground query, but at eight users it saves only about 25 ms at the median and does not improve tail latency. Added Thinker warm-up contention offsets the foreground saving. It remains a default-off research arm and does not replace the formal complete-WAV-at-query baseline. Semantically equivalent benefit requires a native causal/streaming audio encoder or a model-provided streaming cache. Current Qwen can safely finalize only complete roughly eight-second attention windows, which offers little benefit for this workload's short utterances.

This also corrects the earlier attribution. The short baseline still processes a complete WAV at query time yet has a 654 ms p99. The 1,107 ms tail in the 30-turn formal run is therefore not primarily query-time audio; it comes from multi-user Thinker prefill/decode and arrival-warm-up contention at long context.

## Recovery map

| Purpose | Path |
|---|---|
| Session and finite-request lifecycle | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Canonical multimodal history | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Prefix-cache observation | `vllm_omni/worker/gpu_model_runner.py` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload plan and media loading | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| Capacity ladder | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| Formal deployment | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| Run verification | `benchmarks/live_agent/analysis/verify_run.py` |
| GPU sampling | `benchmarks/live_agent/harness/gpu_sampler.py` |
