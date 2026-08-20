# vLLM-Omni Realtime Multi-User Serving: Workflow and Current Findings

This document records the research path, reproducible experiments, and current findings of `thinker-talker-vllm`. The goal is not to improve model quality. It is to serve more concurrent sessions with fewer GPUs while meeting the latency and playback requirements of voice interaction.

No open realtime model is currently available for local deployment, so this project uses the Qwen3-Omni thinker → talker → code2wav pipeline as a realtime-serving surrogate. It is not equivalent to Seed Realtime or Gemini Live, but it is sufficient for studying long-lived sessions, KV residency, pipeline scheduling, and multi-tenant tail latency.

## Quick recovery

- Branch: `thinker-talker-vllm`; formal runs use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`.
- Main implementation: one long-lived engine request per WebSocket, persistent cross-turn KV, and delta-only audio, video, and query appends.
- Only capacity workload: real speech, continuous video, and playback-paced multi-user closed loop; eight users pass and 16 fail.
- Latest fair RCA: clean commit `54d80066`, with exact per-turn media equivalence; query-time consolidation is worse than ordered arrival prefill.
- Application conclusion: retain ordered incremental arrival prefill and the similarity/freshness filters; query-time consolidation is not an engine-tail fix.
- Next engine question: compare Thinker prefill/decode QoS and P/D separation under a fixed media ledger.

Recovery order: read Phase 6 below, then `benchmarks/live_agent/web_client/README.md`; use `run_av_session_ladder.sh` for capacity, `run_prefill_timing_rca.sh` for paired RCA, and `/home/ubuntu/data/results/archive/2026-08-20_prefill_timing_fair_rca/` for the latest formal artifacts.

## Terminology and criteria

- **Stateful incremental**: one WebSocket session owns one persistent engine request; each turn appends only new input while history remains in KV cache.
- **Stateless full replay**: every turn creates a new request and prefills the complete history again.
- **Stateful with KV eviction**: the application retains the session and transcript, but retires the engine request after every turn and rebuilds it from bounded history next time.

The product-level targets are TTFA p99 < 1 s and audible playback stall p99 < 50 ms. Current TTFA starts when the query is sent and does not include product-side VAD or endpointing.

Stateful incremental serving is the primary design because it best matches realtime product semantics and exposes long-lived KV and multi-tenant scheduling behavior. The other two modes are controls for repeated-prefill cost and idle-KV cost; neither is assumed to be universally better.

Every capacity number must be tied to a source commit, deploy YAML, hardware, workload, seeds, and metric definition. Results from different buffering or percentile conventions are not directly comparable.

## Phase 1: Choose the implementation model

Upstream vLLM-Omni already provides the Qwen3-Omni thinker → talker → code2wav pipeline and a video WebSocket endpoint, but its inference unit is still one turn at a time: assemble a prompt after a query, generate one response, then finish the request. It does not provide long-lived session semantics for a realtime product.

The missing capabilities are:

- one engine request and KV state that persist across turns on the same connection;
- incremental audio and video prefill on arrival instead of batch processing after the query;
- explicit session, turn, request-epoch, and output-segment identity;
- long-context roll/compression, reconnect/rebuild behavior, and stale-output isolation;
- KV/slot admission and multi-tenant protection for long-lived users;
- realtime workloads and metrics centered on TTFA and playback stalls.

This branch therefore uses an application-level long-lived session backed by an incremental vLLM engine request:

1. Each turn prefills only new input, avoiding the repeated work of stateless full-history replay.
2. KV lifetime, idle residency, and inter-user contention remain visible to the engine, which makes the system suitable for serving research.
3. Arrival-time prefill can move media encoding off the TTFA critical path.
4. WebSocket sessions, turn fences, and lifecycle events resemble the contracts of products such as Seed Realtime and Gemini Live.
5. The approach is deployable on an existing open Qwen3-Omni model and does not require native realtime model support.

This is a serving-system approximation, not a native duplex model. Model-level full-duplex generation, semantic barge-in, and continuous internal model state are outside the current scope.

Stateful incremental serving is the main implementation. Stateless full replay and application-stateful/KV-evicted serving are retained only as controls.

## Phase 2: Turn per-turn requests into a long-lived session

A fresh request per turn is a useful API baseline but not a realtime session: it has no persistent request, durable KV, arrival-time input, or cross-turn lifecycle.

This branch adds:

- one resumable engine request per session;
- delta-only text, audio, and video submission on each turn;
- optional frame and microphone-audio prefill as media arrives;
- request rolling near the talker token limit, seeded from recent text;
- thinker-context compression through a warmed shadow request and boundary swap;
- admission based on shared KV pools, active sessions, and `max_num_seqs`.

These are application-level session semantics, not native realtime model behavior. KV remains resident while a user is idle, one request failure can end the session, barge-in cannot simply abort accumulated KV, and roll/compression retains only bounded history.

## Phase 3: Bound multimodal and transport cost

- Downscale arriving frames to at most 640×352. Compared with 1280×720, this reduces visual tokens by about 4×.
- Keep at most `max_frames=8`, evicting the oldest frame first.
- Use both `min_gap` and `max_gap` in frame filtering to bound high-motion cost and static-scene blindness.
- Append the newest frame next to the query so freshness comes from prompt position rather than frame rate.
- Move JPEG decoding and resizing into a warmed process pool instead of the WebSocket event loop.
- Send only text and decode fields needed by the talker, not visual payloads or complete output-ID history.
- Use persistent two-slot shared-memory mailboxes and inline sends to remove per-chunk lifecycle and thread-hop overhead.

These changes reduce application and connector overhead without changing vLLM scheduling policy.

## Phase 4: Fix pipeline scheduling rhythm

An active voice stream needs one codec frame every 80 ms.

The critical fix is inline chunk receive inside the scheduler. The old path removed a request from `running` while waiting for a chunk, forcing it to miss at least one scheduling round. After the fix, the legacy deadline-miss rate at 56 sessions fell from 20.1% to 0.9%.

Other verified changes:

- windowed vocoder execution removes repeated convolution while preserving output;
- lean decode payloads omit hidden states and full output-ID lists that the receiver never reads;
- disabling WebSocket deflate in the benchmark client reduced 200-session TTFA p99 from 1654 to 728 ms; server behavior remains unchanged.

The WebSocket result is an important diagnostic: serialized work on one shared event loop can inflate p99 before aggregate CPU utilization reaches saturation.

## Phase 5: Optimize the talker and code2wav

### Pinned host-memory transfer

The talker copies one CPU payload row to the GPU on every step. Replacing synchronous `.to(device)` from pageable memory with a pinned staging buffer and asynchronous H2D reduced the legacy stall rate at 176 sessions from 27.2% to 2.03%.

### Fixed-size KV cache for the code predictor

The original path regenerated the entire 17-position window for each of the remaining 15 codebooks. A fixed 17-slot KV cache with one-position forward steps reduced code-predictor time from about 35.4 to 8.6 ms per round and talker round time from about 76.0 to 34.7 ms. The measured audio-only capacity at that point rose from roughly 180 to 200 sessions. The paths are numerically equivalent, not bitwise identical in bf16.

### Fused Snake activation in code2wav

Twenty-eight multi-kernel Snake activations were replaced with the existing fused Triton implementation. At offline B=80, code2wav fell from 108.7 to 72.0 ms. At 300 online sessions, delivery RTF improved from 0.84 to 1.01.

This improved mid-response playback but did not solve TTFA. Even bypassing code2wav entirely left first-audio p99 above 3 s at 300 sessions. Work performed after first-audio generation cannot fix the first-audio deadline by itself.

### Historical capacity result

The following numbers come from the archived two-GPU experiment near `f91091a6` and use its legacy zero-prebuffer convention. Its old runner, deploy files, and analyzer have been removed from the current tree:

| Workload | Largest passing point | First failing point | Limiting behavior |
|---|---:|---:|---|
| Audio-only | 230 | 240 | TTFA fails before playback continuity |
| AV, synthetic video at 480 ms/frame | 32 | 36 | thinker multimodal-prefill steps block admission |

Slow AV turns carry roughly the same input-token count as fast turns. The difference is that they arrive during a 175–324 ms prefill step. vLLM admits new requests only at step boundaries. Reducing `max_num_batched_tokens` shortens individual steps but repeatedly pays the MoE fixed cost, reducing throughput and worsening TTFA.

The current engine research targets are bounded step duration, finer-grained admission, deadline-aware scheduling, and lower fixed cost for multimodal prefill. Batch-token tuning alone cannot provide both low latency and high throughput.

## Phase 6: Calibrate session baselines and the benchmark

Commit: `b726effe`. This phase changes no scheduler, KV manager, or model execution code. It corrects application semantics and measurement integrity.

### Session correctness

- Add `session_id`, server-generated `incarnation`, request `epoch`, `turn_id`, and `segment_id`.
- Use a typed segment ledger to distinguish normal turns from shadow seeds.
- Attach identity to stateful responses and all session lifecycle events; reject stale epochs, wrong turns, and cross-segment output.
- Clear ledgers on roll, compression, and disconnect so late output cannot contaminate a new turn.

### Three explicit baselines

| Mode | Behavior | Purpose |
|---|---|---|
| `stateless_full_replay` | New request per turn, `history_max_turns=null` | Measure repeated full-history prefill |
| `persistent_incremental` | Keep request and KV across turns | Primary implementation |
| `stateful_evict_rebuild` | Retire request/KV after each turn and rebuild from bounded text history | Measure idle-KV savings and rebuild cost |

The stateless handler previously used `message_history[-2:]`, which retained only the previous turn rather than full history. `history_max_turns` now means: `null` for all history, `0` for none, and a positive integer for the most recent N turns. The compatibility default remains one turn.

### The single capacity workload

Capacity testing keeps one target scenario: a continuous AV session. Each user owns one long-lived WebSocket and mirrors the browser:

- video sends one frame every 500 ms for the full session;
- microphone PCM is sent every 200 ms while the user listens, thinks, or speaks;
- microphone upload pauses at first assistant audio, resumes after actual playback, and observes a 300 ms echo guard;
- every turn uses a real 16 kHz mono PCM16 recording; one session keeps one speaker and never loops a recording;
- 700 ms endpoint silence follows the utterance, then an empty `video.query` is sent, so semantics come only from audio;
- the next think period starts after playback; think time is a deterministic long-tailed distribution with median near 3 s and bounds of 1–12 s;
- all users share one fixed video sequence but start at different offsets.

The effective session configuration is `session_scoped_request=true`, `num_frames=16`, `max_frames=8`, 640×352, similarity threshold 0.95, filter gap `[0,4]`, audio/video arrival prefill enabled, and Talker rolling near 45k tokens. User starts are deterministically staggered over 0–40 seconds. A formal cell has 30 turns per user with two warm-up turns. `run_av_session_ladder.sh` increases 8, 16, 32, ... users and stops a seed at its first SLO failure.

This is a capacity workload, not a causal ablation. Because the next turn begins after actual playback, a slower service keeps a session alive longer and naturally receives more video. Live closed-loop results can compare product capacity, but cannot by themselves prove that one prefill timing is better. Timing ablations must use a fixed trace: record every event from the arrival arm, then replay it in the other arms.

Canonical capacity command:

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/av_capacity_<commit> \
RESULT_PREFIX=av_capacity USERS="8 16 32" SEEDS="7 17" \
TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_av_session_ladder.sh
```

All three session policies consume the same seed-derived `workload_plan.json`. The plan, audio corpus, frame set, source commit, and deploy YAML are hashed so only the session/KV policy changes between arms.

The client uses a 1.4 s smooth-buffer threshold. The current four-frame initial chunk contains only about 217 ms of audio, so playback normally starts when the second chunk arrives rather than after a fixed 1.4 s delay. A cell passes only when every post-warmup turn completes, audible playback-start p99 < 1 s, playback-stall p99 < 50 ms, and there are no protocol, client, or fatal engine errors. Service TTFA is reported separately for stage attribution.

The per-content matrix, audio-only p99 ladder, and synthetic AV cell runner have been removed. Synthetic media in `probe.py` remains only for protocol smoke tests and cannot support a capacity claim.

CPU tests cover real-WAV manifests, speaker/turn plans, media cadence, session identity, and playback timelines.

### Formal capacity results

Formal deployment is fixed to `origin_deploy_3gpu.yaml`: Thinker, Talker, and Code2Wav each use one GPU. Both canonical seeds run 30 turns per user, with two warm-up and 28 measured turns. Every result pins a clean source tree and hashes the corpus, frame set, deploy YAML, and seed-specific `workload_plan.json`.

| Seed | Users | Measured turns | TTFA p99 | Playback-start p99 | Stall p99 | Result |
|---:|---:|---:|---:|---:|---:|---|
| 17 | 8 | 224/224 | 541 ms | 972 ms | 0 | Pass |
| 17 | 16 | 448/448 | 1178 ms | 2062 ms | 0 | Fail |
| 7 | 8 | 224/224 | 479 ms | 824 ms | 0 | Pass |
| 7 | 16 | 448/448 | 1005 ms | 2117 ms | 0 | Fail |

All cells complete without timeout, playback interruption, or protocol error. Sixteen users fail on playback-start tail, not request completion. The stable capacity boundary is 8–16 users, so 32 is not run.

### 16-user chunk-level RCA

The RCA uses clean commit `59b3a033`, the same three-GPU YAML, `seed=17`, and the same workload plan. Each arm runs 10 turns per user with two warm-up turns, for 128 measured turns. It enables `VLLM_OMNI_LOG_SCHED_STEPS=1`, `VLLM_OMNI_LOG_REQ_STEPS=1`, and `VLLM_OMNI_LOG_AUDIO_CHUNKS=1`, then runs `benchmarks/live_agent/analysis/audio_chunk_rca.py`.

The baseline retains real continuous AV behavior: clients upload video at 2 FPS and audio at 5 Hz; `prefill_frames_on_arrival=true` and `prefill_audio_on_arrival=true` append consumable single-session media to the persistent Thinker request without entering Talker or generating a reply.

Baseline observations:

- the first-to-second chunk window contains 25–26 Talker request steps;
- second-chunk gap p50/p99 is 460/1482 ms, with 124/865 ms beyond a 25 ms normal-step envelope;
- 90% of long Talker gaps overlap Thinker work for the same request; Thinker-to-Talker inline-receive hit rate is 42.9%;
- Code2Wav emit-to-waveform p99 is 41 ms and is not the tail source.

#### Query-time prefill diagnostic

This diagnostic changes when one session's media enters Thinker. It is not multi-user batching, does not rebuild session history, and is not a capacity workload.

| Mode | Session override | Server behavior |
|---|---|---|
| Arrival baseline | both flags `true` | Append each consumable audio/video delta on arrival |
| Video query-time | `{"prefill_frames_on_arrival":false,"prefill_audio_on_arrival":true}` | Buffer video until `video.query`; keep audio arrival prefill |
| All query-time | `{"prefill_frames_on_arrival":false,"prefill_audio_on_arrival":false}` | Buffer both audio and video until `video.query` |

The original diagnostic reported:

| 16-user mode | TTFA p99 | Playback-start p99 | Chunk-2 gap p99 | Wait p99 | Inline hit | Code2Wav p99 |
|---|---:|---:|---:|---:|---:|---:|
| Arrival baseline | 1524 ms | 2153 ms | 1482 ms | 865 ms | 42.9% | 41 ms |
| Video query-time | 694 ms | 1067 ms | 611 ms | 264 ms | 72.8% | 29 ms |
| All query-time | 638 ms | 1078 ms | 447 ms | 157 ms | 81.5% | 41 ms |

That comparison was not controlled. With an eight-frame buffer, the three arms logged 24, 886, and 803 dropped frames. Fresh-frame duplication also depended on buffer state. The query-time arms therefore changed both timing and the amount of video delivered, so the apparent improvement cannot establish a prefill/decode contention effect. The similarity filter runs before the delivery-mode branch and remains a valid application optimization.

#### Fair paired rerun

The arrival arm first runs the real closed loop and records every timestamped client frame/audio/query event. Both query-time arms replay that exact trace. All arms keep the same similarity filter; `max_frames=256` prevents eviction and `fresh_frame_force_append_on_query=true` fixes freshness multiplicity. A per-turn ledger records selected and submitted frame identity/content hashes, audio bytes/hashes, and prompt chunk token counts. Results are accepted only if ordered media matches exactly, dropped frames are zero, and replay schedule slips are zero. Chunk boundaries are the treatment: arrival uses small incremental appends, while query-time consolidates them.

The formal rerun uses clean commit `54d80066`, 16 users, `seed=17`, 10 turns per user, and two warm-up turns. Every arm completes 128/128 measured turns. The recorded trace uploads 4,905 frames and 5,875 audio chunks. All arms submit 2,358/2,358 Thinker frame occurrences and 36,884,706/36,884,706 audio bytes; both query-time arms match the arrival arm on all 160 ordered per-turn ledgers, with zero drops and zero replay slips.

| Mode | Prefill chunks | Prompt tokens | TTFA p50/p99 | Playback-start p99 | Chunk-2 gap p99 | Wait p99 | Inline hit | Stall p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arrival | 2783 | 540,534 | 445/1283 ms | 2337 ms | 1240 ms | 683 ms | 44.4% | 0 |
| Video query-time | 967 | 540,556 | 587/2353 ms | 3172 ms | 1371 ms | 715 ms | 65.6% | 0 |
| All query-time | 160 | 538,942 | 566/2416 ms | 2946 ms | 2011 ms | 1594 ms | 71.6% | 223 ms |

Query-time consolidation reduces chunk count, not media volume. With equal media it worsens both TTFA and playback-start tail. Although inline hit rises and median chunk-2 gap falls, the same video prefill moves onto the user-critical query boundary. Thinker SM-active p50/p95 changes from 40%/61% for arrival to 8%/84% for video query-time and 0%/83% for all query-time; p95 inside playback-tail intervals is 67%, 95%, and 93%. Consolidation creates a stronger stage-0 burst. Memory p95 remains about 94–95% in every arm and does not explain the difference.

Application conclusion: retain ordered incremental arrival prefill. A backlog must catch up oldest-frame first; newer frames cannot bypass it. Query-time consolidation is not a latency optimization. The next engine experiment should compare Thinker prefill/decode QoS and P/D separation under the same media ledger.

### Result archives

The formal capacity and original RCA are archived at `/home/ubuntu/data/results/archive/2026-08-20_continuous_av_capacity_rca/`. The fair paired rerun is archived at `/home/ubuntu/data/results/archive/2026-08-20_prefill_timing_fair_rca/`, including the fixed input trace, all three cells, media fairness, chunk RCA, GPU tail attribution, and checksums.

## Current experiment procedure

1. Pin the source commit, deploy YAML, and hardware; record commit, dirty state, and YAML SHA256.
2. Fix user count, turns, speech corpus and turn plan, video sequence and cadence, think time, stagger, and seed.
3. Restart the engine for every cell so KV, requests, or failures cannot leak into the next result.
4. Change only the session policy across the three baselines.
5. Use multiple seeds near the knee. Historical run-to-run variation is 8–17%, so a one-run change below 20% is not a result.
6. Report TTFA, playback stalls, RTF, timeouts, admitted users, identity errors, GPU metrics, and engine probes together.
7. Verify from logs that the workload and mechanism actually ran before interpreting capacity.

New formal capacity claims use only `origin_deploy_3gpu.yaml`, this workload, the 1.4 s smooth buffer, and p99; historical results do not enter the capacity curve.

## Key code

| Area | Path |
|---|---|
| Session lifecycle and configuration | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Stateless history construction | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Session/turn/segment identity | `vllm_omni/entrypoints/openai/video_stream_state.py` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| AV workload planning and media loading | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| AV capacity ladder | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| Fair prefill-timing RCA | `benchmarks/live_agent/web_client/run_prefill_timing_rca.sh` |
| Three session baselines | `benchmarks/live_agent/web_client/run_session_baselines.sh` |
| Playback timeline | `benchmarks/live_agent/playback_metrics.py` |
| Canonical three-GPU deployment | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| Run-mechanism verification | `benchmarks/live_agent/analysis/verify_run.py` |
| Audio-chunk root-cause analysis | `benchmarks/live_agent/analysis/audio_chunk_rca.py` |
| Cross-arm media-equivalence check | `benchmarks/live_agent/analysis/media_fairness.py` |
| Scheduler and connector flags | `vllm_omni/core/sched/runtime_flags.py` |
| Chunk transport | `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` |
| Code predictor KV | `vllm_omni/model_executor/models/common/qwen3_code_predictor.py` |
| Code2wav and Snake | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py` |
