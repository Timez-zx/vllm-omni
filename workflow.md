# vLLM-Omni Realtime Multi-User Serving: Workflow and Current Findings

This document records the research path, reproducible experiments, and current findings of `thinker-talker-vllm`. The goal is not to improve model quality. It is to serve more concurrent sessions with fewer GPUs while meeting the latency and playback requirements of voice interaction.

No open realtime model is currently available for local deployment, so this project uses the Qwen3-Omni thinker → talker → code2wav pipeline as a realtime-serving surrogate. It is not equivalent to Seed Realtime or Gemini Live, but it is sufficient for studying long-lived sessions, KV residency, pipeline scheduling, and multi-tenant tail latency.

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

The following results apply only to the code near `f91091a6`, `deploy_2gpu_seq256.yaml`, 2× RTX PRO 6000, and the old `analyze.py` convention with zero playback prebuffer:

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

### Workload corrections

`mu_bench.py` now models closed-loop users: it waits until the response would finish playing, then waits another random 2–6 seconds before starting the next turn. A faster-delivering engine therefore cannot reduce its own effective concurrency by starting the next turn early.

The benchmark also adds:

- distinct synthetic audio for every user and turn, with SHA256 fingerprints;
- deterministic, auditable video start offsets;
- WAV-header parsing for sample rate and frame count instead of assuming a 44-byte header;
- arrival timestamps and sample counts for every audio delta;
- 1× playback replay with a default 60 ms prebuffer, producing stall count, total, and maximum;
- protocol mismatches, duplicate inputs, source commit/dirty state, and deploy-YAML SHA256.

CPU behavior tests cover session identity, history selection, workload uniqueness, and playback timelines; 38 relevant tests pass.

### Remaining gaps

- `run_session_baselines.sh` applies the same matrix to all three policies, but its default is 1/2/4/8/16 users, 30 turns, video plus text queries, and `audio_input_s=0`. A high-concurrency ladder with audio input is still required for a capacity result.
- The old `analyze.py` starts playback immediately at the first chunk, while the new `mu_bench.py` defaults to a 60 ms prebuffer and summarizes the p95 of per-turn maximum stall. Buffering and percentile conventions must be unified before formal comparison.
- This phase has CPU tests but no new GPU capacity result.

## Current experiment procedure

1. Pin the source commit, deploy YAML, and hardware; record commit, dirty state, and YAML SHA256.
2. Fix user count, turns, audio duration, video content and cadence, question set, think time, stagger, and seed.
3. Restart the engine for every cell so KV, requests, or failures cannot leak into the next result.
4. Change only the session policy across the three baselines.
5. Use multiple seeds near the knee. Historical run-to-run variation is 8–17%, so a one-run change below 20% is not a result.
6. Report TTFA, playback stalls, RTF, timeouts, admitted users, identity errors, GPU metrics, and engine probes together.
7. Verify from logs that the workload and mechanism actually ran before interpreting capacity.

Before publishing a new capacity result, pin one current-branch deployment and workload and unify the 0/60 ms prebuffer and p95/p99 conventions.

## Key code

| Area | Path |
|---|---|
| Session lifecycle and configuration | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Stateless history construction | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| Session/turn/segment identity | `vllm_omni/entrypoints/openai/video_stream_state.py` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Three session baselines | `benchmarks/live_agent/web_client/run_session_baselines.sh` |
| Playback timeline | `benchmarks/live_agent/playback_metrics.py` |
| Legacy SLO analysis | `benchmarks/thinker_talker/analyze.py` |
| Historical two-GPU capacity deployment | `benchmarks/thinker_talker/deploy_2gpu_seq256.yaml` |
| Scheduler and connector flags | `vllm_omni/core/sched/runtime_flags.py` |
| Chunk transport | `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py` |
| Code predictor KV | `vllm_omni/model_executor/models/common/qwen3_code_predictor.py` |
| Code2wav and Snake | `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_code2wav.py` |
