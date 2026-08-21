# vLLM-Omni Realtime Multi-User Serving Workflow

## Goal and current decision

The goal is to serve more continuous audio/video sessions with fewer GPUs while meeting TTFA and speech-smoothness targets. The research target is engine capacity, scheduling, and tail latency, not model quality.

No locally deployable open-source realtime model exposes semantics equivalent to Seed Realtime or Gemini Live. This branch therefore approximates that product workload with the Qwen3-Omni Thinker → Talker → Code2Wav pipeline: clients upload audio and video continuously, while the model still answers in turns. This is sufficient for studying multimodal prefill, speech decode, KV cache, and multi-tenant contention, but it is not model-native full duplex or semantic barge-in.

The current architecture is:

- the WebSocket application owns the session, full multimodal conversation, and media-receive state;
- every turn creates a new, finite, ordinary engine request;
- the engine owns KV within a request and may retain only disposable prefix/KV cache after it ends;
- every turn submits the complete canonical history; a prefix-cache hit reduces work but never determines correctness;
- a cross-turn lifecycle is no longer embedded in one resumable engine request.

This is a better research baseline than an engine-resident session: it matches the general engine request abstraction, makes cache misses performance-only events, and composes with routing, replication, and P/D disaggregation. The application still pays to assemble and preprocess history each turn, and an evicted prefix must be prefilled again. Those costs must be measured rather than hidden behind private engine state.

## Phase 1: application baseline

The implementation lives in `vllm_omni/entrypoints/openai/video_stream_base.py` and `serving_video_stream.py`.

- Each `video.query` receives a unique `video-<uuid>` request ID.
- The application stores completed user/assistant turns. User history preserves accepted text, audio, and selected video rather than degrading to text-only summaries.
- Audio and video received for the current turn are consumed exactly once. Data arriving during generation belongs to the next turn.
- Each turn renders the full canonical prompt. Thinker prefix caching reuses the prefix identical to the preceding request.
- At 49,152 prompt tokens, history is removed from the oldest end only at complete-turn boundaries until at most 16,384 tokens remain. The compacted prompt starts a new cache lineage.
- At most the latest eight frames are retained and submitted per turn, at no more than 640×352.
- The similarity threshold is 0.95 and the freshness gap is `[0,4]`: redundant frames may be dropped, but one frame is forced after four consecutive drops.
- JPEG decode, resize, and thumbnail generation run in a subprocess pool so they do not block the WebSocket event loop.
- Streaming audio DELTAs are forwarded one by one; the client evaluates startup and stalls on a playback timeline.

Removed paths include the cross-turn persistent request, arrival-time prefill-only append, Talker 45k rolling, Thinker shadow compression, the session epoch/segment ledger, and benchmark/diagnostic scripts that required them. Upstream generic streaming/resumable support remains, but this application does not use it.

## Phase 2: fixed deployment

Formal experiments use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`:

| Stage | GPU | Key configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights/KV, prefix caching enabled |
| Talker | 1 | FP8 weights/KV, prefix caching disabled |
| Code2Wav | 2 | separate process |

All stages use separate processes and the YAML remains fixed across user counts. Talker does not use a cross-turn prefix cache; it generates speech within each finite request. Request-lifecycle-independent mailbox, vocoder, Snake, and code-predictor optimizations remain in the tree.

Start the deployment with:

```bash
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

## Phase 3: the only formal workload

Capacity testing models continuous AV sessions. Each user owns one long-lived WebSocket:

- video is uploaded throughout the session at 2 FPS;
- the microphone uploads PCM16 at 5 Hz, pausing during assistant playback plus a 300 ms echo guard;
- each turn uses one real 16 kHz mono recording followed by 700 ms endpoint silence; query text is empty, so semantics come from speech;
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

- **TTFA**: query submission to the first audio arrival. It locates service delay but is not when the user necessarily hears audio.
- **Playback start**: playback begins after 1.4 seconds of audio has buffered; a short reply is released at `audio.done`. The current first granule is about 217 ms, so startup usually waits for a later granule.
- **Stall max**: the largest single underflow when a granule arrives after all previously buffered audio would have played at 1×.
- **RTF deliver**: generated audio duration divided by delivery duration; this measures sustained supply.

A cell passes only when every post-warm-up turn completes, playback-start p99 is below 1 second, stall-max p99 is below 50 ms, and no client, protocol, or fatal engine error occurs. TTFA, GPU utilization, memory, and engine-step data are root-cause signals, not substitutes for the experience SLO.

`analysis/verify_run.py` additionally checks that every turn used a unique finite request, Thinker prefix caching was enabled, later turns produced real prefix hits, and all three stages ran in separate processes.

## Phase 5: archived old-architecture experiments

The following results describe the deleted persistent-request plus arrival-prefill implementation. They must not be merged into the new architecture's capacity curve.

In the 30-turn continuous-AV capacity run, both seeds passed at 8 users and failed at 16. At 8 users, TTFA p99 was 479–541 ms and playback-start p99 was 824–972 ms. At 16 users, TTFA p99 was 1005–1178 ms and playback-start p99 was 2062–2117 ms. The failure was a playback-start tail, not timeouts or playback stalls.

The fair 16-user prefill-timing RCA replayed the same client trace and per-turn media ledger. Every arm submitted 2,358 video occurrences and 36,884,706 audio bytes, with zero dropped frames and zero replay slip.

| Old submission policy | Prefill chunks | TTFA p99 | Playback-start p99 | Second-granule gap p99 | Stall p99 |
|---|---:|---:|---:|---:|---:|
| Arrival incremental | 2783 | 1283 ms | 2337 ms | 1240 ms | 0 |
| Video at query time | 967 | 2353 ms | 3172 ms | 1371 ms | 0 |
| All media at query time | 160 | 2416 ms | 2946 ms | 2011 ms | 223 ms |

The conclusion is limited to the old implementation: concentrating the same media at the query boundary produced a stronger Thinker prefill burst. An earlier apparent improvement was confounded by eviction from an eight-frame buffer and is invalid. Logs also showed that most long Talker waits overlapped a Thinker step, while Code2Wav was not the primary tail source. This motivates studying Thinker prefill/decode interference and P/D disaggregation, but it does not predict the finite-request architecture's capacity.

Archives:

- `/home/ubuntu/data/results/archive/2026-08-20_continuous_av_capacity_rca/`
- `/home/ubuntu/data/results/archive/2026-08-20_prefill_timing_fair_rca/`

## Phase 6: next experiment

The two-turn direct smoke on 2026-08-21 passed: prompts were 280/314 tokens, request IDs differed, Thinker prefix hits were 0/288 tokens, and both turns produced complete speech. This host lacked `nvcc`, so the smoke temporarily used `--attention-backend TRITON_ATTN`. It proves functionality, not capacity. A formal run must first restore the FlashInfer environment or explicitly adopt Triton; results from the two backends cannot be mixed.

The current code has no new formal capacity result. Rebuild the 8 → 16 → 32 curve from a clean commit:

1. Confirm unique request IDs, non-zero prefix hits after the first turn, and the actual processed-frame count.
2. At the first failure point, collect TTFA/playback tails, Thinker prefill/decode steps, Talker waits, GPU SM active, memory, and KV-cache use.
3. Distinguish active GPU computation from memory occupancy while the GPU is idle.
4. Compare P/D disaggregation with the same workload and media trace; implementation difficulty does not lower its experimental priority.

Results are directly comparable only when commit, YAML, input hashes, seed, users, turns, prebuffer, and SLO definitions match. Use at least two seeds near the boundary and do not claim changes smaller than normal run-to-run variation.

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
