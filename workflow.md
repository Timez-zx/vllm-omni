# vLLM-Omni Realtime Multi-User Serving Workflow

## Goal and scope

The goal is to serve more continuous audio-video sessions with fewer GPUs while meeting TTFA and playback-continuity targets. The research focus is engine capacity, scheduling, KV cache behavior, and tail latency rather than model quality.

No locally deployable open realtime model currently matches the interaction semantics of Seed Realtime or Gemini Live. This branch approximates that workload with Qwen3-Omni's Thinker → Talker → Code2Wav pipeline: clients continuously upload audio and video, while the model answers in turns. This produces useful multimodal-prefill, speech-decode, and multi-tenant contention, but it is not native full duplex or semantic barge-in.

## Phase 1: application baseline

The current design is stateful at the application and finite-lived at the engine request layer:

```text
continuous media arrival
  → video frames launch silent finite Thinker requests to warm prefix KV
  → end of user speech submits complete canonical history + complete WAV
  → Thinker → Talker → Code2Wav
  → response finishes and the request is destroyed
  → the application commits the turn; the next turn gets a new request ID
```

Key constraints:

- The WebSocket application owns session state, media reception, and complete canonical multimodal history.
- Every silent warm-up and final response is a new ordinary finite request. The engine owns no live request across turns.
- Every turn submits the complete canonical prompt. Prefix/KV cache entries are disposable; a miss adds prefill work but cannot change correctness.
- The application stores processed canonical message blocks. It renders only the new user and assistant blocks, then assembles the full prompt without reprocessing all historical media.
- Video is append-only within a turn. The similarity/freshness filter decides admission; accepted frames are not subjected to an eight-frame sliding window or another sampling pass.
- New frames launch or coalesce into low-priority `video-warm-<uuid>` requests. They run Thinker only with `max_tokens=1`, emit no text, and never enter Talker. A query immediately cancels unfinished warm-ups instead of waiting for cache fill.
- User audio is a single complete WAV at query time, preserving Qwen's whole-audio semantics. Only the final query may produce speech.
- Media arriving during a response belongs to the next turn, and each media item is consumed once.
- A normal response is capped at 256 Thinker tokens. Video is bounded to 640×352, and JPEG work runs in subprocesses.
- The hard history threshold is 49,152 tokens. A 16,384-token headroom normally causes proactive complete-turn compaction near 32,768 tokens to a target below 16,384, rotating the cache lineage.

`enable_audio_arrival_prefill_approximation` is off by default. It seals audio into one-second chunks for silent arrival prefill and exists only to emulate duplex engine load; it is not semantically equivalent to Qwen whole-audio inference.

The old cross-turn persistent request, resumable append, Talker 45k rolling, Thinker shadow compression, and session ledger have been removed. The current baseline follows a general engine request abstraction and can connect naturally to routing, replication, and P/D separation.

## Phase 2: fixed deployment

Formal experiments use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`:

| Stage | GPU | Configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights/KV, prefix cache, priority scheduler |
| Talker | 1 | FP8 weights/KV, session-isolated conditioning prefix cache |
| Code2Wav | 2 | separate process |

Final responses have priority 0 and silent warm-ups priority 10. The YAML remains unchanged across user counts and session-policy controls. `run_qwen_server.sh` locates the CUDA toolkit and checks that Thinker selected FlashInfer. The multimodal processor cache uses API-side `processor_only` mode.

## Phase 3: formal workload

The only formal workload is continuous AV sessions:

- each user keeps one long-lived WebSocket;
- video uploads continuously at 2 FPS, and accepted frames immediately launch or coalesce into silent Thinker warm-ups;
- the microphone uploads PCM16 at 5 Hz, pausing during assistant playback plus a 300 ms echo guard;
- every turn uses a unique real 16 kHz mono SLURP recording followed by 700 ms endpoint silence, with empty query text;
- audio is appended as one complete WAV to the warmed video prefix at query time, and only this request invokes Talker;
- a session keeps one speaker, while DAVIS video uses a fixed sequence with different starting offsets;
- the next turn starts after response playback completes at 1×, producing a playback-paced closed loop;
- users start at deterministic offsets over 0–40 seconds.

A formal cell has 30 turns per user and excludes the first two as warm-up. The ladder runs 8, 16, 32, ... users and stops each seed at the first SLO failure. The engine restarts for every cell. Synthetic media in `probe.py` is only for protocol validation.

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

## Phase 4: metrics and pass rule

- **TTFA**: query to the first audio packet. It depends on packet size and is only a within-deployment diagnostic.
- **Audio-ready-500**: time until 500 ms of playable audio has accumulated; this is the packetization-independent startup metric.
- **Stall max**: the largest single playback underflow at 1×.
- **RTF deliver**: generated audio duration divided by delivery duration, measuring sustained supply.

A cell passes only when:

- every post-warm-up turn completes;
- Audio-ready-500 p99 is below 1 second;
- Stall-max p99 is below 50 ms;
- there is no client, protocol, or fatal engine error.

Results must use workload schema 4 and pass `benchmarks/live_agent/analysis/verify_run.py`. The verifier checks unique finite requests, arrival warm-ups, the frame ledger, real prefix-cache hits, and separate stage processes.

## Phase 5: current baseline and conclusions

Exact setup for the latest formal control:

- source: clean commit `854535bb85789a882ff5995362e5528a5f52f83d`;
- deploy: `origin_deploy_3gpu.yaml`, SHA256 `ae7cbeb615b24ee8654cf6c867887b2920324fc81ec93be6aaaa8995c55e4e24`;
- workload: schema 4, seed 7, eight users × 30 turns, two warm-up turns, plan SHA256 `99dc083388931ffcbf9479a7f4344613229350546824371fec3744b548db0ed4`;
- default arm: no `MU_SESSION_CFG_JSON`;
- aligned arm: `MU_SESSION_CFG_JSON='{"context_window_trigger_tokens":32000,"context_window_target_tokens":0,"context_window_compaction_headroom_tokens":0}'`.

| Path | Context p50/p95/p99/max | Audio-ready-500 p50/p95/p99/max |
|---|---:|---:|
| Current default history | 22.5k/32.0k/32.4k/32.8k | 478/723/1186/1215 ms |
| Current `32k -> 0` aligned control | 16.8k/30.6k/31.8k/31.9k | 410/635/796/883 ms |
| Archived persistent path | 14.8k/30.9k/34.5k/36.8k | 515/740/824/900 ms |

Result paths:

- default: `/home/ubuntu/data/results/current_default_854535bb_20260822/current_default_seed7_u8`
- aligned: `/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`
- persistent archive: `/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`

The comparison establishes:

1. **Raw TTFA is not comparable.** The persistent first packet held only about 217 ms of audio, producing raw TTFA p50/p95/p99 of 244/392/479 ms. Recomputing its audio deltas at a fixed cumulative 500 ms gives the table's 515/740/824 ms.
2. **History policy materially affects tail latency.** At the same commit, the aligned control reduced p99 from 1186 to 796 ms. GPU0 SM-active p95 in p99-tail windows was 94.7% for default and 52.4% when aligned. The difference includes context length, retained multimodal content, compaction/cache-lineage churn, and closed-loop trajectory; it is not a token-count-only result.
3. **Finite request lifetime is not the performance problem.** After context alignment, current p99 is 796 ms versus 824 ms for persistent, which is the same performance class. Application-owned sessions, finite engine requests, and disposable prefix/KV reuse are a valid architecture.
4. **Repeated application work is no longer the primary tail source.** Processed canonical blocks remove full-history re-rendering, warm-ups do not block queries, and residual upstream work is cleaned after completion. Default tail contains both Thinker delay and post-first-text pipeline waiting, while Talker and Code2Wav GPUs are not saturated.
5. **`32k -> 0` is not a production policy.** It discards completed turns and exists only to isolate request lifetime and compute envelope. A single-seed live closed loop is also not a bit-identical replay.

The application architecture is suitable as the engine-research baseline. The unresolved application decision is production-grade history compression, not request lifetime.

## Phase 6: audio arrival-prefill research arm

Qwen's audio encoder uses bidirectional attention within an approximately eight-second window, so independently encoding one-second audio chunks changes semantics. This path is off by default and exists only to study arrival-prefill load.

This short A/B used schema 4, seed 7, eight users × six turns, one warm-up turn, a 0–8 second stagger, and plan SHA256 `b6160ec3d78a2126fb07830e5c1b75f9319ba579543555c8474852ad9f9ddf9b`. The baseline had no override; the arrival arm used `MU_SESSION_CFG_JSON='{"enable_audio_arrival_prefill_approximation":true}'`. Result metadata reports dirty source `cbf2226a`, so this supports a directional conclusion only and is not a formal result reproducible from one commit.

Short eight-user × six-turn A/B:

| Input mode | Audio-ready-500 p50/p95/p99 |
|---|---:|
| Query-time complete WAV | 368/475/654 ms |
| One-second audio-arrival approximation | 343/554/692 ms |

Arrival saves only about 25 ms at the median and worsens p95/p99 because additional warm-ups compete with foreground queries. The formal workload therefore keeps complete WAV at query time. Semantically equivalent audio arrival prefill requires a native causal/streaming audio encoder.

Results: `/home/ubuntu/data/results/audio_arrival_ab_baseline_20260822/u8_t6` and `/home/ubuntu/data/results/audio_arrival_approx_dev_20260822/u8_t6`.

## Phase 7: next steps

1. Add bounded semantic compaction at the application: a short text summary/seed plus a few recent complete turns. Do not use `32k -> 0` as the formal configuration.
2. Freeze the compaction policy and context envelope, then restart the 8, 16, 32, ... capacity ladder.
3. At the first SLO failure, split Thinker prefill, Thinker decode, post-first-text pipeline waiting, and GPU activity.
4. If concurrent Thinker work still dominates tail latency, compare scheduling isolation and P/D separation rather than hiding the engine issue with application tuning.

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
