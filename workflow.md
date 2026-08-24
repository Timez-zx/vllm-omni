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
- New frames launch or coalesce into low-priority `video-warm-<uuid>` requests. They run Thinker only with `max_tokens=1`, emit no text, and never enter Talker. At most one arrival warm-up is admitted globally. A query closes background admission and immediately cancels registered and same-session pending warm-ups instead of waiting for cache fill.
- The foreground gate reopens after the first engine output. New media may then resume arrival prefill and contend with ongoing Thinker decode. This is intentional engine pressure from the target realtime workload, not a cross-turn persistent request.
- User audio is a single complete WAV at query time, preserving Qwen's whole-audio semantics. Only the final query may produce speech.
- Media arriving during a response belongs to the next turn, and each media item is consumed once.
- A normal response is capped at 256 Thinker tokens. Video is bounded to 640×352, and JPEG work runs in subprocesses.
- The hard history threshold is 49,152 tokens, and a 16,384-token headroom normally triggers proactive compaction near 32,768 tokens. Compaction generates at most 512 tokens of durable text memory, retains the newest two complete turns, and targets a prompt near 16,384 tokens. A summary or rewrite failure leaves history unchanged during proactive maintenance; only the hard limit enables complete-turn dropping as a safe fallback. Compaction and turn commit share one session lock and rotate the cache lineage.

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

### Archived architecture control

The context-aligned eight-user experiment established that finite request lifetime is not the performance problem. Audio-ready-500 p99 was 796 ms for the current finite-request path and 824 ms for the archived persistent path after recomputing both at a fixed 500 ms audio threshold. Application-owned sessions, finite engine requests, and disposable prefix/KV reuse are a valid architecture. Results:

- `/home/ubuntu/data/results/current_context_aligned_854535bb_20260822/context_aligned_seed7_u8`
- `/home/ubuntu/data/results/av_real_formal_4650f134/avreal_formal_seed7_u8`

### 16-user summary + recent diagnostic

Setup:

- source: dirty working tree on `3b661ed19ae343b90576eec39575baf1e1c27a5e`; this is a pre-commit diagnostic, not a clean-commit archive;
- deploy: `origin_deploy_3gpu.yaml`, SHA256 `ae7cbeb615b24ee8654cf6c867887b2920324fc81ec93be6aaaa8995c55e4e24`;
- workload: schema 4, seed 7, 16 users × 30 turns, two warm-up turns, with no `MU_SESSION_CFG_JSON`;
- plan SHA256: `bdaa528c57ba688b3c0d6889d0877029a595a5f249ef81b631e0b8e13b597aa7`;
- input-trace SHA256: `f217432fa3fdc7362f4926f9a7c2c4c9219bc0dd453901d0d93de8f2245fe561`;
- result: `/home/ubuntu/data/results/nonpd_summary_recent_u16_t30_diag_20260823_v2/nonpd_summary_recent_diag_seed7_u16`.

Results:

| Metric | Result |
|---|---:|
| Completion | 448/448, zero timeouts, zero client errors |
| TTFT p50/p99 | 297/914 ms |
| TTFA p50/p95/p99 | 554/1111/1412 ms |
| Audio-ready-500 p99 | 1412 ms |
| Stall-max p99 | 0 ms |
| Prompt tokens p50/p95/p99/max | 20.4k/33.5k/35.4k/37.1k |

The cell failed because Audio-ready-500 p99 exceeded one second, but playback had no stalls after startup. `verify_run.py` confirmed 480 unique finite requests, 2,613 arrival-prefill requests, 6,228 consumed frame occurrences, 3,508/3,538 prefix-cache hits, and three independent stage processes.

An invalid v1 run exposed a P/D admission handshake that had been incorrectly carried into the non-P/D path, producing up to 13.843 seconds of foreground waiting. The non-P/D path now directly cancels the background task and lets local AsyncOmni clean up the request. In v2, query-gate p50/p95/p99/max was 0.2/1.3/2.4/2.9 ms, removing this application-level confounder.

Tail attribution:

1. **Concurrent prefill/decode work on the Thinker GPU is the primary source.** About 63% of tail95 excess is on the Thinker side and 37% is after first text. Thinker TTFT p99 is 914 ms, while median Thinker inter-token latency rises from 18.8 ms overall to 37.8 ms in tail95.
2. **Concurrency explains more than one request's prompt length.** Mean per-request new prefill tokens rise only from 850 overall to 1,025 in tail95. Foreground prefill tokens admitted while waiting for first text rise from 1,135 to 2,612, or 2.3×. When 1/2/3/4 queries arrive within 250 ms, TTFT p50 is 254/369/418/642 ms.
3. **GPU0 is materially busier during tail windows.** Thinker SM-active p50/p95 rises from 38%/66% overall to 53%/87% in tail95. Tail-window values are 14%/24% for Talker and 3%/7% for Code2Wav, so neither downstream GPU is saturated. Allocated memory is not compute utilization.
4. **Queue depth rises with latency.** The mean number of other sessions waiting for first audio at query arrival rises from 0.54 overall to 1.13 in tail95. With 0/1/2/3 such sessions, TTFA p50 is 505/595/696/867 ms.
5. **Summary maintenance is a secondary amplifier.** There were 42 summaries. Using approximate second-resolution server overlap, summaries intersected 72/448 measured turns and 9/23 tail95 turns. TTFA p95 remains about 1,018 ms without summary overlap, so summaries do not explain the primary tail. Forty-five SHM mailbox fallbacks covered only 2/23 tail95 turns and are also not the main cause.

Conclusion: the application is now adequate as the engine-research baseline. At 16 users the dominant delay is not full-history rendering, foreground admission, Talker, or Code2Wav saturation. It is foreground and arrival prefill on GPU0 interfering with Thinker decode; summary maintenance amplifies a minority of tail events.

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

1. Commit the current application baseline, then replay the recorded input trace on that clean commit for a formal 16-user archive.
2. If the first capacity boundary must be stated rigorously, add an eight-user cell at the same commit and seed; this task intentionally ran only 16 users.
3. Engine experiments should first isolate foreground decode from arrival and foreground prefill, then compare P/D separation on its dedicated branch. Do not hide the contention with more application tuning.

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
| Tail attribution | `benchmarks/live_agent/analysis/p99_attribution.py`, `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU sampling | `benchmarks/live_agent/harness/gpu_sampler.py` |
