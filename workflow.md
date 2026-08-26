# vLLM-Omni Realtime Multi-User Serving Workflow

## Goal and scope

The goal is to serve more continuous audio-video sessions with fewer GPUs while meeting TTFA and playback-continuity targets. The research target is engine capacity, scheduling, KV cache behavior, and tail latency rather than model quality.

No locally deployable open model currently matches Seed Realtime or Gemini Live. This branch approximates continuous AV sessions with Qwen3-Omni's Thinker → Talker → Code2Wav pipeline. It targets concurrent continuous-video incremental compute and response decode, not native full duplex or semantic barge-in.

## Phase 1: application baseline

The application owns session state; the engine handles only ordinary finite requests:

```text
video arrival → silent finite Thinker request warms prefix KV
end of speech → complete canonical history + current complete WAV
              → Thinker → Talker → Code2Wav
response end  → destroy request and commit the turn in the application
```

- Every turn and arrival warm-up uses a new request ID. The engine holds no live request across turns.
- Every submission contains the complete canonical prompt. Prefix/KV entries are disposable; a miss changes cost, not semantics.
- The application caches processed canonical message blocks, renders only new blocks, and assembles the complete prompt.
- Video is append-only within a turn after similarity/freshness filtering. There is no eight-frame sliding window or second sampling pass.
- Each session runs at most one arrival request. It runs Thinker with `max_tokens=1`, emits no text, and never invokes Talker.
- Frames accepted while an arrival runs remain separate images and enter the next cumulative prompt snapshot together. Coalescing reduces request count; it does not merge or discard image content.
- Different sessions submit arrivals concurrently to native FCFS. There is no application-wide gate or request priority.
- A final query waits for its session's admitted arrival, then submits the complete prompt, preserving ordered prefix lineage.
- User audio is one complete WAV at query time, preserving Qwen whole-audio semantics. Only the final query invokes Talker.
- Media received during a response belongs to the next turn, and each accepted item is consumed by one final query.
- Thinker responses are capped at 256 tokens. Video is bounded to 640×352, and JPEG work runs in subprocesses.
- Compaction starts only at 49,152 prompt tokens. It retains the newest two completed turns as user audio/text plus assistant text, removes historical images, and keeps only the newest current-turn image. If needed, it drops more complete old turns. It never generates a summary request.

The old cross-turn persistent request, resumable append, Talker rolling, shadow compression, and session ledger have been removed.

## Phase 2: fixed deployment

Formal experiments use only `benchmarks/thinker_talker/origin_deploy_3gpu.yaml`:

| Stage | GPU | Configuration |
|---|---:|---|
| Thinker | 0 | FP8 weights/KV, prefix cache, synchronous scheduling, equal-priority requests |
| Talker | 1 | FP8 weights/KV, conditioning prefix cache |
| Code2Wav | 2 | separate process |

`run_qwen_server.sh` verifies that Thinker selected FlashInfer and creates the stage-0 SHM multimodal cache before its worker starts. The API renderer and input processor share one sender; the worker reuses materialized media rather than restoring the complete media history for every finite request.

## Phase 3: formal workload

- Each user owns one long-lived WebSocket.
- Video uploads continuously at 2 FPS; accepted frames trigger or queue for the next silent Thinker arrival prefill.
- The microphone uploads PCM16 at 5 Hz, pausing during assistant playback plus a 300 ms echo guard.
- Every turn uses a unique real 16 kHz mono SLURP recording followed by 700 ms endpoint silence, with empty query text.
- Audio is appended as one complete WAV to the warmed video prefix at query time.
- Each session keeps one speaker; DAVIS video uses a fixed sequence and different starting offsets.
- The next turn starts after 1× response playback, producing a playback-paced closed loop.
- Users start at deterministic offsets over 0–8 seconds.

This is duplex-like rather than complete AV duplex. Video continues to trigger arrival prefill during user speech and assistant playback. Audio uploads at 5 Hz, but Qwen processes one complete WAV only at query time, and the microphone pauses during playback. The workload targets continuous visual incremental compute; it does not claim to reproduce Gemini Live or Seed Realtime model structure or absolute capacity.

A formal cell has 30 turns per user and excludes the first two from metrics. The engine restarts for every cell. Synthetic media in `probe.py` is only for protocol validation.

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
RESULTS_DIR=/home/ubuntu/data/results/finite_request_capacity_<commit> \
RESULT_PREFIX=finite_request USERS=16 SEEDS=7 TURNS=30 WARMUP_TURNS=2 \
bash benchmarks/live_agent/web_client/run_av_session_ladder.sh
```

## Phase 4: metrics and pass rule

- **TTFT**: query to first text.
- **TTFA / Audio-ready-500**: query until 500 ms of playable audio has accumulated. The current first packet exceeds 500 ms, so they are equal.
- **Stall max**: largest single playback underflow at 1×.
- **RTF deliver**: generated audio duration divided by delivery duration.

A cell passes only when every measured turn completes, Audio-ready-500 p99 is below 1 second, Stall-max p99 is below 50 ms, and there are no client, protocol, or fatal engine errors.

Results must use workload schema 4 and run `benchmarks/live_agent/analysis/verify_run.py`. The verifier checks finite requests, arrival requests, the frame ledger, prefix-cache hits, and three separate stage processes.

## Phase 5: current 16-user result

Setup:

- source: dirty working tree on `726ebd90fe1120d8d0ce1d8c4d99dd54e6703912`;
- deploy: `origin_deploy_3gpu.yaml`, measurement SHA256 `8cfcd31af59031ba20dc822632510a2de721dca9ede8a80070c8cf5647a7787f`;
- workload: schema 4, seed 7, 16 users × 12 turns, two warm-up turns, 0–8 second stagger;
- plan SHA256: `1dfd08e50710f188e5d83526358b9c3aecf4ad0e8616a5b0bb44d6b45d3b0a6f`;
- result: `/home/ubuntu/data/results/nonpd_shm_sync_u16_t12_20260826/nonpd_shm_sync_seed7_u16`.

| Metric | Result |
|---|---:|
| Completion | 160/160, zero timeouts, zero client errors |
| TTFT p50/p99 | 341/2550 ms |
| TTFA p50/p95/p99 | 1371/5371/8212 ms |
| Stall-max p99 | 1972 ms |
| Same-session arrival wait p50/p95/p99 | 0/662/1153 ms |
| Prompt render p50/p95/p99 | 19/81/149 ms |
| Thinker TTFT p50/p95/p99 | 268/1022/1392 ms |
| Thinker ITL p50/p95/p99 | 87/345/544 ms |
| Engine→first audio p50/p95/p99 | 1292/4664/7601 ms |

Verification found 192 final requests, 2,196 arrival requests, 2,712 consumed frame occurrences, and 2,559 prefix-cache hits. There were zero preemptions, zero recomputes, and zero arrival failures.

Tail conclusions:

1. Expanding stagger from 0–8 to 0–40 seconds still produced 7,794 ms TTFA p99; stagger is not the root cause. Control result: `/home/ubuntu/data/results/nonpd_shm_sync_stagger40_u16_t12_20260826/nonpd_shm_sync_stagger40_seed7_u16`.
2. Prompt-render p99 is 149 ms, and same-session arrival-wait p99 is 1,153 ms. Both amplify the tail but cannot explain 8.2 seconds TTFA.
3. Thinker dominates. Formal-request decode ITL reaches 544 ms p99: every generated token repeatedly waits for a heavy batch containing arrival prefills. Talker and Code2Wav chiefly wait for upstream tokens.
4. The application permits one arrival per session, so 16 sessions can create 16 concurrent arrivals. vLLM schedules `running` requests before admitting requests from `waiting`; priority does not displace an existing `running` prefill. Admitted arrivals therefore share token budget and model forwards with formal decode. Continuous replenishment creates decode starvation.

Conclusion: the current application/session implementation is an adequate engine-research baseline. The 16-user tail is not caused by stagger, prompt rendering, summaries, or Talker/Code2Wav saturation. It comes from multi-user fine-grained arrival prefill repeatedly slowing Thinker decode.

## Phase 6: established architecture result

Keep the design stateful in the application, finite at the engine request layer, and dependent only on disposable prefix/KV reuse. The formal workload must not add an application-wide cross-user gate; doing so changes the duplex-like engine load and hides contention. The next step is deadline/QoS-aware incremental-prefill scheduling under a fixed input trace, followed by a 16×30 formal run after the implementation is frozen.

## Recovery map

| Purpose | Path |
|---|---|
| Session and finite-request lifecycle | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload plan and media loading | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| Capacity runner | `benchmarks/live_agent/web_client/run_av_session_ladder.sh` |
| Fixed deployment | `benchmarks/thinker_talker/origin_deploy_3gpu.yaml` |
| Result verification | `benchmarks/live_agent/analysis/verify_run.py` |
| Tail attribution | `benchmarks/live_agent/analysis/p99_attribution.py`, `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU sampling | `benchmarks/live_agent/harness/gpu_sampler.py` |
