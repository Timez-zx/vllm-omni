# vLLM-Omni Real-Time Multi-User Serving Workflow

This document keeps only the current reproducible path, the controls that justify design decisions, and confirmed conclusions. The research branch is `thinker-talker-pd`.

## Phase 1: objective and model boundary

The goal is to serve more long-lived audio/video sessions with fewer GPUs while meeting speech-startup and continuity SLOs. The research target is engine scheduling, KV cache, data movement, capacity, and tail latency—not model quality.

No locally deployable open model currently matches the interaction semantics of Seed Realtime or Gemini Live. This project approximates the target load with Qwen3-Omni's Thinker → Talker → Code2Wav pipeline: clients continuously upload audio/video and the model answers by turn. This produces realistic multimodal prefill, speech decode, and multi-user contention, but it is not a native full-duplex model and does not study semantic barge-in.

## Phase 2: application/engine boundary

The current design keeps the session in the application and gives the engine finite-lived requests:

```text
media arrives continuously
  → filtered video frames trigger or coalesce into silent Thinker warm-ups
  → at end of speech, submit full canonical history + one complete WAV
  → Thinker P → Thinker D → Talker → Code2Wav
  → destroy the request after the answer; persist the turn in the application
```

The invariants are:

- The WebSocket application owns session state, media state, and canonical multimodal history. The engine holds no live cross-turn request.
- Every warm-up and final answer is a new finite request. Each turn submits the full canonical prompt; engine prefix/KV caches are disposable accelerators. A cache miss costs compute but cannot affect correctness.
- The application renders only new message blocks and assembles the full prompt without reprocessing historical media.
- Video enters the current turn append-only after similarity/freshness filtering; the old eight-frame sliding eviction is gone.
- Video warm-ups use priority 10, run Thinker only with `max_tokens=1`, return no text, and never enter Talker. The final query uses priority 0 and cancels unfinished warm-ups.
- Audio is submitted as one complete WAV at query time. Qwen's audio encoder uses bidirectional attention, so independently encoded chunks are not guaranteed to preserve whole-audio semantics.
- Media arriving during an answer belongs to the next turn. History is normally compacted by complete turns near 32k tokens to at most 16k; the hard limit is 49,152 tokens.

This boundary has compute semantics similar to a stateless API backed by prefix caching, while making application ownership of session state explicit. The old cross-turn persistent request, resumable append, Talker rolling, and shadow-request paths have been removed. The application can now use routing, replication, or P/D separation without coupling them to session lifecycle.

## Phase 3: fixed deployment and workload

### Deployment

Upstream vLLM-Omni could not split one Thinker into separate P and D stages while still supplying Talker conditioning states. This branch implements that path without changing the application protocol or finite-request lifecycle.

The formal P/D deployment is fixed at `benchmarks/thinker_talker/pd_deploy_4gpu.yaml`; its current SHA256 is `7bc4502494c19d07046a6ce5e2c272ebf4536108d1336831ea4294d78437ba24`.

| Stage | GPU | Main configuration |
|---|---:|---|
| Thinker P | 0 | FP8 weights/KV, prefix cache, priority, 32k batched tokens |
| Thinker D | 1 | FP8 weights/KV, prefix cache, priority, Delta-KV consumer |
| Talker | 2 | FP8 weights/KV, prefix cache, streaming codec output |
| Code2Wav | 3 | Separate stage, streaming waveform generation |

P→D uses `NixlDeltaPushConnector`; D→Talker→Code2Wav uses shared-memory connectors. Formal experiments vary only users, seed, or an explicitly named experimental variable—not deployment parameters.

### Continuous-AV session workload

- Each user keeps one long-lived WebSocket.
- Video is uploaded continuously at 2 FPS; accepted frames trigger or coalesce into Thinker warm-ups.
- PCM16 microphone audio is uploaded at 5 Hz, paused during assistant playback, with a 300 ms echo guard.
- Each turn uses a unique real 16 kHz mono SLURP utterance plus 700 ms endpoint silence. Query text is empty, and the complete WAV is submitted at query time.
- Each session keeps one speaker. Video comes from fixed DAVIS sequences with different start offsets.
- The next turn starts after the prior answer finishes 1× playback, forming a playback-paced closed loop.
- User starts are deterministically staggered over 0–8 seconds.

Use one or two users for protocol smoke tests. Fixed performance replay uses eight users × 12 turns with the first turn as warm-up. Capacity runs use 30 turns/user, the first two as warm-up, and increase through 8, 16, 32, ... users. Restart the engine for every capacity cell and stop at the first SLO failure.

Canonical eight-user × 12-turn trace:

- File: `/home/ubuntu/data/results/pd_deferred_free_fix_20260823_v1/u8_t12_seed7_u8/input_trace.jsonl.gz`
- SHA256: `03591906c1dd356df55eaac3081dc2efc4f6357f900f3411919c97b55fecf33b`
- Workload-plan SHA256: `7f08495fdc2b9a2ff565deb8f1923347124ae5f24527d2153167332749d5b1bd`
- Fixed ledger: 3,005 frames sent, 1,323 accepted, 1,278 consumed, and 3,606 audio chunks.

Replay command:

```bash
MU_FRAMES_DIR=/home/ubuntu/data/workloads/continuous_av_v1/frames \
MU_AUDIO_MANIFEST=/home/ubuntu/data/workloads/continuous_av_v1/audio_manifest.jsonl \
VLLM_OMNI_BIN=/home/ubuntu/miniconda3/envs/omni/bin/vllm-omni \
MU_PYTHON=/home/ubuntu/miniconda3/envs/omni/bin/python \
MU_INPUT_TRACE_MODE=replay \
MU_REPLAY_INPUT_TRACE=/home/ubuntu/data/results/pd_deferred_free_fix_20260823_v1/u8_t12_seed7_u8/input_trace.jsonl.gz \
RESULTS_DIR=/home/ubuntu/data/results/pd_replay_<commit> \
RESULT_PREFIX=pd_replay USERS=8 SEEDS=7 TURNS=12 WARMUP_TURNS=1 \
bash benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh
```

For the capacity ladder, remove the replay variables and use `USERS="8 16 32" TURNS=30 WARMUP_TURNS=2`. Synthetic media from `probe.py` is only a protocol check and cannot support capacity conclusions.

## Phase 4: metrics and pass criteria

- **TTFA**: query to the first audio packet. It depends on packetization and is compared only within the same deployment.
- **Audio-ready-500**: query to 500 ms of cumulative playable audio. This is the official startup metric.
- **Stall max**: the largest single underflow during 1× playback.
- **RTF deliver**: output-audio duration divided by delivery time, used to test sustained supply.

A capacity cell passes only if every scored turn completes, Audio-ready-500 p99 is below 1 s, Stall-max p99 is below 50 ms, and there is no client, protocol, or fatal engine error.

Results must use workload schema 4 and pass:

```bash
MU_EXPECTED_DEPLOY_BASENAME=pd_deploy_4gpu.yaml \
MU_EXPECTED_STAGE_IDS=0,1,2,3 \
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

The verifier checks finite-request uniqueness, arrival warm-ups, frame accounting, actual prefix-cache reuse, and four independent stages.

## Phase 5: validated design decisions

Each conclusion below is supported only by the aligned or same-configuration control within its row. Absolute latency should not be compared across rows.

| Question | Evidence | Decision |
|---|---|---|
| Are finite requests inherently slower? | After context alignment, current Audio-ready-500 p99 is 796 ms versus 824 ms for the old persistent path | Application-owned sessions plus per-turn finite engine requests are a valid baseline |
| Should audio use arrival prefill? | Eight-user short run: complete-WAV p99 654 ms versus 692 ms for the one-second approximation, which is also not semantically equivalent | Use a complete WAV at query time in the formal workload |
| Does P→D inherently require seconds of transfer? | Old NIXL pull reached about 19.7 s; asynchronous packed push reduced the pipeline to about 1.2 s and measured about 35 GiB/s | The multi-second wait was a connector implementation problem, not an inherent PCIe/P-D cost |
| Why did P input queue? | Full historical media made stage-0 wire p99 253 MiB; ordered mirrored cache reduced it to 6.3 MiB and TTFA p99 from 977 to 714 ms | Keep full token history but omit media tensors already cached by the receiver |
| Why was the P snapshot slow? | Rebuilding and moving full conditioning states every time; delta chunk chains reduced same-trace p99 from 1326 to 988 ms | Transfer lineage deltas and assemble once before the final query |
| Is Delta KV needed? | Theoretical P→D payload over 88 turns fell from about 72.5 GiB to 16.0 GiB, but p99 was 1118 versus 1165/1085 ms | Delta KV removes repeated transfer; bandwidth is not the eight-user tail cause, so re-evaluate capacity benefit at higher load |

The current P/D data path cumulatively retains:

1. an ordered mirrored multimodal cache at stage 0, avoiding repeated historical media tensors;
2. a disposable lineage-delta chunk chain for Talker conditioning snapshots;
3. disposable D prefix KV, with P pushing only the block-aligned missing suffix;
4. P-runner writes of layer-0/layer-24 deltas directly into request-owned shared storage, with handles only on the control plane;
5. a foreground gate that protects media-cache ordering but is not treated as GPU preemption.

## Phase 6: current eight-user P/D baseline and root cause

The latest same-trace comparison preserves all 88 scored turns and the complete media ledger, with no timeout, stall, or replay slip:

| Implementation | TTFA p50/p95/p99 | P p50/p95/p99 | Core-ready→API p50/p95/p99 |
|---|---:|---:|---:|
| Before direct sharing, diagnostics on | 443/681/1002 ms | 141/365/633 ms | 29/98/113 ms |
| Direct-shared P output, diagnostics on | 381/563/770 ms | 87/172/399 ms | 5/15/24 ms |
| Direct-shared P output, diagnostics off | **392/546/815 ms** | — | — |

The diagnostics-off arm has TTFT p50/p99 of 189/512 ms, Audio-ready-500 p99 of 815 ms, and Stall p99 of zero, so it passes the current eight-user SLO. Output encode p99 is 1.8 ms and the ordinary fallback payload is about 0.012 MiB. Direct sharing has removed the main Core→API tail.

The slowest remaining P request is 399 ms. Its foreground runner/Core→API time is only 77/5 ms; roughly 317 ms is spent behind an arrival warm-up that was aborted after entering the GPU. That warm-up processes 13.3k tokens in one step, taking 535 ms in the runner and 441 ms on the CUDA event, and writes an approximately 105 MiB snapshot.

The exact conclusions are:

- The largest remaining component is a **large, non-preemptible warm-up prefill on Thinker P**. Once a low-priority request enters a scheduler step of up to 32k tokens, a later foreground query must wait.
- Talker and Code2Wav are not saturated. P→D Delta-KV transfer and Core→API shared memory are also not the current p99 root cause.
- The foreground gate preserves request/cache order, but an asyncio abort cannot withdraw a GPU kernel that has already started.
- The application lifecycle and workload are now sufficiently sound to treat this as an engine-scheduling research problem instead of changing application semantics again.

Latest result directories:

- Diagnostics: `/home/ubuntu/data/results/pd_direct_shared_long_diag_20260823_v1/u8_t12_direct_shared_seed7_u8`
- Diagnostics off: `/home/ubuntu/data/results/pd_direct_shared_long_clean_20260823_v1/u8_t12_direct_shared_clean_seed7_u8`

The run records `source_commit=2a90a9e2` and `source_dirty=true`. “Diagnostics off” means only that extra logging was disabled; it does not mean a clean git worktree. These numbers are a development baseline. After commit, replay the same trace once to create a formal baseline reproducible from one commit.

## Phase 7: next steps

1. After committing the current implementation, run one diagnostics-off replay of the fixed eight-user × 12-turn trace and pin the formal baseline.
2. On stage 0, A/B a smaller chunked-prefill scheduling quantum, preemptible warm-ups, or deferring cold warm-ups after compaction. A foreground query should wait for at most one small chunk.
3. Verify that aborted warm-ups no longer execute a complete large prefill, then run the 8, 16, 32, ... user capacity ladder with 30 turns/user.
4. At the first SLO failure, decompose tail latency and effective GPU utilization again. Re-evaluate Delta-KV capacity benefit under higher concurrency or cross-node transfer.
5. Enable control-plane diagnostics only for RCA; keep them off in formal capacity runs.

## Recovery map

| Area | Path |
|---|---|
| Session and finite-request lifecycle | `vllm_omni/entrypoints/openai/video_stream_base.py` |
| Canonical multimodal history | `vllm_omni/entrypoints/openai/serving_video_stream.py` |
| P/D orchestrator and snapshots | `vllm_omni/engine/orchestrator.py` |
| P EngineCore IPC | `vllm_omni/engine/stage_engine_core_proc.py` |
| Delta-KV connector | `vllm_omni/engine/nixl_delta_push_connector.py` |
| Shared tensor storage | `vllm_omni/utils/mm_outputs.py` |
| P/D deployment | `benchmarks/thinker_talker/pd_deploy_4gpu.yaml` |
| Multi-user workload | `benchmarks/live_agent/web_client/mu_bench.py` |
| Workload planning and media loading | `benchmarks/live_agent/web_client/continuous_av_workload.py` |
| P/D capacity entry point | `benchmarks/live_agent/web_client/run_pd_av_session_ladder.sh` |
| Run verification | `benchmarks/live_agent/analysis/verify_run.py` |
| Per-stage P/D decomposition | `benchmarks/live_agent/analysis/stage_stats_v2.py` |
| GPU sampling | `benchmarks/live_agent/harness/gpu_sampler.py` |
