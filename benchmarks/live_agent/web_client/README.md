# Qwen3-Omni continuous AV session

This directory contains the browser client and the only supported capacity
workload for `thinker-talker-vllm`.

The model is not natively full duplex. Video is continuous, but microphone
upload pauses while assistant audio is playing and for a 300 ms echo guard.
Turn boundaries come from the client after 700 ms of endpoint silence.

## Canonical deployment

Formal measurements use only:

`benchmarks/thinker_talker/origin_deploy_3gpu.yaml`

It assigns thinker, talker, and code2wav to GPUs 0, 1, and 2. The engine and
capacity launchers reject another deploy basename, and the benchmark records
its SHA256.

Start the engine:

```bash
RESULTS_DIR=/path/to/results \
VLLM_OMNI_BIN=/path/to/bin/vllm-omni \
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

Optional variables are `MU_GPU_IDS` (default `0,1,2`), `MU_PORT`
(default `8091`), `QWEN_MODEL`, and `QWEN_EXTRA_ARGS`.
The launcher refuses inherited stage colocation and engine-ablation flags so
the canonical baseline always uses three separate processes and default engine
features.

Start the page server:

```bash
MU_PYTHON=/path/to/python \
bash benchmarks/live_agent/web_client/run_page_server.sh
```

Forward port 7870 and open `http://localhost:7870/`.

## Capacity workload

Each user owns one long-lived WebSocket:

- video: one JPEG frame every 500 ms for the full session;
- microphone: one PCM chunk every 200 ms except during playback and echo guard;
- turn input: a real mono PCM16 16 kHz recording plus 700 ms endpoint silence;
- query: empty `video.query`; semantics come only from recorded audio;
- pacing: next think time begins after simulated 1x playback completes;
- plan: one speaker per session, no recording repeated within a session,
  deterministic stagger, think time, frame offset, and random seed.

The audio manifest is JSONL:

```json
{"id":"speaker01-turn01","speaker":"speaker01","transcript":"What do you see?","audio":"speaker01/turn01.wav"}
```

Every speaker needs at least `TURNS` recordings. Audio paths are relative to
the manifest.

The canonical corpus uses real close-talk SLURP requests and DAVIS 2017 video.
SLURP real audio is CC BY-NC 4.0. After downloading the official archives,
prepare the deterministic 80-speaker workload with:

```bash
python benchmarks/live_agent/web_client/prepare_slurp_davis.py \
  --slurp-annotations /path/to/slurp-repo \
  --slurp-audio /path/to/slurp_real \
  --davis-jpegs /path/to/DAVIS/JPEGImages/480p \
  --out /path/to/continuous-av-v1
```

It selects 60 distinct, correctly annotated recordings per speaker, balances
assistant scenarios, and uses a fixed microphone profile per session: 50% of
speakers are close-talk and 50% are distant-microphone. It converts audio to
mono PCM16 16 kHz and samples DAVIS at an effective 2 fps.
`corpus_provenance.json` records the selection and source revisions.

Run all three session policies on the same workload plan:

```bash
MU_FRAMES_DIR=/path/to/ordered/jpeg/frames \
MU_AUDIO_MANIFEST=/path/to/utterances.jsonl \
RESULTS_DIR=/path/to/results \
bash benchmarks/live_agent/web_client/run_session_baselines.sh
```

`run_av_session_ladder.sh` runs one policy. A cell passes only if every
post-warmup turn completes, audible playback-start p99 is below 1 s, playback
stall p99 is below 50 ms, and protocol/client/engine correctness checks are
clean. Service TTFA is reported separately for stage attribution. Percentiles
use nearest rank and playback uses the browser's default 1.4 s smooth-buffer
threshold; because chunks arrive discretely, this normally starts on the
second audio delta rather than adding a fixed 1.4 s delay.

Each cell stores `workload_plan.json`, `turns.jsonl`, `summary.json`,
`gpu_samples.jsonl`, and `engine.log`. GPU samples include SM activity,
achieved occupancy, tensor/FP activity, DRAM activity, PCIe traffic, power,
clocks, resident memory, and per-process utilization. Source, deploy, audio
corpus, frames, system prompt, and workload plan are hashed.

Self-repaired engine bookkeeping drift remains in `engine_probes` and
`engine_warning_count` for diagnosis, but does not stop the capacity ladder
unless it causes a missing turn, playback/SLO failure, wedge, or protocol
error.

## Diagnostics

These are diagnostics, not capacity workloads:

```bash
python benchmarks/live_agent/web_client/selftest.py
node benchmarks/live_agent/web_client/playback_test.js
python benchmarks/live_agent/web_client/audio_timeline.py --direct
python benchmarks/live_agent/web_client/probe.py --direct
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

`probe.py` uses synthetic media only to validate the protocol. Capacity claims
must come from `mu_bench.py` with the real manifest and frame sequence.

For chunk-level root-cause runs, start the engine with
`VLLM_OMNI_LOG_SCHED_STEPS=1`, `VLLM_OMNI_LOG_REQ_STEPS=1`, and
`VLLM_OMNI_LOG_AUDIO_CHUNKS=1`, then compare cells with:

```bash
python benchmarks/live_agent/analysis/audio_chunk_rca.py \
  --cell baseline=/path/to/baseline_cell \
  --cell ablation=/path/to/ablation_cell \
  --json-out /path/to/root_cause.json
```

The report separates the fixed Talker AR steps, upstream-chunk waiting, and
Code2Wav latency between the first and second audible chunks.

For the paired arrival-vs-query-time prefill RCA, use
`run_prefill_timing_rca.sh`. It records the arrival arm's real closed-loop
client input, replays the same timestamped frame/audio/query events in both
query-time arms, and fails unless `media_fairness.py` verifies identical
ordered media ledgers, zero frame drops, and zero replay schedule slips.
