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

Run all three session policies on the same workload plan:

```bash
MU_FRAMES_DIR=/path/to/ordered/jpeg/frames \
MU_AUDIO_MANIFEST=/path/to/utterances.jsonl \
RESULTS_DIR=/path/to/results \
bash benchmarks/live_agent/web_client/run_session_baselines.sh
```

`run_av_session_ladder.sh` runs one policy. A cell passes only if every
post-warmup turn completes, TTFA p99 is below 1 s, per-turn maximum playback
stall p99 is below 50 ms, and protocol/client/engine correctness checks are
clean. Percentiles use nearest rank and playback uses a 60 ms prebuffer.

Each cell stores `workload_plan.json`, `turns.jsonl`, `summary.json`,
`gpu.csv`, and `engine.log`. Source, deploy, audio corpus, frames, system
prompt, and workload plan are hashed.

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
