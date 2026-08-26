# Qwen3-Omni continuous AV session

This directory contains the browser client and the only supported capacity
workload for `thinker-talker-vllm`.

## Request and state model

The WebSocket is stateful, but engine requests are not:

- the application retains accepted user audio, selected video, text, and the
  assistant response for the conversation;
- processed canonical message blocks are retained by the application; each
  turn renders only its new user block and completed assistant block, then
  assembles the full token/media prompt for the finite engine request;
- each accepted frame triggers or coalesces a silent, finite Thinker-only
  prefill request over full history plus the cumulative current-turn frames;
- each session runs at most one arrival request at a time; frames accepted while
  it runs remain separate prompt images but are coalesced into the next
  cumulative snapshot;
- different sessions submit arrival requests concurrently to the native FCFS
  scheduler; there is no application-wide gate or request priority;
- a final query waits only for its session's currently admitted arrival request,
  preserving a linear prefix lineage before it submits the full prompt;
- the final query appends one complete WAV and creates a separate finite
  response request;
- the Thinker may reuse identical blocks through vLLM prefix caching;
- cache eviction or a miss changes latency only, never prompt semantics;
- only when the rendered prompt reaches 49,152 tokens, the application retains
  the newest two completed turns as user audio/text plus assistant text,
  discards their historical images, and keeps only the newest accepted image
  in the current turn. It does not generate a summary request.

Frames accepted since the preceding query are consumed by exactly one turn.
Frames arriving during generation accumulate for the next turn. Similarity and
freshness filtering are the only selection policy: accepted frames remain
append-only and are never replaced by a latest-eight sliding window.

The old persistent/resumable engine request, append into that live request,
shadow compression, and Talker rolling paths have been removed. Arrival
warm-ups are independent `output_modalities=["text"]`, prefill-only requests;
the scheduler does not commit their sampled next token and they cannot invoke
Talker. Normal Thinker replies are capped at 256 tokens so one malformed long
answer cannot turn a live voice capacity cell into a minutes-long generation
test.

## Canonical deployment

Formal measurements use only
`benchmarks/thinker_talker/origin_deploy_3gpu.yaml`: GPU 0 is Thinker, GPU 1 is
Talker, and GPU 2 is Code2Wav. Thinker prefix caching is enabled and requests
use native FCFS scheduling. The launcher requires FlashInfer for Thinker.

```bash
RESULTS_DIR=/path/to/results \
VLLM_OMNI_BIN=/path/to/bin/vllm-omni \
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

Start the browser server with:

```bash
MU_PYTHON=/path/to/python \
bash benchmarks/live_agent/web_client/run_page_server.sh
```

Forward port 7870 and open `http://localhost:7870/`.

## Capacity workload

Each user owns one long-lived WebSocket:

- one JPEG frame every 500 ms throughout the session;
- one PCM chunk every 200 ms except during assistant playback and a 300 ms
  echo guard;
- one real mono PCM16 16 kHz recording per turn, followed by 700 ms endpoint
  silence;
- an empty `video.query`, so query semantics come from speech;
- playback-paced closed loop: the next think interval begins after simulated
  1x playback; startup is measured when exactly 500 ms is available;
- deterministic user startup stagger over 0–8 seconds;
- one speaker per session and no recording reuse within that session.

The server scales frames to at most 640x352 and uses similarity threshold 0.95
with freshness gap `[0,4]`. Every retained frame enters the cumulative turn
prefix. Audio remains one complete WAV at query time so Qwen audio semantics do
not depend on synthetic chunks.

The canonical audio path emits an eight-frame initial codec chunk and
four-frame steady chunks. This makes the first useful waveform large enough
to clear the 500 ms prebuffer while later chunks replenish playback faster
than real time.

Prepare the deterministic SLURP/DAVIS workload:

```bash
python benchmarks/live_agent/web_client/prepare_slurp_davis.py \
  --slurp-annotations /path/to/slurp-repo \
  --slurp-audio /path/to/slurp_real \
  --davis-jpegs /path/to/DAVIS/JPEGImages/480p \
  --out /path/to/continuous-av-v1
```

Run the capacity ladder:

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

A cell passes only if every measured turn completes, audio-ready-500 p99 is
below 1 s, playback stall p99 is below 50 ms, and client/engine checks are
clean. Raw TTFA is packetization-dependent and remains diagnostic only. Every
cell records source and deploy hashes, the workload plan,
per-turn output, the engine log, and GPU samples.

Verify response and warm-up request identities, frame-ledger equality, and
Thinker prefix-cache activity:

```bash
python benchmarks/live_agent/analysis/verify_run.py RESULT_DIR
```

## Diagnostics

```bash
python benchmarks/live_agent/web_client/selftest.py
node benchmarks/live_agent/web_client/playback_test.js
python benchmarks/live_agent/web_client/audio_timeline.py --direct
python benchmarks/live_agent/web_client/probe.py --direct
```

`probe.py` uses synthetic media only for protocol validation. Capacity claims
must use `mu_bench.py` with the real audio manifest and frame sequence.
