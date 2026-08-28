# DuplexOmni baseline

This directory validates a three-stage, non-P/D deployment:

- GPU 0: Thinker
- GPU 1: Talker and MTP
- GPU 2: Code2Wav

The client sends only live PCM and video frames over
`/v1/video/chat/stream`. The server application assembles 480 ms slots and
owns dialogue history, context compaction, media filtering, and per-session
backpressure. It submits one ordinary finite engine request per slot; engines
keep only disposable prefix/KV cache. At the 6,144 Thinker-token trigger, the
server drains the current epoch and starts a new cache lineage containing the
system prompt and only the latest complete slot.

The default workload sends one 480 ms PCM slice and one video frame every slot
(about 2.08 FPS). Sessions receive independent random phases. Client frame
arrivals and frames accepted by the server's similarity/freshness filter must
be reported separately. The benchmark no longer builds prompts, stores codec
history, or controls engine lineage.

Thinker text history retains every slot within an epoch. Talker history retains
only slots that completed six codec frames and emitted the validation EOS,
matching the official service. Talker prefix caching uses deterministic cache
identities derived from the exact Thinker prefix and visible media hashes; this
allows historical conditioning KV to be reused without treating constant
placeholder tokens as equivalent dynamic embeddings.

The adapter borrows the official DuplexOmni input dictionary, control fields,
Thinker-to-Talker alignment (token embedding plus final normalized Thinker
hidden), and six-frame codec contract. It also follows the official stage
dependency: the orchestrator returns each Thinker result before its Talker
worker finishes, and the Talker worker owns ordered session history. Here,
Thinker slot `t+1` is independent of Talker/Code2Wav slot `t`; only Talker
slots are serialized. It does not borrow the official global locks or
single-active-session service design.

Source of truth: [DuplexOmni](https://github.com/MuyeHuang/DuplexOmni).

Thinker and Talker use online W8A8 FP8 with per-token/per-head FP8 KV cache by
default. This KV format computes local scales at cache-write time and does not
depend on KV scales in the checkpoint. Code2Wav remains BF16 because it is not
served through the vLLM FP8 LLM quantization path. The BF16 configuration is
retained only as a regression reference.

## Run the default deployment

```bash
benchmarks/duplexomni/run_server.sh
python benchmarks/duplexomni/single_user.py \
  --label fp8 --output /tmp/duplexomni-fp8
benchmarks/duplexomni/stop_server.sh
```

`single_user.py` uses the same server-owned WebSocket session as the capacity
runner; it is not a separate client-owned session implementation.

## Multi-user latency workload

Each session sends one PCM slice every 480 ms on the wall clock. Its phase is
sampled reproducibly from `Uniform[0, 480 ms)`; there is no synchronized
arrival mode. The server starts Thinker slot `t+1` after Thinker slot `t`
produces its control text; it does not wait for Talker/Code2Wav slot `t`. The
orchestrator serializes Talker history. The server applies the four-request
per-session in-flight bound. Media is deterministically distinct across users:
PCM receives inaudible LSB dither and each JPEG receives a tiny corner patch.
This prevents an unrealistic cross-user encoder-cache hit while preserving
semantics and tensor sizes. `--shared-media-across-users` is retained only as
a cache-control A/B.

```bash
python benchmarks/duplexomni/multi_user.py \
  --users 8 --slots 30 --seed 20260827 \
  --output /tmp/duplexomni-load-8x30
```

Use `--no-slot-overlap` only for the fully serial A/B baseline. The manifest
records the pipeline policy, Thinker latency, full-response latency, and how
many slots were submitted before their predecessor audio completed.

The manifest reports wall-clock end-to-end slot latency, finite-request
latency, server application queueing, Thinker latency, and the fraction of
slots missing the 480 ms deadline.

The measured capacity baseline and root-cause breakdown are recorded in
[CAPACITY.md](CAPACITY.md). Use `analyze_capacity.py` to reproduce the stage
and GPU summaries from manifests and telemetry.

For iteration-level prefill/decode attribution, start the server with
`VLLM_OMNI_LOG_PD_ITER=1 VLLM_OMNI_PD_ITER_STAGE=0`, run the workload, then
use `analyze_pd_interference.py --manifest RUN/manifest.json --server-log LOG`.
The trace is disabled by default.

## BF16/FP8 comparison

```bash
benchmarks/duplexomni/run_server.sh bf16
python benchmarks/duplexomni/single_user.py \
  --label bf16 --output /tmp/duplexomni-bf16 \
  --session-id duplexomni-correctness
benchmarks/duplexomni/stop_server.sh

benchmarks/duplexomni/run_server.sh fp8
python benchmarks/duplexomni/single_user.py \
  --label fp8 --output /tmp/duplexomni-fp8 \
  --session-id duplexomni-correctness
benchmarks/duplexomni/stop_server.sh

python benchmarks/duplexomni/compare_runs.py \
  --bf16 /tmp/duplexomni-bf16/manifest.json \
  --fp8 /tmp/duplexomni-fp8/manifest.json
```

The comparison checks structured-control semantics, speaking decisions,
exact `16 x 6` codec shape, and waveform sanity. It does not require FP8 and
BF16 waveforms to be bit-identical.

## Current result

A warmed 300-slot, one-user AV run completed 300/300 valid codec/EOS turns
with no 480 ms deadline miss. E2E p50/p95/p99/max was
`376/442/462/463 ms`; finite-request p99/max was `456/459 ms`, and
application-queue p99 stayed below `0.5 ms`. Eight compactions reduced the
Thinker prompt from about 6.2k tokens to 349 tokens without a latency spike;
canonical prompt rendering had a 16.3 ms p99.

The 6,144-token trigger is performance-derived rather than the model context
limit. In a warmed no-compaction run, the first 480 ms miss appeared near
7.2k prompt tokens and request p95 reached 864 ms by 21k tokens. The trigger
retains about 15% token headroom. This is a one-user long-session correctness
and latency result, not a multi-user capacity result.

Measured on three RTX PRO 6000 Blackwell GPUs with four audio/video slots:

| Mode | Per-slot latency (ms) | Valid slots | Result |
| --- | --- | --- | --- |
| BF16 | 1921 / 367 / 381 / 378 | 4/4 | regression reference |
| online W8A8 FP8 + BF16 KV | 3187 / 350 / 376 / 379 | 4/4 | historical comparison |

The default per-token/per-head FP8 KV path reproduced all eight structured
Thinker outputs from the FP8-weight/BF16-KV audio run. After Triton JIT warm-up,
its eight slot latencies were `357 / 331 / 333 / 308 / 345 / 388 / 385 / 390`
ms. The first cold run paid a one-time 6.95 s Triton compilation cost. On the
Talker stage, usable KV capacity increased from about 3.11M to 6.30M tokens.

An eight-slot audio-only A/B produced identical speaking decisions but different
generated descriptions (3/8 exact TTS fragments; 63.4% character similarity).
That input contained no image or video ground truth, so the result demonstrates
BF16/FP8 non-equivalence but does not establish that either description is more
accurate. FP8 is the selected deployment trade-off.
