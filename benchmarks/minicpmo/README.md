# MiniCPM-o native-duplex P/D benchmark

This benchmark measures sustainable long-session serving capacity for continuous audio-video MiniCPM-o sessions. See the root [workflow](../../workflow.md) for the current result and interpretation.

**Current status (2026-09-10):** the same serving version passes **20 users × 300 s** and fails **24 users × 300 s**. Minimum per-user post-window RTF: 1.000659 / 0.991656; maximum inherited backlog: 217.8 / 7645.8 ms. All 6,000 / 7,200 inputs complete with zero AV fallback, preemption or protocol-audit failures. This is one measurement per load, not an exact capacity ceiling or a dialogue-quality certificate. Older 8/9/10-user results in [sliding-window history](sliding_window.en.md) ([中文](sliding_window.md)) predate these optimizations.

`--functional-only` explicitly excludes capacity certification; `--quality-capture` preserves received text/PCM after measurement without live-path disk writes. `--post-stream-s` controls observation after input ends (default 30), not input cadence or RTF. Internal speech/numerical probes are synchronizing diagnostics and remain off for capacity. Earlier fixes and evidence limits: [generation and serving audit](sliding_quality.en.md) ([中文](sliding_quality.md)).

**Capacity gate:** every user's post-window long RTF must be >1 AND every post-window inherited backlog must be ≤500 ms. Backlog is previous D completion minus current complete-input readiness, clipped at zero, not current-unit execution. Any exceedance fails despite later recovery. `--max-backlog-ms` defaults to 500 and affects offline classification only. These measurements used archived uncommitted source snapshots; subsequently committing the matching code does not retroactively make them clean-build runs.

## Reproduce the current result

Use the `omni` environment (Python 3.12, vLLM 0.26.0) and an unused output directory. The installed vLLM Python files match its distribution RECORD; no unarchived site-packages source patch is required. Four idle 96 GiB GPUs are required for the measured topology.

```bash
OMP_NUM_THREADS=1 /home/ubuntu/miniconda3/envs/omni/bin/python \
  benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 20 --duration-s 300 \
  --kv-window-tokens 18000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --no-triton-force-2d-attention --triton-decode-split-k-threshold 32 \
  --attention-backend TRITON_ATTN --max-backlog-ms 500 \
  --mps --quality-capture --out-dir /path/to/new-run-directory
```

Change only `--users` and the output directory for 24 users. The runner uses independent 12 s warmup sessions, then fresh 300 s histories and 30 s output observation. The 18K window, pin128, BF16 Q and D-only split-K threshold 32 are explicit overrides, not all YAML defaults. Raw results: `/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/batch-policy-clean-{20,24}x300-r1/`.

## Topology

| Stage | GPU |
|---|---:|
| Thinker-P | 0 |
| Thinker-D + Talker | 1 |
| Vision + Audio Encoder sidecars | 2 |
| Code2Wav | 3 |

Use `--topology d-talker` / `deploy_capacity_pd_d_talker_fp8.yaml` for this topology. The base P/D window is 36k; current measurements explicitly override it to 18k. Talker uses a 4k window without cross-request prefix lookup. All stages remain pipelined under private MPS.

For a separate encoder-colocation comparison, use
`deploy_capacity_pd_4gpu_encoders_on_p.yaml`. It preserves the same workload
and request pipeline but places Vision Encoder + Audio Encoder + Thinker-P on
GPU 0, Thinker-D on GPU 1, Talker on GPU 2, and Code2Wav on GPU 3.

The matched FP8 comparison uses `deploy_capacity_pd_4gpu_fp8.yaml` as the
control and `deploy_capacity_pd_4gpu_encoders_on_p_fp8.yaml` as the colocated
variant. FP8 applies to the P, D, and Talker AR stages; the encoder towers and
Code2Wav retain their model-native dtype.

Both placements now use the same background encoder RPC path, one ordered
executor and a private CUDA stream per modality. Cache-ready replies wait for
that stream's completion event, not a device-wide synchronization. P/D KV is
fixed at 64 GiB each; Talker KV is fixed at 8 GiB. Colocation changes neither
these budgets nor the sampling/scheduler settings. The topology still also
moves Talker away from Code2Wav, so a capacity difference alone is **not** an
Encoder/P-only interference measurement.

For an explicitly controlled MPS run (all four stage PIDs must attach):

```bash
python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 --kv-window-tokens 36000 \
  --out-dir /path/to/new-run-directory
```

Use `--topology isolated` for the control. MPS is on by default; `--no-mps`
uses an empty private socket directory to prevent ambient MPS attachment.
The runner archives the fully resolved config and commands, checks that all
four GPUs are free, and stops only its own process group/private MPS instance.
It starts sessions from context zero by default; use the same explicit
`--context-age-max-units` in both arms when preconditioning is desired.

The application maintains session history. Thinker-P appends one native one-second AV unit, transfers only the block-aligned KV delta, and Thinker-D performs finite decode. D output feeds the next P lineage, creating the dependency `D(i-1) -> P(i) -> D(i)`. Talker and Code2Wav do not block the next Thinker unit.

## Production workload

- Real looped MP4 with aligned 16 kHz mono audio.
- 200 ms audio arrivals, 1 FPS video, and one model unit per second.
- Seeded random session phases in `[0, 1 s)` and +/-50 ms arrival jitter.
- Original frame size and official HD slicing with `max_slice_nums=4`.
- Start from context zero; no preconditioning in current capacity runs.
- Current measurement: 18,000-token Thinker window plus 128 pinned prefix tokens; logical history grows to approximately 66k over 300 seconds. Logical-position limit: 262,144.

The current 20/24-user points contain 6,000/7,200 measured units. Every unit must have a client-visible physical-D completion witness and every video frame must be accounted for. Fallback, truncation, missing D-prefix evidence, or an incomplete terminal set invalidates the run.

## Metrics

```text
post-window RTF = completed one-second post-window input budget /
                  wall time from first post-window complete-input readiness
                  through last D completion
```

Capacity requires every user's **unrounded post-window RTF > 1 and inherited backlog ≤500 ms**, all expected D units completed and zero failed users. Every user must reach at least twice the configured window in logical tokens and complete at least 120 post-window units, with bounded physical KV. Exceeding the backlog bound fails even if the session later catches up; current-unit latency, p99 and startup latency are separate diagnostics. The numerator budgets one second for every completed unit; the denominator is one continuous wall-clock span, not the sum of unit latencies. The old first-media-origin `pd_long_horizon.stream_rtf` remains a separate diagnostic. Real-time pacing means RTF near 1 does not demonstrate GPU saturation. This is Thinker-input capacity, not a certificate that audio playback kept up; `end_to_end_capacity_pass` remains unknown without downstream evidence.

The client records absolute send deadlines, actual full-unit send times,
wakeup lag and WebSocket send duration. `--max-send-drift-ms` defaults to 10 ms
(5% of the 200 ms input chunk period). Any larger send drift invalidates the
capacity evidence, separately from a model capacity failure. Deliberate
±50 ms arrival jitter is excluded from this drift. Old artifacts missing
the timing audit cannot silently pass the new validity check. Encoder idle
batching still waits 50 ms in both arms; it is service policy, not noise.

After the last input, the client keeps observing for the full `--post-stream-s`
window instead of closing immediately at D completion. Audio diagnostics use
actual PCM duration and an accumulated 200 ms playback buffer within each
response; pauses between separate responses are not underruns. Unfinished
speech and malformed output are reported. This output window may include
autonomous continuations and is not an internal Talker/Code2Wav queue witness.
GPU telemetry is sampled off the sender event loop.

`analyze_rtf.py` also reports:

- `pd_session_recurrence`: inherited previous-D wait, current pre-D time, current D service, and the fresh serial cycle;
- `physical_d_kv_transfer`: selected KV tokens, blocks, bytes, and validation failures;
- `frame_audit`: arrival-preencode hits and formal fallback frames;
- `benchmark_cleanliness`: source, provenance, diagnostics, completeness, prefix, transfer, and fallback checks.

A formal result requires a clean tree and diagnostics disabled. Diagnostic reruns are analyzable but cannot certify capacity.

## Start a clean server

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/minicpmo/clean_server.py \
  --provenance-out /tmp/minicpm-pd-server-provenance.json -- \
  python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113 \
  2>&1 | tee /tmp/minicpm-pd-server.log
```

Restart the server before each capacity point. `clean_server.py` records import, Git, environment, package, command, and deploy-config provenance and removes hot-path diagnostics. It also exposes the active Python environment's `ninja` and pip-installed CUDA toolkit to stage workers, so a fresh FlashInfer cache does not depend on prior shell activation.

## Smoke test

```bash
python benchmarks/minicpmo/continuous_av.py \
  --url ws://127.0.0.1:8113/v1/realtime \
  --users 1 --duration-s 3 --workload-profile production --seed 20260913 \
  --connect-stagger-s 0 --admission-timeout-s 90 \
  --post-stream-s 30 --close-timeout-s 30 --gpus 0 1 2 3 \
  --media /path/to/omni_duplex1.mp4 --loop-media \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --out /tmp/minicpm-pd-smoke.json
```

## Development screen

Use the same command with `--users 24 --duration-s 30 --seed 20260914 --post-stream-s 60 --close-timeout-s 60`. A 30-second run is useful for regression screening but cannot produce a formal capacity result.

## Current long-session capacity command

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 --kv-window-tokens 36000 \
  --out-dir /home/ubuntu/data/experiments/minicpm-pd-sliding-repeat
```

Use a new directory and change `--users` to 8 for the failure point. The wrapper starts/stops its own server and MPS, warms a separate session, captures provenance/source manifests and runs the completion/window/sender audits. The manual server/smoke commands above are diagnostics, not this capacity setup. For an instrumented rerun, enable only required diagnostics with `--allow-diagnostics`; do not compare profiled latency directly without a matched perturbation check.
