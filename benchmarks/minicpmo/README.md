# MiniCPM-o native-duplex P/D benchmark

This benchmark measures sustainable long-session serving capacity for continuous audio-video MiniCPM-o sessions. See the root [workflow](../../workflow.md) for the current result and interpretation.

## Topology

| Stage | GPU |
|---|---:|
| Thinker-P | 0 |
| Thinker-D | 1 |
| Vision Encoder sidecar | 2 |
| Talker + Code2Wav | 3 |

The application maintains session history. Thinker-P appends one native one-second AV unit, transfers only the block-aligned KV delta, and Thinker-D performs finite decode. D output feeds the next P lineage, creating the dependency `D(i-1) -> P(i) -> D(i)`. Talker and Code2Wav do not block the next Thinker unit.

## Production workload

- Real looped MP4 with aligned 16 kHz mono audio.
- 200 ms audio arrivals, 1 FPS video, and one model unit per second.
- Seeded random session phases in `[0, 1 s)` and +/-50 ms arrival jitter.
- Original frame size and official HD slicing with `max_slice_nums=4`.
- Context rollover trigger at 36,000 estimated tokens; model limit 40,960.
- The 180-second profile randomizes preconditioning age from 0 to 154 units to cover long context and rollover.

The formal 24x180 candidate contains 4,320 measured units. Every unit must have a client-visible physical-D completion witness and every video frame must be accounted for. Fallback, truncation, missing D-prefix evidence, or an incomplete terminal set invalidates the run.

## Metrics

```text
stream RTF = completed one-second input budget /
             wall time from first media arrival through last D completion
```

Capacity requires every session to have `stream RTF >= 1`, all expected D units to complete, and zero failed users. Per-unit `ready_to_d_ms`, P/D service time, and stage RTF are diagnostics.

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

## Long-session candidate

```bash
python benchmarks/minicpmo/continuous_av.py \
  --url ws://127.0.0.1:8113/v1/realtime \
  --users 24 --duration-s 180 --workload-profile production --seed 20260915 \
  --connect-stagger-s 0 --admission-timeout-s 90 \
  --post-stream-s 120 --close-timeout-s 60 --gpus 0 1 2 3 \
  --media /path/to/omni_duplex1.mp4 --loop-media \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --out /tmp/minicpm-pd-24x180.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-server.log \
  --server-provenance-json /tmp/minicpm-pd-server-provenance.json \
  --run-json /tmp/minicpm-pd-24x180.json \
  --out /tmp/minicpm-pd-24x180-analysis.json
```

For a separate instrumented rerun, set only the required diagnostic environment variables and add `--allow-diagnostics` to `clean_server.py`. Never compare instrumented latency directly with a clean run without a matched perturbation check.
