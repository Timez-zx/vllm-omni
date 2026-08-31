# MiniCPM-o native-duplex capacity benchmark

This benchmark measures sustained multi-user input progress for MiniCPM-o 4.5 native duplex serving. The current non-P/D topology uses four RTX PRO 6000 Blackwell 96 GB GPUs:

| GPU | Role |
|---:|---|
| 0 | Thinker |
| 1 | Talker |
| 2 | Code2Wav |
| 3 | Vision encoder/resampler |

Use `deploy_capacity_4gpu.yaml`. The vision sidecar overlaps arrival encoding with Thinker execution without changing TP=1.

## Workload

- Real 960x540 MP4 with aligned 16 kHz mono audio.
- Audio arrives every 200 ms; one frame arrives every second.
- Five audio chunks and one frame form one native one-second model unit.
- `frame_max_side=0`, `max_slice_nums=4`: one global image plus two local crops, 198 vision rows and 211 total steady-state scheduler rows.
- Seeded user phases in `[0, 1 s)`.
- A formal capacity run lasts 360 seconds and loops the media, crossing the 36k context rollover threshold twice per session.

The primary metric is:

```text
stream RTF = input duration /
             wall time from first client AV admission
             through final Thinker runner completion
```

A cell passes only when all input units complete, no user fails, and every session has `stream RTF >= 1`. Per-request latency is diagnostic only: a cumulative snapshot may supersede an older queued snapshot, and a slow unit may recover later without creating long-term backlog.

## Run

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 MINICPMO45_LOG_PREP_DIAG=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113

python benchmarks/minicpmo/continuous_av.py \
  --users 4 --duration-s 360 --phase-window-s 1 --seed 20260839 \
  --connect-stagger-s 0.5 --post-stream-s 30 --close-timeout-s 480 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --gpus 0 1 2 3 \
  --out /tmp/minicpm-nonpd-u4x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-nonpd-server.log \
  --run-json /tmp/minicpm-nonpd-u4x360.json \
  --out /tmp/minicpm-nonpd-u4x360-analysis.json
```

## Final result

| Users | Completed units | Stream RTF min/p50/max | Terminal backlog p99 | Result |
|---:|---:|---:|---:|:---:|
| 4 | 1440/1440 | 1.001/1.001/1.002 | -296 ms | pass |
| 5 | 1800/1800 | 0.999/0.999/1.000 | 522 ms | fail |
| 8 | 2880/2880 | 0.947/0.948/0.949 | 19981 ms | fail |

The strict measured non-P/D capacity is four sessions. The encoder is not the boundary: its p50/p95/p99 is 27/44/55 ms and GPU 3 utilization is 6.7% mean, 32% p95. On GPU 0, mixed AV-prefill/decode steps take 86 ms p50 versus 19 ms for decode-only steps at the passing point. The sustained backlog therefore comes from multimodal prefill and multi-step decode sharing one Thinker GPU.

The same 360-second HD4 workload passes 15 users on the four-GPU P/D deployment, which assigns GPUs to Thinker-P, Thinker-D, Talker, and Code2Wav+Encoder. This is a deployment comparison; P/D has two Thinker GPUs, while non-P/D has one.

Raw runs and analyzed summaries are under `results/`:

- `nonpd_4gpu_hd4_u4_360s_20260831.*.json`
- `nonpd_4gpu_hd4_u5_360s_20260831.*.json`
- `nonpd_4gpu_hd4_u8_360s_20260831.*.json`
- `pd_4gpu_hd4_u15_360s_20260830.*.json`

See `workflow.md` and `workflow.zh.md` at the repository root for the implementation and interpretation.
