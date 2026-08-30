# MiniCPM-o Native-Duplex P/D Workflow

## Goal

Measure how many continuous audio-video sessions one node can sustain in real time, then identify the serving bottleneck. Model quality is out of scope.

## Phase 1: final serving design

```text
one WebSocket per user
  -> upload 200 ms PCM16 audio chunks and one video frame per second
  -> aggregate input into one native one-second model unit
  -> Thinker-P appends only the new AV unit to the session KV lineage
  -> transfer the new KV delta to Thinker-D
  -> Thinker-D performs finite autoregressive decode and decides listen/speak
  -> feed D output into the next P lineage
  -> on speak, run Talker -> Code2Wav
```

- The application owns session, buffering, reconnect, and output state. Engine KV is disposable execution state.
- Sessions enter the engine independently; there is no application-wide gate or cross-user batch.
- Media preparation may run ahead, but `D(i-1)` must finish before `P(i)` because its generated Thinker state is part of the next lineage.
- The next Thinker unit does not wait for Talker or Code2Wav.
- At 36,000 estimated tokens, the application opens a new lineage containing the system/reference input, the previous complete AV unit and confirmed Thinker output, and the current unit. The model limit is 40,960 tokens.

Deployment:

| Stage | GPU | Role |
|---|---:|---|
| Thinker-P | 0 | Incremental multimodal prefill |
| Thinker-D | 1 | Finite autoregressive decode |
| Talker | 2 | Speech-code generation |
| Code2Wav + Vision Encoder | 3 | Waveform generation and stateless video encoding |

P and D use `NixlDeltaPushConnector`. D retains prefix KV, and each unit transfers only the new block-aligned KV suffix.

## Phase 2: formal workload and capacity criterion

- Real looped MP4, 960×540 video, aligned 16 kHz mono audio, and reference audio.
- Audio arrives every 200 ms; video arrives at 1 FPS; the model consumes one one-second unit at 1 Hz.
- `max_slice_nums=4`, so each video unit uses the HD4 path.
- 15 users, 360 seconds, random phases in `[0, 1 s)`, seed `20260839`.
- Total input: 5,400 units. Every session crosses the context threshold twice.

Per-unit RTF diagnoses jitter:

```text
unit RTF = 1000 ms / stage service time
```

Capacity uses sustained progress:

```text
stream RTF = completed one-second input budget /
             wall time from first input-ready to last D completion
```

Capacity passes only when every session has `stream RTF >= 1`, every input unit completes, and no user fails. A single unit above one second is a tail miss, not a capacity failure if later units recover the backlog.

## Phase 3: final measurement

The P/D-only control sets `VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC=1`. It preserves full P computation, KV-delta handoff, full finite D decode, and D-to-next-P feedback, but stops before Talker/Code2Wav. The downstream stages receive zero requests. D output length remains equivalent to the full pipeline: mean/p95/p99 is 3.038/8/8 versus 2.982/8/8 tokens. Vision formal wait p99 is 1 ms, so GPU 3 does not gate this control.

| Metric | Full pipeline | P/D-only control |
|---|---:|---:|
| P service p50/p95/p99 | 310/849/1060 ms | 126/405/686 ms |
| D service p50/p95/p99 | 299/790/1096 ms | 166/833/1110 ms |
| Input-ready to D-done p50/p95/p99 | 1080/2631/3031 ms | 306/1420/2441 ms |
| Previous-D wait p50/p95/p99 | 223/1352/1593 ms | 0.2/643/1223 ms |
| Units above 1 second | 2818/5400 | 605/5400 |
| Per-session stream RTF min/p50/p95 | 1.002/1.002/1.002 | 1.002/1.002/1.003 |

Both runs complete all 5,400 D units without failure and sustain 15 users. The approximately 0.2% RTF headroom means this point is close to the measured boundary. Downstream work materially amplifies latency, but removing it does not eliminate the P/D tail.

## Phase 4: final root cause

The same session has an unavoidable model dependency:

```text
D(i-1) feedback -> P(i) -> KV handoff -> D(i) -> feedback -> P(i+1)
```

P for one session can overlap D for another session, but `P(i)`, `D(i)`, and `P(i+1)` of the same session cannot overlap.

For the slowest 1% of P/D-only units, mean input-ready-to-D latency is 2,895 ms:

| Serial component | Mean | Share |
|---|---:|---:|
| Wait for preceding D feedback | 1,319 ms | 45% |
| Current P service | 541 ms | 19% |
| Current P-done to D-done | 1,048 ms | 36% |

D does not stop an active decode behind a separate prefill-only request. Newly KV-ready requests join active decodes as mixed activation batches. Decode-only runner steps are 20/49/75 ms at p50/p95/p99; mixed steps are 75/165/249 ms.

Final conclusion: concurrency increases P batch cost and mixed D activation/decode cost. Same-session recurrence serializes preceding-D wait, current P, handoff, and current D. Occasional long units cause recoverable jitter; capacity fails only when this backlog no longer drains and a session's long-horizon `stream RTF` falls below one. This is a property of the model dependency plus engine service time, not an application serialization bug.

## Phase 5: reproduction

Start the full pipeline:

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

For the P/D-only control, add `VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC=1` to the server environment.

Run and analyze:

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 15 --duration-s 360 --phase-window-s 1 --seed 20260839 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --close-timeout-s 120 \
  --gpus 0 1 2 3 --out /tmp/minicpm-pd-u15x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-u15-server.log \
  --run-json /tmp/minicpm-pd-u15x360.json \
  --out /tmp/minicpm-pd-u15x360-rtf.json
```

Key files:

- `benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml`
- `benchmarks/minicpmo/continuous_av.py`
- `benchmarks/minicpmo/analyze_rtf.py`
- `vllm_omni/engine/orchestrator.py`
