# MiniCPM-o Non-P/D Native-Duplex Workflow

## 1. Goal

Measure how many continuous audio-video sessions one four-GPU non-P/D deployment can sustain in real time, then compare it with the MiniCPM P/D deployment under the same workload. Model quality is out of scope.

MiniCPM-o 4.5 is used through its native duplex path: the model consumes one-second AV units continuously and decides whether to listen or speak. No synthetic query or forced response is inserted.

## 2. Application design

```text
one WebSocket per user
  -> upload 200 ms PCM16 audio chunks and one video frame per second
  -> combine five audio chunks and one frame into a native one-second unit
  -> append the unit to the session's resident Thinker request/KV lineage
  -> Thinker decides listen or speak
  -> on speak: Talker -> Code2Wav
```

- Sessions enter the engine independently; there is no application-wide admission gate or cross-user batch.
- New input does not wait for Talker, Code2Wav, or audio playback.
- The resident request appends only new input. Generic prefix caching is disabled because the active lineage already owns its incremental KV.
- A newer cumulative session snapshot may supersede an older queued snapshot. The newer snapshot includes the older input; the analyzer credits it only after that cumulative generation actually runs.
- At 36,000 estimated tokens, the next unit starts a fresh lineage containing the fixed system/reference context, the latest complete AV unit and confirmed Thinker output, and the current unit. The model limit is 40,960 tokens.

## 3. Four-GPU non-P/D deployment

| GPU | Role |
|---:|---|
| 0 | Thinker LLM |
| 1 | Talker |
| 2 | Code2Wav |
| 3 | Vision encoder and resampler |

Configuration: `benchmarks/minicpmo/deploy_capacity_4gpu.yaml`.

- Thinker uses `max_model_len=40960`, `max_num_batched_tokens=32768`, `max_num_seqs=64`, and synchronous scheduling.
- The auxiliary GPU is exposed only to the vision tower; it does not change Thinker TP or world size.
- Arrival-side vision work is prepared on CPU, grouped by exact tensor shape across sessions, and encoded in microbatches of at most eight.
- A formal Thinker append consumes the speculative embedding when ready. On a miss it retires that cache key and performs a correctness fallback on GPU 3; late speculative writes cannot overwrite the session state.
- Encoder RPCs bypass the busy Thinker Core loop and are acknowledged when admitted to the sidecar queue. The API does not wait for the shared result queue, and the sidecar remains single-threaded to preserve model safety.
- Session/reference context, CPU audio preparation, stage-output consumption, SDPA vision attention, and the resampler use the same applicable engineering optimizations as the P/D branch. NIXL, KV handoff, and P/D feedback code are intentionally not present.

## 4. Capacity workload and metric

- Source: the real 960x540 `omni_duplex1.mp4` with aligned 16 kHz mono audio and `HT_ref_audio.wav`.
- Arrival rate: audio every 200 ms and video at 1 FPS; the model consumes one one-second unit per user per second.
- `frame_max_side=0`, `max_slice_nums=4`. The reference frame produces one global image and two local crops: 198 vision scheduler rows and 211 steady-state rows per unit.
- Users start at deterministic random phases in `[0, 1 s)`, seed `20260839`.
- Each formal run lasts 360 seconds and loops the media. Every session crosses the 36k context threshold twice.

Capacity uses sustained input progress, not isolated unit latency:

```text
stream RTF = 360 seconds of input budget /
             wall time from the first client AV admission
             through the final unit's Thinker runner completion
```

A concurrency level passes only if every session has `stream RTF >= 1`, all expected input units complete, and no user fails. Per-unit Thinker/Talker latency remains a jitter diagnostic. Talker may be silent for a model-owned listen decision, and Code2Wav spans session idle time, so neither request count nor Code2Wav wall time is the capacity clock.

## 5. Final non-P/D result

Every row below is a 360-second long-session cell. Each session receives 360 one-second AV units and crosses the 36k context rollover threshold twice.

| Users | Completed units | Stream RTF min/p50/max | Terminal backlog p99 | Result |
|---:|---:|---:|---:|:---:|
| 4 | 1440/1440 | 1.001/1.001/1.002 | -296 ms | pass |
| 5 | 1800/1800 | 0.999/0.999/1.000 | 522 ms | fail |
| 7 | 2520/2520 | 0.985/0.986/0.987 | 5666 ms | fail |
| 8 | 2880/2880 | 0.947/0.948/0.949 | 19981 ms | fail |

The measured strict long-session capacity is four users. Five is the first failure under the agreed `RTF >= 1` rule; it misses by only about 0.1%, but must not be rounded up.

At four users, seven older input snapshots were coalesced into later cumulative snapshots. All 1440 inputs completed, so this is legal request replacement rather than data loss. The trace contains eight 490-row rollover admissions, two per session.

The old 30-second capacity table is not comparable and is superseded by this result: a short cell can finish before persistent queue drift or repeated context cycles become visible.

## 6. Bottleneck

The dedicated encoder is not the boundary:

- vision encoder p50/p95/p99: `27/44/55 ms`;
- GPU 3 utilization mean/p95/p99: `6.7%/32%/41%`;
- 1417 of 1440 frames reached the speculative embedding cache; the other 23 used the formal correctness fallback, so no frame was dropped;
- the formal path always consumes either the speculative result or a correctness fallback.

At the four-user point, NVML device-busy mean/p95/p99 is `25.3%/72%/79%` on Thinker, `2.5%/18%/27%` on Talker, and `10.1%/58%/64%` on Code2Wav. These are GPU busy-time samples, not SM occupancy; the bottleneck conclusion comes from the runner traces below, not from treating NVML utilization as raw compute saturation.

A mixed step is one Thinker forward containing both existing decode tokens and newly arrived AV prefill rows. The same-run GPU 0 trace gives the direct comparison:

| Users | Mixed decode steps | Decode-only p50 | Mixed p50 | Slowdown | Stream RTF | Terminal backlog p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7.4% | 18.8 ms | 86.2 ms | 4.6x | 1.001 | -296 ms |
| 5 | 13.0% | 19.4 ms | 81.3 ms | 4.2x | 0.999 | 522 ms |
| 8 | 27.8% | 21.3 ms | 102.2 ms | 4.8x | 0.947 | 19981 ms |

The trace directly proves that AV prefill stretches decode iterations by roughly 4-5x on the shared Thinker GPU. As concurrency raises the mixed-step share, sustained Thinker progress falls below the input rate and backlog accumulates. The reverse direction, decode slowing prefill, is expected from shared execution resources but was not independently isolated by this experiment.

Therefore the current non-P/D capacity limit is concurrent multimodal prefill and multi-step decode sharing GPU 0, not vision preprocessing, Talker, Code2Wav, or an application-wide gate.

## 7. P/D comparison

The comparison uses the same model, byte-identical MP4, HD4 input, one-second cadence, seed, 360-second duration, and 36k rollover policy.

| Deployment | Four-GPU allocation | Verified real-time point |
|---|---|---:|
| Non-P/D | Thinker / Talker / Code2Wav / Encoder | 4 users |
| P/D | Thinker-P / Thinker-D / Talker / Code2Wav+Encoder | 15 users |

The P/D run completed 5400/5400 units with per-session stream RTF `1.002`. Its P and D service p50/p95/p99 were `310/849/1060 ms` and `299/790/1096 ms`.

This is a deployment-level comparison, not an equal-Thinker-GPU efficiency claim: P/D assigns two GPUs to Thinker while non-P/D assigns one and dedicates the fourth GPU to vision. It nevertheless answers the requested four-GPU setup question and shows that separating prefill and decode substantially increases sustainable sessions for this workload.

## 8. Reproduction

Start the non-P/D server:

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 MINICPMO45_LOG_PREP_DIAG=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

Run one capacity cell and analyze it:

```bash
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

Key files:

- `benchmarks/minicpmo/deploy_capacity_4gpu.yaml`
- `vllm_omni/deploy/minicpmo_4_5_4gpu.yaml`
- `benchmarks/minicpmo/continuous_av.py`
- `benchmarks/minicpmo/analyze_rtf.py`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u4_360s_20260831.*.json`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u5_360s_20260831.*.json`
- `benchmarks/minicpmo/results/nonpd_4gpu_hd4_u8_360s_20260831.*.json`
- `benchmarks/minicpmo/results/pd_4gpu_hd4_u15_360s_20260830.*.json`
