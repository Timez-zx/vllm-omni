# MiniCPM-o Native-Duplex P/D Workflow

## Goal

Measure the sustainable long-session capacity of one four-GPU node under continuous audio-video input. Model quality is out of scope.

## Phase 1: serving design

```text
one WebSocket per user
  -> upload 200 ms PCM16 audio chunks and one video frame per second
  -> aggregate them into one native one-second model unit
  -> Thinker-P appends the new AV unit to the session KV lineage
  -> transfer only the new block-aligned KV suffix to Thinker-D
  -> Thinker-D performs finite decode and decides listen/speak
  -> feed D output into the next P lineage
  -> on speak, run Talker -> Code2Wav
```

- The application owns sessions and media buffers; engine KV is disposable execution state.
- Sessions enter the engine independently. There is no application-wide gate or cross-user batch.
- `D(i-1)` must finish before `P(i)` because D output belongs to the next Thinker lineage. The next Thinker unit does not wait for Talker or Code2Wav.
- At 36,000 estimated tokens, the application opens a new lineage containing the system/reference input, the previous complete AV unit and confirmed Thinker output, and the current unit. The model limit is 40,960 tokens.

Deployment:

| Stage | GPU | Role |
|---|---:|---|
| Thinker-P | 0 | Incremental multimodal prefill |
| Thinker-D | 1 | Finite autoregressive decode |
| Vision Encoder | 2 | Stateless HD4 frame encoding |
| Talker + Code2Wav | 3 | Speech-code and waveform generation |

Every arriving video frame is submitted to GPU 2. The Encoder runs one RPC at a time; frames arriving during that RPC form the next microbatch, with no frame drop or per-session lookahead limit. Completed embeddings remain on CPU and move to GPU 0 only when formal P consumes them, so backlog cannot fill Thinker memory. Formal P waits for the matching embedding to become ready; request-local fallback is reserved for an actual encoding failure.

This is a capacity-diagnosis configuration. Production still needs explicit backpressure or a frame-drop policy derived from its memory budget, but silent fallback re-encoding must not hide downstream throughput.

P and D use `NixlDeltaPushConnector`. D retains prefix KV and receives only the new KV suffix.
Both stages store KV in FP8 E4M3. MiniCPM-o 4.5 does not provide KV scales, so P and D use the same deterministic default scale instead of independently calibrating incompatible representations. This increases measured P/D cache capacity from `497,504/490,416` to `995,024/980,848` tokens.

## Phase 2: workload and capacity criterion

- Real looped MP4: 960×540 video, aligned 16 kHz mono audio, and reference audio.
- Audio arrives every 200 ms; video arrives at 1 FPS; the model consumes one one-second unit at 1 Hz.
- `max_slice_nums=4`, so every video unit uses the HD4 path.
- 360 seconds, random phases in `[0, 1 s)`, seed `20260839`.
- Each capacity point uses a cold server followed by one single-user JIT warm-up. Reusing one server across capacity points is invalid because stale session-close state can contaminate the next run.
- The 19-user run contains 6,840 units and two context rollovers per session. The 20-user run expects 7,200 units.
- The latest root-cause run uses 24 users for 360 seconds with seed `20260904`, expects 8,640 formal units, and randomizes history age to cover long context and rollover.
- NVIDIA DCGM profiler counters are sampled at 1 Hz over the input-to-last-D-completion window. `SM active`, `SM occupancy`, `Tensor active`, and `DRAM active` measure actual hardware activity; NVML GPU busy is retained only as a contrast and is not treated as saturation.

Per-unit RTF diagnoses jitter:

```text
unit RTF = 1000 ms / stage service time
```

Capacity uses sustained progress:

```text
stream RTF = completed one-second input budget /
             wall time from first input-ready to last D completion
```

Capacity passes only when every session has `stream RTF >= 1`, every D unit completes, and no user fails. A slow unit is recoverable jitter if later units drain its backlog.

## Phase 3: final measurement

### Capacity-boundary reference

The 19/20-user boundary below used the previous bounded arrival cache. The current unbounded preencode path has not rescanned 19–23 users, so this is a reference boundary rather than a recertification of the latest code.

| Users | Result | Completed D units | Failed users | Stream RTF min/p50/p95 | Cycle RTF min/p50/p95 |
|---:|---|---:|---:|---|---|
| 19 | Pass | 6,840/6,840 | 0 | `1.002/1.002/1.003` | `0.995/1.003/1.018` |
| 20 | Fail | 6,162/7,200 | 17 | `0.777/0.790/0.810` | `0.772/0.772/0.781` |

At 20 users, each session completed only 308.1 units on average and accumulated 80.6 seconds of terminal backlog. The effective D completion rate was about 15.75 one-second units/s versus the required 20 units/s.

| Run | P service p50/p95/p99 | D service p50/p95/p99 | P/D end-to-end p50/p95/p99 | Previous-D wait p50/p95/p99 |
|---|---|---|---|---|
| 19 users | `192/736/1,029 ms` | `237/848/1,175 ms` | `473/2,190/2,770 ms` | `0.177/1,117/1,471 ms` |
| 20 users | `656/1,341/1,482 ms` | `440/946/1,269 ms` | `2,302/3,693/4,123 ms` | `1,152/1,903/2,132 ms` |

### Actual P/D hardware activity at the 20-user failure point

The table covers 392 one-second samples from input start through the final D completion. Percentages are profiler activity ratios, not memory allocation or process residency.

| Counter | Thinker-P mean/p95/max | Thinker-D mean/p95/max |
|---|---:|---:|
| SM active | `27.0/38.1/43.2%` | `15.5/36.1/46.9%` |
| SM occupancy | `3.9/5.5/6.4%` | `1.9/4.5/6.0%` |
| Tensor active | `18.7/27.7/32.1%` | `1.3/2.4/2.8%` |
| DRAM active | `8.8/15.7/17.2%` | `13.9/32.6/43.5%` |
| PCIe TX | `44.3/98.7/230.4 MiB/s` | `3.1/5.9/6.9 MiB/s` |
| PCIe RX | `69.3/89.2/101.9 MiB/s` | `51.7/108.1/241.4 MiB/s` |
| Power | `238/295/326 W` | `138/200/231 W` |
| NVML GPU busy | `36.8/100/100%` | `19.2/51.5/73%` |

P can report 100% NVML busy at p95 while only 38.1% of SM cycles are active and occupancy is 5.5%. Therefore NVML busy would falsely suggest saturation. Neither P nor D saturates SMs, Tensor Cores, DRAM, PCIe, or the 600 W power envelope.

There is no OOM, preemption, or recomputation in the failed run, and D still hits nearly the entire prefix. The failure is therefore not KV capacity or link bandwidth. It is a pipeline-efficiency limit: the recurrent `D(i-1) -> P(i) -> D(i)` dependency, finite request/control overhead, and irregular small P/D batches leave both GPUs idle between bursts. At 20 users, previous-D waiting becomes persistent instead of recoverable, so backlog grows even though raw GPU resources remain available.

### Latest 24-user root-cause run

The run first removes the Encoder confound: every frame is preencoded, ready embeddings stay on CPU, and cache eviction cannot send vision encoding back to the P runner. The short check completed 720/720 frames with arrival-cache hits. The long run had no OOM or formal fallback; Encoder-ready p99 was `608 ms`, while formal P waited only `1 ms` at p99 for its embedding.

The 24-user run sent 8,640 formal units and completed 7,300 D units. Every session had long-horizon RTF below one.

| Metric | Result |
|---|---:|
| Stream RTF min/p50/p95 | `0.776/0.784/0.794` |
| Terminal backlog p50/p95 | `83.6/86.6 s` |
| P/D end-to-end p50/p95/p99 | `2,479/3,219/3,604 ms` |
| Previous-D wait p50/p95/p99 | `1,252/1,694/1,921 ms` |
| P service p50/p95/p99 | `778/1,197/1,368 ms` |
| P-done to D-done p50/p95/p99 | `442/794/1,029 ms` |
| D service p50/p95/p99 | `479/828/1,064 ms` |

P throughput is decisive. During the `390.7 s` formal window, the P runner executed for `383.3 s`, or `98.1%` duty. Its 980 batches averaged `7.52` requests, `1,599` tokens, and `391 ms`, yielding only about `18.8 units/s` against a `24 units/s` input rate. D runner duty was only `56.1%`; D and handoff add per-unit latency but do not set 24-user capacity.

| DCGM counter | Thinker-P mean/p95/max | Thinker-D mean/p95/max |
|---|---:|---:|
| SM active | `30.7/38.4/41.6%` | `15.8/24.9/30.4%` |
| SM occupancy | `4.3/5.3/5.8%` | `1.9/3.1/4.1%` |
| Tensor active | `21.9/28.2/30.2%` | `1.3/1.9/2.3%` |
| DRAM active | `7.8/9.5/10.8%` | `14.5/22.7/28.5%` |
| Power | `256/302/313 W` | `143/173/194 W` |

`98.1% P runner duty` and `30.7% mean SM active` are not contradictory. The first says the engine nearly always has a P batch executing; the second says those small incremental batches activate the SMs for only about one-third of wall time. The current P execution path has no idle capacity, but each batch still uses the hardware inefficiently.

## Phase 4: conclusion

The tail follows the model dependency:

```text
D(i-1) feedback -> P(i) -> KV handoff -> D(i) -> feedback -> P(i+1)
```

Long context increases P and D service time. Once P throughput falls below the arrival rate, `D(i-1)` feedback becomes late and the next unit's wait grows from jitter into persistent backlog. That wait is an effect of insufficient P capacity, not a third independent execution stage.

The latest implementation excludes Encoder-cache eviction, P-side vision re-encoding, GPU embedding leakage, and KV OOM. The primary 24-user limit is Thinker-P runner throughput: it is saturated in time while its small incremental batches underuse SMs, Tensor Cores, and memory bandwidth. Research should target execution efficiency for long-context, small-increment P batches. D/handoff optimization can reduce per-unit latency but cannot by itself close the `18.8 -> 24 units/s` throughput gap.

The previous measured boundary was 19 users passing and 20 failing. The current code confirms that 24 users fail, but its exact boundary still requires a 19–23 rescan. FP8 KV solves residency capacity only; it does not reduce model compute. Default-scale FP8 KV still requires a separate quality evaluation before production use.

## Phase 5: reproduction

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
MINICPMO45_LOG_PREP_DIAG=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113

python benchmarks/minicpmo/continuous_av.py \
  --users 24 --duration-s 360 --phase-window-s 1 --seed 20260904 \
  --connect-stagger-s 0.5 --post-stream-s 30 --close-timeout-s 180 \
  --loop-media --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --context-window-trigger-tokens 36000 --gpus 0 1 2 3 \
  --out /tmp/minicpm-pd-cpu-cache-u24x360.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-cpu-cache-u24-server.log \
  --run-json /tmp/minicpm-pd-cpu-cache-u24x360.json \
  --out /tmp/minicpm-pd-cpu-cache-u24x360-analysis.json
```

Rescan 19–23 users and restart the server before changing the user count. During the client run, collect profiler counters with:

```bash
sudo dcgmi dmon \
  -e 1001,1002,1003,1004,1005,1009,1010,155,203,204 \
  -i 0,1,2,3 -d 1000
```
