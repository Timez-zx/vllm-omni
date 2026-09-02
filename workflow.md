# MiniCPM-o Native-Duplex P/D Workflow

Chinese version: [workflow.zh.md](workflow.zh.md).

## Phase 1: objective

Measure the sustainable long-session capacity of continuous audio-video sessions on one four-GPU node. The study concerns serving latency and capacity, not model quality. Application and connector artifacts must be removed before an engine-level result is claimed.

## Phase 2: current serving design

```text
200 ms audio chunks + 1 FPS video
  -> one native one-second model unit
  -> Vision Encoder sidecar
  -> Thinker-P incremental prefill
  -> block-aligned KV delta transfer
  -> Thinker-D finite decode
  -> D output enters the next P lineage
  -> optional Talker + Code2Wav
```

| Stage | GPU | Role |
|---|---:|---|
| Thinker-P | 0 | Multimodal incremental prefill |
| Thinker-D | 1 | Finite autoregressive decode |
| Vision Encoder | 2 | Stateless HD4 frame encoding |
| Talker + Code2Wav | 3 | Speech-code and waveform generation |

- The application owns session history and media buffers. Engine KV is disposable execution state.
- Sessions enter the engine independently; there is no application-wide admission gate or cross-user batch.
- MiniCPM-o feeds Thinker output back into the next unit, so the real dependency is `D(i-1) -> P(i) -> D(i)`. The next Thinker unit does not wait for Talker or Code2Wav.
- At an estimated 36,000 tokens, the application opens a new lineage containing the system/reference input, the previous complete AV unit and confirmed Thinker output, and the current unit. The model limit is 40,960 tokens.
- P and D use FP8 E4M3 KV and `NixlDeltaPushConnector`. D retains prefix KV and imports only the new block-aligned suffix.

Every arriving frame is pre-encoded on GPU 2. Completed embeddings remain on CPU until the matching P request consumes them. No frame is silently dropped, and an encoding failure is reported instead of being hidden by formal-path re-encoding.

## Phase 3: workload and validity

The capacity candidate is 24 users for 180 seconds:

- real looped 960x540 MP4, aligned 16 kHz mono audio, and reference audio;
- audio arrival every 200 ms, video at 1 FPS, and one model unit per second;
- official HD slicing with `max_slice_nums=4`;
- per-session phase uniformly randomized in `[0, 1 s)` with +/-50 ms arrival jitter;
- seed `20260915`;
- randomized preconditioning age from 0 to 154 units, covering long context and rollover;
- 4,320 measured units plus 1,848 unmeasured preconditioning units.

The primary capacity metric is:

```text
stream RTF = completed one-second input budget /
             wall time from first media arrival through last physical-D completion
```

Capacity requires every session to have `stream RTF >= 1`, all expected physical-D requests to finish, no user failure, and complete frame consumption. Per-unit latency and stage RTF diagnose jitter but do not independently define capacity.

A formal result additionally requires a clean source tree, captured server/client provenance, diagnostics disabled, zero truncation/fallback/preemption, complete D-prefix evidence, and exact physical KV-transfer evidence. A dirty-tree run remains development evidence only.

## Phase 4: engineering confounds removed

| Confound | Current handling |
|---|---|
| Raw AV copied to D | D receives prompt metadata and imported KV only |
| Full historical KV transfer | P sends only the block-aligned delta |
| Historical recomputation on D | D prefix is reused; only a 1-2 token suffix is computed |
| Vision fallback re-encoding | Every measured frame consumes its arrival-preencoded embedding |
| Encoder queue drops | Frame identity is audited end to end; no silent drop |
| GPU2 -> GPU0 -> CPU detour | Sidecar output moves directly from GPU2 to CPU |
| Device copy under a global cache lock | Copy runs outside the lock with pending reservations |
| Late encoder result corrupts a retired session | Session tombstones reject late writes |
| Repeated image/audio metadata decoding | One decode per planning transaction |
| Repeated full-prompt Python copies | Redundant copies are removed and bridge payloads are released after D submit |
| Per-row sampling-metadata sync and duplicate logits clone | Sampling fields move to CPU once per batch; each row has one writable clone and unchanged RNG order |
| FlashInfer cache-miss startup depends on an activated shell | The clean launcher discovers the active Python environment's CUDA toolkit and `ninja`, then records both paths |
| Ambiguous D completion or KV reuse | Every physical D completion carries request, prefix, suffix, block, token, and byte evidence |

A new prepared-request protocol was not introduced: copying/serializing a 17k-token list costs about 0.1 ms and 73 KiB, which cannot explain a 0.5-1 s tail. An unbounded pinned-memory cache was also rejected because it adds memory risk without addressing the measured dominant path.

## Phase 5: latest measurement

Both rows use the same 24x180 workload and seed. The current run completed all 4,320 physical-D requests and consumed all 4,320 frames with zero fallback or user failure.

| Metric | Before cleanup | Current |
|---|---:|---:|
| Ready -> D p50/p95/p99 | `2069/5075/6034 ms` | `533/970/1244 ms` |
| Previous-D inherited wait p50/p95/p99 | `1061/4081/5051 ms` | `0/0/236 ms` |
| Fresh pre-D p50/p95/p99 | `574/772/856 ms` | `273/413/481 ms` |
| Current D service p50/p95/p99 | `388/701/914 ms` | `251/629/819 ms` |
| Fresh serial cycle p50/p95/p99 | `983/1300/1555 ms` | `532/929/1158 ms` |
| Terminal backlog p50/p95/p99 | `2476/3257/3286 ms` | `136/558/620 ms` |
| Stream RTF mean/min | `0.988/0.982` | `0.999/0.997` |

The current run transferred a median of 9 KV tokens and one 1.125 MiB block per unit; p99 was 16 tokens and one block. All 4,320 transfer records were valid.

The old 5-6 second tail was mostly an engineering artifact that recursively carried unfinished work into the next unit. After cleanup, inherited wait averages 7 ms and never exceeds 1 second. In the current top 1% tail, inherited wait contributes 15.6%, fresh pre-D contributes 26.1%, and D service contributes 58.3%.

The current development run narrowly fails the strict finite-run RTF criterion (`min=0.997`) and cannot certify capacity because the source tree is dirty. Its important result is causal: the multi-second backlog has been removed.

### Actual Thinker-P GPU utilization at 28 users

One 28x180 long run collected 1,004 GPU0 hardware-counter samples during the measurement window. Allocated VRAM is not used as a load metric here. `GPU kernel active` only records kernel residency and does not mean that the GPU's compute capacity is fully utilized.

| Metric | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| GPU kernel active | `59.6%` | `60.4%` | `96.1%` | `99.9%` |
| SM active | `39.6%` | `39.0%` | `73.5%` | `79.7%` |
| SM occupancy | `5.4%` | `5.3%` | `10.2%` | `11.2%` |
| Tensor Core active | `29.2%` | `27.8%` | `60.9%` | `66.1%` |
| DRAM bandwidth active | `9.0%` | `9.2%` | `14.7%` | `15.7%` |
| Power (about 600 W limit) | `311 W` | `322 W` | `349 W` | `361 W` |

GPU0 is not continuously at a compute, bandwidth, or power limit: mean SM active is about 40%, Tensor Core active about 29%, and DRAM active only 9%. The higher p95 values show that prefill bursts make the GPU briefly busy, but the pressure is not sustained. SM occupancy is not a direct fraction of peak FLOP/s, but together with the runner batch p50 of one request and about 219 tokens, it shows that most prefill batches expose little GPU parallelism.

The P-side problem at 28 users is therefore not exhausted physical GPU capacity. Fragmented incremental prefills fail to sustain efficient batches: hardware is underused between bursts, while each burst still queues and amplifies the tail. The hardware counters establish the lack of sustained saturation; the batch shapes and runner timings are what attribute the inefficiency to fragmented prefill.

After the final sampler cleanup, a non-diagnostic 24x30 regression on the exact final code completed 720/720 physical-D requests and consumed 720/720 frames with zero fallback or user failure. Ready-to-D p50/p95/p99 was `256/457/557 ms`; current pre-D was `121/215/267 ms`; D service was `126/278/354 ms`; inherited-wait p99 was zero. This matches the cleaned short-run baseline but is not a formal capacity result because it lasts only 30 seconds and the tree is dirty.

A separate 24x30 diagnostic run used the same production workload and seed as the clean short screen. Diagnostics perturb absolute latency, so these values are for attribution only:

| Residual path | p99 | Interpretation |
|---|---:|---|
| Application ready -> P submit | `2.5 ms` | Application admission is not the tail |
| P scheduler queue | `1.7 ms` | The request is selected promptly after Core admits it |
| P runner, all work | `222 ms` | Delta preparation, forward, and sampling/snapshot |
| P result exposure | `33 ms` | Secondary control-plane cost |
| D ingress after IPC decode | `134 ms` | Core waits for the current synchronous runner step before draining its input queue |
| D scheduler queue | `3.6 ms` | Scheduling after admission is prompt |
| D runner, all decode steps | `315 ms` | Sequential autoregressive work; output-token p99 is 8 |
| D result exposure | `18 ms` | Secondary control-plane cost |

Raw StagePool send, Core receive, message decode, and preprocessing are normally below 2 ms. P-to-D `write()` itself has p99 `3.0 ms`; write-to-D completion has p99 `79 ms` and overlaps the compute pipeline. Thus serialization, IPC, KV bandwidth, scheduler queueing, and application gating are not large enough to explain the remaining tail.

The remaining dominant costs are now explicit: real P delta preparation/forward, sequential D decode, and a synchronous Core loop that can admit new arrivals only between runner steps. The first two are model work. The third is an engine scheduling abstraction: the request has already reached and been decoded by Core, but cannot join an in-flight batch. Removing it requires event-driven/thread-safe admission or a different incremental-batching scheduler, not another application-side gate. Random sampling still needs per-row host decisions to preserve each session's RNG sequence; that secondary cost does not explain the measured tail.

## Phase 6: reproduction

Start a fresh non-diagnostic server:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
python benchmarks/minicpmo/clean_server.py \
  --provenance-out /tmp/minicpm-pd-server-provenance.json -- \
  python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113 \
  2>&1 | tee /tmp/minicpm-pd-server.log
```

Run the 24x180 production profile and analyze it:

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

Restart the server for each capacity point. Diagnostic flags are allowed only in a separate run launched with `clean_server.py --allow-diagnostics`.
