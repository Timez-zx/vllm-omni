# MiniCPM-o Native-Duplex Serving Workflow

## Goal and scope

The goal is to measure single-node multi-user capacity under a real-time constraint for continuous audio-video interaction, then locate the engine bottleneck when capacity fails. The research target is scheduling, batching, KV cache behavior, and pipeline latency rather than model quality.

This branch uses the native duplex path of `openbmb/MiniCPM-o-4_5`. Unlike the duplex-like Qwen3-Omni approximation, MiniCPM continuously consumes AV input and decides whether to listen or speak, so it more closely represents the target serving workload.

## Phase 1: application and engine interface

```text
one long-lived WebSocket per user
  → upload one PCM16 audio chunk every 200 ms
  → attach one video frame every second
  → combine five audio chunks into one native 1 s model unit
  → Thinker incrementally processes the unit and decides listen or speak
  → on speak, run Talker → Code2Wav
  → keep accepting input without waiting for audio playback to finish
```

- Each session owns one resumable Thinker request/KV lineage. A new unit appends only new tokens instead of prefilling the complete history again.
- Native `auto_response` remains enabled. The client submits neither synthetic queries nor forced responses.
- Sessions enter the engine concurrently. There is no application-wide gate or cross-user batching.
- Network audio arrives at 5 Hz, but model execution operates at 1 Hz. Partial PCM is combined in the application input buffer and does not create five Thinker prefills.
- User input can enter the existing session while assistant output is active; playback state does not control model admission.
- KV lineage is disposable execution state. The application owns the session, input buffer, reconnect, and output state.

## Phase 2: fixed deployment

Formal baseline experiments use only `benchmarks/minicpmo/deploy_capacity_3gpu.yaml`:

| Stage | GPU | Configuration |
|---|---:|---|
| Thinker | 0 | BF16, one non-disaggregated vLLM engine |
| Talker | 1 | BF16 |
| Code2Wav | 2 | BF16 |

- Hardware: 3 × RTX PRO 6000 Blackwell 96 GB.
- All three stages use `max_num_seqs: 64` and synchronous scheduling.
- `active_stream_window: 0` avoids an application-side cap on concurrent speaking sessions.
- This phase does not use P/D disaggregation. The bottleneck must first be established on this fixed baseline.

Start the server with:

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_3gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

## Phase 3: formal workload and metric

- Every user continuously streams aligned audio and video from the same real MP4.
- Audio: 16 kHz mono PCM16, uploaded every 200 ms.
- Video: 1 FPS at the source 960×540 resolution, with `max_slice_nums=4`.
- The official slicing algorithm produces one global image and two local crops. Each steady-state unit contains 198 vision scheduler tokens and 211 scheduler tokens in total.
- Connections are staggered during setup. Once all sessions are ready, they start behind one barrier at seeded random phases in `[0, 1 s)`.
- Each cell runs for 30 seconds, covering both short and growing context. Media looping is used only for the context-boundary stress test.

For each one-second Thinker and Talker model unit:

```text
RTF = 1000 ms / stage service time
```

Real time requires RTF `> 1`. Strict capacity requires every observed Thinker and Talker unit to stay below 1000 ms; one miss fails that concurrency level. Code2Wav uses one persistent request spanning the entire session, including silent periods, so its request wall time is not a valid unit RTF.

Example formal run:

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 8 --duration-s 30 --phase-window-s 1 --seed 20260829 \
  --connect-stagger-s 0.5 --post-stream-s 4 --gpus 0 1 2 \
  --media /path/to/omni_duplex1.mp4 \
  --ref-audio /path/to/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --out /tmp/minicpm-hd4-u8.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-server.log \
  --run-json /tmp/minicpm-hd4-u8.json \
  --out /tmp/minicpm-hd4-u8-rtf.json
```

## Phase 4: long-session context management

The model limit is 40,960 tokens. Once the estimate reaches 36,000 tokens, the next unit opens a new KV lineage containing only:

- the system prompt and reference audio;
- the previous complete AV unit and its confirmed Thinker output;
- the current AV unit.

The scheduler releases the old lineage's KV blocks before admitting the replacement prompt. The 36k trigger reserves roughly 5k tokens for maximum output, an in-flight unit, and estimation error. It creates no summary and adds no separate model call to the online path.

A 180-second single-user HD4 run rolled over once at unit 157:

| Metric | Result |
|---|---:|
| Replacement prompt | 494 tokens |
| Thinker p50/p95/p99 | 175/407/567 ms |
| Talker p50/p95/p99 | 231/591/621 ms |
| RTF ≤ 1 | 0 |
| Session/server errors | 0 |

The rollover unit took 175 ms. Mean Thinker service time was 394 ms over the preceding 20 units and 199 ms over the following 20. The policy therefore crossed the context limit online and removed the old long-context execution cost. Archived result: `benchmarks/minicpmo/results/long_context_hd4_3gpu_20260829.json`.

## Phase 5: current capacity

Current code on a warmed server with the 30-second HD4 workload:

| Users | Seed | Thinker p50/p95/p99/max | Thinker misses | Talker p50/p95/p99/max | Talker misses | Result |
|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 20260829 | 284/613/706/731 ms | 0/220 | 168/474/687/701 ms | 0/189 | pass |
| 8 | 20260829 | 464/888/959/991 ms | 0/237 | 219/460/609/668 ms | 0/208 | pass |
| 8 | 20260828 | 407/923/981/994 ms | 0/242 | 200/575/737/838 ms | 0/207 | pass |
| 9 | 20260829 | 566/1007/1851/2071 ms | 13/228 | 199/626/1224/1450 ms | 5/230 | fail |

The measured strict capacity is eight sessions, but both eight-user runs have less than 10 ms of maximum-latency headroom. Seven sessions is the practical operating point when a safety margin is required. Nine users is the first clear failure.

## Phase 6: root cause of the nine-user failure

The slowest 5% of nine-user Thinker units average 1493 ms:

| Component | Mean | Share |
|---|---:|---:|
| Application submission to engine admission | 133 ms | 8.9% |
| Scheduler wait | 0.3 ms | <0.1% |
| Runner execution | 1302 ms | 87.2% |
| └ first AV prefill forward | 280 ms | 18.7% |
| └ subsequent decode forwards | 1023 ms | 68.5% |
| Inter-forward control gaps | 3 ms | 0.2% |
| Result exposure | 54 ms | 3.6% |

Key evidence:

1. Within the same nine-user run, decode forwards mixed with another session's AV prefill take 159/414 ms at p50/p95. Decode-only forwards take 24/42 ms, a 6.7×/9.8× difference.
2. Of the slowest units' 1023 ms decode time, 981 ms is inside mixed prefill/decode forwards.
3. A control keeps the same nine users and all 270 HD4 AV prefills but forces every unit to terminate at the listen decision. Thinker p50/p95/p99/max becomes 215/385/405/436 ms with zero misses.
4. Scheduler queue time is only 0.3 ms, excluding application serialization and scheduler admission as the dominant cause.

Conclusion: nine users fail because the Thinker runner places multimodal prefill and other sessions' multi-step decode in the same iterations, repeatedly stretching decode forwards. AV prefill alone meets the one-second budget; the real-time boundary is crossed only when it is mixed with sustained decode. Talker misses are secondary and largely inherit upstream Thinker delay.

GPU 0 NVML busy is 83% at p95 and memory-I/O busy is 57% at p95. These are device busy-time counters, not SM occupancy, so they do not prove complete compute or memory-bandwidth saturation. The supported conclusion is specifically a mixed prefill/decode runner-efficiency and scheduling problem.

Archived result: `benchmarks/minicpmo/results/capacity_hd4_3gpu_20260829.json`.

## Phase 7: research baseline conclusion

Keep the current application design: native one-second duplex units, incremental per-session KV, no cross-session global gate, model-owned listen/speak decisions, and 36k context rollover. It exposes the concurrent continuous-AV prefill/decode load instead of hiding contention through application serialization.

The next engine study should optimize mixed multimodal-prefill/decode batching or deadline/QoS scheduling under a fixed input trace, then compare user capacity under the same real-time SLO.

## Phase 8: MiniCPM P/D disaggregation

The `minicpm-pd` branch splits Thinker into four independent stages:

| Stage | GPU | Lifetime |
|---|---:|---|
| Thinker-P | 0 | One resumable KV lineage per session; append 211 AV tokens each second, run prefill, and sample the boundary token |
| Thinker-D | 1 | One finite request per model unit; import P's KV delta and continue decode |
| Talker | 2 | Serialized within a session and concurrent across sessions |
| Code2Wav | 3 | Consume streaming codec chunks from Talker |

- P and D use `NixlDeltaPushConnector`. D retains received prefix KV and each round transfers only the new block-aligned suffix, not the complete history.
- Thinker tokens generated by D feed the next P lineage, preserving model recurrence.
- A session's next Thinker-P unit does not wait for Talker/Code2Wav. Talker itself remains ordered so one user's speech segments cannot overlap.
- Context rollover, AV cadence, and application-owned session semantics match the non-P/D baseline.

Deploy with:

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113
```

The strict P/D capacity criterion requires every P, D, and Talker unit to stay below one second and each slot from input-ready through D completion to stay below one second. Formal runs disable per-request handoff diagnostics and retain only cadence traces so logging I/O does not affect capacity.

Formal 30-second results:

| Users | P-ready→D p50/p95/p99/max | Late slots | P p95/p99 | D p95/p99 | Talker p95/p99 | Result |
|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 416/614/717/747 ms | 0/247 | 341/411 ms | 319/383 ms | 404/447 ms | pass |
| 9 | 620/1289/1431/1515 ms | 71/288 | 678/742 ms | 512/592 ms | 471/541 ms | fail |

Strict capacity is eight users and nine is the first failure. P/D does not raise the integer ceiling, but it lowers the eight-user maximum from roughly 991 ms for the non-P/D Thinker to 747 ms for the complete P-to-D path. Eight users therefore move from less than 10 ms of headroom to roughly 253 ms.

Slowest-five-percent breakdown at nine users:

| Path | Service | App→Core | Scheduler/KV wait | Runner | Result exposure |
|---|---:|---:|---:|---:|---:|
| Thinker-P | 714 ms | 304 ms | 0.5 ms | 359 ms | 50 ms |
| Thinker-D | 541 ms | 37 ms | 326 ms | 142 ms | 34 ms |

Conclusions:

1. P performs no decode and D performs no multimodal prefill, so the non-P/D mixed prefill/decode contention is removed.
2. At nine users, random-phase collisions combine the 211-token multimodal increments into three- or four-request P batches. P runner time and the application-to-Core input path both grow; P is the limiting stage.
3. D model compute is not the main bottleneck. Its slowest units spend only 142 ms in the runner and 326 ms waiting for ordered KV availability and scheduler admission. This is engineering headroom in the current connector/progress path.
4. A slot must execute P before D. Both stages individually staying below one second does not keep their serial path below one second; bursts cross the deadline and make the next slot wait.
5. P/D GPU-busy p95 is `72%/71%`. This counter is not SM occupancy and does not prove compute or memory-bandwidth saturation.

Archived result: `benchmarks/minicpmo/results/capacity_pd_hd4_4gpu_20260829.json`.

## Recovery map

| Purpose | Path |
|---|---|
| Fixed deployment | `benchmarks/minicpmo/deploy_capacity_3gpu.yaml` |
| P/D deployment | `benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml` |
| Multi-user workload | `benchmarks/minicpmo/continuous_av.py` |
| RTF and tail analysis | `benchmarks/minicpmo/analyze_rtf.py` |
| Current capacity archive | `benchmarks/minicpmo/results/capacity_hd4_3gpu_20260829.json` |
| P/D capacity archive | `benchmarks/minicpmo/results/capacity_pd_hd4_4gpu_20260829.json` |
| Long-context archive | `benchmarks/minicpmo/results/long_context_hd4_3gpu_20260829.json` |
| MiniCPM input aggregation | `vllm_omni/experimental/fullduplex/minicpmo45/input.py` |
| Context rollover | `vllm_omni/experimental/fullduplex/minicpmo45/runtime.py` |
| Thinker multimodal input | `vllm_omni/experimental/fullduplex/minicpmo45/stage0.py` |
| Realtime orchestration | `vllm_omni/experimental/fullduplex/openai/runtime_bridge.py` |
