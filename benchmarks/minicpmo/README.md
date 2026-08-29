# MiniCPM-o native-duplex capacity benchmark

## Workload

- Model: `openbmb/MiniCPM-o-4_5`, BF16.
- Topology: Thinker, Talker, and Code2Wav on GPUs 0, 1, and 2; no P/D disaggregation.
- Each user continuously streams real MP4 audio as 200 ms PCM16 chunks and one real video frame per second.
- Users start at seeded random phases within one second. Setup is staggered, then all users start behind one barrier.
- Native `auto_response` remains enabled. There is no application-side global admission gate or cross-user batching.
- A capacity run lasts 30 seconds, so it includes both short- and longer-context slots.

One native model unit represents one second of input. For Thinker and Talker:

```text
RTF = 1000 ms / stage service time
```

Real-time requires RTF `> 1`. The strict capacity criterion requires every
observed Thinker and Talker unit to pass; one unit at RTF `<= 1` is a capacity
failure. RTF p05 and miss rate are also reported to show how far the system is
from the boundary. Code2Wav uses one persistent request for the whole session,
including idle periods, so its request wall time is not a valid per-unit RTF
and is excluded from this SLO.

Protocol-event gaps are diagnostics only. A model unit may listen silently or
emit several events, so input and output event indices cannot be paired as
slot latency. In particular, gaps between separate speech episodes are not
audio-playback underruns while the model is deliberately listening.

## Long-session context rollover

MiniCPM uses one resumable engine request per session, but its KV lineage is
disposable. Before the estimated Thinker context reaches 36,000 tokens (the
model limit is 40,960), the next one-second unit starts a new lineage containing
only:

- the system prompt and reference-audio context;
- the previous complete AV unit and its confirmed Thinker output;
- the current AV unit.

The scheduler releases all KV blocks owned by the old lineage before admitting
the replacement prompt. Later units resume normal incremental prefill. The
36k trigger is conservative: the estimate reserves the configured maximum
Thinker output for every unit, leaving roughly 5k tokens for estimation error
and an in-flight unit. This is the MiniCPM equivalent of DuplexOmni's bounded
slot-history rollover; the retained model unit differs because MiniCPM has a
native one-second streaming unit.

Use media looping only to cross the context boundary in a reproducible stress
test:

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 1 --duration-s 180 --phase-window-s 0 --seed 20260828 \
  --connect-stagger-s 0 --post-stream-s 4 --gpus 0 1 2 \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 --loop-media \
  --out /tmp/minicpm-long-context.json
```

The 180-second HD4 run completed all 180 input units without a session or
server error. Rollover occurred at Thinker unit 157: the scheduler replaced the
old context with a 494-token prompt (490 retained/current input rows plus four
confirmed output tokens). Its service time was 175 ms, compared with 394 ms
mean over the preceding 20 units and 199 ms mean over the following 20 units.
The full-run Thinker p50/p95/p99 was 175/407/567 ms; no Thinker or Talker unit
exceeded its one-second real-time budget. The rollover therefore both stayed
online and removed the old long-context execution cost. The archived result is
`results/long_context_hd4_3gpu_20260829.json`.

## Run

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_3gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113

python benchmarks/minicpmo/continuous_av.py \
  --users 20 --duration-s 30 --phase-window-s 1 --seed 20260828 \
  --connect-stagger-s 1 --gpus 0 1 2 \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --out /tmp/minicpm-u20.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-server.log \
  --run-json /tmp/minicpm-u20.json \
  --out /tmp/minicpm-u20-rtf.json
```

For the official HD-slicing path, preserve the source frame and set the slice
limit explicitly:

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 7 --duration-s 30 --phase-window-s 1 --seed 20260828 \
  --connect-stagger-s 0.5 --post-stream-s 4 --gpus 0 1 2 \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --out /tmp/minicpm-hd4-u7-30s.json
```

`max_slice_nums` is an upper bound used by the official aspect-ratio grid
algorithm, not a fixed crop count. The 960x540 reference video becomes one
global image plus a 2x1 crop grid: three 64-row vision blocks, or 198 scheduler
tokens including wrappers, per one-second model unit. The default capacity
workload remains the one-block path so existing capacity numbers stay
comparable.

The trace flag adds observation-only admission, scheduling, runner-completion,
and stage-completion timestamps. It does not change scheduling policy.

## P/D deployment and capacity

The P/D variant uses four GPUs: Thinker-P on GPU 0, Thinker-D on GPU 1,
Talker on GPU 2, and Code2Wav on GPU 3. P keeps one resumable lineage per
session. D runs one finite request per model unit and imports only the new
block-aligned KV suffix. D output tokens feed the next P unit. Talker remains
ordered within each session while the next Thinker unit may overlap it.

```bash
VLLM_OMNI_LOG_DUPLEX_CADENCE=1 \
python -m vllm_omni.entrypoints.cli.main serve openbmb/MiniCPM-o-4_5 \
  --omni --deploy-config benchmarks/minicpmo/deploy_capacity_pd_4gpu.yaml \
  --trust-remote-code --host 127.0.0.1 --port 8113

python benchmarks/minicpmo/continuous_av.py \
  --users 8 --duration-s 30 --phase-window-s 1 --seed 20260828 \
  --connect-stagger-s 0.5 --post-stream-s 4 --gpus 0 1 2 3 \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 4 \
  --out /tmp/minicpm-pd-u8-30s.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-server.log \
  --run-json /tmp/minicpm-pd-u8-30s.json \
  --out /tmp/minicpm-pd-u8-30s-rtf.json
```

P/D adds an end-to-end slot criterion: input-ready through D completion must
also remain below one second. Formal 30-second HD4 runs give:

| Users | P-ready→D p50/p95/p99/max | Late slots | P p95/p99 | D p95/p99 | Result |
|---:|---:|---:|---:|---:|:---:|
| 8 | 416/614/717/747 ms | 0/247 | 341/411 ms | 319/383 ms | pass |
| 9 | 620/1289/1431/1515 ms | 71/288 | 678/742 ms | 512/592 ms | fail |

At nine users, the slowest 5% of P units average 714 ms: 304 ms before Core
admission, 359 ms in the multimodal-prefill runner, and 50 ms exposing the
result. The slowest D units average 541 ms, of which 326 ms is ordered
KV/scheduler wait and only 142 ms is runner work. P/D removes mixed
prefill/decode batches, but bursty P work plus the current P/D progress path
still pushes the serial P-to-D slot over its deadline. The archive is
`results/capacity_pd_hd4_4gpu_20260829.json`.

## HD-slicing capacity result

With the 960x540 source preserved and `max_slice_nums=4`, the official grid
algorithm emits one global image plus two local crops. Each one-second unit
therefore carries 198 vision scheduler tokens and 211 total steady-state
scheduler tokens, compared with 66 and 79 on the one-block workload.

Current-code measurements use a warmed server, 30 seconds per run, random
phases in `[0, 1 s)`, and two seeds at the boundary:

| Users | Seed | Thinker p50/p95/p99/max | Thinker misses | Talker p50/p95/p99/max | Talker misses | Result |
|---:|---:|---:|---:|---:|---:|:---:|
| 7 | 20260829 | 284/613/706/731 ms | 0/220 | 168/474/687/701 ms | 0/189 | pass |
| 8 | 20260829 | 464/888/959/991 ms | 0/237 | 219/460/609/668 ms | 0/208 | pass |
| 8 | 20260828 | 407/923/981/994 ms | 0/242 | 200/575/737/838 ms | 0/207 | pass |
| 9 | 20260829 | 566/1007/1851/2071 ms | 13/228 | 199/626/1224/1450 ms | 5/230 | fail |

The measured strict capacity is eight sessions; nine is the first clear
failure. Eight has less than 10 ms of observed max-latency headroom in both
runs, so seven is the sensible operating point when non-zero safety margin is
required.

### Bottleneck evidence

At nine users, the slowest 5% of Thinker units average 1493 ms. The breakdown
is 133 ms from application dispatch, 0.3 ms in the scheduler queue, 1302 ms in
runner execution, 3 ms between forwards, and 54 ms exposing the result. Runner
time is therefore 87% of the tail; admission is not the bottleneck. Inside the
runner, the first AV prefill forward contributes 280 ms and subsequent decode
forwards contribute 1023 ms. Of that decode time, 981 ms is spent in forwards
that also contain another session's prefill.

Forward-level traces provide a direct within-run control. At nine users, 284
decode forwards mixed with prefill have p50/p95 durations of 159/414 ms; 91
decode-only forwards take 24/42 ms. The same decode operation is therefore
6.7x slower at p50 and 9.8x slower at p95 when it must traverse a mixed AV
prefill batch.

A separate diagnostic run keeps the same nine users and all 270 HD4 AV units,
but forces each unit to stop at the model's listen decision. This preserves AV
prefill while removing multi-step decode and Talker. Thinker p50/p95/p99/max
becomes 215/385/405/436 ms with zero misses. This proves that nine-user AV
prefill alone fits the one-second budget; failure appears when those prefills
are interleaved with active Thinker decode.

Reproduce that control by adding `--force-listen-count 10000` to the HD4
capacity command. This option is diagnostic only and must not be used for a
reported end-to-end capacity run.

GPU 0 busy is 83% at p95 and its memory-I/O busy is 57% at p95 in the failing
run. GPU 1 busy is only 27% at p95. These are NVML device busy-time counters,
not SM occupancy, so they do not prove raw hardware saturation. The supported
root cause is narrower: the current Thinker runner makes decode share long
iterations with multimodal prefills, stretching repeated decode forwards past
the model-unit deadline. The archived result is
`results/capacity_hd4_3gpu_20260829.json`.

## One-block baseline capacity result

Measured on three RTX PRO 6000 Blackwell 96 GB GPUs:

| Users | Thinker service p95 | Thinker RTF p05 | Thinker misses | Talker service p95 | Talker RTF p05 | Talker misses | Result |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 15 | 493 ms | 2.027 | 0.00% | 390 ms | 2.565 | 0.00% | pass |
| 16 | 740 ms | 1.352 | 0.20% | 633 ms | 1.579 | 0.00% | fail |
| 17 | 862 ms | 1.160 | 0.80% | 767 ms | 1.304 | 0.47% | fail |
| 18 | 928 ms | 1.078 | 2.79% | 1005 ms | 0.995 | 5.36% | fail |
| 20 | 1072 ms | 0.933 | 8.13% | 1171 ms | 0.854 | 8.20% | fail |

The maximum measured strict capacity is 15 sessions. Under the looser p95 RTF
criterion, 17 sessions still pass; that is reported only as a secondary
engineering reference, not as the capacity claimed here.

## Overload root cause

For the slowest 5% of Thinker units at 20 users, mean service time is 1573 ms:

| Component | Mean | Share |
|---|---:|---:|
| Application dispatch to Core admission | 94 ms | 6.0% |
| Scheduler wait before first selection | 0.4 ms | <0.1% |
| Runner execution across all forwards | 1406 ms | 89.4% |
| └ first incremental-prefill forward | 234 ms | 14.9% |
| └ subsequent decode forwards | 1173 ms | 74.5% |
| Inter-forward scheduler/control gaps | 5 ms | 0.3% |
| Final result exposure | 68 ms | 4.3% |

The tail unit needs 7.5 Thinker forwards on average. Almost every decode forward
shares a batch with another session's incremental prefill: the mixed-step rate
is 93.6%, accounting for 1149 of the 1173 ms decode-runner time. At 20 users,
a mixed decode step has 231 ms p50 runner time, versus 29 ms for a decode-only
step.

Therefore the immediate bottleneck is not application serialization or waiting
for scheduler admission. Continuous one-second AV prefills are mixed into the
same Thinker batches as multi-step decode, lengthening nearly every decode
forward; repeating those forwards pushes a model unit beyond its one-second
budget. GPU 0 busy p95 is only 62%, so this result does not prove raw SM
saturation. The measured runner interval also includes input preparation,
kernel execution, sampling, and worker IPC; improving mixed prefill/decode
runner efficiency remains an engine optimization opportunity.
