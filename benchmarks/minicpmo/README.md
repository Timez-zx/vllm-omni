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

Real-time requires RTF `>= 1`. The strict capacity criterion requires every
observed Thinker and Talker unit to pass; one unit at RTF `< 1` is a capacity
failure. RTF p05 and miss rate are also reported to show how far the system is
from the boundary. Code2Wav uses one persistent request for the whole session,
including idle periods, so its request wall time is not a valid per-unit RTF
and is excluded from this SLO.

An RTF result is reportable only when the run is complete: no user fails and
every client input unit reaches each required Thinker stage. Completeness is a
measurement-validity check, not a second latency threshold; it prevents
dropped or stuck units from making the observed-unit RTF look artificially
good.

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
  --frame-max-side 0 --max-slice-nums 1 \
  --out /tmp/minicpm-pd-u8-30s.json

python benchmarks/minicpmo/analyze_rtf.py \
  --server-log /tmp/minicpm-pd-server.log \
  --run-json /tmp/minicpm-pd-u8-30s.json \
  --out /tmp/minicpm-pd-u8-30s-rtf.json
```

P/D uses the same stage-RTF capacity criterion as the non-P/D deployment.
Input-ready through D completion remains a latency diagnostic, but is not a
capacity gate because P and D are separate pipeline stages. The analyzer
excludes setup and autonomous post-stream continuation slots from the client
input measurements. Earlier 30-second HD4 screening runs on the serial
video-CPU path give:

| Users | Seed | P max | D max | Talker max | RTF misses | Diagnostic P→D p99 | RTF result |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 9 | 20260828 | 372 ms | 323 ms | 363 ms | 0 | 567 ms | pass |
| 10 | 20260828 | 691 ms | 600 ms | 404 ms | 0 | 1063 ms | pass |
| 9 | 20260829 | 481 ms | 353 ms | 329 ms | 0 | 659 ms | pass |
| 10 | 20260829 | 706 ms | 577 ms | 408 ms | 0 | 1029 ms | pass |

P/D pre-registers D destination blocks while P computes and emits only D's
newly scheduled hidden rows; it does not rebuild a full historical side-output
cache. Both 30-second ten-user traces pass the RTF-only criterion. Their P→D
p99 exceeds one second when phases collide, but that path is now diagnostic.

The current implementation batches both sides of video preparation. Eight
persistent CPU workers decode JPEG payloads and run `process_image` across
sessions. GPU inputs are then bucketed by exact pixel-tensor shape and target
patch grid and encoded with a default microbatch of 8. Outputs are scattered
back to their original request/frame/slice positions; any batch failure falls
back to the request-at-a-time path. The default can be overridden with
`MINICPMO45_VISION_ENCODER_BATCH_SIZE`.

The formal long run preserves the 960x540 source, uses one global vision block
per frame (`79` steady scheduler tokens per one-second model unit), loops the
real MP4, and runs 180 seconds plus 60 seconds of drain time. Each boundary
point uses a clean server and a four-second single-user slice1 warm-up first:

```bash
python benchmarks/minicpmo/continuous_av.py \
  --users 1 --duration-s 4 --phase-window-s 0 --seed 20260829 \
  --connect-stagger-s 0 --post-stream-s 10 --close-timeout-s 30 \
  --gpus 0 1 2 3 --loop-media \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 1 \
  --out /tmp/minicpm-pd-slice1-warmup.json

python benchmarks/minicpmo/continuous_av.py \
  --users 17 --duration-s 180 --phase-window-s 1 --seed 20260839 \
  --connect-stagger-s 0.5 --post-stream-s 60 --close-timeout-s 30 \
  --gpus 0 1 2 3 --loop-media \
  --media /path/to/MiniCPM-o-4_5/assets/omni_duplex1.mp4 \
  --ref-audio /path/to/MiniCPM-o-4_5/assets/HT_ref_audio.wav \
  --frame-max-side 0 --max-slice-nums 1 \
  --out /tmp/minicpm-pd-slice1-u17x180.json
```

Repeat at 18 users on another clean server with the same seed. Keeping the seed
fixed makes the 17-user phases an exact subset of the 18-user phases. Diagnostic
runs use `VLLM_USE_FLASHINFER_SAMPLER=0` because this host lacks `nvcc`.

With shape-bucketed GPU vision batching, the 13-user/30-second screen completes
all 390 P and D input units with no stage RTF miss. P/D/Talker p50/p95/p99 are
656/707/711, 437/516/550, and 309/505/522 ms. The same point before GPU
encoder batching had P p99 1251 ms and 115 P misses. A synchronized 8-user
probe forms a 7-request HD4 runner batch whose input preparation is 455–485 ms
(65–69 ms per image), compared with about 648 ms for request-ordered encoding.
The removed heterogeneous all-frame batch took about 1.33 s; exact-shape
bucketing avoids that padding path. These are diagnostic results, not the
formal 180-second capacity boundary.

The direct slice1 control uses the same synchronized 8-user, 12-second probe.
Both modes form a seven-request batch:

| Video mode | New tokens/request | Preparation | Model forward | Runner total | P service p50/p95/p99 |
|---|---:|---:|---:|---:|---:|
| HD4 | 211 | 481 ms | 77.6 ms | 579 ms | 688/748/760 ms |
| slice1 | 79 | 251–253 ms | 33.0–33.1 ms | 301–304 ms | 356/459/476 ms |

Slice1 cuts both preparation and total runner time by about 48%. The reported
preparation is for the whole seven-request batch, about 36 ms per request, and
also includes audio encoding, media assembly, and runner input construction;
it is not pure Vision Encoder time. All 96 P/D input units complete with no RTF
miss.

The clean 180-second boundary is:

After the Talker stop fix, seed-varied 30-second screens pass at 40 users and
fail at 41 due to five Talker units just over one second. This is only a
short-context throughput screen; it does not define an adjacent fixed-seed
boundary or replace the long run below.

| Users | P service p50/p95/p99/max | D service p50/p95/p99/max | Talker service p50/p95/p99/max | RTF misses P/D/Talker | Result |
|---:|---:|---:|---:|---:|:---:|
| 17 | 110/182/205/249 ms | 123/286/646/983 ms | 61/284/423/493 ms | 0/0/0 | pass |
| 18 | 114/184/209/382 ms | 126/355/663/1295 ms | 66/346/583/799 ms | 0/10/0 | fail |

The strict long-session capacity is 17 sessions under seed `20260839`. The old
stage-2 `min_tokens=50` setting was invalid for native duplex: MiniCPMTTS already
limits a codec chunk to 26 steps, while the outer sampler only selects a binary
continue/stop row. Masking stop until step 50 forced 51 useless Talker forwards.
Stage 2 now uses `min_tokens=0`; measured Talker output is 1–26 tokens and has no
RTF miss at either boundary point.

The new limit is Thinker-D long-context decode. All ten 18-user misses occur in
one late burst at about 13,973 context tokens. D scheduler queue p99/max is only
4/10 ms, but runner p99/max reaches 514/1047 ms and decode-runner p99/max reaches
438/944 ms. The burst forms a 14-request batch; 17 users form at most 12 and
remain below one second. P has no miss and a 209 ms p99, so multimodal
preprocessing and P/D transfer are not the boundary. The identical looped AV
trace aligns semantic long-decode positions across users, which is why adjacent
capacity points must use the same phase seed. Current results are archived in
`results/capacity_pd_slice1_optimized_4gpu_20260829.json`.

The long-run lifecycle bug is fixed: an append may commit after the session
fence advances to that append's exact same-epoch target, while input-sequence,
epoch, incarnation, and later-fence changes remain stale. D completion is used
as the completeness witness because resumable P logging can coalesce middle
completion records. One eight-user run hit malformed MessagePack IPC after
reusing a server for two preceding long tests; it is excluded, and the clean
server rerun completed. The superseded serial-fallback archive is
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
