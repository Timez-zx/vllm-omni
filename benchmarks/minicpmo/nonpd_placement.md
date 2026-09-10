# MiniCPM-o 4.5: matched non-P/D module placement

## Question and setup

Compare **all modules on one GPU** against **three GPUs: Encoder / Thinker+Talker / Code2Wav**. Thinker prefill and decode stay together in both cases. Use the same working tree and native duplex pipeline, not an old non-P/D revision versus a newer P/D revision.

| Topology | GPU 0 | GPU 1 | GPU 2 |
|---|---|---|---|
| `all1` | Vision + Audio Encoder + Thinker + Talker + Code2Wav | unused | unused |
| `split3` | Thinker + Talker | Vision + Audio Encoder | Code2Wav |

- Hardware: RTX PRO 6000 Blackwell Server Edition, 96 GiB per GPU.
- Model: `openbmb/MiniCPM-o-4_5`.
- Thinker/Talker weights and KV: FP8. Encoder/Code2Wav retain their native dtype.
- Fixed KV budgets in both topologies: Thinker 32 GiB, Talker 2 GiB. Placement must not implicitly change cache capacity through free-memory profiling.
- Asynchronous vision/audio sidecars, encoder batch limit 16, no arrival-cache entry limit. Native per-session recurrence and downstream audio pipeline remain enabled. Engine scheduling is synchronous in both topologies.
- Configs: `deploy_capacity_nonpd_colocated_fp8.yaml` and its placement-only child `deploy_capacity_nonpd_separated_fp8.yaml`.

## Workload and interpretation

`continuous_av.py --completion-mode nonpd --workload-profile production` sends real audio in 200 ms chunks and one 960×540 video frame each second. `max_slice_nums=4`; model input units are 1 s. Users have seeded random phases, ±50 ms arrival jitter, and different aligned AV offsets within the looping source clip. Equal user counts use identical seeds/media/offsets in A and B.

Short runs only check functionality and locate a boundary. Long runs last **360 s from empty context**, with a 36,000-token rollover trigger and 40,960-token limit. Verify rollover in the trace, not merely elapsed duration. The source is a repeated 35 s clip, not a diverse open-world conversation corpus.

The non-P/D mode has no physical-D completion witness. `analyze_nonpd_placement.py` instead matches real input-unit identities in scheduler admission records to **semantic Thinker unit completion**. A GPU forward alone is not completion. Match queued updates FIFO within each request: the logged admission-generation counter can advance before an earlier unit completes, so it must not be used to deduplicate completion events. Each expected input must have its own completion.

- Thinker RTF = input seconds / (last Thinker unit completion − first client media arrival).
- A separate pipeline-finish ratio includes the latest Talker completion or received audio. This is diagnostic only: the model can automatically continue speaking after the input stream ends, so it is not an input-capacity pass criterion.
- **Capacity requires no input backlog at any time:** at each next complete 1 s input-unit arrival, all preceding units of that session must have completed. Check actual client readiness times, including jitter; 200 ms audio chunks and one currently executing unit are not backlog. The final unit must complete within 1 s of readiness. No startup exclusion or grace period. Compression followed by catch-up does not rescue a failed run, even if terminal RTF is approximately 1. Report RTF as a diagnostic, not the sole pass criterion.
- Coalesced/skipped model units or unfinished admitted Talker work cannot qualify as a pass. Also reject same-session Talker admission while earlier Talker work is unfinished during the complete-input streaming interval; exclude post-input automatic continuation from that queue audit. Inspect audio playback separately; no input backlog is not a certified audio-quality SLO.
- A run with no audio cannot establish full-module capacity. Do not treat input-only throughput as the requested result.
- These are scheduler-traced placement comparisons, not diagnostic-free production capacity certificates. Report both users per deployment and GPU count; 3× more hardware is not a free improvement.

## Reproduce

Run from the repository with the `omni` environment and `PYTHONPATH` pointing to this working tree:

```bash
python benchmarks/minicpmo/run_placement.py \
  --topology all1 --users 1 --duration-s 35 --drain-s 30 \
  --sweep-users 2 4 8 --sweep-duration-s 360 --stop-on-failure \
  --out-dir /home/ubuntu/data/experiments/<unique-run-directory>
```

Repeat with `--topology split3` and a different output directory. The initial 35 s run validates speech and warms the server before capacity points; its cold-start latency/backlog is reported separately, not passed off as a successful capacity run. Each long capacity point uses new sessions from empty context and counts every unit, including startup and rollover. Use distinct user counts within a sweep so session IDs do not collide. Refine skipped intermediate counts near the boundary in separate runs.

The runner refuses occupied target GPUs, records exact commands/runtime provenance, and stops only the process group it started. Keep `server.log`, `provenance.json`, `commands.json`, each `run.json`, and `analysis.json` together. Inspect audio events, errors, unit completeness, and context rollover before accepting a result.

## Bring-up findings

The current vLLM 0.26 environment exposed non-P/D compatibility defects: the eager IPC reader retained a decoder without its output-tensor provider; compact metadata violated the tensor wire contract; and terminal decode steps could retain a previous listen segment's IDs when no new prompt snapshot was emitted. These paths were corrected and covered by regression tests. Runs before normal audio output was restored are invalid for capacity comparison.

The first long run also exposed cumulative hidden-state growth in the non-P/D output path. An API stack sample stopped in `torch.cat` during accumulated-output construction. Native Thinker now emits once per complete unit and releases the consumed unit's latent rows **after** making its immutable output snapshot. KV, model history, and context rollover are unchanged. Ordinary cumulative callers without the native segment contract keep their behavior. Use only matched post-fix A/B runs for the placement result.

## Current measurements (2026-09-08)

All capacity points below use the corrected output path, a warmed server,
360 s of new-session input, and two observed context rollovers per user.
Latency is **complete 1 s input ready → Thinker semantic unit completion**,
not TTFA or complete audio playback. An accumulation event means a new complete
input unit arrived while an older unit of that session was unfinished.

| Placement | Users | p50 / p95 / p99 (ms) | Maximum (ms) | Accumulation events | Maximum older unfinished units | Pass |
|---|---:|---:|---:|---:|---:|---|
| All modules, 1 GPU | 2 | 162 / 440 / 567 | 591 | 0 | 0 | Yes |
| All modules, 1 GPU | 3 | 223 / 826 / 1,023 | 1,361 | 14 | 1 | No |
| All modules, 1 GPU | 4 | 210 / 1,559 / 2,929 | 4,260 | 127 | 4 | No |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 2 | 159 / 356 / 469 | 523 | 0 | 0 | Yes |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 3 | 174 / 486 / 601 | 630 | 0 | 0 | Yes |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 4 | 156 / 396 / 607 | 714 | 0 | 0 | Yes |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 5 | 165 / 383 / 636 | 753 | 0 | 0 | Yes |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 6 | 166 / 500 / 795 | 961 | 0 | 0 | Yes |
| Encoder / Thinker+Talker / Code2Wav, 3 GPUs | 7 | 170 / 784 / 1,436 | 2,336 | 65 | 2 | No |

The observed no-accumulation boundary is **2 users on 1 GPU versus 6 users
on 3 GPUs**. The next tested counts, 3 and 7 respectively, fail. This is
3× deployment capacity using 3× GPUs, not proof of higher per-GPU efficiency.
These are fixed-seed results, not a guarantee over arbitrary conversations
or phases. The 6-user maximum of 961 ms leaves little deadline margin.

The Talker admission audit also found **0** queued same-session segments in
every passing point. The failing 1-GPU 3-/4-user points additionally had
**4 / 36** Talker arrivals with earlier same-session work unfinished. This
does not by itself establish Talker as the original bottleneck: upstream
catch-up can bunch downstream arrivals.
The failing 3-GPU 7-user point had **10** such Talker arrivals, in addition
to its 65 input accumulation events. All its real input units eventually
completed, but recovery does not qualify it as a pass.

The earlier corrected 1-GPU 8-user run illustrates why terminal RTF is
insufficient: RTF minimum **0.99953**, all 2,880 units eventually completed,
but **1,242 accumulation events**, at most **62 older unfinished units**,
and **62.08 s** worst ready-to-completion latency. It fails even though
compression allowed the sessions to catch up. The 12-user run also fails;
only 3,566 of 4,320 units finished within the observation window.

Equal user counts match media offsets, phases, jitter, precision, sampling
settings, and cache budgets. Generated output lengths are not locked: at
2 users, Thinker generated 3,215 tokens on 1 GPU and 3,118 on 3 GPUs.
At 3 users the totals were 5,281 / 5,087; at 4 users, 6,554 / 6,209.
Consequently this is a real-input placement comparison, not an equal-kernel-work
microbenchmark proving that every latency difference is resource contention.

Raw artifacts under `/home/ubuntu/data/experiments/minicpm-nonpd-placement-20260908/`:

- `all1-nobacklog-v3/{2x360,4x360}`; shared trace in the parent directory.
- `all1-nobacklog-3-v3/3x360`; shared trace in the parent directory.
- `split3-nobacklog-v3/{2x360,3x360,4x360,5x360,6x360}`; shared trace in the parent directory.
- `split3-nobacklog-upper-v3/7x360`; shared trace in the parent directory.
- `all1-long-v2/{8x360,12x360}/analysis-no-backlog.json` re-audits the earlier
  corrected long runs using the no-accumulation criterion.

For completed boundary points, use `analysis-audited.json`, which includes
the later Talker queue audit without rerunning or changing the service.

The separate cold-start 1×35 probes show approximately 1.9–2.3 s first-unit
delay and one or two accumulated arrivals. They are **not** capacity passes. The
table evaluates an already warmed service; it does not claim cold-start SLO
compliance. Regression verification: 110 targeted tests passed.

## Which colocated module slows Thinker? (2026-09-08)

**Code2Wav is the largest observed contributor to the all-on-one-GPU tail.**
This is a placement/interference result, not evidence that Thinker alone can
serve only two users or that SM/register capacity is exhausted.

Leave-one-out controls use the same four users, 180 s of input, seed, AV
offsets/phases, precision, fixed KV budgets, slice count, and pipeline. Each
user completes all 180 real input units and crosses one context rollover
(peak approximately 33k tokens). No model stages are disabled. Latency below
is **complete input ready → Thinker unit completion**, not TTFA.

| Placement | p50 / p95 / p99 (ms) | Maximum (ms) | Input accumulation events | Talker accumulation events |
|---|---:|---:|---:|---:|
| All modules on GPU 0, new baseline | 222 / 1,342 / 2,699 | 3,701 | 54 | 22 |
| Only Encoder moved to GPU 1 | 206 / 710 / 1,155 | 1,398 | 13 | 9 |
| Only Talker moved to GPU 1 | 213 / 756 / 1,331 | 2,028 | 19 | 0 |
| Only Code2Wav moved to GPU 2 | 156 / 303 / 472 | 560 | **0** | **0** |

Only the Code2Wav move clears the no-accumulation criterion in these controls.
Moving Encoder or Talker also helps, so the effects are not exclusive or
additive. This is a diagnostic comparison, not a new multi-seed capacity
certificate. The earlier 360 s capacity boundary remains 2 versus 6 users.

Native sampled responses are not bit-identical: real-input Thinker output
totals are 3,103 / 2,722 / 2,925 / 2,692 tokens in the table's order. However,
Encoder-out and Code2Wav-out differ by only 1.1% in token count. Further,
20 long-context pairs match session, input-unit index, exact output-token
count, and context length within 512 tokens. Moving only Code2Wav reduces
their **engine-admission → unit-completion mean from 503 to 108 ms**.
This guards against interpreting shorter responses as the entire effect;
it does not make the whole generated workloads identical.

### Direct GPU evidence

A separate all-colocated **3×180** diagnostic captures 20 s with Nsight
Systems 2025.3.2, beginning at real-input unit 125. Capture is inactive for
the preceding 4-user baseline, although the Nsight injection is loaded.
The separate 3-user trace is not used as a capacity certificate.

The captured processes are Thinker+Encoder PID 123575, Talker PID 124272,
and Code2Wav PID 125116. Within the first process, stream 29 contains the
Thinker's paged-KV attention kernels; stream 7 contains the encoders.
Measured kernel interval unions in the 20.093 s window are Thinker 7.656 s,
Code2Wav 6.963 s, Encoder 2.496 s, and Talker 0.925 s. These are kernel
**wall-clock spans**, possibly including preemption, not SM-active fractions,
FLOPS, bandwidth utilization, or register pressure.

A conservative queued-work witness, correlation ID **982834**:

| Event | Trace-relative time |
|---|---:|
| Thinker kernel launch API returned | 7.845140773 s |
| Previous same-stream GPU work finished | 7.851201862 s |
| Next Thinker kernel started | 7.853561638 s |

After excluding earlier same-stream kernels/copies/memsets, the already
submitted kernel still waits **2.360 ms**. No recorded same-stream event wait
overlaps this interval. Code2Wav kernel spans cover **1.908 ms**, Encoder
0 ms, and Talker 0.002 ms of this gap. This rules out "no Thinker work had
arrived" for this witness. The analyzer excludes CUDA-graph nodes from these
direct-launch witnesses. It does not attribute every millisecond of E2E tail
to this one gap or infer a specific exhausted hardware resource.

Implementation context: Thinker, Talker and Code2Wav use separate CUDA
processes; MPS was not running. Application-level pipelining therefore does
not imply concurrent execution across those CUDA contexts. NVIDIA documents
ordinary multi-context scheduling as time-sliced:
[MPS architecture](https://docs.nvidia.com/deploy/mps/latest/architecture.html).
Code2Wav also performs ten flow-matching steps plus HiFT in its native
precision (`token2wav_float16=false`), not a trivial audio-token lookup.
The ablations and queued-work witness support GPU-sharing interference;
they do **not** establish warp/register saturation or measure an MPS fix.

Artifacts under `/home/ubuntu/data/experiments/minicpm-nonpd-contention-20260908/`:

- `{encoder-out-4x180-v1,talker-out-4x180-v1,code2wav-out-4x180-v1}/4x180/`:
  `run.json`, `analysis.json`, `analysis-units.json`; server traces and
  provenance in each parent directory.
- `all1-baseline-and-trace-v1/4x180/`: fresh baseline, same files.
- `all1-baseline-and-trace-v1/3x180/`: separately profiled diagnostic.
- `all1-baseline-and-trace-v1/nsys-3user-longctx.nsys-rep`, `.sqlite`,
  `.control.json`, `nsys-queue-analysis.json`, and `thinker-code2wav-queued.png`.

Reproduce the placement controls with the earlier `run_placement.py` command,
using `--topology encoder_out|talker_out|code2wav_out`, `--debug-handoff`,
`--sweep-users 4 --sweep-duration-s 180`, and a unique output directory.
For the trace, launch the all1 runner under:

```bash
nsys launch --session-new=minicpm-colocate-contention-20260908 \
  --trace=cuda,nvtx --cuda-event-trace=false --cuda-graph-trace=node --wait=all \
  env PYTHONPATH=/home/ubuntu/vllm-omni python benchmarks/minicpmo/run_placement.py \
  --topology all1 --users 1 --duration-s 35 --drain-s 30 --debug-handoff \
  --sweep-users 4 3 --sweep-duration-s 180 --out-dir <unique-directory>
```

Once `server.log` exists, `capture_placement_trace.py --session <name>
--server-log <path> --out <report-stem> --users 3 --unit 125 --capture-s 20`
starts/stops capture without changing model behavior. Export with
`nsys export --type=sqlite`; use `analyze_placement_nsys.py` with the **new
trace's actual PIDs and stream IDs**, then `plot_placement_nsys.py` with a
correlation ID from its audited witnesses. Exact capture commands and epochs
are retained in `.control.json`. No production runtime optimization, MPS
configuration change, or precision change was made for this investigation.

## MPS follow-up: same GPU, same pipeline

`run_placement.py --topology all1 --mps` starts a **private, same-user MPS
instance on GPU 0**. All three stage processes must appear in its actual
client list before measurement. The daemon and clients select the physical
GPU by UUID; leaving client visibility unrestricted makes PyTorch's NVML
device count disagree with the MPS-visible CUDA device count.

MPS uses 100% default active threads, with no SM partition, priority change,
precision change, or engine-scheduling change. Other GPUs and the system's
default MPS socket are untouched. The runner stops its own stage processes
and then its private MPS instance; `mps.json` retains attachment checks and
lifecycle commands. Fixed KV budgets and the workload above are unchanged.

```bash
python benchmarks/minicpmo/run_placement.py \
  --topology all1 --mps --users 1 --duration-s 35 --drain-s 30 \
  --debug-handoff --sweep-users 3 4 --sweep-duration-s 360 \
  --out-dir /home/ubuntu/data/experiments/<unique-mps-directory>
```

Capacity captures have **no Nsight injection or collection**. A separate
diagnostic run may use the earlier Nsight command with `--mps`; do not use
its profiled interval to certify capacity. Ordinary kernel timestamps and
MPS speedup alone do not establish register, warp, or bandwidth saturation.

### 360 s results

| Users | MPS | Input-ready → Thinker-done p50 / p95 / p99 | Maximum | Input backlog events | Talker backlog events | Pass |
|---:|---|---:|---:|---:|---:|---|
| 3 | off | 223 / 826 / 1,023 ms | 1,361 ms | 14 | 4 | no |
| 3 | on | 172 / 433 / 567 ms | 640 ms | 0 | 0 | yes |
| 4 | off | 210 / 1,559 / 2,929 ms | 4,260 ms | 127 | 36 | no |
| 4 | on | 164 / 505 / 664 ms | 737 ms | 0 | 0 | yes |
| 5 | on | 170 / 465 / 707 ms | 856 ms | 0 | 0 | yes |
| 6 | on | 177 / 547 / 819 ms | 1,318 ms | 6 | 3 | no |
| 8 | on | 205 / 1,934 / 6,054 ms | 8,842 ms | 230 | 26 | no |

All real inputs complete; every session crosses **two context rollovers**.
The completed five-user follow-up resolves the measured no-backlog boundary:
**five users pass; six users fail**, compared with two users without MPS.
This is the observed boundary for this workload/seed and 360 s horizon, not
a phase-independent hardware limit or a repeated-trial reliability guarantee.
The finite-horizon terminal RTF can be fractionally below one (minimum
0.99973 / 0.99994) despite no pending previous unit at any next arrival;
the explicit arrival/backlog audit, not rounding that ratio, determines pass.

The input configurations, media offsets, seeded phases, and unit counts
match exactly. Generated responses are not bit-identical: Thinker output
tokens are 5,281 → 4,462 at three users and **6,554 → 6,663** at four users;
observed audio chunks are 254 → 327 and 257 → 333. Thus the four-user gain
does not come from reducing total Thinker output-token work or disabling
audio. These counts do not guarantee identical acoustic workloads.

The improvement is predominantly **after Thinker engine admission**:

| Users | Mean input-ready → admission, off → on | Mean admission → done, off → on | Admission → done p99, off → on |
|---:|---:|---:|---:|
| 3 | 102 → 91 ms | 208 → 108 ms | 920 → 474 ms |
| 4 | 108 → 89 ms | 302 → 115 ms | 2,691 → 535 ms |

Matching the same session/input-unit index, identical output-token count,
and context within 512 tokens (off context ≥24k) yields 54 three-user pairs
and 76 four-user pairs. Their mean admission-to-done time falls from
478 → 240 ms and **914 → 251 ms**, respectively. This limits response-length
confounding but does not equalize other simultaneously generated work.

Together with the earlier Code2Wav-only placement control, this establishes
that the old two-user boundary was not an intrinsic Thinker compute limit:
cross-process GPU sharing without MPS was a substantial avoidable cost.
Application pipelining was already enabled; MPS changes GPU execution sharing,
not session recurrence or prefill/decode dependencies.

Artifacts: `/home/ubuntu/data/experiments/minicpm-nonpd-mps-20260908/`:

- `all1-mps-3-4x360-v2/{3x360,4x360}/`: `run.json`, `analysis.json`,
  `analysis-units.json`; exact commands, server trace, provenance and
  `mps.json` in the parent.
- `mps-comparison.json`: old/new paths, stage decompositions, and every
  matched long-context pair used above.
- `all1-mps-3-4x360-v1/`: failed startup caused by the device-visibility
  mismatch; **not a workload or capacity result**. Its private MPS instance
  was shut down before the corrected run.

### Capacity boundary with two-user steps

At the user's request, stop the unfinished five-user run and test **8 → 6**,
reusing the same warm server and private MPS instance. The cancelled five
sessions are aborted before the eight-user point starts. The sweep controller
alone is paused during this handover; the server, stages and MPS remain running.
No precision, pipeline, workload, cache budget, or engine setting changes.
Neither new capacity point uses Nsight or per-step runner timing probes.

Both new points run for **360 s**, complete every real input (2,880 / 2,160),
and cross two context rollovers per session. Maximum observed context is
34,016 / 33,595 tokens. At two-user resolution, the verified boundary is
**four users pass, six users fail**; eight users also fail. Five users were
initially interrupted and unclassified. The separate complete follow-up below
resolves that missing point; the cancelled run remains excluded.

Six users are borderline but fail the strict criterion: the p99 is below
1 s, yet six real arrivals find a preceding unit unfinished (about 39–281 ms
late), and three Talker admissions find earlier work unfinished. The slowest
unit is session 2, unit 311, context 33,499, 20 output tokens: **331 ms before
Thinker admission + 988 ms after admission = 1,318 ms**.

Eight users fail clearly: 230 input-backlog events, up to eight older
unfinished units, and 26 Talker-backlog events. The worst real unit reaches
**8,842 ms**, including 8,546 ms after admission. The third-minute p99 is
8,328 ms, versus 344 ms in the fourth minute after rollover. Terminal RTF
near one does not erase earlier backlog. These admission-to-done spans
include queueing behind earlier work; they are not individual GPU-forward
measurements or proof of a particular saturated hardware resource.

Artifacts are in `all1-mps-capacity-from5-v1/` under the MPS experiment root:

- `{8x360,6x360}/`: exact client commands, `run.json`, `analysis.json`, and
  `analysis-units.json`; shared `server.log` and provenance in the parent.
- `even-sweep.json`: revised policy, actual order, MPS client checks, and
  final pass/fail bracket; `even_sweep.py` archives the one-off warm-server
  handover (its original PID values are not portable reproduction inputs).
- `5x360/`: cancelled input/client logs, **not a failed or passed capacity
  point**. The original controller's eventual error is the expected cancelled
  client exit, not a failure of the six-/eight-user server. Its normal cleanup
  shuts down the server and private MPS after both completed measurements.

To reproduce the measured even points on a fresh warmed server, use the
earlier MPS launch command with `--sweep-users 8 6 --sweep-duration-s 360`
and **without** `--stop-on-failure`, so the failed eight-user point does not
prevent the six-user point from running. Keep a unique output directory.

### Completed five-user follow-up

Run only five users on a fresh server after the same 35 s warm-up, retaining
all earlier settings. All **1,800 real input units** complete, every session
crosses **two context rollovers**, and maximum context reaches 33,909 tokens.
There are **zero input-backlog events and zero Talker-backlog events**, with
no client/teardown errors. All admitted Talker work completes and 359 audio
chunks are observed. Thinker produces 7,612 measured output tokens.

Input-ready → Thinker-done p50/p95/p99 is **170/465/707 ms**, maximum
**856 ms**. The slowest unit is session 0, input 156, context 33,507, with
20 output tokens: 208 ms before admission and 648 ms after admission.
Under the strict no-backlog criterion this point **passes**; the six-user
point above fails, establishing the measured five-user boundary.

Artifacts: `all1-mps-5x360-complete-v1/5x360/` under the MPS experiment root
contains `run.json`, `analysis.json`, `analysis-units.json`, and exact client
commands. The parent contains server logs, configuration provenance and
`mps.json`, confirming all three stage clients before/after measurement and
private MPS shutdown. Reproduce with the earlier MPS command, changing only
`--sweep-users 5` and the unique output directory. The interrupted five-user
attempt in `all1-mps-capacity-from5-v1/` is not used for this result.

### Three-GPU MPS follow-up

Each point runs for **360 s** with two context rollovers in every session.
All 2,160 / 2,520 real input units complete with normal audio and no client
or teardown errors. Latency remains input-ready → Thinker unit completion.

| Users | MPS | p50 / p95 / p99 (ms) | Maximum (ms) | Input backlog | Talker backlog | Pass |
|---:|---|---:|---:|---:|---:|---|
| 6 | off | 166 / 500 / 795 | 961 | 0 | 0 | yes |
| 6 | on | 159 / 427 / 668 | 824 | 0 | 0 | yes |
| 7 | off | 170 / 784 / 1,436 | 2,336 | 65 | 10 | no |
| 7 | on | 164 / 593 / 904 | 1,221 | 10 | 2 | no |

**MPS reduces the observed tail but does not change this three-GPU
zero-backlog boundary: six users pass, seven fail.** Stop at the failed
seven-user point; the planned nine-user point is not run. With MPS enabled
in both placements, the measured boundary is **five users on one GPU versus
six users on three GPUs**, for this workload/seed and measurement horizon.

The worst seven-user unit is session 0, input 309, context 33,204 tokens,
20 output tokens: **168 ms before Thinker admission + 1,053 ms after
admission = 1,221 ms**. A p99 below 1 s does not override the ten actual
input-backlog events. These spans include engine waiting and execution;
they do not isolate prefill, decode, or a saturated GPU resource.

Input configurations, phases, media offsets, scheduled jitter and counts
match the old runs exactly. Generated work differs: six-user Thinker output
tokens are 9,487 → 8,824 (audio chunks 448 → 341), while seven-user output
tokens are **11,641 → 11,692** (audio chunks 388 → 458). The seven-user tail
improvement is not explained by fewer total Thinker output tokens or missing
audio, but this is not a bit-identical replay or a repeated-trial guarantee.

#### Reproduction and startup verification

Use `--topology split3 --mps`: **GPU 0 Thinker+Talker, GPU 1 Vision+Audio
Encoder, GPU 2 Code2Wav**. Keep the same FP8 weights/KV, fixed cache budgets,
native encoder/Code2Wav precision, sidecar pipeline and 360 s workload.
MPS has 100% default active threads on all three selected GPUs; GPU 3 and
the system MPS socket remain untouched. This does not enable P/D separation.

```bash
python benchmarks/minicpmo/run_placement.py \
  --topology split3 --mps --users 1 --duration-s 35 --drain-s 30 \
  --debug-handoff --sweep-users 6 7 9 --sweep-duration-s 360 \
  --stop-on-failure \
  --out-dir /home/ubuntu/data/experiments/<unique-split3-mps-directory>
```

Multi-GPU startup required two device-identity corrections, neither of which
changes runtime scheduling or model computation:

- The generated `deploy-mps.yaml` preserves the complete stage environment
  and resolves the auxiliary encoder GPU to a UUID too. Mixed UUID/ordinal
  visibility can hide that second device from PyTorch.
- Initialization locks resolve UUIDs to physical GPU indices. Previously,
  UUID parsing fell back to device 0 for different stages: one initializer
  held that device lock, while another waited for it holding the spawn lock,
  blocking the first initializer from launching. Ordinal launches and
  post-startup execution are unchanged.

Artifacts: `split3-mps-capacity-v3/` under the MPS experiment root above.
`mps-device-verification.json` confirms Thinker and Talker on GPU 0, the
Thinker process's encoder context on GPU 1, and Code2Wav on GPU 2, all as
actual MPS clients. `comparison.json` audits matching input configurations,
phases, media offsets, scheduled jitter and input counts against the earlier
non-MPS six-/seven-user runs; its script is archived alongside it.
`split3-mps-capacity-v1/` and `v2/` were stopped during startup, before any
workload, and are excluded from capacity results.

The benchmark/placement regression suite passes 28 tests. An additional
broader initialization-suite run passes 27 tests and fails its unchanged
`test_build_stage0_input_processor_uses_omni_input_preprocessor` fixture:
the mock configuration lacks `skip_tokenizer_init` required by vLLM 0.26.
That failure is outside the modified device-lock path.

### Nsight verification and remaining tail

The separate MPS diagnostic is `all1-mps-trace-3x180-v1/`. It captures
20 s starting at input unit 125, using the same Nsight version and CUDA-graph
node tracing as the earlier non-MPS three-user trace. It additionally enables
the existing non-blocking `VLLM_OMNI_LOG_PD_ITER=1`,
`VLLM_OMNI_LOG_STEP_GPU=1`, `VLLM_OMNI_LOG_RUNNER_DIAG=1`, and
`VLLM_OMNI_DIAG_STAGE=0` probes; it is **not a capacity certificate**.
No synchronizing AV-preparation probe is enabled.

Actual MPS client PIDs: Thinker+Encoder 145505, Talker 146229, Code2Wav
146909. Thinker is stream 29 and Encoder is stream 7. The captured windows
are both 20.094 s approximately. Apply the conservative direct-launch gap
audit above, including exclusion of graph nodes and recorded event waits:

| Thinker already-submitted gaps | MPS off | MPS on |
|---|---:|---:|
| Direct launches with correlated API records | 84,150 | 90,952 |
| Gaps ≥100 μs per 1,000 direct launches | 20.61 | 1.21 |
| Sum of those qualifying gaps | 678.21 ms | 33.00 ms |
| Gap p99, excluding recorded event waits | 406.2 μs | 40.4 μs |

This is a **95% reduction in the audited cumulative queued gap**, despite
more direct Thinker launches in the MPS window. It corroborates the long-run
improvement and previous placement ablations. These are diagnostic windows
with different generated work, not exact kernel-sequence replays. Rare waits
remain (MPS maximum 10.19 ms); neither MPS nor these data imply every kernel
can execute immediately or that resources never contend. This audit does
not measure hardware occupancy and does not account for all E2E latency.

The slowest unit in the MPS diagnostic is session 1, input 138: 29,471-token
context, 211 new prompt tokens, 20 output tokens, **658.32 ms** total. Its
non-overlapping wall-time breakdown is:

| Part | Time |
|---|---:|
| Complete input ready → Thinker admission | 125.30 ms |
| One prefill batch, including first-token sampling | 87.98 ms |
| Nineteen subsequent decode batches | 434.78 ms |
| Outside those target batches, after admission | 10.26 ms |

Thus the remaining tail sits primarily in repeated decode **execution and
sampling**, not hundreds of milliseconds between target batches. The 19
decode batches contain 56.97 ms of preparation; their forward CUDA-event
spans total 219.97 ms. Do not add GPU event time to all CPU wall times:
the GPU forward runs asynchronously and sampling then waits for its result.
For example, correlation 1708596 is an 8-byte device-to-host copy: the API
takes 14.97 ms but the physical copy takes about 1 μs. The API duration is
not 15 ms of PCIe transfer. This analysis does **not** claim the entire
434.78 ms is irreducible GPU forward or that residual engineering costs have
all been eliminated.

Keep `nsys-3user-longctx.nsys-rep`, `.sqlite`, `.control.json`,
`nsys-queue-analysis.json`, and `tail-runner-decomposition.json` in the
diagnostic directory. `nsys-off-queue-normalized.json` in the experiment root
re-analyzes the earlier non-MPS trace using the same updated gap analyzer.
