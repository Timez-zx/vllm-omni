# Reproducing "Audio+video: capacity 32, and the wall is vLLM's engine loop"

This document reproduces `workflow.md`'s AV-capacity finding at four points (16, 24, 32, 40
concurrent sessions, 3 seeds each: 7, 11, 23 — the same seed set the original capacity-32
measurement used) and extends its evidence chain with independently-derived measurements at
every load point instead of the doc's single 62-step snapshot.

**Setup**: `benchmarks/thinker_talker/deploy_2gpu_seq256.yaml` (thinker on GPU0, talker+code2wav
as separate processes on GPU1), `VLLM_OMNI_ADMIT_MAX_SESSIONS=<N>`, `VLLM_OMNI_POOL_SHARE=0.8`,
`VLLM_OMNI_CP_KV_CACHE=1`, `VLLM_OMNI_LOG_SCHED_STEPS=1`, `VLLM_OMNI_LOG_STEP_GPU=1`. Client:
`mu_bench.py --content synthetic --video-interval-ms 480 --turns 10 --audio-input-s 3
--think 2,6`, `MU_QUESTIONS=mixed`, `MU_STAGGER_S=0,40`. This is the doc's own "full setup
behind one capacity cell" recipe (workflow.md ~line 496), not `run_cell.sh`'s defaults, which
the script's own header says will not reproduce the numbers in the doc.

All timing analysis uses two facts about this deployment's logging: the client (`mu_bench.py`)
and the engine share one physical box's `CLOCK_MONOTONIC`, so `t_q` (client sends) and every
`mono=` timestamp in the engine log are directly comparable with no clock conversion; and the
engine emits `[turnprobe] recv`/`first-text`, `[OmniARScheduler] ADMIT`, and `[STEP-GPU]` lines
that make admission and per-step GPU time individually attributable to a specific turn, once
matched carefully (a turn is matched to an event only when exactly one candidate falls in an
80ms window — ambiguous matches are dropped, not guessed at). Match rate across all cells:
2,787-3,709 of ~3,360-4,800 turns depending on the exact metric (76-83%); the remainder are
genuinely ambiguous under concurrent traffic, not silently misattributed.

---

## Capacity table

Criteria: TTFA p99 < 1000 ms, stutter silence p99 < 50 ms.

| users | seed | ttfa_p99 | stall_p99 | verdict |
|---:|---:|---:|---:|:---:|
| 16 | 7  | 638.9 ms | 0.00 ms | PASS |
| 16 | 11 | 525.1 ms | 0.00 ms | PASS |
| 16 | 23 | 562.2 ms | 0.00 ms | PASS |
| 24 | 7  | 808.0 ms | 0.00 ms | PASS |
| 24 | 11 | 864.0 ms | 0.00 ms | PASS |
| 24 | 23 | 851.3 ms | 0.00 ms | PASS |
| 32 | 7  | 877.5 ms | 0.00 ms | PASS |
| 32 | 11 | 920.3 ms | 0.00 ms | PASS |
| 32 | 23 | 847.8 ms | 0.00 ms | PASS |
| 40 | 7  | **1006.4 ms** | 0.00 ms | **FAIL** |
| 40 | 11 | 944.5 ms | 0.00 ms | PASS |
| 40 | 23 | 993.2 ms | 0.00 ms | PASS |

**Analysis.** Stutter is a clean 0.00ms at every single cell across all four loads — confirms
the doc's claim that stutter never binds at this scale; first audio is what fails, when
anything fails. 16/24/32 pass cleanly on all three seeds (9/9); 40 is the first point where a
seed crosses the line (7 FAILs at 1006.4ms, 11 and 23 pass at 944-993ms). This does **not**
match the doc's own sharp-wall claim ("32 passes on all three seeds; 36 fails on all three") —
on this hardware the boundary is fuzzier and sits closer to 40 than to 32. It is, however,
consistent with the doc's own noise analysis, which reports 17.5% run-to-run TTFA-p99 variance
specifically at 40 sessions and calls that "where it flips the verdict" — this run reproduces
exactly that instability, just with the specific seed that flips (7, not 23) differing from
whichever seed flipped in the doc's own measurement. Two earlier attempts in this investigation
(different from the numbers reproduced here) found u=32 itself flipping on seed 23 — the
takeaway across all attempts is that the real transition zone spans roughly 32-40, not a single
point, and any one run's exact pass/fail split at the edge should not be trusted without
multiple seeds, exactly as the doc itself warns.

---

## Evidence 1: do slow turns carry more of their own work?

"Slow" = TTFA > 500ms, matching the doc's own threshold. "Own work" here is each turn's
`prompt_tokens` value, read directly from its `ADMIT` log line — i.e. how large that specific
request's own prompt was, independent of anything else happening on the card.

| users | n fast | fast median prompt_tokens | n slow | slow median prompt_tokens | slow % of turns |
|---:|---:|---:|---:|---:|---:|
| 16 | 446 | 483 | 13  | 697 | 2.8%  |
| 24 | 557 | 482 | 72  | 258 | 11.4% |
| 32 | 665 | 483 | 112 | 222 | 14.4% |
| 40 | 742 | 484 | 180 | 222 | 19.5% |

**Analysis.** At u=16 slow turns do carry somewhat more of their own prompt (697 vs 483
tokens) — but from u=24 onward this inverts: slow turns' own prompts are *smaller* than fast
turns' (222-258 vs ~482-484), not larger. This is a stronger version of the doc's own finding
("they are not doing more work; the problem is not theirs") — at higher load, being slow is
actively *anti-correlated* with how much work the turn itself brought to the table. The
straightforward reading: at higher concurrency, whether a turn is fast or slow is governed
almost entirely by what it lands behind, not what it is. Also notable: the fraction of slow
turns rises monotonically with load (2.8% → 19.5%), the same trend the capacity table's
tightening margin already showed.

---

## Evidence 2: step end → admitted

Four columns, matching the doc's own table shape: time from arrival to admission (the whole
wait), time from arrival to when the nearest single step finishes, the leftover
(`step_end→admitted`), and that step's own token count.

| users | class | got→admitted | got→step_end | step_end→admitted | tokens in step |
|---:|:---|---:|---:|---:|---:|
| 16 | fast (n=446) | 14.0 ms  | 35.2 ms | **−19.8 ms** | 480  |
| 16 | slow (n=13)  | 15.8 ms  | 54.3 ms | **−11.2 ms** | 697  |
| 24 | fast (n=557) | 15.9 ms  | 30.6 ms | **−10.7 ms** | 272  |
| 24 | slow (n=72)  | 60.6 ms  | 43.7 ms | **+16.6 ms** | 682  |
| 32 | fast (n=665) | 17.3 ms  | 27.3 ms | **+5.9 ms**  | 272  |
| 32 | slow (n=112) | 86.0 ms  | 49.3 ms | **+34.0 ms** | 1444 |
| 40 | fast (n=742) | 19.8 ms  | 19.1 ms | **+8.2 ms**  | 272  |
| 40 | slow (n=180) | 113.0 ms | 60.2 ms | **+31.0 ms** | 1554 |

Doc's own values for comparison: fast −0.6ms / 20 tokens, slow −1.5ms / 2443 tokens — always
essentially zero.

**Analysis.** At u=16, `step_end→admitted` is negative for both classes (−12 to −20ms),
matching the doc's near-zero finding: admission really does happen (at worst) the instant the
nearest step finishes. From u=24 up this breaks down — the residual turns positive and grows
with load, reaching +34ms for slow turns at u=32. Directly counting how many `[STEP-GPU]` steps
actually elapse between arrival and the *real* `ADMIT` event (not just the nearest one) explains
why: median steps-before-admit is 0 at u=16 (most turns admitted at the very first step after
arrival) but shifts to 1 at u=24 and above, with a real tail reaching 8-13 extra steps for the
unluckiest turns. The doc's single-step model ("wait for the step you landed on, then you're
in") is the *low-load* special case of a more general mechanism: at higher concurrency, one
step usually is not enough to free an admission slot, because other already-admitted requests
are still consuming the shared per-step token budget across several consecutive steps. The
metric doesn't change; the number of steps it takes to satisfy it does.

---

## Evidence 3: the blocking step is prefill, not decode

For each turn, "the blocking step" is the first `[STEP-GPU]` entry (by completion time) at or
after the turn's arrival — the step it was structurally forced to wait behind. Its tokens split
into `decode = nreq` (one token per already-running request) and `prefill = ntok − nreq`
(everything else), matching the doc's own definition exactly.

### 3a. Tokens

| users | class | n | decode tokens (median) | prefill tokens (median) | prefill share |
|---:|:---|---:|---:|---:|---:|
| 16 | fast | 446 | 1.0 | 479.0  | 99.8% |
| 16 | slow | 13  | 1.0 | 696.0  | 99.9% |
| 24 | fast | 577 | 1.0 | 271.0  | 99.6% |
| 24 | slow | 76  | 3.0 | 790.0  | 99.6% |
| 32 | fast | 726 | 1.0 | 271.0  | 99.6% |
| 32 | slow | 124 | 5.0 | 1436.5 | 99.7% |
| 40 | fast | 830 | 1.0 | 269.5  | 99.6% |
| 40 | slow | 212 | 7.0 | 1436.5 | 99.5% |

### 3b. Time (proportional allocation of the blocking step's measured `gpu_ms`, split by each
token category's share of that step's total tokens — a step is one combined GPU forward pass,
so this is an estimate assuming roughly linear cost per token, not an independently measured
split)

| users | class | n | step_ms (median) | prefill_ms | decode_ms | prefill share |
|---:|:---|---:|---:|---:|---:|---:|
| 16 | fast | 446 | 42.7 | 42.57 | 0.13 | 99.7% |
| 16 | slow | 13  | 68.6 | 68.48 | 0.16 | 99.8% |
| 24 | fast | 577 | 36.3 | 36.14 | 0.22 | 99.4% |
| 24 | slow | 76  | 60.9 | 60.65 | 0.44 | 99.3% |
| 32 | fast | 726 | 34.9 | 34.79 | 0.46 | 98.7% |
| 32 | slow | 124 | 96.0 | 95.62 | 0.62 | 99.4% |
| 40 | fast | 830 | 29.9 | 29.72 | 0.76 | 97.5% |
| 40 | slow | 212 | 85.8 | 85.26 | 0.66 | 99.2% |

Doc's own value, for comparison: of the 62 steps that blocked a slow turn, p50 total tokens
2443, decode 14, prefill 2431 (99.5%).

**Analysis.** Decode is not a meaningful contributor to blocking-step composition at any load,
in tokens or in time — it stays essentially pinned near 1 token / under 1ms regardless of how
loaded the system is, while prefill is what scales (271→1554 tokens, 30ms→96ms, fast to slow,
low load to high). This independently reproduces the doc's headline conclusion ("this is
prefill blocking prefill, not prefill blocking decode") with a much larger, load-swept sample
(the doc's own evidence here was one 62-step snapshot; this is thousands of turns across four
load points) and a different measurement path (per-turn blocking-step attribution vs. the doc's
own step-size table), landing on the same number to within a percentage point. One difference
worth flagging: our fast-side prefill share stays ≥97.5% even at u=40, notably higher than it
might be expected to drop given decode contributes marginally more (0.76ms) at higher load —
the effect is real but small next to prefill's dominance.

---

## Evidence 4: the application layer vs. the engine

The doc's own evidence-4 table is a **segment timeline**: query received → handler entered →
compress/roll ladder → build chunk → into queue → picked up → admitted → first text token,
each segment tagged app or engine. We do not have this app's internal timers for the
`compress/roll ladder`/`build chunk`/`into queue→picked up` sub-segments — those need custom
instrumentation the doc's own author added that isn't present in this log. What we do have,
directly measured, are the network hop (`net`, client→server) and the two **engine** segments,
which in the doc's own table already carry the overwhelming majority of the total time.

| users | class | n | net (app, query→server) | **picked up → admitted** | **admitted → first text** | total |
|---:|:---|---:|---:|---:|---:|---:|
| 16 | fast | 446 | 0.20 ms | 14.0 ms  | 73.3 ms  | 87.6 ms  |
| 16 | slow | 13  | 0.29 ms | 15.8 ms  | 145.1 ms | 161.2 ms |
| 24 | fast | 557 | 0.20 ms | 15.9 ms  | 75.0 ms  | 91.1 ms  |
| 24 | slow | 72  | 0.90 ms | 60.6 ms  | 250.8 ms | 312.3 ms |
| 32 | fast | 665 | 0.27 ms | 17.3 ms  | 77.5 ms  | 95.1 ms  |
| 32 | slow | 112 | 2.19 ms | 86.0 ms  | 271.0 ms | 359.2 ms |
| 40 | fast | 742 | 0.28 ms | 19.8 ms  | 77.7 ms  | 97.8 ms  |
| 40 | slow | 180 | 1.46 ms | 113.0 ms | 260.7 ms | 375.2 ms |

Doc's own slow-turn values, for comparison: app-layer segments sum to 34.4ms, `picked up→admitted`
=178ms, `admitted→first text`=331ms, total=543.4ms, app share=6.3%.

**Analysis.** The app layer is even more negligible here than in the doc — our measurable proxy
(`net_ms`) is 0.18-0.61% of the total, versus their 6.3% — though that comparison isn't quite
apples-to-apples, since their 6.3% includes `build chunk` (34ms), a real app-side cost invisible
to this log, so our number understates the true app-layer share rather than proving it's
genuinely smaller. What *is* directly comparable is the shape: in every cell, `admitted→first
text` dominates `picked up→admitted` by roughly 3-5x (73ms vs 14ms fast at u16, 261-271ms vs
86-113ms slow at u32/40) — the same structural point the doc's own 331ms-vs-178ms split makes
(~1.9x there), just proportionally larger here. Both agree: most of the wait happens *after*
admission, not before it — the compute itself (chunked prefill spanning multiple steps, per
evidence #2's steps-before-admit count) is the bigger cost, not the admission gate. The
slow-turn totals also track the capacity table cleanly: 161ms at u16 (comfortably inside
budget) climbing to 359-375ms at u32/40 (where the wall starts to bite).

---

## Evidence 5: does the one latency knob help?

**Not reproduced.** The doc's finding here — sweeping `max_num_batched_tokens` from 16384 down
to 512, showing the blocking-step interval shrinks but TTFA p99 degrades monotonically by 60%
regardless of direction — requires new engine boots at each `mbt` value, which this
investigation did not run. This is the one piece of the doc's evidence chain left untested by
this reproduction.

---

## Additional finding: GPU time is linear in step tokens, fit fresh from this data

```
gpu_ms ≈ 6.94 + 0.0507 ms/token × tokens        (R² = 0.866, n = 107,040 steps,
                                                  pooled across all 12 cells)
```

Doc's own fit, for comparison: `15.3 + 0.0422 ms/token × tokens`.

| tokens in step | predicted gpu_ms (this fit) |
|---:|---:|
| 20 (typical low-load fast block) | 8.0 ms |
| 272 (typical fast block, u24-40) | 20.7 ms |
| 921 (a real example turn's own big prefill) | 53.6 ms |
| 2443 (doc's own reported slow-block size) | 130.7 ms |

**Analysis.** Same shape as the doc's fit — a small fixed per-step overhead plus a genuinely
linear marginal cost per token — and R²=0.866 confirms the linear model is a good fit, not a
loose correlation. This is the mechanism underneath everything else in this document: a step
with 100x more tokens costs roughly 100x more time (modulo the small fixed overhead), which is
why the *size* of the step a request lands behind, not whether it's labeled prefill or decode,
is what actually predicts TTFA.

---

## Additional finding: stage-0 window composition (queue vs. compute)

Not part of the doc's own evidence chain — a complementary decomposition built with the same
event data. Every request's full stage-0 contribution (`recv`→`first-text`) split into
`queue_ms` (no logged GPU activity of any kind during that stretch — not specifically "queue
for prefill", undifferentiated idle/untracked time) and `prefill_ms`/`decode_ms` (GPU-busy
time, split by each overlapping step's token composition). Means, not medians — mean is
additive (`queue+prefill+decode == window` exactly), medians of separately-aggregated columns
are not.

### Full population

| cell | n | queue_ms | prefill_ms | decode_ms | window_ms |
|---|---:|---:|---:|---:|---:|
| u16 seed11 | 152 | 33.3 | 72.2  | 6.3 | 111.8 |
| u16 seed23 | 153 | 32.1 | 75.4  | 5.7 | 113.1 |
| u16 seed7  | 154 | 39.3 | 74.3  | 5.8 | 119.5 |
| u24 seed11 | 213 | 42.9 | 90.2  | 7.9 | 141.0 |
| u24 seed23 | 203 | 41.2 | 90.5  | 7.1 | 138.8 |
| u24 seed7  | 213 | 50.1 | 87.5  | 8.3 | 145.9 |
| u32 seed11 | 265 | 48.0 | 93.7  | 8.1 | 149.8 |
| u32 seed23 | 258 | 49.9 | 100.2 | 7.4 | 157.5 |
| u32 seed7  | 254 | 53.5 | 106.4 | 8.7 | 168.6 |
| u40 seed11 | 316 | 60.7 | 121.2 | 8.3 | 190.2 |
| u40 seed23 | 305 | 59.9 | 105.9 | 9.0 | 174.9 |
| u40 seed7  | 301 | 58.6 | 104.3 | 9.3 | 172.2 |

### P95 tail only (worst 5% by TTFA, threshold computed per-cell)

| cell | n | queue_ms | prefill_ms | decode_ms | window_ms |
|---|---:|---:|---:|---:|---:|
| u16 seed11 | 8  | 71.9  | 155.3 | 6.0 | 233.2 |
| u16 seed23 | 8  | 48.0  | 161.6 | 4.8 | 214.4 |
| u16 seed7  | 8  | 73.8  | 185.6 | 4.0 | 263.4 |
| u24 seed11 | 11 | 80.2  | 310.2 | 1.9 | 392.3 |
| u24 seed23 | 11 | 101.1 | 232.2 | 8.4 | 341.8 |
| u24 seed7  | 11 | 104.7 | 233.2 | 2.5 | 340.4 |
| u32 seed11 | 14 | 117.8 | 309.9 | 2.5 | 430.3 |
| u32 seed23 | 13 | 113.7 | 266.4 | 1.8 | 381.9 |
| u32 seed7  | 13 | 122.8 | 348.7 | 3.4 | 475.0 |
| u40 seed11 | 16 | 157.4 | 361.0 | 6.7 | 525.2 |
| u40 seed23 | 16 | 144.1 | 327.8 | 7.2 | 479.2 |
| u40 seed7  | 16 | 136.8 | 315.0 | 7.2 | 459.1 |

**Analysis.** `decode_ms` stays under 10ms everywhere, both full population and P95 tail,
confirming (via yet another independent path) that decode was never a real contributor to this
window — `prefill_ms` is what grows, in both the average case and the tail. `queue_ms` and
`prefill_ms` grow together from average case to P95 tail (queue ~1.5-2.5x, prefill ~2-3x) at
every load point, so the tail isn't disproportionately more *queued* — it's riding a bigger
version of the same composition, because it landed behind a larger prefill chunk. (An earlier
draft of this table also reported a `gpu_util%` = `(prefill_ms+decode_ms)/window_ms` column here
and called it flat across load — that ratio is a real number, but it is **not** GPU hardware
utilization: it's this request's own busy-fraction of its own wait window, and since many
requests' windows overlap in real time, averaging it across requests double-counts the same
GPU-busy milliseconds once per concurrently-waiting request. The corrected, hardware-true version
of GPU utilization is in the next section.)

---

## Additional finding: true GPU utilization (time-based, hardware-wide)

Computed correctly this time: **once per cell, over wall-clock time**, not averaged across
per-request windows. For every `[STEP-GPU]` entry on a given stage, sum `gpu_ms`; divide by the
cell's total wall-clock span. This counts every busy millisecond exactly once, regardless of how
many requests were concurrently waiting through it — the property an aggregate hardware
utilization number actually needs.

```
utilization% = 100 × (Σ gpu_ms across all steps) / (span of the cell, in ms)
```

`stage=0` = GPU0 (thinker), `stage=1` = GPU1 (talker). Code2wav is not a scheduled
autoregressive step and has no `[STEP-GPU]` entries at all, so the GPU1 numbers below are a
**lower bound** on true GPU1 occupancy (talker + code2wav together), not the full picture.

| cell | span_s | GPU0 busy_s | **GPU0 util%** | GPU1 busy_s | **GPU1 util%** |
|---|---:|---:|---:|---:|---:|
| u16 seed7  | 263.3 | 72.2  | **27.4** | 24.4 | **9.3**  |
| u16 seed11 | 272.5 | 73.3  | **26.9** | 24.9 | **9.1**  |
| u16 seed23 | 273.0 | 72.8  | **26.7** | 24.3 | **8.9**  |
| u24 seed7  | 263.2 | 97.5  | **37.0** | 30.9 | **11.8** |
| u24 seed11 | 290.0 | 97.3  | **33.6** | 30.8 | **10.6** |
| u24 seed23 | 284.0 | 92.2  | **32.5** | 30.8 | **10.8** |
| u32 seed7  | 267.1 | 113.9 | **42.6** | 37.0 | **13.9** |
| u32 seed11 | 308.8 | 112.4 | **36.4** | 36.2 | **11.7** |
| u32 seed23 | 294.5 | 116.1 | **39.4** | 36.8 | **12.5** |
| u40 seed7  | 274.5 | 126.5 | **46.1** | 39.5 | **14.4** |
| u40 seed11 | 277.5 | 126.9 | **45.7** | 40.5 | **14.6** |
| u40 seed23 | 275.9 | 125.9 | **45.6** | 40.1 | **14.5** |

**Time-series, u=40 (all three seeds, phase-aligned to each run's own start)**: [published
chart](https://claude.ai/code/artifact/2e0666c6-ef27-48f8-9be4-c82dcf64bc2a). Bucketed into 5s
windows, it shows the same phase structure on every seed — a ramp-up (0-40s, matching the
client's 40-second connection stagger), a steady state (~40-210s) holding GPU0 in the 45-67%
band and GPU1 in the 14-20% band, and a ramp-down as sessions finish their 10 turns.

**Analysis.** This is a materially different, and more useful, picture than the flawed
per-request metric gave. Two real findings: (1) **true utilization climbs monotonically with
load** — GPU0 26.7% → 46.1% from u16 to u40, GPU1 8.9% → 14.6% — a genuine trend the old metric's
double-counting bias had flattened out. (2) **Even at u=40, sitting right at the capacity wall,
GPU0 is idle more than half the time (54%) and GPU1 more than 85% of the time.** That is the
opposite signature of hardware saturation — a genuinely compute-bound system would show
utilization climbing toward 90-100% as the wall approaches. Instead this confirms the doc's
"the wall is vLLM's engine loop, not the hardware" framing more sharply than the per-request
metric could: there is substantial spare GPU capacity sitting idle at the exact load where the
system starts failing its TTFA criterion. The failure is entirely about *when* work gets
scheduled, not whether the hardware can keep up with the volume. GPU0 also runs at roughly 3x
GPU1's utilization throughout — architecturally expected (the thinker's multimodal prefill+decode
is heavier per step than the talker's codec-token generation), though GPU1's true number is
understated here since code2wav's own compute isn't captured by this log source at all.

---

## Summary

| doc's evidence point | reproduced? | note |
|---|:---:|---|
| Capacity is bound by first audio, not stutter | ✅ | stutter 0.00ms at every cell, all loads |
| Capacity number itself (32 sharp wall) | ⚠️ partial | wall is fuzzier here, spans ~32-40 not a single point |
| #1: slow turns aren't doing more of their own work | ✅ stronger | slow turns' own prompts are *smaller* at u≥24 |
| #2: admission ≈ instant after the blocking step | ✅ at u=16 only | breaks down at u≥24; generalizes to "instant after enough steps" |
| #3: blocking step is prefill, not decode | ✅ | 97.5-99.9% prefill share, tokens and time, every load |
| #4: application layer is negligible | ✅ | net_ms 0.18-0.61% of total; app-layer sub-segments not directly measurable, engine segments (picked up→admitted, admitted→first text) directly measured and dominate as in the doc |
| #5: mbt knob fails both directions | ❌ not tested | requires new engine boots, out of scope here |

The core mechanism reproduces cleanly and, in several places (evidence #1, #3, and the true
GPU-utilization finding — climbing with load but still well under 50% even at the wall, the
opposite signature of a hardware-bound system), reproduces *more strongly* than the doc's own
single-sample evidence could show. The one genuine divergence is the sharpness of the capacity
wall itself — on this hardware it is a zone (32-40), not a single passing/failing boundary —
which is itself consistent with, not contradictory to, the doc's own stated 17.5% run-to-run
noise finding at exactly that load.
