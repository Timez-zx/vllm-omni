# MiniCPM P/D: sliding KV window

**Latest status, 2026-09-09: at 18K × 300 s, 8/9 users pass and 10 users fail.** Content degeneration remains separate from serving errors; see the [generation and serving audit](sliding_quality.en.md).

New gate: after the window fills, every user must have long RTF strictly **>1**, and every unit's `max(0, previous D completion − current complete-input readiness)` must be **≤500 ms**. The first sliding unit retains its pre-window predecessor, so existing backlog is not hidden at the boundary. This excludes current-unit execution and is not TTFA. Any exceedance fails, even if later recovered. `--max-backlog-ms 500` changes offline classification only, never input admission or dropping. Related regression: 137 passes.

Offline reclassification of 10 users: **403 violating units, 10/10 violating users, maximum backlog 7,098.226 ms**. Original `analysis.json` and RTF-only summaries remain unchanged; the new result is `analysis-backlog500.json` in the same directory. RTF-only success below no longer means capacity success.

## Latest runs: 18K × 8/9 users × 300 s

Same four-GPU topology, MPS, pipeline, HD4 AV, FP8/BF16 Q, Triton automatic selection and pin128 as the 10-user control below. Serving and installed-vLLM sources are unchanged; the benchmark only adds the offline gate and recording fields. All points use 300 s of measured input; the eight/nine-user source manifests and resolved deployment configurations match exactly.

| Users | Minimum per-user long RTF | Maximum inherited backlog | Units exceeding 500 ms | Verdict |
|---:|---:|---:|---:|---|
| 8 | 1.001756 | 127.641 ms | 0 | Pass |
| 9 | 1.001732 | 301.425 ms | 0 | Pass |
| 10 | 1.001471 | 7,098.226 ms | 403 | Fail |

All **2,400/2,400 inputs complete** at eight users. Each enters the window at unit 84–85, covers 216–217 post-window units and reaches 65.0k–65.9k logical tokens. Post-window complete-AV readiness→D completion p50/p95/p99 is **413/792/894 ms**, maximum 1,072 ms; unlike inherited backlog, this includes current-unit execution. P/D residency peaks at 1,134 blocks, D computes one input-prefill token per unit, and KV delta median/p99 is 227/245 tokens. No AV fallback, runtime error or KV preemption; received-event/export audit has 0 FAIL/UNKNOWN. Maximum sender drift is 6.25 ms and source hashes remain stable. This establishes the two input-capacity criteria for this workload and horizon, not model quality or the full audio SLO.

Reproduce with the command below using `--users 8 --duration-s 300 --max-backlog-ms 500`, leaving other arguments unchanged. Archive: `/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/users-8x300-w18000-backlog500-r1/`. The new policy is recorded in `analysis.json`, under `bounded_backlog_audit` and `sliding_window_audit.long_horizon_rtf`.

The valid nine-user repeat completes **2,700/2,700 inputs**. Each enters the window at unit 84–85, covers 216–217 post-window units and reaches 65.3k–66.1k logical tokens. Post-window complete-AV readiness→D completion p50/p95/p99 is **488/975/1,065 ms**, maximum 1,227 ms. P/D residency peaks at 1,134 blocks, D computes one input-prefill token per unit, and KV delta median/p99 is 228/245 tokens. No AV fallback, runtime error or KV preemption; received-event/export audit has 0 FAIL/UNKNOWN. Maximum sender drift is **5.86 ms**, with stable pre/post-run source hashes.

The first nine-user attempt had a nearly simultaneous client wake-up delay affecting two chunks, maximum **25.37 ms**, violating the predeclared 10 ms sender-quality gate. Raw artifacts remain archived but are excluded from accepted capacity points. No code or threshold was changed for the valid repeat. Accepted archive: `users-9x300-w18000-backlog500-r2/` under the same root; `r1/TIMING_INVALID.md` explains the excluded attempt. Reproduce by changing only `--users` to 9. Nine is the highest passing tested point for this workload/horizon, not a capacity guarantee for other inputs or unlimited sessions.

## Ten-user control: 18K × 300 s

At the user's request, this run uses an **18,000-token window + pin128** and **300 s input**. Four-GPU placement, MPS, FP8, BF16 Q, native Triton automatic selection and the HD4 AV workload are unchanged. A separate session warms up for 12 s; output observation lasts 30 s after input. The deployment default remains 36K; this run overrides it on the command line.

- **3,000/3,000 inputs complete; 10/10 users have long RTF≥1, minimum 1.0014706348.** Each enters the window at unit 85, covers 216 post-window units and reaches 64.9k–66.3k logical tokens, including full-window replacement.
- Post-window complete-AV readiness→D completion p50/p95/p99: **585/4,578/6,167 ms**, maximum **8,087 ms**. Temporary backlog is recovered by the end. Long-RTF success does not establish low-tail latency or maximum capacity at 10 users.
- P/D residency peaks at **1,134 blocks**. D computes one input-prefill token per unit; transferred-delta median/p99 is **229/245 tokens**. No AV fallback, runtime error or KV preemption; received-event/export audit has **0 FAIL/UNKNOWN**. Maximum sender drift is 6.77 ms.
- Source hashes remain stable during the run. This is an uncommitted development-snapshot measurement, not clean-build, semantic-quality or complete audio-playback certification. The expanded 18K GPU-attention/cache/config regression has **65 passes**. Window, concurrency and duration differ from the 36K controls below; their outcome difference cannot be assigned to one isolated factor.

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 10 --duration-s 300 \
  --kv-window-tokens 18000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --no-triton-force-2d-attention --quality-capture --out-dir /path/to/new-run
```

Archive: `/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/users-10x300-w18000-native-attention-r1/`; its sibling `capacity-summary-10-w18000.json` is the compact summary. The original 420 s wrapper was cancelled during service initialization, before measurement.

## 36K controls: native attention selection

Use the four-GPU topology and workload below, with explicit **36k+pin128, BF16 Q and FP8 KV**, normal compile/graphs, private MPS and no synchronizing probes. Remove the earlier numerical-diagnostic `triton_force_2d_attention` override: it disabled decode split-K. One attention layer at one request/36k history measures 1.258→0.137 ms. This establishes an avoidable diagnostic cost, not the cause of all end-to-end backlog.

| Users | Completed inputs | Minimum post-window long RTF | Full-window latency p50 / p95 / p99 | Verdict |
|---:|---:|---:|---|---|
| 6 | 2,520/2,520 | 0.985506 | 745 / 9,661 / 11,006 ms | Fail |
| 7 | 2,940/2,940 | 0.983531 | 1,621 / 20,998 / 24,309 ms | Fail |
| 8 | 3,360/3,360 | 0.951623 | 19,735 / 37,194 / 39,317 ms | Fail |

Latency is complete one-second AV readiness→D completion, **including inherited backlog, not current-unit GPU time or TTFA**. RTF is each user's post-window input-second budget/corresponding wall time; the unrounded value must be ≥1. Individual late-unit counts are not the gate. Respectively 5/5/6 users fail at 6/7/8 concurrency.

- Each user covers 254–257 full-window units and reaches 91.8k–94.0k logical tokens. P/D retain at most 2,259 valid blocks, including 8 pinned blocks; no prompt rebuilding.
- All 8,820 D input prefills compute exactly 1 token. Post-window transferred-delta median/p99 are 230/245 tokens. No AV fallback, runtime error or KV preemption; received-event/export audits have 0 FAIL/UNKNOWN.
- Separate direct P execution evidence comes from 32 earlier sampled formal forwards: 18 post-window samples compute only 211–230 contiguous new positions, up to 91k history. Source/regression and current D accounting provide complementary evidence; D accounting is not a per-position probe of every P forward in these runs.
- Per-user D completion→client receipt p99 ranges are 15–19/20–25/29–39 ms at 6/7/8 users. The old multi-second output blockage does not recur. Maximum send drifts 6.43/6.67/6.95 ms are below 10 ms.
- Omni/vLLM source manifests match across all three runs, with stable pre/post-run hashes. These are uncommitted development-snapshot results, not semantic-quality or complete audio-SLO certification. The 4-user startup was cancelled by user request, not measured as a failure.

Artifacts: `/home/ubuntu/data/experiments/minicpm-pd-capacity-review-20260909/`. Each `users-{6,7,8}x420-native-attention-r1/` contains resolved config, source, MPS, send/completion records, text/PCM and audits. `capacity-summary.json` is the combined result; `summarize_capacity.py` reproduces it offline.

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --no-triton-force-2d-attention --quality-capture --out-dir /path/to/new-run
```

Change only `--users` to 7 or 8 for those points. Older capacity tables below are not current-version results.

## Implementation

- P/D use native token-level sliding-window attention with **36,000 tokens**. Expired KV blocks are reclaimed; the application does not rebuild a short prompt or re-prefill the whole window.
- Logical token history, positions and full-history hashes continue growing. vLLM's `SlidingWindowManager` matches resident blocks without treating cropped tokens as an identical new prefix.
- Native AV scheduler tokens are placeholders, not media identities. Every new lineage gets a random cache namespace, preserved by P, early D registration and formal D. Later units within that session keep the same salt; cross-user placeholder matches cannot reuse KV.
- NIXL registrations carry **absolute logical block indices**. Only corresponding valid P blocks are transferred; expired null blocks must never be copied or confused with a shifted source list.
- P-only/D-only modes and `D(i-1) -> P(i)` dependence are preserved. Legacy prompt rollover remains available only for deployments without this sliding-window flag.
- All 20 Talker layers use a separate **4,096-token** window with a 65,536 logical-position limit, avoiding overflow of the original 4k input array during long speech. Its binary scheduler IDs do not identify actual conditioning/code, so cross-request prefix lookup is disabled. Live-request KV retention and sliding remain enabled.

This changes attention semantics and is not equivalent to official whole-unit cropping and RoPE realignment; lossless model quality is not established. With default pin=0, initial system/reference KV eventually leaves the window; the latest validation explicitly uses pin128 to protect eight initial blocks. Logical positions are capped at **262,144**: this is not an unlimited session. The 36k limit bounds resident attention/KV, not logical prompt length.

## Setup and criteria

GPU0 Thinker-P; GPU1 Thinker-D + Talker; GPU2 Vision + Audio Encoders; GPU3 Code2Wav. Private MPS and pipelining remain enabled. P/D/Talker weights and P/D KV use FP8. P/D each have a fixed 64 GiB KV budget; Talker has 8 GiB (BF16 KV).

Real 960×540 AV with `max_slice_nums=4`; 200 ms audio chunks, 1 FPS video and one-second model units. Random session phases, ±50 ms arrival jitter, seed `20260908`. Media preprocessing finishes before timing; open-loop sends do not wait for responses. Unintended send drift must stay within 10 ms.

Each point restarts the server and uses a separate one-user 12-second warm-up. Measured users then start from zero context, stream for **420 seconds**, and observe output for 30 more seconds. Warm-up provides no measured-session history. Repo and installed-vLLM source snapshots are archived before the run and checked for stability afterward.

Capacity criterion (updated 2026-09-09): every user must satisfy **post-window long RTF > 1** AND **maximum inherited backlog ≤ 500 ms**, using unrounded values. RTF is:

```text
RTF_sw = N completed one-second post-window input units /
         (last D completion time − first post-window complete-input readiness)
```

The numerator budgets N seconds, including one second for the first unit. The denominator is one continuous wall-clock span including waits, P/D and handoff—not summed unit latencies or an average of unit RTFs. Its origin is **complete-input readiness**, unlike the legacy `stream_rtf` measured from the first media chunk. Legacy values remain diagnostic and must not be interchanged.

Every user must have at least 120 post-window units and reach at least twice the window's token count, replacing the original window fully. Any post-window inherited backlog above 500 ms fails even if later recovered; current-unit execution time and p99 remain separate. Since inputs are paced in real time, RTF near 1 does not imply GPU saturation, and passing a finite horizon does not establish unlimited-session stability.

Also verify bounded live P/D blocks for every user, no prompt recycling, complete inputs/frames, no fallback/preemption/runtime errors and valid sender timing. This certifies only Thinker input consumption; audio playback continuity is evaluated separately, not assumed to pass.

## Reproduction

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 6 --duration-s 420 --kv-window-tokens 36000 \
  --out-dir /home/ubuntu/data/experiments/minicpm-pd-sliding-repeat
```

Use a new output directory. Preserve resolved config, commands, warm-up/run/analysis JSON, server log, MPS evidence, source snapshots and `source-stability.json`.

## Archive: earlier RTF-only results (2026-09-09)

Offline reanalysis of the same long runs: **6 users passed; 8 failed; 7 was not tested.** No server rerun or raw-record modification was made. This applies to the stated configuration, input and 420-second horizon, not unlimited sessions or every workload.

| Users | Limiting user's post-window input budget | Corresponding wall time | Minimum long-horizon RTF | Capacity |
|---:|---:|---:|---:|---|
| 6 | 253 s | 252.845578 s | 1.000611 | Pass |
| 8 | 252 s | 253.211870 s | 0.995214 | Fail |

Decisions use unrounded values. The 8-user point fails because the limiting session does not sustain the input rate over approximately 253 seconds, not because the 123 individual deadline misses below are a capacity gate.

Latency below is **complete one-second AV unit ready → physical D completion**, restricted to each session's post-window units. It is not TTFA.

| Users | Completed units, entire run | Post-window units per user | Post-window latency p50 / p95 / p99 | Late units (diagnostic) | Maximum unit lateness |
|---:|---:|---:|---|---:|---:|
| 6 | 2,520 / 2,520 | 252–254 | 284 / 706 / 814 ms | 0 | 0 ms |
| 8 | 3,360 / 3,360 | 252–254 | 322 / 1,064 / 2,002 ms | 123 | 1,400 ms |

- Whole-run p99 was **785 / 1,780 ms** for 6/8 users. Neither run missed the first unit. The 8-user run first missed input 394; all 123 misses occurred after window saturation.
- Final logical history was approximately **90k–92k per user**, without prompt recycling. Each P/D session retained at most **2,251 live 16-token blocks**. Post-window KV transfer had a median of **225 tokens per unit** in both runs and p99 of 244/245 tokens—not a 36k-window retransmission.
- Every video/audio unit used its arrival sidecar result; zero fallback, runtime errors or observed KV preemptions. Every user's first D request had zero local prefix hits. Maximum unintended send drift was **6.38 / 6.94 ms**, below 10 ms.
- For the slowest 1% of 8-user units, mean latency decomposed into **1,029 ms** inherited previous-D wait, **138 ms** current pre-D work, and **908 ms** D-submit-to-completion: **2,074 ms** total. This localizes delay to the D path and inherited backlog. D service still includes ingress, KV readiness, scheduling and decode; it is **not a pure GPU-decode measurement**.

Speech output remained active, but playback observation still recorded buffer underruns (193/315 total for 6/8 users). These results do not certify uninterrupted end-to-end audio; the capacity claim is limited to input consumption.

Artifacts under `/home/ubuntu/data/experiments/minicpm-pd-sliding-20260908/`:

- Passing point: `users-6x420-w36000-v7b`; failing point: `users-8x420-w36000-v7`.
- The code/config manifests match between points and remained unchanged during each run. Four-stage private-MPS attachment passed. 283 related regression tests passed.
- New results are saved as `analysis-long-window-rtf.json` in each directory. Original `analysis.json` retains the old zero-backlog criterion. `input_capacity_pass` now reports long-horizon RTF; `input_deadline_pass` reports only the old per-unit diagnostic. Only analysis logic changed; serving code and measured configuration did not.
- The worktree is uncommitted, so formal `capacity_pass` remains gated by the clean-tree check. The table reports the **archived development snapshot's long-horizon input RTF result**, not certification of a committed baseline.

Earlier v1–v5 attempts are invalid due to configuration, cache-isolation or Talker boundary errors. The v6 4/8-user points guided the search but predate the Talker cross-request cache guard. Original artifacts and explanations remain in the experiment root and are excluded from the final table. A separate fix preserves D's final sampled token when a segment ends without a native terminator; no padding is used.

Do not carry forward the old one-block-per-unit or capacity claims: those tests rebuilt short context and lacked cross-user AV-placeholder cache isolation.
