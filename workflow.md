# MiniCPM-o Native-Duplex P/D Workflow

Chinese version: [workflow.zh.md](workflow.zh.md).

**Current capacity (2026-09-10): 21 users, the highest passing tested load under this configuration; 22 narrowly exceeds the backlog limit.** Criteria and the 300 s cross-window scope are below; this is not an unlimited-duration guarantee or a theoretical hardware limit. Engineering repairs supersede the earlier 12-user failure. P GPU queueing/execution and multi-step D execution now dominate, with CPU preparation/control still present.

## Current setup and criteria

GPU0=Thinker-P, GPU1=Thinker-D+Talker, GPU2=Vision/Audio Encoder, GPU3=Code2Wav; private MPS and pipelined stages. P/D/Talker use FP8 weights, P/D use FP8 KV, Encoder/Code2Wav retain native precision, and Talker uses BF16 KV. BF16 Q and native Triton attention; latest runs explicitly set D's split-K threshold to 32.

Real HD4 video at 1 FPS and 200 ms audio chunks form native one-second units; random phases, ±50 ms jitter, seed=20260908. Independent warmup, fresh histories, 18K+pin128 sliding window, 300 s input plus 30 s output observation. Every user's post-window **RTF must exceed 1 and inherited backlog must stay ≤500 ms**. Sender drift above 10 ms invalidates a capacity point. Prefill and KV transfer remain incremental after sliding begins.

## Uninstrumented long runs

| Current version / users | Completed inputs | Minimum post-window RTF | Maximum inherited backlog | Units >500 ms | Result |
|---|---:|---:|---:|---:|---|
| O, 20 users | 6000/6000 | 1.000659 | 217.8 ms | 0 | Pass in this run |
| 38157bae, 21 users | 6300/6300 | 1.000787 | 339.3 ms | 0 | Pass; current capacity |
| 38157bae, 22 users | 6600/6600 | 1.000383 | 523.0 ms | 5 | Fail |
| O, 24 users | 7200/7200 | 0.991656 | 7645.8 ms | 1889 | Fail |

All four loads use identical serving sources and resolved deployment configurations. Sources remain stable during measurement; sender timing and complete AV consumption pass, with zero fallback/preemption and 0 FAIL/UNKNOWN protocol checks. At 21 users, every user completes at least 216 post-window units and stays within 339.3 ms backlog. At 22, all users pass long RTF, but two users exceed 500 ms in five units. At 24, 22 users eventually achieve RTF>1, but all exceed the backlog limit. Catching up does not erase violations.

The 21/22-user runs use committed `38157bae` with clean worktrees. At 21, maximum sender drift is 6.628 ms and P/D residency is bounded at 1134 blocks. Raw evidence and commands: [21-user result](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/batch-policy-clean-21x300-r1/RESULTS.md), [22-user result](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/batch-policy-clean-22x300-r1/RESULTS.md). One 300 s run per load establishes 21 as the highest passing tested point, not unlimited-duration stability. Periodic inputs do not imply constant autonomous decode work.

## Repairs and remaining issues

Retained repairs independently progress KV completion; remove repeated full-history detokenization/hashing/conversion and Python-list tensor transport; reduce output copies and redundant GPU control metadata; and batch identical sampling rules across users.

Newly confirmed issues: P echoed full histories every unit, now replaced by a suffix+offset with strict prefix validation. D changed to a slower attention path at effective batch 16→17. Identical Q/K/V controls at 18K measure 0.894→0.457 ms per layer for 17 requests using native split-K. The explicit threshold is 32; 48/64-request controls no longer benefit. Window and inputs are unchanged, but reduction orders are not bit-equivalent. All 42 sliding-window GPU numerical checks pass.

**Latest repair O is long-run validated:** forbidden-token masks, per-user repetition-penalty indices and temperature processing now join the existing sampling batch; shared/unseeded RNG retains original row order. A full-policy 24-row microbenchmark falls from 4.53 to 2.03 ms/step with 4,400 identical tokens and matching RNG states. Related regressions: 169 passed, one inapplicable case skipped. No new kernel or reduced generation budget. Live D forward means at batch 16/17 are 22.7/23.5 ms, without the previous approximately 39 ms cliff; these live groups are not identical-input operator controls.

An independent 24-user diagnostic of the same version averages **1640 ms = 833 before D submission + 807 afterward** for its 72 slowest fresh cycles, excluding inherited backlog. P admission waits 309 ms and P execution through Core takes 446; D admission/scheduling takes 41 and execution/inter-iteration control 749, with approximately 95 ms elsewhere. P's own forward event is 377 ms; 279 ms of admission wait overlaps the preceding batch's GPU-completion-wait phase. D averages 19.8 steps and 421 ms of forward events. D also accumulates 124 ms preparation, 76 ms sampling/state work and 100 ms inter-iteration control: these are not all GPU compute, and GPU waits must not be added twice.

In that slow cohort, audio's critical-path contribution is only 3.6 ms, KV post→notification 8.6 ms and P runner→Core 2.0 ms. P output averages approximately 123 KiB with 2.3 ms encoding-thread CPU time. Unready encodings, KV notifications deferred to the next batch and full-history output messages are no longer dominant. Post-window P increments have median/p99 220/230 tokens; D's local prefill remains one token, KV deltas have median/p99 229/245 tokens, and P/D residency is bounded at 1134 blocks. There is no full-window recomputation.

**Keep diagnostic and capacity evidence separate:** the uninstrumented 24-user run's 72 slowest fresh cycles average 1449 ms (788 before D, 661 afterward), versus 1640 ms with diagnostics. Diagnostic overhead is not production latency, and different generation decisions prevent correcting individual phases by simply subtracting the run totals. GPU execution/queueing is now the main component, not proof that all non-GPU overhead has disappeared.

Same-run 60 s hardware means for P / D+Talker: GPU busy 81.5% / 74.1%, SM active 76.9% / 50.9%, DRAM read active 4.6% / 49.0%. These do not prove continuous saturation or theoretical peak throughput. P attention uses 128 threads and 28,928 B shared memory per block; 102,400 B per SM permits at most three blocks, or 25% warp capacity. Low warp occupancy therefore does not mean 75% freely usable resources, nor prove an optimal kernel. Earlier 10 s CUDA tracing confirms that forward spans mostly execute kernels. Profiler-stop and native-stack-capture stalls and subsequent backlog are excluded from capacity evidence.

Evidence, repairs, failed attempts, tests and commands: [diagnostic record](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/WRITE_PROGRESS.md); the current reproduction command is in the [benchmark README](benchmarks/minicpmo/README.md#reproduce-the-current-result). This is input-consumption/protocol evidence, not certification of dialogue semantics, complete playback SLOs or unlimited duration. The 20/24-user runs used uncommitted sources, so their archived clean-build certification remains false; committing later does not rewrite old records. The 21/22-user follow-ups are valid uninstrumented capacity measurements from clean source trees.

## Pre-commit functional verification (2026-09-10)

- Current serving sources match both 20/24-user snapshots file by file. Installed vLLM 0.26.0 Python sources match the distribution RECORD; no local source patch is missing from the handoff.
- Re-ran all 53 added/modified test files: **1531 passed, one inapplicable case skipped**, including explicit GPU sliding-window numerics, cross-user cache/RNG isolation, P/D handoff, sampling, output protocol and cancellation. Only the CPU tests' incorrect assumption that preceding tests cannot initialize CUDA was fixed; serving logic was unchanged.
- Re-audited both long runs: input, session/response ownership, text/PCM exports and KV accounting all have **0 FAIL and 0 UNKNOWN**. No new serving-functionality error was found. Sliding still uses incremental prefill/KV delta, not full-window recomputation.
- Scope: this does not guarantee dialogue semantics or uninterrupted playback for all inputs. Earlier independent official long-history generation also degraded; see the [generation audit](benchmarks/minicpmo/sliding_quality.en.md). This is a research baseline with verified serving contracts, not comprehensive model-quality certification.

Verification artifacts are `precommit-functional-regression-r2.log` and `precommit-functional-audit-{20,24}.json` in the diagnostic directory above.

## Earlier repairs in this round (superseded by current results)

Two fixes: the existing NIXL writer progresses outgoing status independently instead of delaying completion notification until another P batch; duplex text output uses vLLM's native incremental detokenizer with bounded prompt-boundary decoding. Existing optimizations, complete logical histories, 18K sliding incremental prefill/KV delta, inputs and sampling policy remain intact. One traced KV copy took approximately 0.47 ms but notification waited until 338 ms; the repaired 16-user slow cohort observes notification in approximately 4.1 ms. Full-history detokenization previously accounted for 62% of API GIL-holder stack samples; that hotspot disappears.

| Version/load | Completed inputs | Minimum post-window RTF | Maximum inherited backlog | Units >500 ms | Post-window Ready→D p50/p95/p99 |
|---|---:|---:|---:|---:|---|
| Both fixes, 12 users, uninstrumented | 3600/3600 | 1.001887 | 4.6 ms | 0 | 435/759/848 ms |
| Same version, 16 users, diagnostic | 4800/4800 | 1.000238 | 392.7 ms | 0 | 546/1116/1231 ms |
| Same version, 20 users, uninstrumented | 6000/6000 | 0.945461 | 26,143 ms | 2538 | 2720/17384/21409 ms |

All use the unchanged four-GPU/FP8/MPS/pipelined, HD4, 18K+pin128, 300 s setup below. Sender timing is valid; video/audio consumption is complete with zero fallback; observable protocol audits have 0 FAIL/UNKNOWN; sources remain unchanged within runs. The instrumented 16-user result does not replace a clean capacity point. At 20 users, the 60 slowest post-window fresh cycles average 1759 ms: 585 before D submission and 1174 afterward, excluding inherited backlog. D service is not pure forward time.

Further validation: reuse vLLM random_sample for GPU sampling; 540 fixed-input token/RNG comparisons match, and full 12-row sampling falls from 7.81 to 6.54 ms. End-to-end validation of this change is pending and is not mixed into the table. PyTorch 2.11 multinomial checks are already asynchronous; the benefit is fewer repeated checks/small kernels, not removal of a proven per-user host synchronization. Evidence and reproduction: [current diagnostics](/home/ubuntu/data/experiments/minicpm-pd-backlog-diag-20260910/WRITE_PROGRESS.md). These are archived development-snapshot input-capacity results, not dialogue-semantic or full audio-playback SLO certification.

## Previous repair record (superseded by the table above)

Existing KV delta transfer, immediate finished-block publication, Encoder pipelining, background output draining, forbidden-token index caching and vectorized repetition penalty remain intact. Three gaps are addressed: retire ready P results before executing another batch; collect independent users' sampling decisions once per phase instead of repeatedly synchronizing GPU/CPU per user; let the existing NIXL writer check every 1 ms while D awaits remote KV and sleep when idle. Model, sampling policy/RNG, inputs and generation budgets are unchanged; no new thread or transfer protocol. All 209 regression tests pass. Fixed-logit sampling for 12 rows falls from 8.975 to 8.093 ms with identical tokens/RNG; sampling overhead is not eliminated.

Unchanged setup: GPU0=P, GPU1=D+Talker, GPU2=AV Encoder, GPU3=Code2Wav; private MPS, pipelined stages, FP8 weights/KV, 64 GiB P/D KV each, BF16 Q and native Triton attention dispatch. Real HD4/1 FPS video and 200 ms audio chunks, random phases and ±50 ms jitter, seed=20260908. Zero initial history, explicit 18K+pin128 sliding window, 300 s input plus 30 s observation. Every user must have post-window RTF >1 and maximum inherited backlog ≤500 ms.

| 12-user version | Completed inputs | Minimum post-window RTF | Maximum backlog | Units >500 ms | Post-window Ready→D p50/p95/p99 | Result |
|---|---:|---:|---:|---:|---|---|
| Before: index cache + vectorized repetition penalty | 3600/3600 | 1.001755 | 1322 ms | 82 | 696/1368/1751 ms | Fail |
| Add prompt P delivery + batched sampling feedback | 3600/3600 | 1.001828 | 1108 ms | 25 | 516/1077/1489 ms | Fail |
| Current: also progress D notifications independently | 3600/3600 | 1.001286 | 503.6 ms | 1 | 489/1002/1251 ms | Fail |

All three uninstrumented runs have identical deployment-config hashes, valid sender timing, complete video/audio consumption, zero AV fallback/runtime errors/KV preemptions, observable functional audits with 0 FAIL/UNKNOWN, and unchanged source/config within each run. Prefill and KV transfers remain incremental after sliding starts; P/D residency peaks at 1134 blocks, without whole-window recomputation. The current single violation follows user 1's unit 186, whose fresh processing takes 1344 ms (602 before D submission and 742 afterward). Combined with existing waiting, unit 187 inherits 503.6 ms backlog; unit 187 itself is already catching up.

A separate nonblocking 12×300 diagnostic confirms **P runner completion→Core delivery p99 falls from 190 to 2.24 ms**: the old delay from executing another batch before delivering a ready result is removed. The new 36 slowest fresh cycles, all post-window, average 1222 ms: 289 before P admission, 175 for P scheduling/runner/delivery, 27 to D submission, 141 from D submission to admission, 570 for D scheduling and multistep execution, and 19 for delivery to the API. These are additive segments from the same requests, excluding inherited backlog. Within D execution, GPU-event forward totals approximately 298 ms, sampling 113 ms and preparation 81 ms; CPU/GPU subspans may overlap and must not be mechanically added again.

**Unresolved overhead:** in that slow-request cohort, KV write→D readiness still takes approximately 151 ms: 142 before notification forwarding and 9 afterward. Pending-receive polling removes reliance on another engine wake, but has not demonstrated a significant reduction of that tail; the interval is not pure PCIe copying. Another 238 ms precedes P input parsing, spanning AV cache readiness, application/control-plane work and IPC; reliable per-request attribution is missing, so this path was not speculatively changed. Remaining latency cannot all be attributed to prefill or declared free of engineering issues. The diagnostic also has one backlog violation (540 ms), and does not replace the uninstrumented result. An accidental device-synchronizing diagnostic was aborted and excluded. Full evidence and replay: [results](/home/ubuntu/data/experiments/minicpm-pd-delivery-sampler-20260910/RESULTS.md). These are archived development-snapshot input-capacity results, not normal-dialogue quality or full audio-playback SLO certification.

**Pre-fix records below (2026-09-09): with 18K windows and 300 s input, 8/9 users passed and 10 users failed.** Backlog is `max(0, previous D completion − current complete-AV readiness)`, excluding current-unit execution.

Valid nine-user repeat: **2,700/2,700 complete, minimum RTF 1.001732, maximum backlog 301.425 ms, zero violating units**. Post-window latency p50/p95/p99 is **488/975/1,065 ms**, with 216–217 post-window units per user. Maximum sender drift is 5.86 ms, below the 10 ms limit. No AV fallback, runtime error or KV preemption; received audit has 0 FAIL/UNKNOWN. Source snapshots and deployment configuration match the eight-user run. The first nine-user attempt had 25.37 ms sender drift and remains archived but excluded from accepted capacity points; no timing threshold was relaxed. Nine users pass this workload and 300 s horizon, not every input or unlimited duration.

Eight users: **2,400/2,400 complete, minimum RTF 1.001756, maximum backlog 127.641 ms, zero violating units**. Post-window complete-AV readiness→D completion p50/p95/p99 is **413/792/894 ms**, with 216–217 post-window units per user. Configuration matches the 10-user run; serving/installed-vLLM sources are unchanged, with only analysis policy and recording fields updated. No AV fallback, runtime error or KV preemption; received audit has 0 FAIL/UNKNOWN. All ten users pass RTF but violate the backlog bound: 403 units exceed it, maximum 7,098 ms; ten users are reclassified offline from unchanged records.

The 18,000+pin128 run keeps the same four-GPU topology, MPS, FP8 and native Triton automatic selection, with no serving-algorithm change. All 3,000 inputs complete; every user enters the window at unit 85, covers 216 post-window inputs and reaches 64.9k–66.3k logical tokens. Minimum long RTF is **1.001471**, but post-window complete-AV readiness→D completion p50/p95/p99 is **585/4,578/6,167 ms**: temporary backlog is recovered, not absent. No AV fallback, runtime error or KV preemption; received-event/export audit has 0 FAIL/UNKNOWN. P/D residency peaks at 1,134 blocks; KV delta median/p99 is 229/245 tokens. Reproduction is in the linked window document.

Earlier 36K controls:

- Four GPUs: GPU0=P, GPU1=D+Talker, GPU2=AV Encoder, GPU3=Code2Wav; MPS, pipelining, FP8 weights/KV, explicit 36k+pin128 and BF16 Q.
- One measurement confound was removed: previous numerical diagnostics forced 2D attention, disabling decode's historical-KV split-K parallelism. At one request and 36k history, isolated attention-layer time was **1.258 ms versus native split-K 0.137 ms**; this is not a whole-model speedup. Capacity runs restore native automatic selection without changing sampling, inputs, windows or the serving algorithm.
- Identical source snapshots, **420 s input + 30 s observation** per point: 6/7/8 users complete **2,520/2,940/3,360** inputs, with minimum post-window long-horizon RTF **0.985506/0.983531/0.951623**. All fail the per-user unrounded RTF≥1 criterion. The 4-user startup was cancelled at the user's request and is excluded.
- Every user covers 254–257 full-window units and reaches 91.8k–94.0k logical tokens; P/D residency is bounded at 2,259 blocks per request. No encoder fallback, runtime error or KV preemption; received-protocol/export audits have 0 FAIL/UNKNOWN in all three runs. D computes one input-prefill token per unit; post-window KV delta median/p99 are 230/245 tokens, not whole-window retransfers.
- The old output-drain blockage does not recur: per-user D completion→client receipt p99 ranges are **15–19/20–25/29–39 ms** for 6/7/8 users. Input processing still accumulates backlog; complete execution is not real-time success. CPU regression: 1,166 passed; separate GPU/cache checks: 53 passed, including the actual 36k window and 72k logical positions.

Detailed setup, criteria and reproduction: [sliding-window experiment](benchmarks/minicpmo/sliding_window.en.md). These are archived uncommitted development snapshots, not clean-build certification. Official-path long-history degeneration remains separate: these runs measure input consumption under current autonomous generation, not normal-dialogue quality or the full audio-playback SLO. Earlier fixes and evidence limits: [generation and serving audit](benchmarks/minicpmo/sliding_quality.en.md).

Older capacity and exploration records below are archived evidence, not functional or usable-capacity claims for the repaired version.

## Phase 1: objective

Measure sustainable continuous-AV session capacity on one four-GPU node. First validate input, state, KV and output-delivery correctness, then study serving latency and capacity. Content degeneration also reproduced by the official path is recorded separately, not made a model-improvement prerequisite. Application and connector artifacts must still be excluded before an engine-level claim.

## Phase 2: current serving design

```text
200 ms audio chunks + 1 FPS video
  -> one native one-second model unit
  -> Vision + Audio Encoder sidecars
  -> Thinker-P incremental prefill
  -> block-aligned KV delta transfer
  -> Thinker-D finite decode
  -> D output enters the next P lineage
  -> optional Talker + Code2Wav
```

| Stage | GPU | Role |
|---|---:|---|
| Thinker-P | 0 | Multimodal incremental prefill |
| Thinker-D + Talker | 1 | Finite autoregressive decode and speech-code generation, sharing via MPS |
| Vision + Audio Encoders | 2 | Stateless HD4 frame and streaming audio encoding |
| Code2Wav | 3 | Speech codes to waveform |

- The application owns session history and media buffers. Engine KV is disposable execution state.
- Thinker requests enter independently without a global admission gate. Encoders have separate cross-session batching: a 50 ms idle-to-busy window, then continuous dispatch while backlogged.
- MiniCPM-o feeds Thinker output back into the next unit, so the real dependency is `D(i-1) -> P(i) -> D(i)`. The next Thinker unit does not wait for Talker or Code2Wav.
- This measurement explicitly uses an 18,000-token sliding attention/KV window and 128 pinned-prefix tokens (the launcher's default remains 36,000; pass the override when reproducing). Logical history and full-history hashes keep growing; D reuses resident window blocks and NIXL transfers delta blocks by absolute logical position. The logical position limit is 262,144. Admission reserves the complete current generation budget; exceeding the limit explicitly closes only that session, without silently rebuilding history. This is neither unlimited-duration support nor the official whole-unit/sink policy, and does not claim lossless quality.
- Talker uses a separate 4,096-token window and 65,536 logical-position limit. Cross-request prefix lookup based on its binary placeholder tokens is disabled, while live-request KV remains retained. Long speech no longer overflows the original 4k input array.
- P and D use FP8 E4M3 KV and `NixlDeltaPushConnector`. D retains prefix KV and imports only the new block-aligned suffix.

Every arriving frame and audio unit is pre-encoded on GPU 2. Completed embeddings remain on CPU until the matching P request consumes them. No input is silently dropped. Correctness fallback remains available after an encoding failure, but any fallback invalidates capacity certification.

The current config is `benchmarks/minicpmo/deploy_capacity_pd_d_talker_fp8.yaml`: FP8 P/D/Talker weights and P/D KV, with native Encoder/Code2Wav precision. P and D each have 64 GiB of KV; Talker has 8 GiB of BF16 KV. Encoders use background threads, private per-modality CUDA streams and completion events. `run_pd_placement.py` defaults to private MPS, archives the resolved config and verifies all four stage clients.

Major forwards support cross-user batching: Vision buckets compatible slice shapes, Audio buckets compatible inputs/cache states, P/D/Talker use engine batching, and Code2Wav buckets compatible codec/cache states. Some sampling remains per-user. Batching support does not guarantee large batches or optimal efficiency. Next-unit encoding can overlap the previous Thinker unit; Talker/Code2Wav do not block the next Thinker unit. Same-unit P→D remains dependent.

Other retained configs are different topologies: `isolated` gives P, D and Encoders separate GPUs and shares GPU3 between Talker/Code2Wav; `encoders-on-p` places Encoders on GPU0 and Talker on GPU2. Their earlier capacity numbers do not describe this run.

## Phase 3: archived 240-second workload and criteria

This search uses 8→16→12→10 users, a fresh server for each point, 240 seconds of input and 30 seconds of output observation:

- real looped 960x540 MP4, aligned 16 kHz mono audio, and reference audio;
- audio arrival every 200 ms, video at 1 FPS, and one model unit per second;
- official HD slicing with `max_slice_nums=4`;
- per-session phase uniformly randomized in `[0, 1 s)` with +/-50 ms arrival jitter;
- seed `20260908`, with dispersed aligned media offsets;
- zero initial context, with the first unit included. Each session actually rolls over near input 157, after a peak prompt of about 33.4k–33.8k;
- autonomous listen/speak behavior; absolute-time open-loop sends without waiting for responses. Media decoding/JPEG preparation finishes before timing.

The primary capacity metric is:

```text
stream RTF = completed one-second input budget /
             wall time from first media arrival through last physical-D completion
```

Input capacity requires every session to have `stream RTF >= 1`, all expected physical-D requests to finish, no user failure, and complete frame consumption. In addition, each D unit must finish by the next complete-input arrival; the final unit has one second from readiness. Catching up later cannot erase transient backlog. This certifies Thinker input consumption, not the full audio pipeline; `end_to_end_capacity_pass` remains unknown without downstream evidence.

Record planned/actual sends, client wakeup lag and WebSocket send duration. Unintended send drift above 10 ms (5% of a 200 ms chunk period) invalidates capacity evidence; configured ±50 ms arrival jitter is excluded. Old artifacts without these audit fields cannot pass the new checks.

The first 16-user run had a 177.7 ms client stall and is invalid. Subsequent runs defer client cyclic GC outside timing, retain normal reference counting and leave the server unchanged; individual late sends are now recorded. Maximum drift in the 10/12/16-user reruns was 7.05/6.89/6.62 ms. RTF certification now uses the unrounded value, not a three-decimal display rounded to `1.000`.

The client observes audio throughout `post-stream-s` instead of disconnecting at D completion. Playback diagnostics accumulate actual PCM duration with a 200 ms buffer per response; pauses between responses are not underruns, and unfinished speech is reported separately. This window may include autonomous continuations and is not an internal Talker/Code2Wav queue witness. GPU sampling runs off the sender event loop.

A formal result additionally requires a clean source tree, captured server/client provenance, diagnostics disabled, zero truncation/fallback/preemption, complete D-prefix evidence, and exact physical KV-transfer evidence. A dirty-tree run remains development evidence only.

## Phase 4: engineering confounds removed

| Confound | Current handling |
|---|---|
| Raw AV copied to D | D receives prompt metadata and imported KV only |
| Full historical KV transfer | P sends only the block-aligned delta |
| D incorrectly replays the last media position | Cache import preserves all computed positions. Native P/D initially computes only P's one generated token per unit; the old two-token case was a correctness bug, not normal overhead |
| Missing P/D sampling and turn state | Bounded state returns to P; seed/offset preserve RNG continuity across units and restore once per session/unit identity, without media or encoder state |
| Answer termination confused with unit termination | Continue after TURN_EOS to the native unit boundary; use final turn state rather than an intermediate TURN_EOS; D only acknowledges units already ended by P |
| Chat EOS/stop rules truncate native units | Native Thinker clears inherited chat stopping rules and stops only on the three native unit tokens; ordinary EOS generation is not masked |
| Extra 28-character cutoff | Remove the non-official character cap and retain the native maximum of 20 tokens per unit |
| D→Talker token/hidden misalignment | Pair each text token with its own post-forward hidden state, not the preceding position that predicted it; skip only P's initial decision token |
| LISTEN consumes Talker's new-response marker | Commit Talker lifecycle only upon actual handoff; reset KV for a new response and retain incremental KV within that response |
| Old buffered audio acquires new response text | Fence outputs by epoch/model turn, reject late old-turn results, and never retain unowned audio across responses |
| Text gating discards audio-only responses | Explicitly owned native non-LISTEN audio creates/continues a response; only actual turn-end ends it |
| Local audio/text coordinates labelled cumulative | Use the playback cursor's cumulative boundaries; do not invent word-level alignment |
| Worker drops chunk identity and repeated text is incorrectly deduplicated | Preserve cache_epoch/chunk_seq/delta metadata; deduplicate by actual chunk identity and reject missing required identity |
| Detached P→D routing reads another stage's tokens | Read the P output's own boundary token, not shared stage scratch state; retain rejection of invalid identities and empty boundaries |
| Empty-text units incorrectly interrupt speech continuation | Non-LISTEN empty-text units still advance TTS as in the official implementation; final turn state controls termination |
| Incomplete chunked prefill corrupts sampling state | P/D/Talker receive actual sampling eligibility through the outer model; incomplete rows do not sample, consume RNG or update turn state, and bookkeeping does not rewind untouched RNG |
| Vision fallback re-encoding | Every measured frame consumes its arrival-preencoded embedding |
| Concurrent first-use audio/vision initialization replaces a cache | One locked runtime initialization preserves the first-frame cache |
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

2026-09-08 validation: 177 regression tests passed. Both placements ran `2 users × 10 s input + 30 s output observation`, with FP8, HD4, zero initial context and seed `20260908`. Each completed 20/20 D inputs and 20/20 frames, with no audio/video fallback, user failure or input backlog. Maximum send drift was 3.75/4.11 ms for colocated/isolated respectively; all four stages attached to private MPS. P and D each actually allocated 932,064 KV tokens.

These are functional probes, not capacity certification: output observation still recorded 5/4 playback-buffer underruns for colocated/isolated, and one isolated response had no observed terminal event by the window end. Their cause is not yet established; the full speech pipeline is not certified. Resolved configs, logs and input/output audits are archived in `/home/ubuntu/data/experiments/minicpm-pd-fairness-20260908/{colocated-2x10-v3,isolated-2x10-v1}/`.

## Phase 5: archived context-rebuild long runs (2026-09-08)

All use MPS, D+Talker sharing, FP8, HD4 and 240 seconds of input as above. Input completion, frame conservation, audio/video arrival-cache hits, context rollover and MPS attachment were verified; no runtime errors, fallback or preemption. Source is uncommitted but snapshotted: these remain development evidence, not formal certification.

| Users | Completed inputs | Ready→D p50/p99 | Late inputs | Maximum lateness | Minimum stream RTF (raw value displayed to six decimals) |
|---:|---:|---:|---:|---:|---:|
| 8 | 1920/1920 | 206/604 ms | 0 | 0 ms | 0.999648 |
| 10 | 2400/2400 | 200/343 ms | 1 | 6 ms | 0.999743 |
| 12 | 2880/2880 | 223/386 ms | 3 | 176 ms | 0.999475 |
| 16 | 3840/3840 | 234/577 ms | 9 | 442 ms | 0.999574 |

Every late input is the first unit; subsequent 239 units, including rollover, have no input backlog. The slowest first unit at 16 users took 1444 ms Ready→D: 1227 ms before D submission and 217 ms of D service. This localizes the delay to the earlier path, not specifically to P forward, encoder first-use initialization or another startup operation.

Conclusion: the per-unit zero-backlog check passed at 8 users; even 10 users had one 6 ms miss and cannot pass that strict rule. **This does not establish an eight-user GPU capacity limit.** The search hit startup jitter before finding a sustained service limit. Minimum finite-window RTF is slightly below 1 at every point, so strict `RTF>=1` also remains unpassed. Rounding cannot erase that difference, nor can this tiny finite-window deficit alone establish growing long-term backlog.

Full speech delivery remains uncertified. The 8/10/12/16-user runs observed 225/229/559/199 underruns with a 200 ms playback buffer and 5/10/10/1 responses without an observed terminal event; malformed PCM chunks were zero. The observation window includes autonomous continuation. These counts neither prove Talker/Code2Wav saturation nor establish smooth playback.

Artifacts: `/home/ubuntu/data/experiments/minicpm-pd-d-talker-capacity-20260908/`. Timing-valid points are `users-{8,10,12}-fresh-v1` and `users-16-fresh-v2`; `users-16-fresh-v1` is invalid due to sender lag. `search-v1` only ran warm-up: a downstream request gauge did not drain, so server reuse was abandoned, not counted as capacity failure. Original `analysis.json` files are preserved; `analysis-exact-rtf.json` is offline reanalysis with the corrected RTF comparison. Input records were not modified afterward.

## Phase 6: existing measurements (before placement-fairness fixes)

The numbers below predate fixed KV budgets and unified encoder execution paths; they are not a new capacity result under the revised checks. Validate functionality with short runs before repeating the matched capacity comparison.

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

## Phase 7: reproduction

Each point starts and stops its own server and private MPS. For example, repeat 16 users:

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 16 --duration-s 420 --kv-window-tokens 36000 --seed 20260908 \
  --out-dir /home/ubuntu/data/experiments/minicpm-pd-sliding-repeat
```

The output directory must be new. The current runner defaults to a separate 12-second warm-up session, then 420 seconds of zero-context measured input and 30 seconds of output observation. Real MP4/reference audio, HD4, random phases and jitter are unchanged. Capacity uses per-session post-window long-horizon RTF; unit deadlines and backlog are diagnostic only. This command does not reproduce the old short-context rebuild policy. Config, commands, provenance, MPS, warm-up, input/output audits and source-stability evidence are archived.

Source snapshots are in `users-8-fresh-v1/{source,vllm-source}` and `users-16-fresh-v2/{source,vllm-source}`; the latter also matches the source during the 10/12-user timed runs. The later RTF precision correction changes only offline analysis. Use diagnostic flags in separate attribution runs, never in capacity timing.
