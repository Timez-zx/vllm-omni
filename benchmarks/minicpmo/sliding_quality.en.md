# MiniCPM P/D: generation and sliding-window verification

Chinese version: [sliding_quality.md](sliding_quality.md).

## Current status (2026-09-09)

**The latest 36k long run passes observable serving contracts; generation quality and serving correctness are evaluated separately. No new capacity is certified.** The official raw-media BF16 encoder/generator/basic-window 420-unit control also enters long control-only output at unit150/context32,504, before eviction. It uses neither vLLM/P-D nor our attention mask: this degeneration does not require our KV-transfer implementation. This does not separate checkpoint from native generation-policy responsibility or prove all serving code correct.

Further work targets input, state, KV and output delivery, not sampling changes to correct content. The default remains 36k; 8k is an archived control, not a training limit or the production fix. Archived generation failures below must not be read as uniquely attributed serving bugs.

## Setup and workload

- GPU0: Thinker-P; GPU1: Thinker-D + Talker; GPU2: Vision/Audio Encoder; GPU3: Code2Wav. Private MPS and pipelined modules.
- P/D/Talker FP8 weights; P/D FP8 KV. The latest controls explicitly retain BF16 Q and force Triton 2D attention on P/D; both diagnostic controls default off.
- Original 960×540 HD4 video at 1 FPS; 16 kHz audio arrives every 200 ms and forms one-second model units. Aligned audio/video loop over the same 35-second clip.
- Seed 20260908, different user phases/media offsets, fresh histories. Separate 12-second warmup, 420-second measured input per user, then 30-second observation.
- Explicitly pin 128 initial tokens; window 36,000 or 8,000. Prefill remains incremental and P→D transfers block-aligned KV deltas, without rebuilding the window.
- Sliding bounds physical KV, not absolute logical positions; the latter are guarded at 262,144. This differs from native whole-unit eviction with K repositioning and is not claimed numerically equivalent or quality-neutral.

## Latest serving validation and output-consumer fix

An independent application bug was found: `_maybe_continue_native_response` awaited the silence-continuation scheduler inside output consumption; that scheduler can wait one input period or a prior silence append. Already-completed D results and audio consequently queued. There is now at most one background scheduling task per session. Output keeps draining while real-input ordering, stale-owner checks and the continuation policy remain. Shielding prevents cancellation of the scheduling wait from cancelling a submitted append; closing the session retires its timer.

A CPU reproducer using the actual runtime bridge confirms the output consumer was blocked by a pending scheduler before the fix and returns immediately afterward. Regression: 1,166 passed, 18 skipped.

All three runs use the setup above: 2×420 s, 36k/pin128, FP8 weights/KV, BF16 Q, Triton 2D, MPS and normal CUDA Graph. No synchronizing KV/speech probes; only post-run client export. Inputs/config match, but freely generated token trajectories are not assumed identical.

| Run | Post-input observation | Actual D completions | Per-user D completion → client receipt p99 |
|---|---:|---:|---|
| r1, before fix | 30 s | 839/840; user 0 lacks its final unit | 34.34 s / 962 ms |
| r2, before fix | 90 s | 840/840 | 17.15 s / 12.30 s |
| r3, after fix | 30 s | 840/840 | **12.95 ms / 12.13 ms** |

This measures completion-event delivery, not TTFA, GPU forward or pure network time. Event-receive monotonic timestamps are aligned offline using the corresponding input-send wall/monotonic anchor; clock-pair and export-rounding error remain. r1 stays incomplete; longer r2 observation was not a performance fix. r3 completes with the original 30-second observation.

Final r3 checks:

- All 840 frames and 840 audio units use arrival encoders, with no fallback, input truncation or request runtime error. Maximum sender drift is 5.64 ms; sources remain unchanged throughout execution.
- Each user covers 254 post-eviction units; final logical prompts are 91,773 / 92,369. P/D have at most 2,259 resident blocks, including eight protected initial blocks. The complete window is replaced without prompt recycling.
- All 840 D input-prefill steps compute only P's newly sampled token. `cached = local + external`, `prompt = cached + 1`, and `transferred = external` hold. Post-eviction delta medians are 226 / 230 tokens, maximum 245, not full-window retransfers.
- Received protocol, session/response ownership, text/audio coordinates and PCM-to-WAV bytes pass: 80 normally completed replies and two separately accounted cancelled replies. Received totals are 1,569 characters and 424.24 seconds of PCM, including delivered cancelled prefixes. Audit: zero FAIL/UNKNOWN.
- **Minimum post-window long-horizon RTF is 0.943; input-processing backlog remains.** Removing output-consumer blocking is not a real-time capacity pass; this experiment does not reattribute the remaining P/D waits.

Event audits do not prove every historical KV value identical or that upstream never omitted a chunk. This run does not recapture source codec or redo ASR/semantic scoring; earlier sampled numerical KV evidence remains separately archived. Degeneration may change decode/speech demand, so `--functional-only` explicitly prevents completed execution being reported as certified normal-dialogue capacity.

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 2 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 --kv-cache-dtype fp8 \
  --triton-disable-q-quantization --triton-force-2d-attention \
  --quality-capture --functional-only --out-dir /path/to/new-run
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/functional_audit.py /path/to/new-run
```

Evidence root: `/home/ubuntu/data/experiments/minicpm-pd-serving-contracts-20260909/`. Runs are `fp8-pin128-w36000-2x420-r1`, `fp8-pin128-w36000-2x420-drain90-r2`, and `fp8-pin128-w36000-2x420-output-drain-fix-r3`. Each preserves config, commands, source snapshots, complete events/WAV, `functional-audit.json`, `analysis.json` and source stability. `final-summary.json` is generated by `summarize_contract_runs.py`; `check_output_drain_scheduler_wait.py` and before/after logs reproduce the blocking. Official raw-media control: adjacent `minicpm-pd-native-reference-20260909/bf16-native-420-topk100-basic36000/`.

Obsolete audit rules were also corrected: current D does not replay the remote final token, so `transferred = external + 1` or two-token replay is no longer accepted. Missing evidence remains UNKNOWN. `--post-stream-s` still defaults to 30 and changes observation only, not input cadence or the RTF formula.

## Archived attribution: identical inputs, generation outside serving

Two additional controls on 2026-09-09 do not certify 36k functionality or capacity:

1. **Fixed-history numerical replay.** A two-user 36k BF16-weight/KV capture records actual P/D input embeddings and output hidden states, with only layer0 KV to bound probe size. Native HF replays 6,503 forwards. The following P unit's committed history excludes 174 discarded D-suffix forwards. Among 3,002 valid D sampling positions beyond absolute position 36k, raw top1 agrees at 2,878 positions; native top1 is always a control token. This is conditional on the serving history and cannot alone establish where that bad history originated.
2. **Native independent generation.** Reuse only one user's system/reference and per-unit AV embeddings, never serving-generated feedback. Released `streaming_generate` and `StreamDecoder.feed/decode` maintain their own outputs and sampling state: BF16/SDPA, top-k100, at most 20 tokens/unit, 240 units. Retain the full physical KV and apply an explicit pin128 + recent-window mask, without P/D, KV compaction/movement/repositioning, or TTS.

| Native independent control | Long control-only units | Longest consecutive span | First masked-out historical KV / last ordinary text |
|---|---:|---|---|
| 36k, seed42 | 81/240 | units 157–223, 67 units | unit168 / unit156 |
| 36k, seed7 | 127/240 | units 149–204, 56 units | unit165 / unit148 |
| 8k mask, seed42, diagnostic only | 0/240 | None | unit39 / unit237 |

Here a long control-only unit has at least ten generated control tokens and no ordinary text, excluding the closing `</unit>`; a normal single LISTEN is not a failure. This unit-level definition differs from earlier TTS-handoff counts. The seed42 8k/36k controls match generated tokens exactly for43 units and first diverge at unit44; the 8k control reaches absolute position50,925. **Visible-long-history generation instability does not require vLLM/P-D or faulty KV movement**, and cannot simply be blamed on an absolute-position overflow. This neither establishes an8k training limit nor fixes36k; the8k control does not replace production configuration.

The capture requests2×300 seconds, but D completes only254/300 and214/300 within client observation. Probes also record teardown execution; those records are not completed capacity work. All native controls complete240 units but do not validate audio output. Scripts, input ledgers, per-position comparisons, raw per-unit tokens, and commands are archived in `/home/ubuntu/data/experiments/minicpm-pd-generation-fix-20260909/teacher-ledger-findings.md`. The selected CPU regression has379 passes this turn. New code adds explicit diagnostics, not a generation repair.

## Confirmed repairs

| Boundary | Repair and evidence |
|---|---|
| Input | Preserve the complete reference WAV instead of truncating 6.016 seconds to 6.000; budget actual processor rows. Last reference-row cosine against native improves from 0.750 to 0.999 |
| Multi-user audio | Each session owns its CPU audio processor rather than sharing mutable fixed/dynamic normalization. Real concurrent Mel maximum error drops from 0.589 to zero; live-session mode changes no longer affect reference encoding |
| Autonomous replies | Remove implicit RMS/silence-to-LISTEN and its state-clearing side effect; retain explicit force_listen. The formal media does not cross the silence threshold, so this is not an established cause of its long-run failure |
| P/D state | Carry bounded sampling/repetition/RNG state and unit identity. D computes only P's sampled token, not media again |
| Detached routing | Read P's own output token, not shared scratch state overwritten by another stage; retain strict identity and boundary checks |
| Native generation | Separate answer-end from unit-end, remove inherited chat EOS/stop and extra character caps; partial prefill does not advance sampling |
| Talker lifetime | Pair tokens with their own hidden rows; reset KV when a new spoken response actually reaches Talker, not on LISTEN; retain incremental intra-response execution |
| Client output | Isolate session/epoch/model-turn/chunk; deduplicate by identity, not text; preserve legitimately owned audio-only output and actual cumulative coordinates |
| Cancellation | ACK_ONLY commits only client-confirmed playback. Actual writer/cancel regressions exclude queued, unplayed tails from history |
| Diagnostics | Do not coerce opaque NIXL handles to integers; missing KV samples are UNKNOWN; incomplete measurements exit nonzero |

Processor-isolation and native-policy long controls are complete: 8k passes input, state and delivery checks, while 36k still degenerates. The combined CPU regression has 997 passes and three skips; test counts do not replace generation verification.

## Latest evidence

| Control | Result |
|---|---|
| 36k, same 2D attention, short run | Same-input/history P/D final-hidden discrepancy drops from 32.07% to zero; inspected complete KV and actual WRITE chains agree. No sliding or normal speech coverage |
| 36k, before full-reference repair, 2×420 seconds | All 840 inputs complete; 32 actual transfers and 469,368 layer/position raw-KV/scale comparisons match. Generation still loops; all 52 WAVs receive ASR |
| 36k, complete reference input, 2×420 seconds | D completes only 407/420 and 402/420 inputs before observation ends. First abnormalities near 141/59 seconds precede sliding. From about 157 seconds, users produce 56/59 consecutive control-only handoffs and never recover normal dialogue |
| Delivery audit of that run | All 109 normally completed replies conserve text/codec/PCM; all 111 client WAVs receive ASR. Two cancelled replies deliver only continuous prefixes; two additional background replies are not delivered and are accounted separately. 86/109 normal replies contain PCM without ordinary text |
| 36k, complete reference + isolated audio + native policy, 2×420 seconds | D completes 391/420 and 401/420. Source abnormalities begin near53–55 seconds and11k context, before sliding at unit166, and persist afterward.74 normal replies conserve delivery; all76 WAVs receive ASR. Usability fails |
| 8k, same source/precision/input, 2×420 seconds | All840 inputs complete,383 post-window units each, at most509 KV blocks; no input backlog, fallback or runtime errors. No prolonged control-only loop in89/183 source handoffs |
| 8k speech audit | All44 replies finish normally, without cancelled tails.771 characters,5,911 codecs and241.72 seconds of PCM are conserved; all44 Talker turns start at position0. All44 WAVs receive ASR and broadly correspond to source text. Off-topic and counting errors remain; this is not perfect model capability |
| 36k, same-topology BF16 Thinker weights/KV, 2×420 seconds | D completes 418/420 and 420/420. First control-only output appears near 86/93 seconds, before sliding at units 165/167. Longest consecutive control-only streaks are 57/42 handoffs, without subsequent normal-dialogue recovery; usability fails |
| BF16 speech audit | All 53 Talker turns start at position0; 50 normal replies conserve 417 characters and 425.56 seconds of PCM. All 51 WAVs receive ASR, with degenerate speech still present. One cancelled reply's unreceived 180-character, 10-second tail is accounted separately, not as normal-reply loss |
| 8k, no internal probes, normal CUDA Graph, 2×420 seconds | P/D compile and capture succeed; all840 inputs and383 post-window units per user complete, with no input backlog, fallback or runtime errors; protocol audit has zero FAIL/UNKNOWN. All42 replies finish normally, delivering1,067 characters and279.12 seconds of PCM. All42 WAVs receive ASR; late replies remain responsive |

The initial control-only signal is a non-LISTEN TTS handoff with no ordinary text and at least ten control tokens. One signal is not automatically a failure; sustained runs and subsequent recovery are checked separately. These intrusive diagnostics show tens of seconds of backlog and cannot establish production capacity or a unique performance cause.

Earlier 8k BF16-KV and FP8-KV/BF16-Q controls each complete 840 inputs and 383 post-window units per user without reproducing the old 182-second phrase loop. They predate the complete-reference repair and support retesting 8k, not certification of the latest implementation.

The latest 8k/36k FP8 controls have identical source snapshots and differ materially only in window configuration. However, free-running sampled outputs already differ before the first slide; one comparison cannot establish a unique numerical cause. 8k has positive operating evidence, not proof that 36k is fixed; defaults remain unchanged.

The numerical-probe controls above use `enforce_eager=true`. An additional8k control therefore removes numerical/speech probes and retains only post-run client export, exercising normal compilation/graphs. Basic input, sliding and delivery checks pass, but source codec/hidden is not recaptured. ASR disagrees with parts of long number sequences; Whisper-small alone cannot distinguish TTS pronunciation from recognition error. All recordings remain available, without verbatim-speech or MOS certification. The original capacity report also remains restricted by dirty source trees and its obsolete remote-last-token replay check; it is not relabelled as passing capacity.

The BF16 control changes P/D weights and KV to BF16 and explicitly reserves 48 GiB of KV per worker to accommodate the colocated Talker, still enough for both users' windows. Talker remains FP8; topology, media, pin128, 2D attention and model/serving sources remain unchanged. This rules out FP8 as a necessary cause and first eviction as the initial trigger, not all serving bugs. Shared static KV scales were separately evaluated on short CPU probes, not loaded into serving or presented as a quality repair.

## What is and is not established

- Inspected KV transfers are correct; they do not support transport corruption as the explanation. Long-run byte comparisons cover their explicitly recorded sampled positions, not every position.
- Matching 2D attention removes the short-run discrepancy but not long-run degeneration. After sliding, 16-token block shifts change 32-token reduction-tile grouping; equal visible history can still produce numerical differences. Causation of generation failure is unproven.
- Official BF16 inference without vLLM/P-D/TTS also degenerates with the large window, while the 8k control continues answering. This does not exonerate serving. Official long controls use top-k100 versus serving20; a native150-unit comparison matches token-for-token, but is not full sampling equivalence.
- `sent_ms` measures enqueueing, not successful network delivery. Cancelled tails must be audited by reply ownership, not treated as losses from normally completed replies.
- ASR assists text/speech checking; it is not MOS, factual accuracy or universal correctness.
- Normal-dialogue capacity is not certified. Future serving measurements must explicitly describe output-load changes caused by degeneration, without treating content errors as direct evidence of broken KV/protocol handling.

## Reproduction and artifacts

```bash
/home/ubuntu/miniconda3/envs/omni/bin/python benchmarks/minicpmo/run_pd_placement.py \
  --topology d-talker --users 2 --duration-s 420 \
  --kv-window-tokens 36000 --pinned-prefix-tokens 128 \
  --kv-cache-dtype fp8 --triton-disable-q-quantization \
  --triton-force-2d-attention --quality-capture \
  --speech-probe-dir /path/to/new-speech-probe \
  --numerical-probe-dir /path/to/new-numerical-probe \
  --numerical-probe-seqs 1-3 --out-dir /path/to/new-run
```

This is a correctness diagnostic, not a capacity command. An 8k control explicitly changes the window. Deliberately interrupted warmups are not formal results.

The BF16 control uses `--thinker-quantization none --kv-cache-dtype bfloat16 --thinker-kv-cache-memory-gib 48`, retaining other arguments. The reservation option affects P/D only, not other stages or deployment defaults.

For the no-internal-probe control, set the window to `8000`, omit `--speech-probe-dir`, `--numerical-probe-dir` and `--numerical-probe-seqs`, and retain `--quality-capture` plus the precision/attention flags.

- Root: `/home/ubuntu/data/experiments/minicpm-pd-generation-fix-20260909/`.
- Latest complete-reference failure: `pin128-w36000-fullref-qbf16-kvfp8-2d-2x420-r1/`, containing setup, source/MPS records, logs, input completions, WAVs/ASR, `source-generation-findings.json`, `speech-delivery-owner-audit.json` and `functional-audit-v4.json`.
- Latest isolation controls: `pin128-w{36000,8000}-fullref-isolated-qbf16-kvfp8-2d-2x420-r2/`.8k has zero FAIL/UNKNOWN protocol checks; speech and source-generation reports are in each run. The deliberately stopped8k r1 contains warmup only and is not this result.
- BF16 control: `pin128-w36000-fullref-isolated-bf16all-2d-2x420-r1/`, retaining source-generation, per-reply delivery, WAV/ASR and protocol audits. Only 838/840 inputs complete; no capacity certification.
- Normal-graph control: `pin128-w8000-fullref-isolated-qbf16-kvfp8-2d-noprobe-2x420-r1/`; see its `final-functional-findings.md` for results and evidence limits.
- Numerical evidence: root `attention-2d-short-result.md`, `attention-2d-long-numerical-result.md`; reference evidence: that run's `reference-input-comparison.json`.
- Native policy and controls: root `native-context-policy-audit.md` and neighboring `minicpm-pd-native-reference-20260909/`.
- Protocol audit: `python benchmarks/minicpmo/functional_audit.py RUN --out RUN/functional-audit-v4.json`. Passing observable contracts does not certify generation quality.
