# Where this stands

## Read this first: THREE separate faults sat on the audio path, all fixed

"The first reply makes some sound, later replies make none, and what you do hear is
only the start of a sentence" was not one bug. It was three, in three different
components, and each is written up below:

| | where | effect on its own |
|---|---|---|
| Playback prebuffer + underrun re-arm | browser | deliveries fall silent in turn; what plays is the previous one |
| Leaked streaming-parked counter | engine scheduler | text-only forever after exactly `max_num_seqs` sessions |
| Audio delta extractor | API server | every reply capped at its first granule, 0.22 s |

That is why it looked so erratic, and why each partial fix looked like it had not
worked. **All three are fixed and verified on a running server.**

**Verified after the restart: 8 consecutive sessions, every one delivered audio**
(6 direct + 2 through the page-server proxy), first audio 319–434 ms after a
1,324 ms first-request warm-up. Before the fix the fifth session would have been
text-only. And the fix is doing work rather than the leak having gone away — the
engine log shows the clamp firing, three times, each repairing a leak of exactly
one slot on stage 1:

```
stage 1 streaming-parked counter had leaked 1 slot(s) (was 1, actually parked 0 of 0 tracked); clamped.
```

with zero occurrences of `has sampled ZERO output tokens` or `looks WEDGED`. So the
leak is still real and still happens once per session; it is now repaired every pass
instead of accumulating to `max_num_seqs`.

**Cause 1, in the browser: the playback prebuffer.** `PLAYBACK_PREBUFFER_MS` was
250 ms while the delivery it had to start on was ~220 ms, and an underrun re-armed
the threshold. Neither is the bug alone: an oversized threshold alone just delays the
start by one delivery, and re-arming alone is harmless below one delivery. Together,
each delivery has to be paid for out of the next. Fixed, and
`node playback_test.js` reproduces all three cases against the real worklet.

**Cause 2, in the engine: a leaked slot counter closes admission for good, after
exactly `max_num_seqs` sessions.** `num_waiting_for_streaming_input` is upstream's
hand-maintained count of requests parked for streaming input, and every decrement
is guarded on the status, so any path that changes the status before removing the
request leaks one. `get_num_unfinished_requests` already derives around this leak —
but there is a **second consumer** that derivation does nothing for. Upstream's
waiting loop gates admission on

```
num_running = len(self.running) + self.num_waiting_for_streaming_input
if num_running >= self.max_num_running_reqs: break
```

so once the counter reaches `max_num_seqs` the loop breaks on its first pass,
forever. Measured on stage 1 with `max_num_seqs: 4` — the counter climbed **1, 2,
3, 4 across four sessions**, and the fifth got text from stage 0 and never one
audio token:

```
stage 1 num_waiting_for_streaming_input is 4 but 0 request(s) ... are actually parked
req=video-sess-... has been tracked for 45s and has sampled ZERO output tokens
no request advanced for 45s ... this stage looks WEDGED, not idle (stage 1)
```

**The engine survived exactly `max_num_seqs` sessions and then served text-only
forever.** That reads as a client bug from the browser: the reply arrives, only the
voice is missing. Fixed by `OmniARScheduler._clamp_streaming_parked_counter()`,
which counts parked requests over `self.requests` — not over the queues, because
the chunk transfer adapter holds parked requests out of both, and a queue-derived
repair was tried before and left the counter at -1. Four unit tests cover it.

Neither `selftest.py` nor `probe.py` could have caught either one: the first checks
the protocol, the second never plays a sample and, at the time, ran while the
counter was still under 4.

## Prefill-on-arrival: measured wins, then a diagnosed architectural limit. OFF.

`prefill_frames_on_arrival` appends each retained frame to the live session request as it
ARRIVES. It works, it is measurably better on latency, **and it kills the talker.** All three
statements are supported; the last one decides the default.

### What it buys (measured, before the crash was understood)

| frames/turn | OFF median | ON median | ON − OFF |
|---|---|---|---|
| ~1 (`frame_filter_min_gap: 8`) | 355.4 ms | 345.5 ms | −9.8 ms |
| 3 (`min_gap: 2`) | 419.2 ms | 349.9 ms | **−69.2 ms (−16.5%)** |

Tripling frames costs OFF +63.8 ms and ON +4.4 ms — **~16× less sensitive to frame rate**,
which is the decoupling the feature exists for. Spread also tightens 6× (OFF 344–379 ms,
ON 343–349 ms).

### Why it is off: stage-0 / stage-1 context divergence

The append advances **stage 0's** context and, by design, ships **nothing** to stage 1. But
the talker consumes thinker embeddings *per position*: the positions the append created never
reach it, so the two stages' token accounting diverges. The talker then indexes its ~4k-row
codec embedding table with text-vocabulary ids and

```
Indexing.cu:1515: indexSelectSmallIndex: Assertion `srcIndex < srcSelectDimSize` failed.
gpu_model_runner.py:1790 in _preprocess
    self.talker_mtp_input_ids.gpu[decode_slice].copy_(req_input_ids)
```

poisons the CUDA context and the stage-1 process dies, taking the engine with it.

**Controlled, because "it crashed" is not evidence on its own** (`crash_repro.py`, identical
hostile pacing — continuous frames, mic pausing during replies, barge-in queries every 2.5 s):

| arm | result |
|---|---|
| `--no-prefill` | **survived 6 turns / 398 frames** |
| prefill on | **died on turn 1**, repeatedly |

And it still dies with **all three payload gates confirmed firing** (`not prewarming`,
`dropping stage`, `withholding content` in the log). That is what rules out the whole class of
"a payload leaked through" explanations and points at the divergence itself. Withholding a
segment from the talker is not a gap to be plugged; it is the design.

### Three real defects found on the way, all fixed and all independent of this feature

1. **Per-chunk `max_tokens` was silently ignored.** Upstream carries it on every
   `StreamingUpdate` and never applies it, so `Request.max_tokens` keeps the FIRST chunk's
   value for the whole session while the stop check compares per-segment output counts
   against it. Measured: an append submitted with `max_tokens=1` generated ~20 tokens. Fixed
   in `_update_request_as_session`; a no-op for ordinary turns, which all carry the same value.
2. **A cross-thread read of per-segment state.** `save_async` enqueues a *reference* to the
   request; by the time the save thread dequeues it, the next streaming update may have
   replaced `sampling_params`. Both directions of that race were observed within an hour. The
   decision is now snapshotted at enqueue time, on the scheduler's thread.
3. **The launcher truncated the log**, destroying a crash's stack the moment the natural next
   step (restart) was taken. It now keeps `$LOG.prev`.

Plus two things that make the next attempt cheaper: the session's non-default config is logged
once per session (a post-mortem could not tell which features a browser tab had enabled), and
`CUDA_LAUNCH_BLOCKING` in the launcher's environment **does not reach the stage processes** —
their env is rebuilt at spawn, and the stage-level `runtime.env` block did not apply either.

### The crash: root-caused, fixed, and the feature is ON

`prefill_frames_on_arrival` is **enabled by default**. The engine no longer dies.

**The out-of-range index was `-1`, not a text-vocabulary id, and there were TWO defects, each
hiding the other** — which is why four earlier attempts all "failed identically" while failing
for two different reasons at once.

**Defect 1 — `-1` reaching `codec_embedding`.** Under async scheduling `token_ids_cpu` never
holds a sampled id: vLLM writes the sentinel `-1` and patches the real value onto the GPU row
from `prev_sampled_token_ids`, which reaches only requests present in the **previous forward's
batch**. An append parks the talker, so it leaves the persistent batch and is re-admitted on an
**output** row — readable only from CPU, and reads `-1`. `codec_embedding` is
`nn.Embedding(3072, hidden)`, so `indexSelectSmallIndex: srcIndex < srcSelectDimSize`.
Independently confirmed: `nn.Embedding(3072,16).cuda()` indexed with `-1` reproduces that assert
at the **same file and line**; index 2150 (the largest id the talker can sample) is fine, which
is why the shipping path never tripped it. **Fixed by `async_scheduling: false` on stage 1.**

**Defect 2 — a prefill tensor labelled as a decode payload.** An append stops on the very
forward that prefills it, and the scheduler clears `_output_token_ids` before `save_async`, so
`thinker2talker_async_chunk` took its decode-shaped path and shipped the append's `[222,1024]`
prefill tensor as `embed.decode`; the runner copies that into a **one-row** decode slot:
`output with shape [1,1024] doesn't match the broadcast shape [222,1024]`. Invisible until
defect 1 was fixed, because the CUDA assert killed the process first, on the same turn.

**The discriminator is STRUCTURAL, not a transported marker.** Three marker channels were tried
and all three silently read False at the point of use (`additional_information` as a dict and as
a real payload, `SamplingParams.extra_args`, and an enqueue-time snapshot): `sampling_params` has
already been replaced by the next streaming update by the time either the scheduler or the save
thread looks. What cannot be lost is a property of the forward itself — **more than one row of
thinker embeddings AND the segment ending** means the segment produced no text, so there is no
talker obligation. With the feature off, a segment's first forward never stops, so the branch is
unreachable — verified by windowed log count (0 hits), not by argument.

**Three further real fixes, all independent of this feature:**

1. Per-chunk `max_tokens` was silently ignored — upstream carries it on every `StreamingUpdate`
   and never applies it, so `Request.max_tokens` kept the first chunk's value for the whole
   session. Measured: an append submitted with `max_tokens=1` generated ~20 tokens.
2. The turn was claimed only *inside* the task `create_task` schedules, so this turn's outputs
   arriving in that window were swallowed by the append suppression (2 bad turns in 16). Claimed
   synchronously now, with a separate flag so the overlap refusal is unaffected.
3. `t_first_text` was stamped on the first non-empty **delta**, not the first text output. The
   first stage-0 output often carries no new text, so `first_text` landed ~17 ms *after*
   `first_audio` and a health check called a clean turn misattributed. Measuring the wrong
   instant is not attributing to the wrong turn.

**Appends require an empty queue.** They share the session queue with turn deltas, so anything
still queued means a turn's chunk has not been consumed and an append would land between that
chunk and its answer — measured on the proxy path as `Overlapping turn` plus a turn reporting
`first_text=-1.000s chars=0 audio_chunks=1`. `turn_busy` alone does not cover it: the flag clears
when the turn's body returns, while its chunk may still be queued.

### Verified

| check | result |
|---|---|
| proxy path (what the browser uses), 6 turns | **6/6 clean**, 0 anomalies, no overlap, first audio 340–348 ms |
| hostile pacing, 8 turns / 90 frames | engine alive |
| hostile pacing, 398 frames | engine alive |
| feature OFF control | clean, and the new branch unreachable (windowed count 0) |
| `selftest.py` / `playback_test.js` / scheduler units | 17/17, 13/13, 14/14 |
| A/B, direct path | ON 83–256 ms vs OFF 378 ms |

**Known residual, disclosed rather than hidden.** Under the A/B's deliberately hostile pacing
(query every 2.5 s, media never pausing), 3 turns in 16 show `first_audio` before `first_text`
by **9–59 ms**: a trailing audio output from the previous turn landing on the new turn's state
after the segment boundary rotated it. It is a tail of tens of milliseconds, not a misattributed
reply, and it inflates the A/B's ON figure — so treat −78% as an upper bound and the −32%
measured with clean accounting as the honest one. The proxy path shows none of it.

## Done and verified

| | |
|---|---|
| Branch `live-agent-web` | upstream `main` + our 26 Qwen commits. **One** merge conflict, and it was not a real one |
| Optimisations survive the merge | all 19 `StreamingVideoSessionConfig` fields, `TALKER_TEXT_ONLY` on, the session roll / orphan recovery / heartbeat intact — checked by import, not assumed |
| Browser client | written; `node --check` clean on all three JS files |
| Page server | renders, injects its config, serves all four assets — verified with a FastAPI TestClient, no GPU |
| `selftest.py` | **14/14 passing**, no GPU |
| `playback_test.js` | **5/5 passing**, no GPU and no browser — runs the real worklet under Node |
| `probe.py` | written; drives the whole chain with synthetic media, no browser |
| The page | rewritten layout: transcript-dominant split, state as coloured pills, light **and** dark |

Neither conda env has `pytest`, and adding one would mutate the env the 0.24.0
measurements depend on. To run the scheduler tests, stub `pytest.mark` and call the
functions directly — `/tmp/.../scratchpad/run_sched_tests.py` does this, 14/14.

## In progress when this was written

**The Qwen server coming up.** Two real obstacles were hit and fixed on the way,
both of them consequences of moving from vLLM 0.24.0 to 0.26.0:

1. **A 402-second silent phase** that looks exactly like a hang. Qwen3-Omni is 30B
   MoE and FlashInfer JIT-compiles a CUTLASS fused-MoE kernel on first use. GPU at
   0%, log silent, process in `pipe_read` — and `ptxas` at 100% five forks down.
   Cached after the first build. See the README for the one command that tells
   compiling from hung.
2. **Stage 2 OOM, then my own overcorrection.** The measurement config
   (`harness/deploy_pc_stage0.yaml`, 0.74/0.12/0.06) fits on 0.24.0 and does not on
   0.26.0 — `Tried to allocate 1.41 GiB ... 1.04 GiB is free`, after stage 1's CUDA
   graph capture had taken resident memory to 85 GB. FlashInfer's kernels want
   workspace the fractions did not budget, and another user's MPS server holds a
   slice they cannot see.

   Cutting stage 0 to 0.62 to make room **killed stage 0 instead**: its weights are
   59.4 GiB and 0.62 × 94.97 = 58.9, so it reported
   `Available KV cache memory: 0.0 GiB` and died. **Stage 0's fraction is a floor,
   not a preference.**

   The working shape is `deploy_web_demo.yaml`: stage 0 back at **0.74**, and the
   headroom taken from three other places — `enforce_eager` on stages 1 and 2,
   `max_num_seqs` 16→4, and `max_num_batched_tokens` 16384→8192 on the thinker.
   A **new** file rather than a retune of the measurement config, which must stay
   byte-identical or every number in `workflow.md` loses its baseline.

Check where it got to:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8091/health   # want 200
tail -5 /data/zx/results/qwen_live.log
```

## Then, in order

```bash
cd /home/zx/voice-agent/vllm-omni
PYTHONPATH=$PWD python benchmarks/live_agent/web_client/probe.py --direct   # no browser
bash benchmarks/live_agent/web_client/run_page_server.sh                    # then ssh -L 7870
```

## The truncation: SOLVED

**Was:** the reply's audio was ~0.22 s no matter how long the reply, so only the first
small codec granule ever reached the client. **Now:** 13.58 s delivered of 13.66 s
produced on a 151-character reply, and audio duration tracks the reply (2.11–13.58 s
across turns) instead of being pinned. First-audio latency unchanged, 344–402 ms.
The 0.08 s difference is the CausalConv frame stripped from the first granule, by
design.

### What it was

The delta extractor read a bare tensor as a transient first state that would "become
a list", emitted it once, and answered `None` to every output after it:

```python
if not isinstance(audio_data, list):
    if chunks_drained >= 1:
        return None, chunks_drained
```

It never becomes a list. Any streaming request is coerced to
`RequestOutputKind.DELTA` (`entrypoints/utils.py`, `maybe_coerce_to_message_type`),
and under DELTA the output processor calls `drain_delta_payload()` after every
snapshot, which pops the audio key outright. **Each output therefore carries only
what stage 2 produced since the previous one** — a fresh granule every time, with
nothing cumulative to index into.

The measured shape of one reply, per stage-2 output:

```
7,125 samples        <- the deliberately small first granule
48,000  x 6          <- 25 codec frames x 1920
32,640  finish_reason=stop
= 13.66 s produced,  0.22 s delivered
```

A cumulative payload cannot look like that — lengths would grow monotonically and
could never drop to 32,640 — so the contract is per-step by measurement as well as by
construction.

### Two things that made this take longer than it should have

**The A/B that cleared the delivery path was worthless, and looked authoritative.**
`fast` and `slow` both gave 0.22 s, which was read as "the delivery path is innocent".
Both arms had the same bug: `slow` fell into `full_np[0:0]` the moment
`chunks_drained` reached 1. **Two implementations of one wrong assumption agree with
each other, so their agreement proves nothing.** An A/B only rules a component out if
the arms fail independently — check that before trusting one.

**The diagnostic actively pointed away from the bug.** The `[session-out]` log
reported `output.audio_data`, which is not the field the extractor reads. It printed
`audio_n=0` on every audio output, including the ones carrying the reply — asserting
that no audio was produced while 13.66 s was being produced. Reporting a different
field from the one that matters is worse than reporting nothing. It now summarises the
real source, shapes only, no device-to-host copy.

### Guarded against

`selftest.py` replays the measured granule shape through both delta modes and asserts
the whole reply survives. The old logic delivers 5,205 of 133,845 samples against it,
so the check has teeth rather than merely passing.

### A number this unlocked: rtf below 1

With the whole reply finally delivered, real-time factor is measurable. Per turn,
wall time spanned by the stage-2 granules against the audio they contain:

| audio | span | rtf |
|---|---|---|
| 9.10 s | 6 s | 0.66 |
| 9.95 s | 6 s | 0.60 |
| 6.91 s | 5 s | 0.72 |
| 2.46 s | 1 s | 0.41 |
| 2.19 s | 1 s | 0.46 |

**So this pipeline generates speech faster than real time**, where MiniCPM-o measured
**rtf 1.37** and could not. Two caveats, both real: log timestamps are
second-resolution so this is ±1 s (trust the longer turns), and the MiniCPM number
came from a different harness, so this is indicative rather than a matched comparison.
A matched measurement is now worth doing — it is the project's sharpest question.

## Genuinely unknown

The Qwen numbers — 513 ms TTFA, 40/40 turns with 2 rolls, the 817→33 talker cut —
were **all measured on vLLM 0.24.0**, and this branch runs **0.26.0** with a
different deploy config (fewer sequences, stage 1 eager). **They are not
comparable here until re-measured.** The merge proves the code is present; it
proves nothing about performance. `benchmarks/live_agent/harness/` is
branch-agnostic, so re-running it is how to find out — and `DEPLOY_CONFIG=` on the
launcher can select the original config if the card is free enough to take it.

## The honest regression

The turn trigger is in the browser, not the model. Qwen3-Omni has no listen/speak
token and `should_trigger_turn()` returns `False`, so nothing server-side can
decide you have stopped talking. `selftest.py` asserts that is still true, so if
upstream ever changes it, the test says so.

The MiniCPM arm (`live-agent-minicpm`) has the model deciding, measured at 470 ms
to first audio and **rtf 1.37 — it cannot keep pace with real time**. That is the
arm to compare against, and the difference is real rather than an implementation
shortcut.
