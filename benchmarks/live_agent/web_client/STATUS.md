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
