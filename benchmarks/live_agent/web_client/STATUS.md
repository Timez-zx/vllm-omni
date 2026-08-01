# Where this stands

## Read this first: the silence had TWO causes, both fixed and both verified

The reported symptom — the first reply makes some sound, later replies make none —
had two independent causes, either of which produces roughly that.

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
250 while a turn delivers ~220 ms of audio, and an underrun re-armed the threshold.
Neither is the bug alone: an oversized threshold alone just runs a turn behind, and
re-arming alone is harmless below one turn. Together, turns fall silent and what
plays is the previous reply. Fixed, and `node playback_test.js` reproduces all
three cases against the real worklet.

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

## A REAL BUG, found by the probe

**The reply's audio is truncated to about 0.22 s regardless of how long the reply
is.** Established, not suspected:

| text length | audio delivered |
|---|---|
| 34 chars — "Hello! How can I assist you today?" | 0.22 s |
| 104 chars — "One, two, three, … fifteen." | **0.22 s** |

0.22 s at 24 kHz is ~5,280 samples, which is exactly
`initial_codec_chunk_frames: 4` (4 × 1920) minus the one frame the first emit
strips as a CausalConv artifact. **Only the first small codec granule reaches the
client.**

Server-side accounting agrees that more was produced: `audio_chunks=3` per turn,
three `type=audio stage=2` outputs with the third carrying `finish_reason=stop`.
So stage 2 ran three times and only the first delta carried new samples.

Latency itself is fine and repeatable: **first audio 340–376 ms** across five
turns, which is the same order as the 513 ms baseline.

### A/B 1: our talker change is NOT the cause — ruled out

```
TALKER_TEXT_ONLY=1 (default) :  0.22 s
TALKER_TEXT_ONLY=0 (control) :  0.22 s     identical
```

Same probe, same query, same config otherwise. So the truncation is **upstream's
behaviour on this path**, not something the merge introduced. Our change stays on.

That also rules out the obvious reading of `initial_codec_chunk_frames: 4`. A small
FIRST granule is by design; the bug is that **the granules after it never arrive**.
Server-side, `audio_chunks=3` while only one delta carried new samples — so
`_extract_audio_delta_b64` found `audio_data[chunks_drained:]` empty twice, meaning
stage 2's audio tensor list did not grow across its three outputs.

### A/B 2: the delta path is NOT the cause either — also ruled out

```
VLLM_VIDEO_AUDIO_DELTA_MODE=fast (default) :  0.22 s
VLLM_VIDEO_AUDIO_DELTA_MODE=slow           :  0.22 s     identical
```

`slow` re-concatenates the whole audio buffer on every call instead of emitting
only the new tail, so if the fast path's `chunks_drained` bookkeeping were dropping
samples, `slow` would have recovered them. It did not.

### So the waveform really is 0.22 s, and the loss is upstream of delivery

Both plausible client- and serving-side causes are eliminated by experiment. What
remains is the **talker → code2wav** path: either the talker sends codec chunks for
only the first granule, or code2wav stops after producing one. Everything after that
— the delta extraction, the WAV encoding, the websocket, the page — is demonstrably
faithful to what it is given.

Where to look next, in order of how cheap it is:

1. `stage_input_processors/qwen3_omni.py` — `talker2code2wav_async_chunk`. There is a
   known-suspicious line there: `chunk_length = length % chunk_size_config` computed
   over `code_prompt_token_ids[request_id]`, which is a modulo over the request's
   **whole lifetime** and is only reset on `finished`. A per-turn accumulator that
   never resets would produce exactly this shape.
2. The `codec_left_context_frames: 25` value, which differs from MiniCPM's `3`.
3. Whether the same truncation happens on the **older** engine with the same probe —
   that separates "0.26.0 regression" from "always been like this and the harness
   never noticed because it measured time-to-first-audio, not total audio".

Number 3 is the one that decides whether this is new. `git checkout live-agent`,
the `omni` env, the same probe.

**Until this is resolved, do not judge audio quality in the browser** — you would be
listening to a fifth of a second. Everything else about the chain is verified.

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
