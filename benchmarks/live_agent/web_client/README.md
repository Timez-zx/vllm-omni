# Browser live session for Qwen3-Omni

Talk to the Qwen3-Omni pipeline — with the session-scoped request, the automatic
session roll, the frame downscale and filter bounds, and the text-only speech
stage — from a browser, using your own camera and microphone.

Runs on branch `live-agent-web`: upstream `main` (which has the browser-client
machinery and the newer engine) merged with the optimisations that were developed
against v0.24.0.

---

## What this is, and what it is not

**It is** continuous streaming: frames and audio go up as independent messages
and never stop, including while the model is talking — anything you say then is
kept for the next turn.

**It is not** a natively full-duplex model. Qwen3-Omni has no listen/speak token
in its vocabulary, and this server's `should_trigger_turn()` returns `False`
unconditionally, so **nothing on the server side will ever decide you have
finished speaking.** This page therefore decides, with either a silence detector
or a hold-to-talk button, and it says so on screen. That is a genuine difference
from `examples/online_serving/minicpmo/realtime_web`, where the model itself
owns the decision — do not confuse the two when comparing them.

---

## Run it

### 1. Server (about 2–3 minutes)

```bash
bash benchmarks/live_agent/web_client/run_qwen_server.sh
```

It refuses to start if the GPU is not free — the card is shared, so check rather
than assume, and never kill someone else's job. It also asserts that the
importable `vllm_omni` is this checkout and that the merged config fields exist,
because a server on `site-packages` would come up healthy with none of the
optimisations and nothing would say so.

### 2. Page server

```bash
PYTHONPATH=/home/zx/voice-agent/vllm-omni \
  /home/zx/miniconda3/envs/omni-minicpm/bin/python \
  benchmarks/live_agent/web_client/server.py --port 7870 --ws-backend ws://127.0.0.1:8091
```

It serves the page **and** same-origin-proxies the websocket to
`/v1/video/chat/stream`, so only one port has to reach your Mac.

### 3. Your Mac

```bash
ssh -N -L 7870:127.0.0.1:7870 <server>
```

Open **`http://localhost:7870/`**. `localhost` is a secure context, so the
browser grants camera and microphone without any certificate.

Press **Start call**, then speak. Turn the **Camera** on to let it see you.

---

## Before you trust it: four checks that cost nothing

```bash
# 1. the client's assumptions against the real server code -- no GPU needed
PYTHONPATH=/home/zx/voice-agent/vllm-omni python benchmarks/live_agent/web_client/selftest.py

# 2. the playback worklet, driven over several turns -- no GPU, no browser
node benchmarks/live_agent/web_client/playback_test.js

# 3. can the speech actually PLAY without gaps -- needs the server, no browser
PYTHONPATH=/home/zx/voice-agent/vllm-omni python benchmarks/live_agent/web_client/audio_timeline.py --direct

# 4. the whole chain without a browser: synthetic frames + audio in, audio out
PYTHONPATH=/home/zx/voice-agent/vllm-omni python benchmarks/live_agent/web_client/probe.py
PYTHONPATH=/home/zx/voice-agent/vllm-omni python benchmarks/live_agent/web_client/probe.py --direct
```

`selftest.py` catches the failures that are invisible at runtime: a config field
the server does not have (the session silently runs on defaults), a message type
it does not dispatch, a wrong audio container.

`playback_test.js` runs the real worklet class under Node with the two
`AudioWorklet` globals stubbed. It exists because the bug that made later turns
silent was reachable by neither of the other two: `selftest.py` checks the
protocol and `probe.py` never plays a sample, so a browser was the only thing that
could catch it. Now it is caught in 200 ms.

`audio_timeline.py` answers a question `probe.py` structurally cannot: **"did audio
arrive" is not "can it play straight through."** It records the arrival time and
duration of every delta, then replays the client's buffering rule over those
timestamps to find where a player would run dry and for how long. A turn can deliver
every sample it produced and still be unlistenable, because continuity is a property
of the arrival schedule, not the total.

`probe.py` localises everything else. Run it **through the page server** and then
`--direct` to the engine: if `--direct` passes and the other does not, the proxy
is at fault; if both pass and the browser does not, it is the page or the tunnel.

---

## The first start is much slower than you expect, and looks hung

Qwen3-Omni is 30B **MoE**, and FlashInfer JIT-compiles a CUTLASS fused-MoE CUDA
kernel the first time it runs. After `torch.compile took Ns` the log goes silent
for **many minutes**, GPU utilisation reads 0%, and the process sits in
`pipe_read`. Every one of those says "hung" and all of them are wrong.

The work is several forks down: Python -> ninja -> nvcc -> sh -> **ptxas**, and
only the leaf burns CPU. To tell compiling from hung:

```bash
ps -eo pid=,pcpu=,comm= --sort=-pcpu | head -3     # want ptxas or cicc near 100%
sudo env "PATH=$PATH" py-spy dump --pid $(pgrep -f StageEngineCore | head -1)
```

If the stack shows `run_ninja` / `get_cutlass_fused_moe_module`, it is compiling —
wait. It is cached afterwards, so later starts skip it.

MiniCPM-o never hits this: it is a dense 9B with no fused MoE. So do not use its
startup time (~2 min) as the expectation for this one.

## Reading the page

| Panel | What it tells you |
|---|---|
| Model | Listening → Thinking → Speaking |
| Playback | `Underrun xN` means audio arrived slower than it played. Between turns this is normal; during one, see the prebuffer trade below |
| Events | Every protocol message, including `session rolled` |

**`session rolled` is expected, not an error.** It is the mechanism that lets the
conversation outlive the speech stage's length limit: the engine request is
retired while healthy and reopened seeded with the last 8 turns of text. That one
turn is slower, and older *visual* detail does not cross.

---

## Knobs

Client-side, in `app/static/app.js`:

| Constant | Default | When to change it |
|---|---|---|
| `FRAME_INTERVAL_MS` | 500 | ~2 fps. Raise for less video cost, lower for a fresher view |
| `SILENCE_RMS` | 0.012 | Raise it if a noisy room keeps triggering turns |
| `SILENCE_HANG_MS` | 700 | How long a pause has to be before it counts as your turn ending |
| `MIN_SPEECH_MS` | 400 | Ignores coughs and door slams |
| `PLAYBACK_PREBUFFER_MS` | `{fast: 60, smooth: 1400}` | `fast` is the default and is gap-free at `codec_chunk_frames: 4`; see below |
| `ECHO_GUARD_MS` | 300 | Mic upload resumes this long after the assistant stops |

### Codec chunk size sets the startup quantum

One codec frame is 1920 samples at 24 kHz = **80 ms of audio**, and the talker emits one
per decode step at a measured **52.5 ms/frame** (36 granules, R² 0.99,
`gap = 0.656 × audio_duration − 0.005 s`). So the talker is faster than real time per
frame. What used to stall speech was waiting for a whole granule before the vocoder was
called at all.

`codec_chunk_frames` is that granule: how many NEW frames to accumulate per code2wav
call. Measured both ways:

| chunk | granule | takes | first sound | stalls |
|---|---|---|---|---|
| 25 | 2.00 s | 1.31 s | 0.35 s **or** 1.70 s | one ~1.1 s stall, **or** none if you wait |
| **4** (shipped) | 0.32 s | 0.21 s | **0.35 s** | **none** |

At 4, delivery outruns playback from the first boundary, so there is no trade left to
make: earliest and smoothest are the same setting.

**What it costs.** Every call re-processes `codec_left_context_frames` (25) frames of
history for seam continuity *without emitting them*, so work per call is `(25 + N)`
frames to produce `N` — 2× at N=25, **7.25× at N=4**, about 3.6× more vocoder compute per
second of speech. Measured at one user this is invisible: `gap/audio` was 0.647 at N=4
against 0.656 at N=25, and rtf stayed 0.69–0.70 against 0.67. That is because stage 2
pipelines behind stage 1's next-frame decode (hence the ~0 intercept in the fit). **It is
still real GPU work on a shared card, so expect to pay it in multi-user capacity rather
than in milliseconds — that is unmeasured.**

**Two things measured and worth knowing:**

* **The first boundary is tight.** 0.217 s of playable audio against 0.213 s to produce
  the next granule is a 4 ms margin, and one turn in nine showed exactly a 4 ms stall.
  Inaudible, but do not shave `PLAYBACK_PREBUFFER_MS.fast` below 60 ms. To widen it,
  raise `initial_codec_chunk_frames` server-side instead.
* **Seams do not click.** 6× more joins, so `audio_timeline.py` checks numerically:
  sample-to-sample jump across each join against the distribution within chunks. Across
  ~250 joins, one exceeded the within-chunk p99.9 — not systematic. Ears are still the
  judge; this only rules out the obvious failure.

The page keeps the buffer choice because it cannot see the server's chunk size: at 25 the
early start stutters, and the switch is the fix. **If speech ever stutters one word in,
that is what it means.**

To re-measure after any config change:

```bash
PYTHONPATH=/home/zx/voice-agent/vllm-omni python benchmarks/live_agent/web_client/audio_timeline.py --direct
node benchmarks/live_agent/web_client/playback_test.js
```

`audio_timeline.py` prints the arrival schedule, the stalls, the seam check, and what
every candidate prebuffer would cost — the trade is a table rather than an argument.

Server-side, in `buildSessionConfig()` — `max_frame_width/height`,
`session_roll_at_talker_tokens`, the filter gaps. These are the merged
optimisations, and `selftest.py` verifies each name against the real Pydantic
model rather than trusting it.

---

## If something is wrong

| Symptom | Check |
|---|---|
| Page loads, nothing happens on Start call | Browser console; then `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7870/healthz` |
| `/ws` answers **403 with an empty body** and nothing is logged | The handler was never entered. `from __future__ import annotations` makes annotations strings, and FastAPI resolves `client: WebSocket` from **module** globals — if the fastapi imports are inside a function they are locals, the lookup fails, `client` is treated as a missing query parameter, and the socket is closed. Keep those imports at module level |
| Connects, then an `error` event about reaching the engine | The engine is not up: `curl .../8091/health` |
| It answers, but never about what you said | Speak longer; the meter must move. If it does, the turn may be firing early — try hold-to-talk |
| Audio clicks every fraction of a second | WAV headers are reaching playback; `selftest.py` covers this, so re-run it |
| The first reply makes sound, later ones are silent | `node playback_test.js`. This exact shape was the prebuffer/re-arm interaction, and the test reproduces it |
| One turn wedges the page and nothing recovers | A missing `response.audio.done`. The 45 s watchdog in `endTurn()` releases it; look for `turn ended (watchdog…)` in the log |
| It never answers | Nothing is sending `video.query`. Switch the mode to hold-to-talk and press it |
| Speech starts, stalls about a second, then continues | Expected in **as early as possible** mode and structural, not a fault: the first granule is 0.217 s and the second takes 1.34 s to make. Switch **Start speaking…** to smooth, or run `audio_timeline.py` to see the schedule |
| A fix seems to have no effect at all | Compare the `client assets <hash>` line in the event log with the stamp the page server prints at startup. Responses are `no-store` and every asset URL is stamped, but `addModule()` for the playback worklet is cached hardest and used to survive reloads |
| Turn 20 much slower than turn 2 | `session_scoped_request` did not take effect — check the server log for `[session] turn=` lines |
