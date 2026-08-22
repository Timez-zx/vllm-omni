#!/usr/bin/env python3
"""Measure whether speech can play WITHOUT GAPS, which "it works" never tells you.

probe.py answers "did audio arrive and how much". That is not the same question as "can
a player play it straight through", and the difference is exactly what a listener hears
as a stall. This drives one turn, records the arrival time and duration of every
`response.audio.delta`, then replays the client's own buffering rule against those
timestamps to find where the player would run dry -- and for how long.

    python audio_timeline.py --direct

The model: playback starts once PREBUFFER_MS of audio is queued, then consumes audio at
exactly 1x. A stall happens whenever the queue empties before the next delta lands. That
is a property of the ARRIVAL SCHEDULE, not of the total, so a turn can deliver every
sample it produced and still be unlistenable.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import wave

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from probe import session_config, synth_frame_jpeg, synth_speechlike_pcm  # noqa: E402

DEFAULT_PREBUFFER_MS = 500.0  # keep in step with app.js smooth mode


def wav_seconds(raw: bytes) -> float:
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.getnframes() / w.getframerate()


def wav_samples(raw: bytes) -> list[int]:
    import struct

    with wave.open(io.BytesIO(raw), "rb") as w:
        data = w.readframes(w.getnframes())
    return list(struct.unpack(f"<{len(data) // 2}h", data))


def seam_report(chunks: list[list[int]]) -> str:
    """Are the joins between granules discontinuous enough to click?

    Smaller codec chunks mean MORE joins, and each join is where the vocoder's left
    context has to reproduce what the previous call already emitted. A bad seam is a
    step discontinuity in the waveform, which is audible as a click -- and it is the one
    audio-quality question that can be answered numerically instead of by ear.

    Compares the sample-to-sample jump ACROSS each join against the distribution of
    jumps WITHIN the chunks. A seam that is indistinguishable from ordinary signal
    movement cannot click; one far out in the tail can.
    """
    within = []
    for c in chunks:
        step = max(1, len(c) // 2000)  # sample the interior, do not walk 300k
        within.extend(abs(c[i + 1] - c[i]) for i in range(0, len(c) - 1, step))
    across = [abs(chunks[i + 1][0] - chunks[i][-1]) for i in range(len(chunks) - 1) if chunks[i] and chunks[i + 1]]
    if not within or not across:
        return "  seams: not enough data"
    within.sort()

    def q(p: float) -> int:
        return within[min(len(within) - 1, int(len(within) * p))]

    worse = sum(1 for a in across if a > q(0.999))
    return (
        f"  seams: {len(across)} join(s); worst jump {max(across)}, "
        f"median {sorted(across)[len(across) // 2]}\n"
        f"         within-chunk jumps: p50 {q(0.5)}, p99 {q(0.99)}, p99.9 {q(0.999)}, "
        f"max {within[-1]}\n"
        f"         {worse}/{len(across)} join(s) exceed the within-chunk p99.9 "
        f"-> {'AUDIBLE CLICKS LIKELY' if worse else 'no seam stands out'}"
    )


def simulate(deltas: list[tuple[float, float]], prebuffer_s: float) -> dict:
    """Replay the client's buffering rule over measured arrivals.

    `deltas` is [(arrival_s_since_query, audio_s)]. Returns the stalls a listener would
    hear: playback begins when the queue first holds `prebuffer_s`, then drains at 1x.
    """
    queued = 0.0
    start = None
    for t, dur in deltas:
        queued += dur
        if queued >= prebuffer_s:
            start = t
            break
    if start is None:
        return {"start": None, "stalls": [], "stall_total": 0.0}

    # `played_until` is wall-clock time up to which audio is covered.
    played_until = start
    stalls = []
    for t, dur in deltas:
        if t < start:
            # Already counted into the prebuffer that triggered the start.
            played_until += dur
            continue
        if t > played_until:
            stalls.append((played_until - start, t - played_until))
            played_until = t
        played_until += dur
    return {"start": start, "stalls": stalls, "stall_total": sum(d for _, d in stalls)}


async def run(args) -> int:
    import websockets

    url = args.url or ("ws://127.0.0.1:8091/v1/video/chat/stream" if args.direct else "ws://127.0.0.1:7870/ws")
    print(f"connecting to {url}")
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        await ws.send(
            json.dumps(
                session_config(
                    "You are a friendly voice assistant in a live video call. Always answer with both text and speech."
                )
            )
        )
        state: dict = {"deltas": [], "pcm": [], "text": "", "t0": None, "done": False}

        async def reader() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "response.audio.delta":
                    raw_wav = base64.b64decode(msg.get("data", ""))
                    state["deltas"].append((time.monotonic() - state["t0"], wav_seconds(raw_wav)))
                    state["pcm"].append(wav_samples(raw_wav))
                elif t == "response.text.delta":
                    state["text"] += msg.get("delta", "")
                elif t == "response.audio.done":
                    state["done"] = True
                elif t == "error":
                    print(f"  !! server error: {msg.get('message')}", file=sys.stderr)
                    state["done"] = True

        task = asyncio.create_task(reader())

        for _ in range(4):
            await ws.send(json.dumps({"type": "video.frame", "data": base64.b64encode(synth_frame_jpeg("t")).decode()}))
            await ws.send(
                json.dumps({"type": "audio.chunk", "data": base64.b64encode(synth_speechlike_pcm(0.5)).decode()})
            )
            await asyncio.sleep(0.1)

        # Several turns, because the arrival schedule of turn 2 is not something turn 1
        # can tell you -- and a per-turn client rule (the playback re-arm) is only as
        # good as the schedule staying the same. Measuring one turn is what let a
        # "smooth on the first reply only" bug pass every check here.
        turns = []
        for turn in range(args.turns):
            state["deltas"], state["pcm"] = [], []
            state["text"], state["done"] = "", False
            state["t0"] = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": args.query}))
            deadline = time.monotonic() + args.timeout_s
            while not state["done"] and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            turns.append((state["deltas"][:], state["text"], state["pcm"][:]))
            # Speak again between turns, the way a live client would.
            for _ in range(3):
                await ws.send(
                    json.dumps({"type": "audio.chunk", "data": base64.b64encode(synth_speechlike_pcm(0.5)).decode()})
                )
                await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "video.done"}))
        task.cancel()

    if not any(d for d, _, _ in turns):
        print("no audio at all -- this is a probe.py problem, not a continuity one")
        return 1

    worst = 0
    for turn_idx, (deltas, text, pcm) in enumerate(turns):
        if not deltas:
            print(f"\nturn {turn_idx}: NO AUDIO")
            worst = 1
            continue
        print(f"\n--- turn {turn_idx}: {text[:70]}")
        report(deltas, args)
        print(seam_report(pcm))
    return worst


def report(deltas, args) -> None:

    print(f"{'delta':>5} {'arrived':>9} {'audio_s':>8} {'gap_since_prev':>15}")
    prev = 0.0
    for i, (t, dur) in enumerate(deltas):
        print(f"{i:>5} {t:>8.3f}s {dur:>8.3f} {t - prev:>14.3f}s")
        prev = t
    total_audio = sum(d for _, d in deltas)
    print(
        f"\ntotal audio {total_audio:.2f}s delivered over {deltas[-1][0]:.2f}s wall "
        f"(rtf {deltas[-1][0] / total_audio:.2f})"
    )

    sim = simulate(deltas, args.prebuffer_ms / 1000.0)
    print(f"\nplayback simulation at prebuffer={args.prebuffer_ms:.0f}ms:")
    if sim["start"] is None:
        print("  never starts on the threshold alone -- response.audio.done releases it")
        return
    print(f"  starts at {sim['start']:.3f}s after the query")
    if not sim["stalls"]:
        print("  NO STALLS -- plays straight through")
    else:
        for at, dur in sim["stalls"]:
            print(f"  STALL {dur * 1000:>6.0f}ms at {at:.3f}s into the speech")
        print(f"  {len(sim['stalls'])} stall(s), {sim['stall_total'] * 1000:.0f}ms of silence total")

    # The trade the prebuffer makes, priced rather than argued about.
    print("\nwhat other prebuffers would do (start latency vs stalls):")
    for ms in (60, 250, 500, 1000, 1500, 2000):
        s = simulate(deltas, ms / 1000.0)
        if s["start"] is None:
            print(f"  {ms:>5}ms  never starts on the threshold alone")
            continue
        print(
            f"  {ms:>5}ms  start {s['start']:.3f}s  {len(s['stalls'])} stall(s), {s['stall_total'] * 1000:.0f}ms silent"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None)
    p.add_argument("--direct", action="store_true", help="bypass the page-server proxy")
    p.add_argument("--query", default="Please count slowly from one to fifteen.")
    p.add_argument("--prebuffer-ms", type=float, default=DEFAULT_PREBUFFER_MS)
    p.add_argument("--turns", type=int, default=3, help="later turns are the interesting ones")
    p.add_argument("--timeout-s", type=float, default=120.0)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
