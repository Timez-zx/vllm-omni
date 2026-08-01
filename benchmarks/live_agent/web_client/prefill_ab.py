#!/usr/bin/env python3
"""Does prefilling frames on arrival help, once there is idle time to move work into?

    PYTHONPATH=/home/zx/voice-agent/vllm-omni python prefill_ab.py --direct

The first measurement of this feature said it made things 6x WORSE, and that number was
an artefact of how it was measured: probe.py sends its frames ~0.1 s before the query, so
there was no gap to prefill into and one prefill simply became two engine round-trips.

This paces the way a browser does instead -- frames and microphone audio streaming
continuously for SPEAK_S seconds, then the query -- which is the only timing under which
the feature can pay. Both arms run against the same live engine, because
prefill_frames_on_arrival is a per-session config field, so no restart separates them and
no restart can be blamed for the difference.

Arms alternate rather than running in blocks: the engine warms up and the GPU is shared,
so a block design would hand the whole trend to whichever arm ran second.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from probe import session_config, synth_frame_jpeg, synth_speechlike_pcm  # noqa: E402

FRAME_INTERVAL_S = 0.5      # 2 fps, matching app.js FRAME_INTERVAL_MS
AUDIO_CHUNK_S = 0.2         # matching SEND_INTERVAL_MS


async def one_turn(ws, state, *, speak_s: float, label: str) -> float | None:
    """Stream media for speak_s, then query. Returns ms from query to first audio."""
    state.update(first_audio_ms=None, done=False, t_query=None)

    # Stream frames and audio the way a live client does: continuously, while "speaking".
    frames = 0
    t_end = time.monotonic() + speak_s
    while time.monotonic() < t_end:
        # Vary the frame so the similarity filter has something to retain. Identical
        # frames are ~0.998 similar and would be dropped, which would make this an
        # experiment about an empty buffer.
        await ws.send(json.dumps({
            "type": "video.frame",
            "data": base64.b64encode(synth_frame_jpeg(f"{label}-{frames}")).decode(),
        }))
        frames += 1
        await ws.send(json.dumps({
            "type": "audio.chunk",
            "data": base64.b64encode(synth_speechlike_pcm(AUDIO_CHUNK_S)).decode(),
        }))
        await asyncio.sleep(FRAME_INTERVAL_S)

    state["t_query"] = time.monotonic()
    await ws.send(json.dumps({"type": "video.query", "text": "Count from one to six."}))
    deadline = time.monotonic() + 120
    while not state["done"] and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return state["first_audio_ms"]


async def run_arm(url: str, *, prefill: bool, turns: int, speak_s: float) -> list[float]:
    import websockets

    cfg = session_config(
        "You are a friendly voice assistant in a live video call. "
        "Reply out loud in one short sentence. Always answer with both text and speech."
    )
    cfg["prefill_frames_on_arrival"] = prefill

    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps(cfg))
        state: dict = {"first_audio_ms": None, "done": False, "t_query": None}

        async def reader() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "response.audio.delta":
                    if state["first_audio_ms"] is None and state["t_query"]:
                        state["first_audio_ms"] = (time.monotonic() - state["t_query"]) * 1000
                elif t == "response.audio.done":
                    state["done"] = True
                elif t == "error":
                    print(f"  !! {msg.get('message')}", file=sys.stderr)
                    state["done"] = True

        task = asyncio.create_task(reader())
        # A warm-up turn, discarded. Arrival prefill needs first_sent -- there is nothing
        # to append to before the session's first real chunk -- so turn 0 can never show
        # the effect and including it would dilute both arms unequally.
        await one_turn(ws, state, speak_s=speak_s, label="w")
        out = []
        for i in range(turns):
            ms = await one_turn(ws, state, speak_s=speak_s, label=f"{int(prefill)}{i}")
            print(f"  {'ON ' if prefill else 'OFF'} turn {i}: "
                  f"{'no audio' if ms is None else f'{ms:7.1f} ms'}")
            if ms is not None:
                out.append(ms)
        await ws.send(json.dumps({"type": "video.done"}))
        task.cancel()
    return out


async def main_async(args) -> int:
    url = args.url or ("ws://127.0.0.1:8091/v1/video/chat/stream" if args.direct
                       else "ws://127.0.0.1:7870/ws")
    print(f"{url}\nspeaking {args.speak_s}s before each query "
          f"({int(args.speak_s / FRAME_INTERVAL_S)} frames sent), {args.turns} turns per arm, "
          f"alternating\n")

    on: list[float] = []
    off: list[float] = []
    for rnd in range(args.rounds):
        print(f"round {rnd}")
        # Alternate the order too, so a warming trend cannot favour one arm.
        order = [False, True] if rnd % 2 == 0 else [True, False]
        for prefill in order:
            got = await run_arm(url, prefill=prefill, turns=args.turns, speak_s=args.speak_s)
            (on if prefill else off).extend(got)

    def show(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"  {name}: no data")
            return
        print(f"  {name}: n={len(xs)}  median {statistics.median(xs):7.1f} ms  "
              f"min {min(xs):7.1f}  max {max(xs):7.1f}")

    print("\nfirst audio after the query")
    show("prefill OFF", off)
    show("prefill ON ", on)
    if on and off:
        mo, mn = statistics.median(off), statistics.median(on)
        delta = mn - mo
        print(f"\n  ON - OFF = {delta:+.1f} ms ({delta / mo * 100:+.1f}%)")
        # Say which way it went in words, so a sign error cannot be read as a win.
        print("  -> prefilling on arrival is " + (
            "FASTER" if delta < 0 else "SLOWER") + " under this pacing")
        if len(on) < 3 or len(off) < 3:
            print("  (too few samples to call it; raise --rounds)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--speak-s", type=float, default=6.0,
                   help="how long media streams before the query -- this IS the idle time")
    p.add_argument("--turns", type=int, default=2, help="measured turns per arm per round")
    p.add_argument("--rounds", type=int, default=2)
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
