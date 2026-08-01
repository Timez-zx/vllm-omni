#!/usr/bin/env python3
"""Reproduce the stage-1 device-side assert that prefill-on-arrival triggers.

    PYTHONPATH=/home/zx/voice-agent/vllm-omni python crash_repro.py --direct

Run the engine with CUDA_LAUNCH_BLOCKING=1 first, so the assert's stack points at the
kernel's launch site instead of at the next event record.

Why prefill_ab.py did NOT reproduce this: it stops sending media the moment the query is
submitted and waits quietly for the reply. A browser never stops -- frames and microphone
audio keep arriving THROUGH the model's reply and through the gap after it. The crash hit
a hand-driven browser session ~10 s in, so continuous media is part of the recipe until
proven otherwise. This driver therefore streams without pause and issues a query every
--turn-every seconds, until the engine dies or --max-turns complete.

Exit code 2 = engine died (reproduction succeeded), 0 = survived, 1 = other failure.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from probe import session_config, synth_frame_jpeg, synth_speechlike_pcm  # noqa: E402


async def main_async(args) -> int:
    import websockets

    url = args.url or ("ws://127.0.0.1:8091/v1/video/chat/stream" if args.direct
                       else "ws://127.0.0.1:7870/ws")
    cfg = session_config(
        "You are a friendly voice assistant in a live video call. "
        "Reply out loud in one short sentence. Always answer with both text and speech."
    )
    cfg["prefill_frames_on_arrival"] = not args.no_prefill
    if args.min_gap:
        cfg["frame_filter_min_gap"] = args.min_gap
        cfg["frame_filter_max_gap"] = args.min_gap * 2

    state = {"turns_done": 0, "engine_error": None, "last_event": None}

    print(f"{url}  min_gap={args.min_gap or 'default'}  "
          f"query every {args.turn_every}s, up to {args.max_turns} turns")
    try:
        async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
            await ws.send(json.dumps(cfg))

            async def reader() -> None:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    t = msg.get("type")
                    state["last_event"] = t
                    if t == "response.audio.done":
                        state["turns_done"] += 1
                        print(f"  turn {state['turns_done']} completed")
                    elif t == "error":
                        m = msg.get("message", "")
                        print(f"  server error: {m[:140]}")
                        if "Engine" in m or "engine" in m:
                            state["engine_error"] = m
                            return

            rtask = asyncio.create_task(reader())

            t0 = time.monotonic()
            frame_i = 0
            # The browser's actual rhythm, which prefill_ab.py and the first version of
            # this script both simplified away:
            #   * the auto trigger sends video.query with EMPTY text -- the buffered
            #     audio IS the question;
            #   * the echo guard stops mic upload while the assistant is speaking, so
            #     audio chunks PAUSE during replies while frames keep flowing;
            #   * a user can speak into the reply, so some turns barge in: a query is
            #     sent while the previous turn is still in flight.
            speaking = True
            speak_until = t0 + 4.0
            next_action = None
            while state["turns_done"] < args.max_turns and state["engine_error"] is None:
                now = time.monotonic()
                await ws.send(json.dumps({
                    "type": "video.frame",
                    "data": base64.b64encode(synth_frame_jpeg(f"c{frame_i}")).decode(),
                }))
                frame_i += 1
                if speaking:
                    await ws.send(json.dumps({
                        "type": "audio.chunk",
                        "data": base64.b64encode(synth_speechlike_pcm(0.4)).decode(),
                    }))
                    if now >= speak_until:
                        speaking = False
                        await ws.send(json.dumps({"type": "video.query", "text": ""}))
                        next_action = now + args.turn_every
                elif next_action is not None and now >= next_action:
                    # Start "speaking" again for the next turn -- sometimes into the
                    # middle of the previous reply (a barge-in), sometimes after it.
                    speaking = True
                    speak_until = now + (2.0 if (state["turns_done"] % 2) else 4.0)
                await asyncio.sleep(0.5)
                if rtask.done():
                    break
                if now - t0 > args.max_turns * args.turn_every + 180:
                    print("  timed out")
                    break
            rtask.cancel()
    except Exception as exc:
        # A dropped socket right after an engine error IS the reproduction.
        if state["engine_error"] is None:
            print(f"connection failed: {exc}")
            state["engine_error"] = f"socket: {exc}"

    if state["engine_error"]:
        print(f"\nREPRODUCED: {str(state['engine_error'])[:160]}")
        return 2
    print(f"\nsurvived {state['turns_done']} turns, {frame_i} frames -- no crash")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--no-prefill", action="store_true",
                   help="same hostile pacing WITHOUT arrival prefill: the control arm that "
                        "decides whether the crash is the append's fault at all")
    p.add_argument("--min-gap", type=int, default=None,
                   help="lower = more retained frames = more appends per second")
    p.add_argument("--turn-every", type=float, default=8.0)
    p.add_argument("--max-turns", type=int, default=6)
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
