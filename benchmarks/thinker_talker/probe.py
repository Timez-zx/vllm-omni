#!/usr/bin/env python3
"""Drive the live session the way the browser does, without a browser.

Exists so a failure can be localised. If this passes and the page does not, the
problem is in the page or the tunnel; if this fails, the problem is the server or
the protocol, and no amount of browser debugging will find it.

It speaks the same messages the page speaks, in the same order, with synthetic
media so no device or permission is involved:

    session.config -> audio.chunk* + video.frame* -> video.query -> audio back

    PYTHONPATH=/home/zx/voice-agent/vllm-omni python probe.py            # via the page server
    PYTHONPATH=/home/zx/voice-agent/vllm-omni python probe.py --direct   # straight to the engine
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import struct
import sys
import time
import wave

RATE = 16000


def synth_speechlike_pcm(seconds: float) -> bytes:
    """Amplitude-modulated tone: not speech, but not silence either.

    Real silence would be a fair test of the transport and an unfair test of the
    model, which may reasonably answer a silent turn with nothing. Something with
    energy in it keeps the failure modes separable.
    """
    n = int(RATE * seconds)
    out = bytearray()
    for i in range(n):
        t = i / RATE
        env = 0.35 * (1.0 + math.sin(2 * math.pi * 3.0 * t)) / 2.0
        sample = env * (0.6 * math.sin(2 * math.pi * 220 * t) + 0.4 * math.sin(2 * math.pi * 480 * t))
        out += struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
    return bytes(out)


def synth_frame_jpeg(label: str, width: int = 640, height: int = 360) -> bytes:
    import cv2
    import numpy as np

    img = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.putText(img, label, (30, int(height * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (15, 15, 15), 9)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return enc.tobytes()


def session_config(system_prompt: str, overrides: dict | None = None) -> dict:
    # Deliberately the same values app.js sends, so this probe and the page
    # exercise one configuration rather than two.
    cfg = {
        "type": "session.config",
        "system_prompt": system_prompt,
        "modalities": ["text", "audio"],
        "num_frames": 16,
        # 8, not 256: the buffer holds frames whose on-arrival prefill was refused
        # (a turn is in flight, or a compression shadow is warming), and a turn
        # submits the WHOLE buffer at once. Measured single-user cost of a turn is
        # 348 ms + 59 ms per frame, so an unbounded buffer turns a stalled moment
        # into a 13-19 frame sweep (876 ms single-user, ~4.7 s at 13 users). The
        # cap evicts the OLDEST frame, which is the right one to lose on a live
        # feed. See workflow section 16.
        "max_frames": 8,
        "max_frame_width": 640,
        "max_frame_height": 352,
        "frame_jpeg_quality": 90,
        "enable_frame_filter": True,
        "frame_filter_threshold": 0.95,
        "frame_filter_min_gap": 8,
        "frame_filter_max_gap": 16,
        "use_audio_in_video": True,
        "session_scoped_request": True,
        # OFF: crashes stage 1 with a CUDA device-side assert under continuous frames.
        "prefill_frames_on_arrival": True,
        "session_roll_at_talker_tokens": 45000,
        "session_roll_history_turns": 8,
    }
    # Overrides exist so one probe run can exercise context compression on a short
    # session. Left alone the trigger is derived from the KV pool share and never
    # fires inside ten turns, so a compression path can only be reached by waiting
    # out a real workload -- too slow to iterate on.
    if overrides:
        cfg.update(overrides)
    return cfg


async def run(args) -> int:
    import websockets

    url = args.url
    print(f"connecting to {url}", file=sys.stderr)
    try:
        ws = await asyncio.wait_for(websockets.connect(url, max_size=None), timeout=15)
    except Exception as exc:
        print(f"\n!! cannot connect: {exc}\n"
              f"   engine health:  curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:8091/health\n"
              f"   page server:    curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:7870/healthz\n",
              file=sys.stderr)
        return 2

    turns_ok = 0
    async with ws:
        _overrides: dict = {}
        if getattr(args, "compress_at", 0):
            _overrides["context_compression_trigger_tokens"] = int(args.compress_at)
        if getattr(args, "compress_target", 0):
            _overrides["context_compression_target_tokens"] = int(args.compress_target)
        if _overrides:
            print(f"session overrides: {_overrides}", file=sys.stderr)
        await ws.send(json.dumps(session_config(
            "You are a friendly voice assistant in a live video call. Reply out loud in one or two "
            "short sentences. Always answer with both text and speech.",
            _overrides,
        )))

        state = {"text": "", "wavs": [], "t_query": None, "first_audio_ms": None, "done": False}

        async def reader() -> None:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "response.text.delta":
                    state["text"] += msg.get("delta", "")
                elif t == "response.audio.delta":
                    if state["first_audio_ms"] is None and state["t_query"]:
                        state["first_audio_ms"] = (time.monotonic() - state["t_query"]) * 1000
                    state["wavs"].append(base64.b64decode(msg.get("data", "")))
                elif t == "response.audio.done":
                    state["done"] = True
                elif t == "session.rolled":
                    print(f"  [session rolled] {msg}", file=sys.stderr)
                elif t == "error":
                    print(f"  !! server error: {msg.get('message')}", file=sys.stderr)
                    state["done"] = True

        task = asyncio.create_task(reader())

        for turn in range(1, args.turns + 1):
            state.update(text="", wavs=[], t_query=None, first_audio_ms=None, done=False)

            # Stream a couple of seconds of media the way a live client would,
            # frames and audio as independent messages.
            frames = 0
            pcm = synth_speechlike_pcm(args.speak_s)
            chunk = int(RATE * 0.1) * 2
            next_frame_at = 0.0
            elapsed = 0.0
            for off in range(0, len(pcm), chunk):
                await ws.send(json.dumps({
                    "type": "audio.chunk",
                    "data": base64.b64encode(pcm[off:off + chunk]).decode(),
                }))
                elapsed += 0.1
                if elapsed >= next_frame_at:
                    next_frame_at = elapsed + 0.5
                    await ws.send(json.dumps({
                        "type": "video.frame",
                        "data": base64.b64encode(synth_frame_jpeg(str(turn))).decode(),
                    }))
                    frames += 1
                await asyncio.sleep(0.1)

            state["t_query"] = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": args.query}))

            deadline = time.monotonic() + args.timeout_s
            while time.monotonic() < deadline and not state["done"]:
                await asyncio.sleep(0.1)

            samples = 0
            rate = 0
            for blob in state["wavs"]:
                try:
                    with wave.open(io.BytesIO(blob), "rb") as w:
                        rate = w.getframerate()
                        samples += w.getnframes()
                except Exception:
                    pass
            secs = samples / rate if rate else 0.0
            ok = samples > 0
            turns_ok += 1 if ok else 0
            print(f"turn {turn}: frames={frames} chunks={len(state['wavs'])} "
                  f"audio={secs:.2f}s@{rate or '?'}Hz "
                  f"first_audio={state['first_audio_ms'] and round(state['first_audio_ms']) or '-'}ms "
                  f"{'OK' if ok else 'NO AUDIO'}")
            if state["text"]:
                print(f"         text: {state['text'][:110]}")

        await ws.send(json.dumps({"type": "video.done"}))
        await asyncio.sleep(0.4)
        task.cancel()

    print(f"\n{turns_ok}/{args.turns} turn(s) produced audio")
    return 0 if turns_ok == args.turns else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=None)
    p.add_argument("--direct", action="store_true",
                   help="bypass the page server and talk to the engine")
    p.add_argument("--turns", type=int, default=2)
    p.add_argument("--speak-s", type=float, default=2.0)
    p.add_argument("--query", default="", help="empty means the question is in the audio")
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--compress-at", type=int, default=0,
                   help="context_compression_trigger_tokens for this session "
                        "(0 = leave the server default, which never fires in a short run)")
    p.add_argument("--compress-target", type=int, default=0,
                   help="context_compression_target_tokens (0 = server default)")
    args = p.parse_args()
    if args.url is None:
        args.url = ("ws://127.0.0.1:8091/v1/video/chat/stream" if args.direct
                    else "ws://127.0.0.1:7870/ws")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
