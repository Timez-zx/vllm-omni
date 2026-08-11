#!/usr/bin/env python3
"""Reproduce the browser's AUTOMATIC turn trigger, which probe.py bypasses.

probe.py sends an explicit video.query, so it exercises the engine but not the
client-side silence detector -- and "first utterance of a session is ignored"
lives precisely in that detector. This speaks the wire the way app.js does under
the default 'auto' trigger: stream speech-like audio, go quiet, and let the
SERVER-SIDE analogue of the VAD decide when the turn fires. But there is no VAD
on the server; the trigger is the client's. So this script reimplements app.js's
updateSilenceDetector exactly and sends video.query itself at the same instant
app.js would, then checks whether a reply comes back -- for TWO back-to-back
sessions on one proxy connection is not possible (each session is its own ws),
so it opens two sessions in sequence and reports each.

Pass = both sessions' first triggered turn produces audio.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct
import sys
import time

RATE = 16000
# Mirror app.js
SILENCE_RMS = 0.012
SILENCE_HANG_MS = 700
MIN_SPEECH_MS = 400


def frame_rms(pcm16: bytes) -> float:
    if not pcm16:
        return 0.0
    n = len(pcm16) // 2
    vals = struct.unpack("<%dh" % n, pcm16[: n * 2])
    s = sum((v / 32768.0) ** 2 for v in vals)
    return math.sqrt(s / max(1, n))


def speechlike(seconds: float, rate: int = RATE) -> bytes:
    n = int(rate * seconds)
    out = bytearray()
    for i in range(n):
        t = i / rate
        env = 0.5 * (1.0 + math.sin(2 * math.pi * 3.0 * t))
        s = env * (0.6 * math.sin(2 * math.pi * 200 * t) + 0.4 * math.sin(2 * math.pi * 450 * t))
        out += struct.pack("<h", max(-32768, min(32767, int(s * 20000))))
    return bytes(out)


def silence(seconds: float, rate: int = RATE) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


async def one_session(url: str, label: str, timeout_s: float) -> dict:
    import websockets

    speech_ms = 0.0
    silence_ms = 0.0
    saw_speech = False
    triggered = False
    trigger_at = None
    got_audio = False
    first_audio_at = None
    started = time.monotonic()

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "session.config",
            "system_prompt": "You are a friendly assistant. Answer in one short sentence.",
            "sample_rate": RATE,
            "use_audio_in_video": True,
        }))
        # One camera frame, like the page.
        await ws.send(json.dumps({"type": "video.frame", "data": base64.b64encode(_jpeg()).decode()}))

        # 1.2 s of speech, then silence -- exactly the app.js detector's input.
        chunk_ms = 100
        chunk = speechlike(chunk_ms / 1000)
        seq = [("speech", 1200), ("silence", 1500)]
        reader_done = asyncio.Event()

        async def reader():
            nonlocal got_audio, first_audio_at
            try:
                while not reader_done.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                    if isinstance(raw, (bytes, bytearray)):
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") == "response.audio.delta":
                        got_audio = True
                        if first_audio_at is None:
                            first_audio_at = round(time.monotonic() - started, 3)
            except (asyncio.TimeoutError, Exception):
                pass

        rtask = asyncio.create_task(reader())

        for kind, dur in seq:
            elapsed = 0
            while elapsed < dur:
                pcm = chunk if kind == "speech" else silence(chunk_ms / 1000)
                await ws.send(json.dumps({"type": "audio.chunk", "data": base64.b64encode(pcm).decode()}))
                # Replicate updateSilenceDetector
                rms = frame_rms(pcm)
                if not triggered:
                    if rms >= SILENCE_RMS:
                        speech_ms += chunk_ms
                        silence_ms = 0
                        if speech_ms >= MIN_SPEECH_MS:
                            saw_speech = True
                    elif saw_speech:
                        silence_ms += chunk_ms
                        if silence_ms >= SILENCE_HANG_MS:
                            await ws.send(json.dumps({"type": "video.query", "text": ""}))
                            triggered = True
                            trigger_at = round(time.monotonic() - started, 3)
                await asyncio.sleep(chunk_ms / 1000)
                elapsed += chunk_ms

        # Give the reply time to come back.
        await asyncio.sleep(6.0)
        reader_done.set()
        rtask.cancel()
        try:
            await rtask
        except Exception:
            pass

    return {
        "label": label,
        "saw_speech": saw_speech,
        "triggered": triggered,
        "trigger_at_s": trigger_at,
        "got_audio": got_audio,
        "first_audio_at_s": first_audio_at,
    }


def _jpeg() -> bytes:
    # Minimal valid gray JPEG (1x1) is enough; the server only counts frames.
    return base64.b64decode(
        b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////"
        b"////////////////////////////////////////////////////wgALCAABAAEBAREA"
        b"/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAA"
        b"AAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBAB"
        b"AAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAA"
        b"gBAQABPyF//9oADAMBAAIAAwAAABAf/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEB"
        b"PxB//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPxB//8QAFBABAAAAAAAAAAAAAA"
        b"AAAAAAAP/aAAgBAQABPxB//9k="
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:7870/ws")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    results = []
    for i in (1, 2):
        r = await one_session(args.url, f"session{i}", args.timeout)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    ok = all(r["triggered"] and r["got_audio"] for r in results)
    print("VAD_PROBE=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
