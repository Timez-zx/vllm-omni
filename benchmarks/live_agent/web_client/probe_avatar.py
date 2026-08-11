#!/usr/bin/env python3
"""probe.py's avatar-mode sibling: does the bridge actually deliver a person?

Runs the same synthetic session probe.py runs, through the page proxy started
with --avatar, and verifies the three things the bridge added:

  * avatar.turn arrives BEFORE that turn's response.start (ordering is what the
    client's audio-holds-for-video anchor is built on);
  * avatar.frame messages arrive, carry the turn's rid, and their pts advance;
  * the first talking-rid frame lands within the client's fallback window, so a
    real browser would have coupled A/V instead of degrading to voice-only.

Exit 0 only if all three hold.

    python probe_avatar.py                  # via the page proxy on :7870
    python probe_avatar.py --url ws://...   # anywhere else
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

from probe import session_config, synth_frame_jpeg, synth_speechlike_pcm

FALLBACK_MS = 3500  # mirror app.js AVATAR_AUDIO_FALLBACK_MS


async def run(url: str, timeout_s: float) -> int:
    import websockets

    started = time.monotonic()
    now = lambda: round(time.monotonic() - started, 3)  # noqa: E731

    turn_rid = None
    turn_seen_at = None
    response_start_at = None
    first_frame_at = None
    frame_count = 0
    frame_rids = set()
    last_pts = -1
    pts_monotonic = True
    audio_ms = 0.0
    audio_done_at = None
    statuses = []

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(session_config(
            "You are a friendly voice assistant. Answer in one short sentence."
        )))
        pcm = synth_speechlike_pcm(2.0)
        step = 3200 * 2  # 200 ms of 16 kHz PCM16
        for offset in range(0, len(pcm), step):
            await ws.send(json.dumps({
                "type": "audio.chunk",
                "data": base64.b64encode(pcm[offset:offset + step]).decode(),
            }))
        await ws.send(json.dumps({
            "type": "video.frame",
            "data": base64.b64encode(synth_frame_jpeg("probe")).decode(),
        }))
        await ws.send(json.dumps({"type": "video.query", "text": "你好，请简短介绍你自己。"}))

        while time.monotonic() - started < timeout_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
            except asyncio.TimeoutError:
                break
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            t = msg.get("type")
            if t == "avatar.turn":
                turn_rid = msg.get("rid")
                turn_seen_at = now()
            elif t == "response.start":
                response_start_at = now()
            elif t == "avatar.frame":
                frame_count += 1
                frame_rids.add(msg.get("rid"))
                pts = msg.get("pts", 0)
                if msg.get("rid") == turn_rid:
                    if first_frame_at is None:
                        first_frame_at = now()
                    if pts < last_pts:
                        pts_monotonic = False
                    last_pts = pts
            elif t == "avatar.status":
                statuses.append(msg.get("event"))
            elif t == "response.audio.delta":
                wav = base64.b64decode(msg.get("data") or "")
                audio_ms += max(0, (len(wav) - 44)) / 2 / 24000 * 1000
            elif t == "response.audio.done":
                audio_done_at = now()
            elif t == "error":
                print(f"!! server error: {msg.get('message')}")
                return 1
            # Stop once the turn's audio is done AND we have given the avatar
            # a fallback window's worth of chance to produce its first frame.
            if audio_done_at is not None and (
                first_frame_at is not None
                or now() - audio_done_at > FALLBACK_MS / 1000
            ):
                # Drain a little longer for trailing frames, then stop.
                if now() - audio_done_at > 3.0:
                    break

    ordered = (
        turn_seen_at is not None
        and response_start_at is not None
        and turn_seen_at <= response_start_at
    )
    coupled = (
        first_frame_at is not None
        and response_start_at is not None
        and (first_frame_at - response_start_at) * 1000 <= FALLBACK_MS
    )
    print(json.dumps({
        "avatar_turn_at": turn_seen_at,
        "response_start_at": response_start_at,
        "first_talking_frame_at": first_frame_at,
        "frames": frame_count,
        "frame_rids": sorted(r for r in frame_rids if r is not None),
        "turn_rid": turn_rid,
        "pts_monotonic": pts_monotonic,
        "audio_ms": round(audio_ms),
        "audio_done_at": audio_done_at,
        "statuses": statuses[:8],
        "check_order(avatar.turn<=response.start)": ordered,
        "check_frames_arrived": frame_count > 0,
        "check_couple_within_fallback": coupled,
        "check_pts_monotonic": pts_monotonic,
    }, ensure_ascii=False, indent=2))
    ok = ordered and frame_count > 0 and coupled and pts_monotonic
    print("AVATAR_PROBE=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:7870/ws")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
