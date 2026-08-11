#!/usr/bin/env python3
"""Tee a live session's reply audio to a SoulX-LiveAct avatar server.

The avatar model is audio-driven: given a PCM timeline (plus a reference image
it already holds), it emits JPEG video frames of a person saying that audio.  It
does not care who synthesized the audio, which is what makes this a bridge
rather than an integration: the engine keeps producing exactly what it produces
today, and the avatar consumes a copy.

Wire mapping, engine protocol -> avatar protocol:

    {"type": "response.start"}                  -> {"event": "start", "rid": N}
    {"type": "response.audio.delta", ...}       -> b"\\x01" + PCM16 mono 24 kHz
    {"type": "response.audio.done"}             -> {"event": "end"}

and back, avatar -> browser (injected into the downstream websocket):

    0x21 | rid u32le | pts_ms u32le | JPEG      -> {"type": "avatar.frame",
                                                    "rid": N, "pts": ms,
                                                    "data": base64 JPEG}
    {"event": "avatar_*", ...}                  -> {"type": "avatar.status", ...}

The bridge also tells the client which rid belongs to the turn it just opened
({"type": "avatar.turn", "rid": N}); the client uses that to anchor audio
playback to the first video frame of that turn, because the avatar needs about a
second of audio timeline before its first talking block exists and lips must not
lag the voice.

Pacing note: the engine synthesizes audio several times faster than real time
and the avatar buffers internally (it generates a block whenever enough timeline
has accumulated), so audio chunks are forwarded on arrival with no throttling.

Failure stance: the avatar is OPTIONAL.  Every failure here degrades the call to
audio-only and says so via avatar.status; nothing in this file may take the
session down with it.
"""

from __future__ import annotations

import array
import asyncio
import base64
import json
import logging
import struct

LOG = logging.getLogger("avatar-bridge")

AVATAR_SAMPLE_RATE = 24000


def wav_to_pcm24k(wav: bytes) -> bytes:
    """Extract PCM16 mono 24 kHz samples from one RIFF/WAVE payload.

    The engine sends each response.audio.delta as a complete little WAV file
    (the browser client parses it the same way, see decodeWav in app.js).  The
    avatar wants raw PCM16 mono at 24 kHz, so parse the fmt chunk, take the data
    chunk, and convert only if the stream ever deviates from that -- today it is
    already 24 kHz mono 16-bit, so the common path is a plain slice.
    """
    if len(wav) < 12 or wav[0:4] != b"RIFF" or wav[8:12] != b"WAVE":
        # Some server variants ship raw PCM16 mono 24 kHz instead of a WAV
        # container (see the protocol note at the top of app.js).  Raw samples
        # are indistinguishable from garbage by inspection, so trust the
        # protocol's stated default rather than refusing to speak.
        return wav[: len(wav) - (len(wav) % 2)]
    channels, rate, bits = 1, AVATAR_SAMPLE_RATE, 16
    data = None
    offset = 12
    while offset + 8 <= len(wav):
        chunk_id = wav[offset : offset + 4]
        (size,) = struct.unpack_from("<I", wav, offset + 4)
        body = wav[offset + 8 : offset + 8 + size]
        if chunk_id == b"fmt ":
            _fmt, channels, rate = struct.unpack_from("<HHI", body, 0)
            (bits,) = struct.unpack_from("<H", body, 14)
        elif chunk_id == b"data":
            data = body
        # Chunks are word-aligned.
        offset += 8 + size + (size & 1)
    if data is None:
        raise ValueError("WAV payload has no data chunk")
    if bits != 16:
        raise ValueError(f"unsupported sample width: {bits} bits")

    samples = array.array("h")
    samples.frombytes(data[: len(data) - (len(data) % 2)])
    if channels > 1:
        samples = array.array(
            "h",
            (
                sum(samples[i : i + channels]) // channels
                for i in range(0, len(samples) - channels + 1, channels)
            ),
        )
    if rate != AVATAR_SAMPLE_RATE and len(samples) > 1:
        # Linear resample in pure Python.  This path is not expected to run
        # (the engine emits 24 kHz); it exists so a config change upstream
        # degrades pitch-correctly instead of playing at the wrong speed.
        out_len = round(len(samples) * AVATAR_SAMPLE_RATE / rate)
        step = (len(samples) - 1) / max(out_len - 1, 1)
        resampled = array.array("h", bytes(2 * out_len))
        for i in range(out_len):
            pos = i * step
            idx = int(pos)
            frac = pos - idx
            nxt = samples[idx + 1] if idx + 1 < len(samples) else samples[idx]
            resampled[i] = int(samples[idx] * (1 - frac) + nxt * frac)
        samples = resampled
    return samples.tobytes()


class AvatarBridge:
    """One bridge per browser connection; owns one websocket to the avatar."""

    def __init__(self, avatar_url: str, send_client_json):
        self.avatar_url = avatar_url
        # Serialized sender into the browser websocket, provided by the proxy.
        # The proxy's own downstream pump and this bridge's frame pump both
        # write to the same client socket, hence the shared lock lives there.
        self._send_client_json = send_client_json
        self._ws = None
        self._recv_task = None
        self._rid = 0
        self._failed_notified = False
        self._audio_format_logged = False

    async def start(self) -> None:
        try:
            import websockets

            self._ws = await websockets.connect(
                self.avatar_url, max_size=None, ping_interval=20
            )
            hello = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=10))
            self._recv_task = asyncio.create_task(self._recv_loop())
            # Nudge the avatar into its breathing loop right away so the person
            # is visibly alive before the first reply.
            await self._ws.send(json.dumps({"event": "idle"}))
            await self._status(
                event="avatar_connected",
                ready=bool(hello.get("ready")),
                resolution=hello.get("resolution"),
                fps=hello.get("fps"),
            )
        except Exception as exc:  # noqa: BLE001 - avatar is optional by design
            self._ws = None
            await self._fail(f"avatar unavailable: {exc}")

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"event": "end"}))
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
            self._ws = None

    async def on_downstream(self, text: str) -> None:
        """Inspect one engine->browser message and tee what the avatar needs.

        Called by the proxy for every downstream text frame BEFORE it is
        forwarded to the browser, so the browser sees the original stream
        unmodified and in order.
        """
        if self._ws is None:
            return
        try:
            msg = json.loads(text)
        except (ValueError, TypeError):
            return
        msg_type = msg.get("type")
        try:
            if msg_type == "response.start":
                self._rid += 1
                await self._ws.send(json.dumps({"event": "start", "rid": self._rid}))
                # Tell the browser which rid to couple its A/V anchor to.
                await self._send_client_json(
                    {"type": "avatar.turn", "rid": self._rid}
                )
            elif msg_type == "response.audio.delta":
                try:
                    pcm = wav_to_pcm24k(base64.b64decode(msg.get("data") or ""))
                except ValueError as exc:
                    if not self._audio_format_logged:
                        self._audio_format_logged = True
                        await self._status(event="avatar_audio_skip", message=str(exc))
                    return
                if pcm:
                    await self._ws.send(b"\x01" + pcm)
            elif msg_type == "response.audio.done":
                await self._ws.send(json.dumps({"event": "end"}))
        except Exception as exc:  # noqa: BLE001 - degrade, never break the call
            self._ws = None
            await self._fail(f"avatar send failed: {exc}")

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    if len(message) > 9 and message[0] == 0x21:
                        rid, pts = struct.unpack_from("<II", message, 1)
                        await self._send_client_json(
                            {
                                "type": "avatar.frame",
                                "rid": rid,
                                "pts": pts,
                                "data": base64.b64encode(message[9:]).decode(),
                            }
                        )
                    continue
                try:
                    event = json.loads(message)
                except (ValueError, TypeError):
                    continue
                name = event.pop("event", "avatar_event")
                # Forward the useful telemetry, drop the chatty per-block noise
                # the browser has no use for.
                if name in (
                    "avatar_block",
                    "avatar_done",
                    "avatar_error",
                    "avatar_gesture",
                ):
                    await self._status(event=name, **event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._ws = None
            await self._fail(f"avatar stream ended: {exc}")

    async def _status(self, **payload) -> None:
        try:
            await self._send_client_json({"type": "avatar.status", **payload})
        except Exception:  # noqa: BLE001 - client is going away; nothing to do
            pass

    async def _fail(self, message: str) -> None:
        LOG.warning("%s", message)
        if not self._failed_notified:
            self._failed_notified = True
            await self._status(event="avatar_error", message=message)
