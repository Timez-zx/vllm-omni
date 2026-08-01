#!/usr/bin/env python3
"""Real-time-paced client emulator for vllm-omni /v1/video/chat/stream.

Why this exists
---------------
The shipped reference client (examples/online_serving/qwen3_omni/
streaming_video_client.py) contains no sleep anywhere: it blasts every frame
onto the socket as fast as the event loop allows, then sends audio, then a
query. That measures a bulk upload, not a live session. Any latency or
capacity number taken from it is meaningless for a camera-on voice agent.

This client instead drives an OPEN-LOOP schedule against absolute monotonic
deadlines:

  * video.frame is emitted every 1/fps seconds,
  * audio.chunk is emitted every chunk_ms of real time (silence included, so
    the always-on audio path is exercised the way a real client exercises it),
  * both continue on schedule regardless of what the server is doing.

Send lag (deadline vs actual send time) is recorded rather than absorbed: if
the client itself becomes the bottleneck, that has to be visible in the data
instead of being misattributed to the server.

Every event is written to a JSONL trace with a single monotonic clock so the
latency metrics can be reconstructed offline.

Turn structure per turn:
    ... continuous frames+audio ...
    [speech window: real speech PCM instead of silence]
    video.query          <- proxy for "user stopped speaking"
    wait for response.text.done + response.audio.done
    >= inter_turn_idle_s of idle       <- REQUIRED: PR #2342 documents an
                                          engine-layer scheduler race on
                                          back-to-back short replies; the
                                          upstream workaround is >=200 ms idle.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import pathlib
import sys
import time
import wave

import websockets

SILENCE = b"\x00\x00"

# Qwen-Omni will not reliably produce speech without this system prompt. The
# vllm-omni streaming-video handler defaults system_prompt to None and
# create_message_history() returns [], so a default session gets NO system
# prompt at all and the talker emits a fraction of a second of audio. The
# string below is the one used in vllm-omni's own curl example and benchmark
# data modules; supplying it is required for the audio path to be measurable.
QWEN_OMNI_SPEECH_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# A short rotation of plausible live-assistant questions. Turns use distinct
# queries so that a replayed or cached response is visible as such rather than
# being mistaken for a fast one.
DEFAULT_QUERIES = [
    "What is happening right now?",
    "Describe what you can see in front of you.",
    "Is anything in the scene changing?",
    "What is the person doing at the moment?",
    "Tell me what stands out in this room.",
    "Has anything moved since a moment ago?",
]


# ---------------------------------------------------------------------------
def load_frames(frame_dir: pathlib.Path) -> list[bytes]:
    return [p.read_bytes() for p in sorted(frame_dir.glob("*.jpg"))]


def load_pcm16_mono16k(path: pathlib.Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise ValueError(
                f"{path}: need mono/16kHz/16-bit, got "
                f"{w.getnchannels()}ch/{w.getframerate()}Hz/{w.getsampwidth()*8}bit"
            )
        return w.readframes(w.getnframes())


class Trace:
    """Append-only event log on one monotonic clock."""

    def __init__(self, path: pathlib.Path, meta: dict) -> None:
        self.f = path.open("w")
        self.t0 = time.monotonic()
        self.rec("meta", **meta, wall_start=time.time())

    def rec(self, kind: str, **kw) -> None:
        self.f.write(json.dumps({"t": time.monotonic() - self.t0, "k": kind, **kw}) + "\n")

    def close(self) -> None:
        self.f.flush()
        self.f.close()


# ---------------------------------------------------------------------------
async def receiver(ws, tr: Trace, state: dict) -> None:
    """Log every inbound event; flag the first text/audio delta per turn."""
    try:
        async for raw in ws:
            t = time.monotonic() - tr.t0
            try:
                msg = json.loads(raw)
            except Exception:
                tr.rec("rx_unparseable", n=len(raw))
                continue
            mt = msg.get("type", "?")
            turn = state["turn"]
            if mt == "response.start":
                tr.rec("rx_response_start", turn=turn)
            elif mt == "response.text.delta":
                d = msg.get("delta", "")
                if not state["got_text"]:
                    state["got_text"] = True
                    tr.rec("rx_first_text_delta", turn=turn, nchars=len(d))
                tr.rec("rx_text_delta", turn=turn, nchars=len(d))
            elif mt == "response.text.done":
                state["text_done"] = True
                tr.rec("rx_text_done", turn=turn, text=msg.get("text", "")[:2000])
            elif mt == "response.audio.delta":
                raw_au = base64.b64decode(msg.get("data", "") or "")
                nb = len(raw_au)
                # each delta is its own WAV container; parse it so audio
                # duration is exact instead of assuming a sample rate
                dur = None
                try:
                    with wave.open(io.BytesIO(raw_au), "rb") as w:
                        dur = w.getnframes() / w.getframerate()
                        state["audio_sr"] = w.getframerate()
                        state["audio_width"] = w.getsampwidth()
                except Exception:
                    pass
                if dur is not None:
                    state["audio_seconds"] += dur
                if not state["got_audio"]:
                    state["got_audio"] = True
                    tr.rec("rx_first_audio_delta", turn=turn, nbytes=nb,
                           sr=state.get("audio_sr"), dur_s=dur)
                state["audio_bytes"] += nb
                tr.rec("rx_audio_delta", turn=turn, nbytes=nb, dur_s=dur)
            elif mt == "response.audio.done":
                state["audio_done"] = True
                tr.rec("rx_audio_done", turn=turn,
                       total_audio_bytes=state["audio_bytes"],
                       total_audio_seconds=state["audio_seconds"],
                       sr=state.get("audio_sr"))
            elif mt == "session.done":
                tr.rec("rx_session_done")
                state["session_done"] = True
                return
            elif mt == "error":
                tr.rec("rx_error", message=str(msg.get("message"))[:500])
                state["error"] = str(msg.get("message"))
            else:
                tr.rec("rx_other", type=mt)
    except websockets.exceptions.ConnectionClosed as e:
        tr.rec("rx_closed", code=getattr(e, "code", None), reason=str(e)[:200])


# ---------------------------------------------------------------------------
async def run_session(args) -> int:
    frames = load_frames(pathlib.Path(args.frames))
    if not frames:
        print(f"no frames in {args.frames}", file=sys.stderr)
        return 1
    speech = load_pcm16_mono16k(pathlib.Path(args.speech)) if args.speech else b""

    bytes_per_chunk = int(16000 * 2 * args.chunk_ms / 1000)
    silence_chunk = SILENCE * (bytes_per_chunk // 2)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "uri": args.uri,
        "model": args.model,
        "fps": args.fps,
        "chunk_ms": args.chunk_ms,
        "audio_mode": args.audio_mode,
        "num_frames": args.num_frames,
        "max_frames": args.max_frames,
        "evs": args.evs,
        "evs_threshold": args.evs_threshold,
        "turns": args.turns,
        "turn_period_s": args.turn_period_s,
        "speak_dur_s": args.speak_dur_s,
        "inter_turn_idle_s": args.inter_turn_idle_s,
        "frames_available": len(frames),
        "speech_bytes": len(speech),
        "warmup_turns": args.warmup_turns,
        "system_prompt": args.system_prompt,
    }
    tr = Trace(outp, meta)

    async with websockets.connect(args.uri, max_size=args.ws_max_mb * 1024 * 1024,
                                  ping_interval=None) as ws:
        cfg = {
            "type": "session.config",
            "model": args.model,
            "modalities": ["text", "audio"] if args.audio_out else ["text"],
            "num_frames": args.num_frames,
            "max_frames": args.max_frames,
            "enable_frame_filter": args.evs,
            "frame_filter_threshold": args.evs_threshold,
            "use_audio_in_video": bool(speech) or args.audio_mode == "continuous",
        }
        if args.system_prompt:
            cfg["system_prompt"] = args.system_prompt
        await ws.send(json.dumps(cfg))
        tr.rec("tx_session_config", **{k: v for k, v in cfg.items() if k != "type"})

        state = {"turn": 0, "got_text": False, "got_audio": False,
                 "text_done": False, "audio_done": False, "audio_bytes": 0,
                 "audio_seconds": 0.0, "audio_sr": None, "audio_width": None,
                 "session_done": False, "error": None}
        rx = asyncio.create_task(receiver(ws, tr, state))

        loop = asyncio.get_running_loop()
        t_start = loop.time()
        next_frame_deadline = t_start
        next_audio_deadline = t_start
        frame_i = 0
        audio_pos = 0            # cursor into the speech PCM
        speech_until = -1.0      # absolute loop time the current utterance ends
        frame_period = 1.0 / args.fps
        queries = ([args.query] if args.query else DEFAULT_QUERIES)
        audio_period = args.chunk_ms / 1000.0

        for turn in range(args.turns):
            state.update(turn=turn, got_text=False, got_audio=False,
                         text_done=False, audio_done=False, audio_bytes=0,
                         audio_seconds=0.0)

            # --- perceive phase: stream frames+audio for turn_period_s, with a
            #     speech window at the end of the window ---
            turn_end = loop.time() + args.turn_period_s
            speech_start = turn_end - args.speak_dur_s
            tr.rec("turn_begin", turn=turn, speak_dur_s=args.speak_dur_s)
            spoke = False

            while loop.time() < turn_end:
                now = loop.time()
                # ---- video on its own deadline ----
                if now >= next_frame_deadline:
                    blob = frames[frame_i % len(frames)]
                    lag = now - next_frame_deadline
                    await ws.send(json.dumps({
                        "type": "video.frame",
                        "data": base64.b64encode(blob).decode(),
                    }))
                    tr.rec("tx_frame", turn=turn, i=frame_i,
                           nbytes=len(blob), lag_s=lag)
                    frame_i += 1
                    # absolute schedule: never accumulate drift
                    next_frame_deadline += frame_period
                    if lag > frame_period:  # fell a whole period behind
                        skipped = int(lag // frame_period)
                        next_frame_deadline += skipped * frame_period
                        tr.rec("frame_schedule_slip", turn=turn,
                               skipped_deadlines=skipped, lag_s=lag)

                # ---- audio on its own deadline ----
                if args.audio_mode != "off" and now >= next_audio_deadline:
                    in_speech = speech and now >= speech_start
                    if in_speech:
                        if not spoke:
                            tr.rec("speech_begin", turn=turn)
                            spoke = True
                        chunk = speech[audio_pos:audio_pos + bytes_per_chunk]
                        if len(chunk) < bytes_per_chunk:
                            audio_pos = 0
                            chunk = speech[0:bytes_per_chunk]
                        audio_pos += bytes_per_chunk
                    else:
                        if args.audio_mode == "speech-only":
                            next_audio_deadline += audio_period
                            continue
                        chunk = silence_chunk
                    lag = now - next_audio_deadline
                    await ws.send(json.dumps({
                        "type": "audio.chunk",
                        "data": base64.b64encode(chunk).decode(),
                    }))
                    tr.rec("tx_audio", turn=turn, nbytes=len(chunk),
                           speech=bool(in_speech), lag_s=lag)
                    next_audio_deadline += audio_period
                    if lag > audio_period:
                        skipped = int(lag // audio_period)
                        next_audio_deadline += skipped * audio_period
                        tr.rec("audio_schedule_slip", turn=turn,
                               skipped_deadlines=skipped, lag_s=lag)

                # yield to the event loop until the nearer deadline
                nxt = next_frame_deadline
                if args.audio_mode != "off":
                    nxt = min(nxt, next_audio_deadline)
                await asyncio.sleep(max(0.0, min(nxt, turn_end) - loop.time()))

                if state["error"] or state["session_done"]:
                    break

            if state["error"] or state["session_done"]:
                break

            # --- query: proxy for "user stopped speaking" ---
            q = queries[turn % len(queries)]
            await ws.send(json.dumps({"type": "video.query", "text": q}))
            tr.rec("tx_query", turn=turn, text=q, frames_sent_total=frame_i,
                   warmup=turn < args.warmup_turns)

            # --- wait for the turn to complete ---
            want_audio = args.audio_out
            t_wait = loop.time()
            while loop.time() - t_wait < args.turn_timeout_s:
                if state["error"] or state["session_done"]:
                    break
                if state["text_done"] and (state["audio_done"] or not want_audio):
                    break
                await asyncio.sleep(0.005)
            else:
                tr.rec("turn_timeout", turn=turn)

            tr.rec("turn_end", turn=turn, text_done=state["text_done"],
                   audio_done=state["audio_done"],
                   audio_bytes=state["audio_bytes"])

            if state["error"] or state["session_done"]:
                break

            # --- mandatory idle gap (upstream race workaround) ---
            tr.rec("inter_turn_idle_begin", turn=turn, s=args.inter_turn_idle_s)
            await asyncio.sleep(args.inter_turn_idle_s)
            # re-anchor the schedules so the idle gap is not "caught up" in a burst
            now = loop.time()
            next_frame_deadline = max(next_frame_deadline, now)
            next_audio_deadline = max(next_audio_deadline, now)

        await ws.send(json.dumps({"type": "video.done"}))
        tr.rec("tx_done")
        try:
            await asyncio.wait_for(rx, timeout=args.turn_timeout_s)
        except asyncio.TimeoutError:
            tr.rec("rx_task_timeout")
            rx.cancel()

    tr.rec("session_end", error=state["error"])
    tr.close()
    print(f"wrote {outp}")
    return 2 if state["error"] else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8091/v1/video/chat/stream")
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--frames", default="/data/zx/stimuli/frames/talkinghead")
    ap.add_argument("--speech", default="/data/zx/stimuli/audio_talkinghead.wav")
    ap.add_argument("--out", default="/data/zx/results/trace.jsonl")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--audio-mode", choices=["continuous", "speech-only", "off"],
                    default="continuous",
                    help="continuous = send silence between utterances too "
                         "(what a real client does)")
    ap.add_argument("--audio-out", action="store_true", default=True)
    ap.add_argument("--no-audio-out", dest="audio_out", action="store_false")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--evs", action="store_true", default=True)
    ap.add_argument("--no-evs", dest="evs", action="store_false")
    ap.add_argument("--evs-threshold", type=float, default=0.95)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--turn-period-s", type=float, default=10.0,
                    help="wall-clock seconds of streaming before each query")
    ap.add_argument("--speak-dur-s", type=float, default=3.0)
    ap.add_argument("--inter-turn-idle-s", type=float, default=0.25,
                    help=">=0.2 required: upstream scheduler race workaround")
    ap.add_argument("--turn-timeout-s", type=float, default=120.0)
    ap.add_argument("--query", default=None,
                    help="fixed query for every turn; default rotates DEFAULT_QUERIES "
                         "so a replayed response is distinguishable from a fast one")
    ap.add_argument("--system-prompt", default=QWEN_OMNI_SPEECH_SYSTEM_PROMPT,
                    help="REQUIRED for Qwen-Omni speech output; the handler "
                         "defaults to no system prompt at all")
    ap.add_argument("--no-system-prompt", dest="system_prompt",
                    action="store_const", const=None,
                    help="reproduce the shipped default (no system prompt)")
    ap.add_argument("--warmup-turns", type=int, default=1,
                    help="turns marked as warmup; the engine JIT-compiles Triton "
                         "kernels on the first inference, costing seconds")
    ap.add_argument("--ws-max-mb", type=int, default=32)
    args = ap.parse_args()

    if args.inter_turn_idle_s < 0.2:
        print("WARNING: inter-turn idle < 200 ms hits a known upstream "
              "scheduler race (PR #2342); results will be contaminated.",
              file=sys.stderr)
    return asyncio.run(run_session(args))


if __name__ == "__main__":
    raise SystemExit(main())
