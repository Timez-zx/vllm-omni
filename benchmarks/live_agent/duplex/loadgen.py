"""Parametric multi-user load generator for the MiniCPM-o native duplex stack.

Each simulated user opens one Realtime duplex WebSocket session and loops a
question/silence cycle at true realtime pacing (200 ms PCM chunks, 16 kHz):

    [question wav, ~Q s speech] -> commit -> [G s silence units]

With auto_response enabled the server emits whole 1 s model units for speech
AND silence, so every user costs one unit per second regardless of talking --
the duplex duty-cycle workload shape. Knobs:

  --users N              concurrent sessions
  --gap-s G              silence seconds between questions (speak fraction)
  --video                attach one camera frame per second of audio
  --cycles C             question cycles per user
  --barge-in             send the next question while response audio is still
                         arriving (interrupt pressure)

Per-user audit (JSON lines): per-turn commit->first-audio latency, audio
delta arrival times vs realtime playback deadline (underruns), event counts,
errors. Summary aggregates p50/p95 across users.

Usage:
  python benchmarks/live_agent/duplex/loadgen.py \
      --users 4 --cycles 2 --gap-s 6 --out /data/zx/results/duplex/loadgen_u4.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    read_pcm16_wav,
)

CHUNK_MS = 200
CHUNK_BYTES = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * CHUNK_MS // 1000


def _data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


def _b64_jpeg(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


class UserSession:
    def __init__(self, uid: int, args, question_pcm: bytes, ref_url: str, frame_b64: str | None):
        self.uid = uid
        self.args = args
        self.question_pcm = question_pcm
        self.ref_url = ref_url
        self.frame_b64 = frame_b64
        self.turns: list[dict] = []
        self.errors: list[str] = []
        self.units_sent = 0
        self.audio_deltas = 0
        self.underruns = 0
        self.admitted = False
        self.rolls: list[dict] = []
        self.last_close_at = 0.0
        self.matched_responses: list[dict] = []

    async def _stream_pcm(self, client: RealtimeDuplexClient, pcm: bytes) -> None:
        """Send pcm at realtime pace; attach one frame per second when enabled."""
        chunk_index = 0
        for offset in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[offset : offset + CHUNK_BYTES]
            duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
            event: dict[str, object] = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode("ascii"),
                "input_audio_format": "pcm16",
                "sample_rate_hz": PCM16_SAMPLE_RATE,
                "duration_ms": duration_ms,
            }
            if self.frame_b64 is not None and chunk_index % 5 == 0:
                event["video_frames"] = [self.frame_b64]
                event["max_slice_nums"] = 1
            await client.send(event)
            chunk_index += 1
            if duration_ms >= CHUNK_MS:
                self.units_sent += duration_ms / 1000.0
            await asyncio.sleep(duration_ms / 1000)

    def _order_match_responses(self, client: RealtimeDuplexClient) -> None:
        """Order-based attribution: k-th response belongs to k-th commit.

        Windowed attribution mislabels late responses (a cycle-0 answer landing
        during cycle 1 shows up as a 5 s cycle-1 answer). Responses are FIFO per
        session, so order matching is ground truth.
        """
        commits = [t["commit_at"] for t in self.turns if "commit_at" in t]
        responses: list[dict] = []
        seen: dict[str, dict] = {}
        for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=False):
            etype = event.get("type")
            rid = client.events.response_id(event)
            if etype == "response.created" and rid and rid not in seen:
                seen[rid] = {"rid": rid, "created_at": received_at, "first_audio_at": None, "audio_s": 0.0}
                responses.append(seen[rid])
            elif etype == "response.audio.delta" and rid in seen:
                delta = event.get("delta") or event.get("audio")
                if isinstance(delta, str):
                    raw = len(base64.b64decode(delta))
                    rate = event.get("sample_rate_hz")
                    rate = rate if isinstance(rate, int) and rate > 0 else 24_000
                    if seen[rid]["first_audio_at"] is None:
                        seen[rid]["first_audio_at"] = received_at
                    seen[rid]["audio_s"] += raw / (2 * rate)
        matched = []
        for k, resp in enumerate(responses):
            if k >= len(commits):
                break
            matched.append(
                {
                    "commit_idx": k,
                    "created_ms": round((resp["created_at"] - commits[k]) * 1000, 1),
                    "first_audio_ms": (
                        round((resp["first_audio_at"] - commits[k]) * 1000, 1)
                        if resp["first_audio_at"] is not None
                        else None
                    ),
                    "audio_s": round(resp["audio_s"], 2),
                }
            )
        self.matched_responses.extend(matched)

    def _watch_audio(self, client: RealtimeDuplexClient, turn: dict) -> None:
        """Update turn audio stats from events collected since turn start."""
        first_audio_s = None
        total_audio_s = 0.0
        deltas = 0
        for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=False):
            if received_at < turn["commit_at"]:
                continue
            if event.get("type") != "response.audio.delta":
                continue
            delta = event.get("delta") or event.get("audio")
            if not isinstance(delta, str):
                continue
            raw_len = len(base64.b64decode(delta))
            rate = event.get("sample_rate_hz")
            rate = rate if isinstance(rate, int) and rate > 0 else 24_000
            if first_audio_s is None:
                first_audio_s = received_at
            # underrun: this delta arrived after the moment its predecessor
            # audio finished playing (200 ms client jitter allowance)
            playback_deadline = first_audio_s + total_audio_s + 0.2
            if received_at > playback_deadline and deltas > 0:
                self.underruns += 1
            total_audio_s += raw_len / (2 * rate)
            deltas += 1
        turn["first_audio_ms"] = (
            round((first_audio_s - turn["commit_at"]) * 1000, 1) if first_audio_s is not None else None
        )
        turn["audio_s"] = round(total_audio_s, 2)
        turn["audio_deltas"] = deltas
        self.audio_deltas += deltas

    def _transcript_tail(self, client: RealtimeDuplexClient, max_chars: int = 400) -> str:
        parts = [
            e.get("delta", "")
            for e in client.events.events
            if e.get("type") in ("response.audio_transcript.delta", "response.output_text.delta")
            and isinstance(e.get("delta"), str)
        ]
        return "".join(parts)[-max_chars:]

    async def _run_cycles(self, client: RealtimeDuplexClient, cycles: list[int]) -> None:
        silence_chunk = b"\x00" * CHUNK_BYTES
        for cycle in cycles:
            turn: dict = {"cycle": cycle}
            await self._stream_pcm(client, self.question_pcm)
            turn["commit_at"] = time.monotonic()
            await client.send({"type": "input_audio_buffer.commit", "final": True})
            # keep the duty cycle: stream silence while the model answers
            gap_chunks = int(self.args.gap_s * 1000 / CHUNK_MS)
            for i in range(gap_chunks):
                await client.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(silence_chunk).decode("ascii"),
                        "input_audio_format": "pcm16",
                        "sample_rate_hz": PCM16_SAMPLE_RATE,
                        "duration_ms": CHUNK_MS,
                    }
                )
                self.units_sent += CHUNK_MS / 1000.0
                await asyncio.sleep(CHUNK_MS / 1000)
                if self.args.barge_in and i == gap_chunks // 2:
                    break  # cut the gap short: next question interrupts
            self._watch_audio(client, turn)
            self.turns.append(turn)

    async def run(self) -> dict:
        url = (
            f"{self.args.url}?duplex=1&model=openbmb%2FMiniCPM-o-4_5"
            "&minicpmo45_native_duplex=1&autostart=0"
        )
        url = url.replace("http://", "ws://").replace("https://", "wss://")
        roll_every = getattr(self.args, "roll_every_cycles", 0) or 0
        try:
            all_cycles = list(range(self.args.cycles))
            segments = (
                [all_cycles[i : i + roll_every] for i in range(0, len(all_cycles), roll_every)]
                if roll_every > 0
                else [all_cycles]
            )
            seed_instructions: str | None = None
            for seg_idx, segment in enumerate(segments):
                async with RealtimeDuplexClient(url) as client:
                    session_payload_extra = {}
                    if seed_instructions:
                        session_payload_extra["instructions"] = seed_instructions
                    await client.send(
                        {
                            "type": "session.update",
                            "session": {
                                "model": "openbmb/MiniCPM-o-4_5",
                                "modalities": ["audio", "text"],
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "turn_detection": None,
                                "overlap_policy": "listen_only",
                                "playback_commit_policy": "ack_only",
                                "ref_audio": self.ref_url,
                                "extra_body": {
                                    "auto_response": True,
                                    "minicpmo45_native_duplex": True,
                                    "force_listen_count": 0,
                                },
                                **session_payload_extra,
                            },
                        }
                    )
                    from vllm_omni.experimental.fullduplex.client import wait_for

                    reopen_started = time.monotonic()
                    await wait_for(
                        lambda: client.events.count("session.created") > 0,
                        timeout_s=30,
                        label="session.created",
                    )
                    if seg_idx > 0:
                        self.rolls.append(
                            {
                                "gap_ms": round((time.monotonic() - self.last_close_at) * 1000, 1),
                                "reopen_ms": round((time.monotonic() - reopen_started) * 1000, 1),
                                "seed_chars": len(seed_instructions or ""),
                            }
                        )
                    self.admitted = True
                    await self._run_cycles(client, segment)
                    self._order_match_responses(client)
                    if seg_idx < len(segments) - 1:
                        tail = self._transcript_tail(client)
                        seed_instructions = (
                            "Streaming Omni Conversation.\n"
                            f"(之前对话中你已经说过: {tail})" if tail else None
                        )
                    await client.close_session(timeout_s=10.0)
                    self.last_close_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001 -- audit everything, fail nothing
            self.errors.append(f"{type(exc).__name__}: {exc}")
        latencies = [t["first_audio_ms"] for t in self.turns if t.get("first_audio_ms") is not None]
        return {
            "uid": self.uid,
            "admitted": self.admitted,
            "turns": self.turns,
            "spoke_turns": len(latencies),
            "first_audio_ms": latencies,
            "units_sent": round(self.units_sent, 1),
            "audio_deltas": self.audio_deltas,
            "underruns": self.underruns,
            "rolls": self.rolls,
            "matched_responses": self.matched_responses,
            "errors": self.errors,
        }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 1)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime")
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--gap-s", type=float, default=6.0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--barge-in", action="store_true")
    parser.add_argument(
        "--roll-every-cycles",
        type=int,
        default=0,
        help="Session-level roll: after N cycles close the session and reopen "
        "seeded with the transcript tail as instructions (0 = off)",
    )
    parser.add_argument("--question-wav", default="/data/zx/results/duplex/user_q_padded.wav")
    parser.add_argument(
        "--ref-audio",
        default=(
            "/data/zx/hf/hub/models--openbmb--MiniCPM-o-4_5/snapshots/"
            "073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1/assets/HT_ref_audio.wav"
        ),
    )
    parser.add_argument("--frame-jpg", default="/data/zx/results/duplex/frame.jpg")
    parser.add_argument("--stagger-s", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    question_pcm = read_pcm16_wav(Path(args.question_wav))
    ref_url = _data_url(Path(args.ref_audio))
    frame_b64 = _b64_jpeg(Path(args.frame_jpg)) if args.video else None

    users = [UserSession(i, args, question_pcm, ref_url, frame_b64) for i in range(args.users)]

    async def staggered(u: UserSession):
        await asyncio.sleep(u.uid * args.stagger_s)
        return await u.run()

    started = time.monotonic()
    results = await asyncio.gather(*(staggered(u) for u in users))
    wall_s = round(time.monotonic() - started, 1)

    all_lat = [x for r in results for x in r["first_audio_ms"]]
    matched_lat = [
        m["first_audio_ms"]
        for r in results
        for m in r.get("matched_responses", [])
        if m.get("first_audio_ms") is not None
    ]
    summary = {
        "config": {
            "users": args.users,
            "cycles": args.cycles,
            "gap_s": args.gap_s,
            "video": args.video,
            "barge_in": args.barge_in,
        },
        "wall_s": wall_s,
        "admitted": sum(1 for r in results if r["admitted"]),
        "rejected_or_failed": sum(1 for r in results if not r["admitted"] or r["errors"]),
        "total_turns": sum(len(r["turns"]) for r in results),
        "spoke_turns": sum(r["spoke_turns"] for r in results),
        "first_audio_ms_p50": _percentile(all_lat, 0.50),
        "first_audio_ms_p95": _percentile(all_lat, 0.95),
        "first_audio_ms_max": _percentile(all_lat, 1.0),
        "matched_responses": sum(len(r.get("matched_responses", [])) for r in results),
        "matched_first_audio_p50": _percentile(matched_lat, 0.50),
        "matched_first_audio_p95": _percentile(matched_lat, 0.95),
        "matched_first_audio_max": _percentile(matched_lat, 1.0),
        "underruns_total": sum(r["underruns"] for r in results),
        "errors": [e for r in results for e in r["errors"]],
        "users_detail": results,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    concise = {k: v for k, v in summary.items() if k != "users_detail"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
