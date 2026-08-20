#!/usr/bin/env python3
"""Multi-user benchmark for one target scenario: continuous AV sessions.

Every simulated user mirrors the browser client:

* one long-lived WebSocket;
* JPEG frames for the full session at 500 ms cadence;
* PCM16 microphone chunks at 200 ms cadence while listening/thinking/speaking;
* microphone pause from first assistant audio through playback plus a 300 ms echo guard;
* a real recorded utterance followed by 700 ms endpoint silence;
* an empty ``video.query`` cut marker, because the question is in the audio;
* playback-paced closed-loop turns.

The audio manifest is JSONL with ``id``, ``speaker``, ``transcript``, and a
16 kHz mono PCM16 WAV path in ``audio``. Paths are relative to the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from continuous_av_workload import (  # noqa: E402
    AUDIO_CADENCE_MS,
    AUDIO_RATE,
    ECHO_GUARD_MS,
    ENDPOINT_SILENCE_MS,
    VIDEO_INTERVAL_MS,
    TurnPlan,
    UserPlan,
    build_user_plan,
    load_audio_manifest,
    load_frame_set,
    make_room_tone_chunks,
    plan_sha256,
)
from playback_metrics import simulate_playback  # noqa: E402
from probe import session_config  # noqa: E402

URL = os.environ.get("MU_URL", "ws://127.0.0.1:8091/v1/video/chat/stream")
LOG = pathlib.Path(os.environ.get("MU_ENGINE_LOG", "/tmp/vllm-omni-results/qwen_live.log"))
TURN_TIMEOUT_S = 180.0
WARMUP_S = 4.0
GIVE_UP_AFTER = 3
PLAYBACK_PREBUFFER_S = 0.060
WS_DEFLATE = False
SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a live video call. You can see the camera and hear the user. "
    "Reply out loud conversationally, usually in one or two short sentences. Always answer with text and speech."
)

LOG_PROBES_BAD = {
    "unowned_audio": r"UNOWNED",
    "torch_cat_error": r"expected a non-empty list of Tensors",
    "counter_leak_clamped": r"streaming-parked counter had leaked",
    "zero_output_wedge": r"sampled ZERO output tokens",
    "negative_slice": r"scope drift; shipping unadjusted",
}
LOG_PROBES_INFO = {
    "segment_stops": r"\[session\] audio segment stop",
    "arrival_prefill": r"prefill-on-arrival",
    "compress_warm": r"COMPRESS: warming shadow",
    "compress_swap": r"COMPRESS #\d+ at turn",
    "warmup_queued": r"warm-up queued",
    "blocking_roll": r"(?i)blocking roll",
    "preempted_reqs": r"preemptions=[1-9]",
    "recompute": r"(?i)recomput",
}


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


class User:
    """One browser-shaped continuous AV session."""

    def __init__(self, *, plan: UserPlan, frames: list[str], opts: argparse.Namespace) -> None:
        self.plan = plan
        self.name = plan.user
        self.frames = frames
        self.opts = opts
        self.frame_pos = plan.frame_start_offset
        self.records: list[dict] = []
        self.errors: list[str] = []
        self.protocol_mismatches = 0
        self.session_incarnation: str | None = None
        self.session_epoch: int | None = None
        self.expect_session_identity = True
        self.acks_accepted = 0
        self.acks_filtered = 0
        self.rolls = 0
        self.stray_audio = 0
        self.cur: dict | None = None
        self.done_evt = asyncio.Event()

        self.frames_sent = 0
        self.mic_chunks_sent = 0
        self.mic_samples_sent = 0
        self.mic_speech_samples_sent = 0
        self.mic_endpoint_samples_sent = 0
        self.mic_ambient_samples_sent = 0
        self.mic_paused = False
        self.mic_pause_started: float | None = None
        self.mic_paused_s = 0.0
        self.room_tone = make_room_tone_chunks(variant=int(hashlib.sha256(plan.user.encode()).hexdigest()[:8], 16))
        self.room_index = 0
        self.pending_mic: bytes | None = None
        self.pending_mic_offset = 0
        self.pending_speech_bytes = 0
        self.utterance_done = asyncio.Event()
        self.session_started: float | None = None
        self.session_ended: float | None = None

    def _set_mic_paused(self, paused: bool) -> None:
        if paused == self.mic_paused:
            return
        now = time.monotonic()
        self.mic_paused = paused
        if paused:
            self.mic_pause_started = now
        elif self.mic_pause_started is not None:
            self.mic_paused_s += now - self.mic_pause_started
            self.mic_pause_started = None

    def _queue_utterance(self, turn: TurnPlan) -> None:
        if self.pending_mic is not None:
            raise RuntimeError("microphone utterance already queued")
        endpoint_samples = AUDIO_RATE * ENDPOINT_SILENCE_MS // 1000
        self.pending_mic = turn.utterance.pcm + bytes(endpoint_samples * 2)
        self.pending_mic_offset = 0
        self.pending_speech_bytes = len(turn.utterance.pcm)
        self.utterance_done.clear()

    async def run(self) -> None:
        import websockets

        cfg = session_config(SYSTEM_PROMPT)
        cfg.update(
            {
                "session_id": self.name,
                "frame_filter_min_gap": 0,
                "frame_filter_max_gap": 4,
                "prefill_frames_on_arrival": True,
                "prefill_audio_on_arrival": True,
            }
        )
        extra = os.environ.get("MU_SESSION_CFG_JSON")
        if extra:
            cfg.update(json.loads(extra))
        self.expect_session_identity = bool(cfg.get("session_scoped_request", True))

        try:
            await asyncio.sleep(self.plan.start_delay_s)
            async with websockets.connect(URL, max_size=None, compression=None) as ws:
                self.session_started = time.monotonic()
                await ws.send(json.dumps(cfg))
                tasks = [
                    asyncio.create_task(self._reader(ws)),
                    asyncio.create_task(self._frame_pump(ws)),
                    asyncio.create_task(self._microphone_pump(ws)),
                ]
                try:
                    await asyncio.sleep(WARMUP_S)
                    await self._turn_loop(ws)
                    await ws.send(json.dumps({"type": "video.done"}))
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            self.errors.append(f"connection: {exc!r}"[:200])
            self._mark_skipped(reason="connection_lost")
        finally:
            self._set_mic_paused(False)
            self.session_ended = time.monotonic()

    async def _frame_pump(self, ws) -> None:
        deadline = asyncio.get_running_loop().time()
        while True:
            await ws.send(
                json.dumps(
                    {
                        "type": "video.frame",
                        "data": self.frames[self.frame_pos],
                        "frame_id": f"{self.name}-f{self.frames_sent}",
                    }
                )
            )
            self.frames_sent += 1
            self.frame_pos = (self.frame_pos + 1) % len(self.frames)
            deadline += VIDEO_INTERVAL_MS / 1000.0
            await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))

    async def _microphone_pump(self, ws) -> None:
        samples_per_chunk = AUDIO_RATE * AUDIO_CADENCE_MS // 1000
        bytes_per_chunk = samples_per_chunk * 2
        deadline = asyncio.get_running_loop().time()
        while True:
            if not self.mic_paused:
                if self.pending_mic is not None:
                    start = self.pending_mic_offset
                    stop = min(len(self.pending_mic), start + bytes_per_chunk)
                    chunk = self.pending_mic[start:stop]
                    self.pending_mic_offset = stop
                    speech_bytes = max(0, min(stop, self.pending_speech_bytes) - start)
                    endpoint_bytes = len(chunk) - speech_bytes
                    self.mic_speech_samples_sent += speech_bytes // 2
                    self.mic_endpoint_samples_sent += endpoint_bytes // 2
                    if stop >= len(self.pending_mic):
                        self.pending_mic = None
                        self.pending_mic_offset = 0
                        self.pending_speech_bytes = 0
                        self.utterance_done.set()
                else:
                    chunk = self.room_tone[self.room_index % len(self.room_tone)]
                    self.room_index += 1
                    self.mic_ambient_samples_sent += len(chunk) // 2
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "data": base64.b64encode(chunk).decode(),
                        }
                    )
                )
                self.mic_chunks_sent += 1
                self.mic_samples_sent += len(chunk) // 2
            deadline += AUDIO_CADENCE_MS / 1000.0
            await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))

    async def _reader(self, ws) -> None:
        async for raw in ws:
            received_at = time.monotonic()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            event_type = msg.get("type")
            if event_type == "session.created":
                if msg.get("session_id") != self.name:
                    self.protocol_mismatches += 1
                    self.errors.append("session.created id mismatch")
                    continue
                self.session_incarnation = msg.get("incarnation")
                self.session_epoch = msg.get("epoch")
                continue
            if event_type in ("session.rolled", "session.compressed"):
                if msg.get("incarnation") != self.session_incarnation:
                    self.protocol_mismatches += 1
                    self.errors.append(f"{event_type} incarnation mismatch")
                    continue
                self.session_epoch = msg.get("epoch")
                self.rolls += int(event_type == "session.rolled")
                continue
            if event_type == "video.frame.ack":
                self.acks_accepted += int(bool(msg.get("accepted")))
                self.acks_filtered += int(not msg.get("accepted"))
                continue
            if event_type == "error":
                self.errors.append(str(msg.get("message"))[:200])
                continue

            cur = self.cur
            if cur is None:
                if event_type == "response.audio.delta":
                    self.stray_audio += 1
                continue
            if event_type.startswith("response.") and self.expect_session_identity:
                got = (msg.get("session_id"), msg.get("incarnation"), msg.get("turn_id"))
                expected = (self.name, self.session_incarnation, cur["expected_turn_id"])
                if got != expected:
                    self.protocol_mismatches += 1
                    self.errors.append(f"{event_type} identity mismatch: got={got} expected={expected}")
                    continue
                if cur["segment_id"] is None:
                    cur["segment_id"] = msg.get("segment_id")
                    cur["epoch"] = msg.get("epoch")
                elif msg.get("segment_id") != cur["segment_id"] or msg.get("epoch") != cur["epoch"]:
                    self.protocol_mismatches += 1
                    self.errors.append(f"{event_type} segment/epoch changed mid-turn")
                    continue

            if event_type == "response.text.delta":
                if cur["t_first_text"] is None:
                    cur["t_first_text"] = received_at
                cur["text_stream"] += msg.get("delta") or ""
            elif event_type == "response.text.done":
                cur["text_at_first_sound"] = msg.get("text") or ""
            elif event_type == "response.audio.delta":
                if cur["t_first_audio"] is None:
                    cur["t_first_audio"] = received_at
                    self._set_mic_paused(True)
                try:
                    data = base64.b64decode(msg["data"])
                    with wave.open(io.BytesIO(data), "rb") as wav:
                        if wav.getframerate() != 24_000:
                            raise ValueError(f"unexpected output sample rate {wav.getframerate()}")
                        samples = wav.getnframes()
                        pcm = wav.readframes(samples)
                except Exception as exc:
                    self.errors.append(f"invalid audio delta: {exc!r}"[:200])
                    continue
                cur["deltas"].append((received_at, samples))
                cur["audio_samples"] += samples
                cur["n_deltas"] += 1
                if self.opts.save_wav_dir:
                    cur["pcm"] += pcm
            elif event_type == "response.audio.done":
                cur["t_done"] = received_at
                self.done_evt.set()

    async def _turn_loop(self, ws) -> None:
        consecutive_timeouts = 0
        for index, turn in enumerate(self.plan.turns):
            if index > 0:
                await asyncio.sleep(turn.think_s)
            self._queue_utterance(turn)
            await self.utterance_done.wait()
            self.cur = {
                "t_first_text": None,
                "t_first_audio": None,
                "t_done": None,
                "text_stream": "",
                "text_at_first_sound": "",
                "audio_samples": 0,
                "n_deltas": 0,
                "deltas": [],
                "pcm": bytearray(),
                "expected_turn_id": index,
                "segment_id": None,
                "epoch": None,
            }
            self.done_evt.clear()
            queried_at = time.monotonic()
            await ws.send(json.dumps({"type": "video.query", "text": ""}))
            try:
                await asyncio.wait_for(self.done_evt.wait(), timeout=TURN_TIMEOUT_S)
            except asyncio.TimeoutError:
                self._set_mic_paused(False)
                consecutive_timeouts += 1
                self.records.append(self._timeout_record(turn, queried_at))
                self.cur = None
                if consecutive_timeouts >= GIVE_UP_AFTER:
                    self._mark_skipped(start=turn.turn + 1, reason="gave_up")
                    return
                continue

            consecutive_timeouts = 0
            cur, self.cur = self.cur, None
            record, playback_end = self._complete_record(turn, queried_at, cur)
            self.records.append(record)
            if self.opts.save_wav_dir and cur["pcm"]:
                path = self.opts.save_wav_dir / f"{self.name}_t{turn.turn:02d}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24_000)
                    wav.writeframes(bytes(cur["pcm"]))
            remaining = playback_end - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            await asyncio.sleep(ECHO_GUARD_MS / 1000.0)
            self._set_mic_paused(False)

    def _timeout_record(self, turn: TurnPlan, queried_at: float) -> dict:
        cur = self.cur or {}
        return {
            **turn.event_fields(),
            "user": self.name,
            "q": turn.utterance.transcript,
            "status": "timeout",
            "t_q": queried_at,
            "t_ft": cur.get("t_first_text"),
            "t_fa": cur.get("t_first_audio"),
            "t_done": None,
            "ttfa_ms": None,
            "ttft_ms": None,
            "wall_s": None,
            "audio_s": cur.get("audio_samples", 0) / 24_000.0,
            "session_id": self.name,
            "incarnation": self.session_incarnation,
            "epoch": cur.get("epoch"),
            "segment_id": cur.get("segment_id"),
        }

    def _complete_record(self, turn: TurnPlan, queried_at: float, cur: dict) -> tuple[dict, float]:
        ttfa = (cur["t_first_audio"] - queried_at) * 1000 if cur["t_first_audio"] else None
        ttft = (cur["t_first_text"] - queried_at) * 1000 if cur["t_first_text"] else None
        audio_s = cur["audio_samples"] / 24_000.0
        deliver_s = cur["t_done"] - cur["t_first_audio"] if cur["t_first_audio"] else None
        playback = simulate_playback(
            [(stamp - queried_at, samples) for stamp, samples in cur["deltas"]],
            sample_rate=24_000,
            prebuffer_s=PLAYBACK_PREBUFFER_S,
        )
        playback_end = (
            queried_at + (playback.start_s or 0.0) + audio_s + playback.stall_total_s
            if playback.start_s is not None
            else time.monotonic()
        )
        return (
            {
                **turn.event_fields(),
                "user": self.name,
                "q": turn.utterance.transcript,
                "status": "ok",
                "t_q": queried_at,
                "t_ft": cur["t_first_text"],
                "t_fa": cur["t_first_audio"],
                "t_done": cur["t_done"],
                "ttfa_ms": ttfa,
                "ttft_ms": ttft,
                "wall_s": cur["t_done"] - queried_at,
                "audio_s": audio_s,
                "rtf_deliver": audio_s / deliver_s if deliver_s and deliver_s > 0 else None,
                "playback_start_ms": playback.start_s * 1000 if playback.start_s is not None else None,
                "stall_count": len(playback.stalls_s),
                "stall_total_ms": playback.stall_total_s * 1000,
                "stall_max_ms": playback.stall_max_s * 1000,
                "session_id": self.name,
                "incarnation": self.session_incarnation,
                "epoch": cur["epoch"],
                "segment_id": cur["segment_id"],
                "deltas": [[round(stamp - queried_at, 4), samples] for stamp, samples in cur["deltas"]],
                "n_deltas": cur["n_deltas"],
                "chars_stream": len(cur["text_stream"]),
                "text": cur["text_stream"][:200],
            },
            playback_end,
        )

    def _mark_skipped(self, *, start: int = 1, reason: str) -> None:
        completed = {record["turn"] for record in self.records}
        for turn in self.plan.turns:
            if turn.turn >= start and turn.turn not in completed:
                self.records.append(
                    {
                        **turn.event_fields(),
                        "user": self.name,
                        "q": turn.utterance.transcript,
                        "status": "skipped",
                        "reason": reason,
                    }
                )

    def stats(self) -> dict:
        wall_s = (self.session_ended or time.monotonic()) - (self.session_started or time.monotonic())
        return {
            "user": self.name,
            "session_wall_s": wall_s,
            "frames_sent": self.frames_sent,
            "mic_chunks_sent": self.mic_chunks_sent,
            "mic_audio_s_sent": self.mic_samples_sent / AUDIO_RATE,
            "mic_speech_s_sent": self.mic_speech_samples_sent / AUDIO_RATE,
            "mic_endpoint_s_sent": self.mic_endpoint_samples_sent / AUDIO_RATE,
            "mic_ambient_s_sent": self.mic_ambient_samples_sent / AUDIO_RATE,
            "mic_paused_s": self.mic_paused_s,
        }


def benchmark_provenance() -> dict:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
    except Exception:
        commit, dirty = None, None
    deploy_raw = os.environ.get("MU_DEPLOY_CONFIG")
    deploy = pathlib.Path(deploy_raw).resolve() if deploy_raw else None
    return {
        "source_commit": commit,
        "source_dirty": dirty,
        "deploy_config": str(deploy) if deploy else None,
        "deploy_config_sha256": hashlib.sha256(deploy.read_bytes()).hexdigest()
        if deploy and deploy.is_file()
        else None,
        "engine_log": str(LOG),
    }


def summarize(records: list[dict], users: list[User], meta: dict, log_slice: str, warmup_turns: int) -> dict:
    measured = [record for record in records if record.get("turn", 0) > warmup_turns]
    ok = [record for record in measured if record.get("status") == "ok"]
    ttfa = [record["ttfa_ms"] for record in ok if record.get("ttfa_ms") is not None]
    ttft = [record["ttft_ms"] for record in ok if record.get("ttft_ms") is not None]
    stalls = [record["stall_max_ms"] for record in ok]
    probes = {
        key: len(re.findall(pattern, log_slice)) for key, pattern in {**LOG_PROBES_BAD, **LOG_PROBES_INFO}.items()
    }
    user_stats = [user.stats() for user in users]
    expected = len(users) * max(0, meta["turns_per_user"] - warmup_turns)
    protocol_mismatches = sum(user.protocol_mismatches for user in users)
    stray_audio_deltas = sum(user.stray_audio for user in users)
    client_errors = sum(len(user.errors) for user in users)
    ttfa_p99 = pctl(ttfa, 0.99)
    stall_p99 = pctl(stalls, 0.99)
    capacity_pass = bool(
        len(ok) == expected
        and len(ttfa) == expected
        and ttfa_p99 is not None
        and ttfa_p99 < 1000
        and stall_p99 is not None
        and stall_p99 < 50
        and protocol_mismatches == 0
        and stray_audio_deltas == 0
        and client_errors == 0
        and all(probes[key] == 0 for key in LOG_PROBES_BAD)
    )
    return {
        **meta,
        "warmup_turns": warmup_turns,
        "expected_measured_turns": expected,
        "n_ok": len(ok),
        "n_timeout": sum(record.get("status") == "timeout" for record in measured),
        "n_skipped": sum(record.get("status") == "skipped" for record in measured),
        "ttfa_p50_ms": pctl(ttfa, 0.50),
        "ttfa_p95_ms": pctl(ttfa, 0.95),
        "ttfa_p99_ms": ttfa_p99,
        "ttft_p50_ms": pctl(ttft, 0.50),
        "ttft_p99_ms": pctl(ttft, 0.99),
        "stall_max_ms_p50": pctl(stalls, 0.50),
        "stall_max_ms_p95": pctl(stalls, 0.95),
        "stall_max_ms_p99": stall_p99,
        "rtf_deliver_p50": pctl([record["rtf_deliver"] for record in ok if record.get("rtf_deliver")], 0.50),
        "input_audio_s_p50": pctl([record["input_audio_s"] for record in ok], 0.50),
        "output_audio_s_p50": pctl([record["audio_s"] for record in ok], 0.50),
        "protocol_identity_mismatches": protocol_mismatches,
        "stray_audio_deltas": stray_audio_deltas,
        "client_errors": client_errors,
        "per_user_errors": {user.name: user.errors[:5] for user in users if user.errors},
        "session_rolls": sum(user.rolls for user in users),
        "frames_sent": sum(item["frames_sent"] for item in user_stats),
        "mic_chunks_sent": sum(item["mic_chunks_sent"] for item in user_stats),
        "mic_audio_s_sent": sum(item["mic_audio_s_sent"] for item in user_stats),
        "mic_speech_s_sent": sum(item["mic_speech_s_sent"] for item in user_stats),
        "mic_ambient_s_sent": sum(item["mic_ambient_s_sent"] for item in user_stats),
        "mic_paused_s": sum(item["mic_paused_s"] for item in user_stats),
        "per_user_media": user_stats,
        "engine_probes": probes,
        "capacity_pass": capacity_pass,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--repeat-sessions", type=int, default=1)
    parser.add_argument("--warmup-turns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stagger", default=os.environ.get("MU_STAGGER_S", "0,40"))
    parser.add_argument("--frames-dir", type=pathlib.Path, required=True)
    parser.add_argument("--audio-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--save-wav-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.users <= 0 or args.turns <= 0 or args.repeat_sessions <= 0:
        parser.error("users, turns, and repeat-sessions must be positive")
    if not 0 <= args.warmup_turns < args.turns:
        parser.error("warmup-turns must be in [0, turns)")
    try:
        stagger_s = tuple(float(value) for value in args.stagger.split(","))
        if len(stagger_s) != 2 or stagger_s[0] < 0 or stagger_s[1] < stagger_s[0]:
            raise ValueError
    except ValueError:
        parser.error("stagger must be LO,HI with 0 <= LO <= HI")
    if args.save_wav_dir:
        args.save_wav_dir.mkdir(parents=True, exist_ok=True)

    frames, frame_meta = load_frame_set(args.frames_dir)
    utterances, audio_meta = load_audio_manifest(args.audio_manifest)
    plans = [
        build_user_plan(
            uid=uid,
            rep=rep,
            users=args.users,
            turns=args.turns,
            seed=args.seed,
            utterances=utterances,
            frame_count=len(frames),
            stagger_s=stagger_s,
        )
        for rep in range(args.repeat_sessions)
        for uid in range(args.users)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    plan_payload = [plan.event_fields() for plan in plans]
    (args.out / "workload_plan.json").write_text(json.dumps(plan_payload, indent=1))

    log_offset = LOG.stat().st_size if LOG.exists() else 0
    started_at = time.monotonic()
    all_users: list[User] = []
    for rep in range(args.repeat_sessions):
        first = rep * args.users
        cohort_plans = plans[first : first + args.users]
        cohort = [User(plan=plan, frames=frames, opts=args) for plan in cohort_plans]
        await asyncio.gather(*(user.run() for user in cohort))
        all_users.extend(cohort)

    records = [record for user in all_users for record in user.records]
    records.sort(key=lambda record: (record["user"], record["turn"]))
    with (args.out / "turns.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")

    log_slice = ""
    if LOG.exists():
        with LOG.open("rb") as engine_log:
            engine_log.seek(log_offset)
            log_slice = re.sub(rb"\x1b\[[0-9;]*m", b"", engine_log.read()).decode(errors="replace")

    meta = {
        **benchmark_provenance(),
        **frame_meta,
        **audio_meta,
        "workload_schema": 2,
        "percentile_method": "nearest_rank",
        "scenario": "continuous_av_session",
        "users": args.users,
        "turns_per_user": args.turns,
        "repeat_sessions": args.repeat_sessions,
        "seed": args.seed,
        "stagger_s": list(stagger_s),
        "video_interval_ms": VIDEO_INTERVAL_MS,
        "audio_cadence_ms": AUDIO_CADENCE_MS,
        "endpoint_silence_ms": ENDPOINT_SILENCE_MS,
        "echo_guard_ms": ECHO_GUARD_MS,
        "playback_prebuffer_ms": PLAYBACK_PREBUFFER_S * 1000,
        "query_transport": "audio; empty video.query is the client-side cut marker",
        "websocket_compression": WS_DEFLATE,
        "workload_plan_sha256": plan_sha256(plans),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "session_config_overrides": os.environ.get("MU_SESSION_CFG_JSON"),
        "wall_s": time.monotonic() - started_at,
    }
    summary = summarize(records, all_users, meta, log_slice, args.warmup_turns)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"== continuous AV x {args.users}: ok={summary['n_ok']}/{summary['expected_measured_turns']} "
        f"timeout={summary['n_timeout']} skipped={summary['n_skipped']}"
    )
    print(
        f"   ttfa p50/p99={summary['ttfa_p50_ms']}/{summary['ttfa_p99_ms']} ms "
        f"stall-max p99={summary['stall_max_ms_p99']} ms pass={summary['capacity_pass']}"
    )
    return 0 if records and summary["client_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
