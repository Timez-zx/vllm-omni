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
import gzip
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
# Capacity is judged at one fixed amount of playable audio, independent of how
# the server packetizes codec frames. Raw TTFA remains a transport diagnostic;
# it is not comparable across different initial_codec_chunk_frames values.
AUDIO_READY_THRESHOLD_MS = 500.0
PLAYBACK_PREBUFFER_S = AUDIO_READY_THRESHOLD_MS / 1000.0
WS_DEFLATE = False
SYSTEM_PROMPT = (
    "You are a friendly voice assistant in a live video call. You can see the camera and hear the user. "
    "Reply out loud conversationally, usually in one or two short sentences. Always answer with text and speech."
)

LOG_PROBES_BAD = {
    "unowned_audio": r"UNOWNED",
    "torch_cat_error": r"expected a non-empty list of Tensors",
    "arrival_prefill_failed": r"\[arrival-prefill\] failed",
}
LOG_PROBES_WARN = {
    "late_stage_output": r"Dropping output for unknown req",
}
LOG_PROBES_INFO = {
    "finite_requests": r"\[finite-request\]",
    "arrival_prefills": r"\[arrival-prefill\].*frames=",
    "history_compactions": r"\[session-history\] compact",
    "prefix_cache_hits": r"\[prefix-cache\].*hit_tokens=[1-9]",
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

    def __init__(
        self,
        *,
        plan: UserPlan,
        frames: list[str],
        opts: argparse.Namespace,
        replay_events: list[dict] | None = None,
    ) -> None:
        self.plan = plan
        self.name = plan.user
        self.frames = frames
        self.opts = opts
        self.frame_pos = plan.frame_start_offset
        self.records: list[dict] = []
        self.errors: list[str] = []
        self.acks_accepted = 0
        self.acks_filtered = 0
        self.frames_consumed = 0
        self.stray_audio = 0
        self.cur: dict | None = None
        self.done_evt = asyncio.Event()
        self.session_done_evt = asyncio.Event()

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
        self.replay_events = replay_events
        self.input_trace: list[dict] = []
        self.trace_started: float | None = None
        self.send_lock = asyncio.Lock()
        self.replay_schedule_slips = 0

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

    async def _run_turn_loop_with_background(
        self,
        ws,
        background: list[asyncio.Task],
    ) -> None:
        """Run live turns while treating a stopped media task as fatal.

        A reader or cadence pump that exits can otherwise leave the turn loop
        waiting forever for an utterance or response event that nobody can
        produce.
        """
        turn_task = asyncio.create_task(self._turn_loop(ws), name=f"{self.name}-turn-loop")
        try:
            done, _ = await asyncio.wait(
                [turn_task, *background],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if turn_task in done:
                await turn_task
                return

            stopped = next(task for task in background if task in done)
            if stopped.cancelled():
                raise RuntimeError(f"background task {stopped.get_name()} was cancelled")
            error = stopped.exception()
            if error is None:
                raise RuntimeError(f"background task {stopped.get_name()} stopped early")
            raise RuntimeError(f"background task {stopped.get_name()} failed: {error!r}") from error
        finally:
            if not turn_task.done():
                turn_task.cancel()
            await asyncio.gather(turn_task, return_exceptions=True)

    async def run(self) -> None:
        import websockets

        cfg = session_config(SYSTEM_PROMPT)
        cfg.update(
            {
                "session_id": self.name,
                "frame_filter_min_gap": 0,
                "frame_filter_max_gap": 4,
            }
        )
        extra = os.environ.get("MU_SESSION_CFG_JSON")
        if extra:
            cfg.update(json.loads(extra))

        try:
            await asyncio.sleep(self.plan.start_delay_s)
            async with websockets.connect(URL, max_size=None, compression=None) as ws:
                self.session_started = time.monotonic()
                await ws.send(json.dumps(cfg))
                self.trace_started = time.monotonic()
                tasks = [asyncio.create_task(self._reader(ws), name=f"{self.name}-reader")]
                if self.replay_events is None:
                    tasks.extend(
                        [
                            asyncio.create_task(self._frame_pump(ws), name=f"{self.name}-frame-pump"),
                            asyncio.create_task(self._microphone_pump(ws), name=f"{self.name}-microphone-pump"),
                        ]
                    )
                try:
                    if self.replay_events is None:
                        await asyncio.sleep(WARMUP_S)
                        await self._run_turn_loop_with_background(ws, tasks)
                        # Freeze the recorded input trace before closing the
                        # session. Otherwise the cadence pumps can append media
                        # after video.done while the reader waits for its ack.
                        for task in tasks[1:]:
                            task.cancel()
                        await asyncio.gather(*tasks[1:], return_exceptions=True)
                        await self._send_input(ws, {"type": "video.done"})
                    else:
                        await self._replay_inputs(ws)
                    await asyncio.wait_for(self.session_done_evt.wait(), timeout=30.0)
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

    async def _send_input(self, ws, message: dict, *, trace_record: dict | None = None) -> None:
        """Serialize sends and optionally record their actual client-side order."""
        async with self.send_lock:
            await ws.send(json.dumps(message))
            if self.opts.record_input_trace:
                if self.trace_started is None:
                    raise RuntimeError("input trace clock was not initialized")
                record = dict(trace_record if trace_record is not None else message)
                record["at_s"] = round(time.monotonic() - self.trace_started, 6)
                record["seq"] = len(self.input_trace)
                self.input_trace.append(record)

    async def _frame_pump(self, ws) -> None:
        deadline = asyncio.get_running_loop().time()
        while True:
            frame_index = self.frame_pos
            frame_id = f"{self.name}-f{self.frames_sent}"
            await self._send_input(
                ws,
                {
                    "type": "video.frame",
                    "data": self.frames[frame_index],
                    "frame_id": frame_id,
                },
                trace_record={"type": "video.frame", "frame_index": frame_index, "frame_id": frame_id},
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
                speech_bytes = 0
                endpoint_bytes = 0
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
                encoded = base64.b64encode(chunk).decode()
                await self._send_input(
                    ws,
                    {"type": "audio.chunk", "data": encoded},
                    trace_record={
                        "type": "audio.chunk",
                        "data": encoded,
                        "speech_bytes": speech_bytes,
                        "endpoint_bytes": endpoint_bytes,
                    },
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
            if event_type == "session.done":
                self.session_done_evt.set()
                continue
            if event_type == "video.frame.ack":
                self.acks_accepted += int(bool(msg.get("accepted")))
                self.acks_filtered += int(not msg.get("accepted"))
                continue
            if event_type == "video.frames.consumed":
                self.frames_consumed += len(msg.get("frame_ids") or [])
                continue
            if event_type == "error":
                self.errors.append(str(msg.get("message"))[:200])
                continue

            cur = self.cur
            if cur is None:
                if event_type == "response.audio.delta":
                    self.stray_audio += 1
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
                    if self.replay_events is None:
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
            }
            self.done_evt.clear()
            queried_at = time.monotonic()
            await self._send_input(
                ws,
                {"type": "video.query", "text": ""},
                trace_record={"type": "video.query", "text": "", "turn": index},
            )
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

    def _start_replay_turn(self, turn_index: int) -> float:
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
        }
        self.done_evt.clear()
        return time.monotonic()

    async def _finish_replay_turn(self, turn_index: int, queried_at: float) -> None:
        turn = self.plan.turns[turn_index]
        try:
            await asyncio.wait_for(self.done_evt.wait(), timeout=TURN_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.records.append(self._timeout_record(turn, queried_at))
            self.cur = None
            return
        cur, self.cur = self.cur, None
        if cur is None:
            self.errors.append(f"replay turn {turn_index} lost response state")
            return
        record, _ = self._complete_record(turn, queried_at, cur)
        self.records.append(record)

    async def _replay_inputs(self, ws) -> None:
        if self.trace_started is None or self.replay_events is None:
            raise RuntimeError("replay was not initialized")
        finalizer: asyncio.Task | None = None
        loop = asyncio.get_running_loop()
        for event in self.replay_events:
            deadline = self.trace_started + float(event["at_s"])
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            event_type = event["type"]
            if event_type == "video.frame":
                frame_index = int(event["frame_index"])
                await self._send_input(
                    ws,
                    {
                        "type": "video.frame",
                        "data": self.frames[frame_index],
                        "frame_id": event["frame_id"],
                    },
                )
                self.frames_sent += 1
            elif event_type == "audio.chunk":
                await self._send_input(ws, {"type": "audio.chunk", "data": event["data"]})
                pcm_bytes = base64.b64decode(event["data"])
                self.mic_chunks_sent += 1
                self.mic_samples_sent += len(pcm_bytes) // 2
                self.mic_speech_samples_sent += int(event.get("speech_bytes", 0)) // 2
                self.mic_endpoint_samples_sent += int(event.get("endpoint_bytes", 0)) // 2
                ambient_bytes = len(pcm_bytes) - int(event.get("speech_bytes", 0)) - int(event.get("endpoint_bytes", 0))
                self.mic_ambient_samples_sent += max(0, ambient_bytes) // 2
            elif event_type == "video.query":
                if finalizer is not None:
                    if not finalizer.done():
                        self.replay_schedule_slips += 1
                    await finalizer
                turn_index = int(event["turn"])
                queried_at = self._start_replay_turn(turn_index)
                await self._send_input(ws, {"type": "video.query", "text": event.get("text", "")})
                finalizer = asyncio.create_task(self._finish_replay_turn(turn_index, queried_at))
            elif event_type == "video.done":
                if finalizer is not None:
                    await finalizer
                await self._send_input(ws, {"type": "video.done"})
                return
            else:
                raise ValueError(f"unsupported replay event {event_type!r}")

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
            release_at_s=cur["t_done"] - queried_at,
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
                "audio_ready_500_ms": playback.start_s * 1000 if playback.start_s is not None else None,
                # Compatibility alias for older analysis scripts. Its contract
                # is now the fixed 500 ms threshold above, not a tunable client
                # configuration.
                "playback_start_ms": playback.start_s * 1000 if playback.start_s is not None else None,
                "stall_count": len(playback.stalls_s),
                "stall_total_ms": playback.stall_total_s * 1000,
                "stall_max_ms": playback.stall_max_s * 1000,
                "session_id": self.name,
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
            "frames_accepted": self.acks_accepted,
            "frames_filtered": self.acks_filtered,
            "frames_consumed": self.frames_consumed,
            "mic_chunks_sent": self.mic_chunks_sent,
            "mic_audio_s_sent": self.mic_samples_sent / AUDIO_RATE,
            "mic_speech_s_sent": self.mic_speech_samples_sent / AUDIO_RATE,
            "mic_endpoint_s_sent": self.mic_endpoint_samples_sent / AUDIO_RATE,
            "mic_ambient_s_sent": self.mic_ambient_samples_sent / AUDIO_RATE,
            "mic_paused_s": self.mic_paused_s,
            "replay_schedule_slips": self.replay_schedule_slips,
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


def write_input_trace(path: pathlib.Path, users: list[User], metadata: dict) -> None:
    """Write a compact replayable trace; frames reference the pinned frame set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as output:
        output.write(json.dumps({"kind": "header", **metadata}, sort_keys=True) + "\n")
        for user in users:
            for event in user.input_trace:
                output.write(json.dumps({"kind": "event", "user": user.name, **event}, sort_keys=True) + "\n")


def load_input_trace(path: pathlib.Path) -> tuple[dict, dict[str, list[dict]]]:
    metadata: dict | None = None
    events: dict[str, list[dict]] = {}
    with gzip.open(path, "rt") as source:
        for line_no, raw in enumerate(source, 1):
            item = json.loads(raw)
            if item.get("kind") == "header":
                if metadata is not None:
                    raise ValueError(f"{path}:{line_no}: duplicate trace header")
                metadata = {key: value for key, value in item.items() if key != "kind"}
            elif item.get("kind") == "event":
                user = str(item.pop("user"))
                item.pop("kind", None)
                events.setdefault(user, []).append(item)
            else:
                raise ValueError(f"{path}:{line_no}: invalid trace record")
    if metadata is None or not events:
        raise ValueError(f"{path}: trace is empty or missing its header")
    for user, user_events in events.items():
        expected = list(range(len(user_events)))
        actual = [int(event["seq"]) for event in user_events]
        if actual != expected:
            raise ValueError(f"{path}: non-contiguous event sequence for {user}")
    return metadata, events


def summarize(records: list[dict], users: list[User], meta: dict, log_slice: str, warmup_turns: int) -> dict:
    measured = [record for record in records if record.get("turn", 0) > warmup_turns]
    ok = [record for record in measured if record.get("status") == "ok"]
    ttfa = [record["ttfa_ms"] for record in ok if record.get("ttfa_ms") is not None]
    ttft = [record["ttft_ms"] for record in ok if record.get("ttft_ms") is not None]
    stalls = [record["stall_max_ms"] for record in ok]
    audio_ready = [
        record.get("audio_ready_500_ms", record.get("playback_start_ms"))
        for record in ok
        if record.get("audio_ready_500_ms", record.get("playback_start_ms")) is not None
    ]
    probes = {
        key: len(re.findall(pattern, log_slice))
        for key, pattern in {**LOG_PROBES_BAD, **LOG_PROBES_WARN, **LOG_PROBES_INFO}.items()
    }
    user_stats = [user.stats() for user in users]
    expected = len(users) * max(0, meta["turns_per_user"] - warmup_turns)
    stray_audio_deltas = sum(user.stray_audio for user in users)
    client_errors = sum(len(user.errors) for user in users)
    ttfa_p99 = pctl(ttfa, 0.99)
    stall_p99 = pctl(stalls, 0.99)
    audio_ready_p99 = pctl(audio_ready, 0.99)
    capacity_pass = bool(
        len(ok) == expected
        and len(ttfa) == expected
        and len(audio_ready) == expected
        and audio_ready_p99 is not None
        and audio_ready_p99 < 1000
        and stall_p99 is not None
        and stall_p99 < 50
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
        "audio_ready_threshold_ms": AUDIO_READY_THRESHOLD_MS,
        "audio_ready_500_p50_ms": pctl(audio_ready, 0.50),
        "audio_ready_500_p95_ms": pctl(audio_ready, 0.95),
        "audio_ready_500_p99_ms": audio_ready_p99,
        # Compatibility aliases. New comparisons should use audio_ready_500.
        "playback_start_p50_ms": pctl(audio_ready, 0.50),
        "playback_start_p95_ms": pctl(audio_ready, 0.95),
        "playback_start_p99_ms": audio_ready_p99,
        "first_audio_chunk_ms_p50": pctl(
            [record["deltas"][0][1] / 24.0 for record in ok if record.get("deltas")],
            0.50,
        ),
        "stall_max_ms_p50": pctl(stalls, 0.50),
        "stall_max_ms_p95": pctl(stalls, 0.95),
        "stall_max_ms_p99": stall_p99,
        "rtf_deliver_p50": pctl([record["rtf_deliver"] for record in ok if record.get("rtf_deliver")], 0.50),
        "input_audio_s_p50": pctl([record["input_audio_s"] for record in ok], 0.50),
        "output_audio_s_p50": pctl([record["audio_s"] for record in ok], 0.50),
        "stray_audio_deltas": stray_audio_deltas,
        "client_errors": client_errors,
        "per_user_errors": {user.name: user.errors[:5] for user in users if user.errors},
        "frames_sent": sum(item["frames_sent"] for item in user_stats),
        "frames_accepted": sum(item.get("frames_accepted", 0) for item in user_stats),
        "frames_filtered": sum(item.get("frames_filtered", 0) for item in user_stats),
        "frames_consumed": sum(item.get("frames_consumed", 0) for item in user_stats),
        "mic_chunks_sent": sum(item["mic_chunks_sent"] for item in user_stats),
        "mic_audio_s_sent": sum(item["mic_audio_s_sent"] for item in user_stats),
        "mic_speech_s_sent": sum(item["mic_speech_s_sent"] for item in user_stats),
        "mic_ambient_s_sent": sum(item["mic_ambient_s_sent"] for item in user_stats),
        "mic_paused_s": sum(item["mic_paused_s"] for item in user_stats),
        "replay_schedule_slips": sum(item.get("replay_schedule_slips", 0) for item in user_stats),
        "per_user_media": user_stats,
        "engine_probes": probes,
        "engine_warning_count": sum(probes[key] for key in LOG_PROBES_WARN),
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
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--record-input-trace",
        action="store_true",
        help="record the actual continuous client input as out/input_trace.jsonl.gz",
    )
    trace_group.add_argument(
        "--replay-input-trace",
        type=pathlib.Path,
        help="replay a previously recorded input trace exactly",
    )
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
    trace_metadata: dict | None = None
    replay_by_user: dict[str, list[dict]] = {}
    if args.replay_input_trace:
        trace_metadata, replay_by_user = load_input_trace(args.replay_input_trace)
        expected_trace = {
            "frame_set_sha256": frame_meta["frame_set_sha256"],
            "workload_plan_sha256": plan_sha256(plans),
            "users": args.users,
            "turns_per_user": args.turns,
            "repeat_sessions": args.repeat_sessions,
            "seed": args.seed,
        }
        mismatches = {
            key: (trace_metadata.get(key), value)
            for key, value in expected_trace.items()
            if trace_metadata.get(key) != value
        }
        if mismatches:
            parser.error(f"input trace does not match this workload: {mismatches}")
        missing_users = {plan.user for plan in plans} - set(replay_by_user)
        extra_users = set(replay_by_user) - {plan.user for plan in plans}
        if missing_users or extra_users:
            parser.error(f"input trace user mismatch: missing={sorted(missing_users)} extra={sorted(extra_users)}")
    args.out.mkdir(parents=True, exist_ok=True)
    plan_payload = [plan.event_fields() for plan in plans]
    (args.out / "workload_plan.json").write_text(json.dumps(plan_payload, indent=1))

    log_offset = LOG.stat().st_size if LOG.exists() else 0
    started_at = time.monotonic()
    all_users: list[User] = []
    for rep in range(args.repeat_sessions):
        first = rep * args.users
        cohort_plans = plans[first : first + args.users]
        cohort = [
            User(
                plan=plan,
                frames=frames,
                opts=args,
                replay_events=replay_by_user.get(plan.user) if args.replay_input_trace else None,
            )
            for plan in cohort_plans
        ]
        await asyncio.gather(*(user.run() for user in cohort))
        all_users.extend(cohort)

    records = [record for user in all_users for record in user.records]
    records.sort(key=lambda record: (record["user"], record["turn"]))
    with (args.out / "turns.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    trace_path: pathlib.Path | None = args.replay_input_trace
    if args.record_input_trace:
        trace_path = args.out / "input_trace.jsonl.gz"
        write_input_trace(
            trace_path,
            all_users,
            {
                "trace_schema": 1,
                "frame_set_sha256": frame_meta["frame_set_sha256"],
                "workload_plan_sha256": plan_sha256(plans),
                "users": args.users,
                "turns_per_user": args.turns,
                "repeat_sessions": args.repeat_sessions,
                "seed": args.seed,
            },
        )

    log_slice = ""
    if LOG.exists():
        with LOG.open("rb") as engine_log:
            engine_log.seek(log_offset)
            log_slice = re.sub(rb"\x1b\[[0-9;]*m", b"", engine_log.read()).decode(errors="replace")

    meta = {
        **benchmark_provenance(),
        **frame_meta,
        **audio_meta,
        "workload_schema": 4,
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
        "playback_prebuffer_ms": AUDIO_READY_THRESHOLD_MS,
        "query_transport": "audio; empty video.query is the client-side cut marker",
        "websocket_compression": WS_DEFLATE,
        "workload_plan_sha256": plan_sha256(plans),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "session_config_overrides": os.environ.get("MU_SESSION_CFG_JSON"),
        "input_trace_mode": "record" if args.record_input_trace else "replay" if args.replay_input_trace else "live",
        "input_trace": str(trace_path.resolve()) if trace_path else None,
        "input_trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest() if trace_path else None,
        "wall_s": time.monotonic() - started_at,
    }
    summary = summarize(records, all_users, meta, log_slice, args.warmup_turns)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(
        f"== continuous AV x {args.users}: ok={summary['n_ok']}/{summary['expected_measured_turns']} "
        f"timeout={summary['n_timeout']} skipped={summary['n_skipped']}"
    )
    print(
        f"   service-ttfa p50/p99={summary['ttfa_p50_ms']}/{summary['ttfa_p99_ms']} ms "
        f"audio-ready-500 p99={summary['audio_ready_500_p99_ms']} ms "
        f"stall-max p99={summary['stall_max_ms_p99']} ms pass={summary['capacity_pass']}"
    )
    if summary["capacity_pass"]:
        return 0
    # Keep an SLO boundary distinct from a broken harness/protocol. The
    # capacity ladder treats 3 as an expected stopping condition.
    if records and summary["client_errors"] == 0:
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
