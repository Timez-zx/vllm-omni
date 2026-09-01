"""Capacity workload for MiniCPM-o 4.5 native duplex serving.

Each user streams the audio and matching camera frames from one real MP4 at
wall-clock pace.  Audio is sent every 200 ms and one frame is attached to the
first append of every 1 s model unit.  Native auto-response stays enabled, so
the model continuously decides whether to listen or speak without synthetic
query commits between units.

Capacity is computed from per-unit stage service traces by ``analyze_rtf.py``.
Protocol progress is secondary: a listen result represents one 1 s model unit,
while speech may be projected as multiple audio deltas or internal continuation
units. Consequently, the Nth input unit cannot be paired with the Nth protocol
event. Audio playback slack is a diagnostic rather than slot latency.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    build_realtime_url,
)

CHUNK_MS = 200
UNIT_MS = 1000
CHUNKS_PER_UNIT = UNIT_MS // CHUNK_MS
CHUNK_BYTES = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * CHUNK_MS // 1000
MODEL = "openbmb/MiniCPM-o-4_5"


@dataclass(frozen=True)
class MediaAsset:
    path: Path
    pcm16: bytes
    frames: list[str]
    duration_s: float


def _percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    index = max(0, math.ceil(q * len(clean)) - 1)
    return round(clean[min(index, len(clean) - 1)], 2)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 2) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": _percentile(values, 1.0),
    }


def _ref_audio_data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _jpeg_b64(image: Image.Image, *, max_side: int = 448, quality: int = 80) -> str:
    image = image.convert("RGB")
    if max_side > 0:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _load_media(path: Path, *, frame_max_side: int = 448) -> tuple[bytes, list[str], float]:
    """Decode a real MP4 into 16 kHz PCM16 and one matching frame per second."""
    audio_parts: list[bytes] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"{path} has no audio stream")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM16_SAMPLE_RATE)
        for frame in container.decode(audio=0):
            converted = resampler.resample(frame)
            for out in converted if isinstance(converted, list) else [converted]:
                if out is not None:
                    # AudioPlane buffers are aligned and may include padding;
                    # ndarray contains exactly ``out.samples`` PCM samples.
                    audio_parts.append(out.to_ndarray().tobytes())
        flushed = resampler.resample(None)
        for out in flushed if isinstance(flushed, list) else [flushed]:
            if out is not None:
                audio_parts.append(out.to_ndarray().tobytes())

    pcm16 = b"".join(audio_parts)
    duration_s = len(pcm16) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
    frame_by_second: dict[int, str] = {}
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            timestamp = frame.time
            if timestamp is None:
                timestamp = float(frame.pts * stream.time_base) if frame.pts is not None else 0.0
            second = max(0, int(timestamp))
            if second not in frame_by_second:
                frame_by_second[second] = _jpeg_b64(frame.to_image(), max_side=frame_max_side)

    unit_count = int(duration_s)
    frames: list[str] = []
    previous: str | None = None
    for second in range(unit_count):
        previous = frame_by_second.get(second, previous)
        if previous is None:
            raise ValueError(f"no video frame available for second {second}")
        frames.append(previous)
    return pcm16, frames, duration_s


class GPUSampler:
    def __init__(self, gpu_ids: list[int], interval_s: float = 0.2) -> None:
        self.gpu_ids = gpu_ids
        self.interval_s = interval_s
        self.samples: dict[int, list[dict[str, float]]] = {gpu_id: [] for gpu_id in gpu_ids}
        self._stop = asyncio.Event()

    async def run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = {gpu_id: pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in self.gpu_ids}
            while not self._stop.is_set():
                now = time.monotonic()
                for gpu_id, handle in handles.items():
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    self.samples[gpu_id].append(
                        {
                            "at_s": now,
                            "gpu_pct": float(util.gpu),
                            "memory_pct": float(util.memory),
                            "memory_gib": memory.used / (1024**3),
                            "power_w": power,
                        }
                    )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except TimeoutError:
                    pass
        except Exception as exc:  # GPU metrics must not invalidate the workload.
            self.samples[-1] = [{"error": str(exc)}]  # type: ignore[list-item]

    def stop(self) -> None:
        self._stop.set()

    def report(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for gpu_id, samples in self.samples.items():
            if gpu_id < 0:
                result["error"] = samples
                continue
            result[str(gpu_id)] = {
                "sample_count": len(samples),
                "gpu_util_pct": _summary([sample["gpu_pct"] for sample in samples]),
                "memory_io_pct": _summary([sample["memory_pct"] for sample in samples]),
                "memory_used_gib": _summary([sample["memory_gib"] for sample in samples]),
                "power_w": _summary([sample["power_w"] for sample in samples]),
            }
        return result


class UserSession:
    def __init__(
        self,
        uid: int,
        args: argparse.Namespace,
        media: MediaAsset,
        ref_audio: str,
        phase_s: float,
        context_age_units: int,
        media_offset_units: int,
        session_id: str,
    ) -> None:
        self.uid = uid
        self.args = args
        self.media = media
        self.ref_audio = ref_audio
        self.phase_s = phase_s
        self.context_age_units = context_age_units
        self.media_offset_units = media_offset_units
        self.session_id = session_id
        self.formal_epoch_s = 0.0
        self.jitter_rng = random.Random((args.seed + 1) * 1_000_003 + uid)

    def _media_chunk(self, chunk_index: int) -> bytes:
        source_chunk = self.media_offset_units * CHUNKS_PER_UNIT + chunk_index
        source_offset = source_chunk * CHUNK_BYTES
        if self.args.loop_media:
            source_offset %= len(self.media.pcm16)
        chunk = self.media.pcm16[source_offset : source_offset + CHUNK_BYTES]
        if self.args.loop_media and len(chunk) < CHUNK_BYTES:
            chunk += self.media.pcm16[: CHUNK_BYTES - len(chunk)]
        if len(chunk) < CHUNK_BYTES:
            raise ValueError("media is shorter than preconditioning plus --duration-s")
        return chunk

    def _media_frame(self, unit_index: int) -> str:
        frame_index = self.media_offset_units + unit_index
        if self.args.loop_media:
            frame_index %= len(self.media.frames)
        if frame_index >= len(self.media.frames):
            raise ValueError("media has no frame for the requested unit")
        return self.media.frames[frame_index]

    async def run(
        self,
        ready_queue: asyncio.Queue[int],
        measurement_done_queue: asyncio.Queue[int],
        start_event: asyncio.Event,
    ) -> dict[str, Any]:
        # Session admission is setup, not the workload. Spread handshakes to
        # avoid measuring a reference-audio initialization burst, then hold a
        # barrier so every admitted session begins media in the same 1 s phase
        # window.
        await asyncio.sleep(self.uid * self.args.connect_stagger_s)
        url = build_realtime_url(
            self.args.url,
            MODEL,
            autostart=False,
            session_id=self.session_id,
        )
        unit_ready_at: list[float] = []
        send_drift_ms: list[float] = []
        arrival_jitter_ms: list[float] = []
        admitted = False
        errors: list[str] = []
        teardown_errors: list[str] = []
        client = RealtimeDuplexClient(url, open_timeout_s=self.args.timeout_s)
        started_at = time.monotonic()
        formal_started_at = math.inf
        measurement_done_at = math.inf
        ready_reported = False
        measurement_done_reported = False
        precondition_units_sent = 0
        formal_units_sent = 0
        precondition_frames_sent = 0
        formal_frames_sent = 0
        try:
            async with client:
                extra_body: dict[str, object] = {}
                if self.args.context_window_trigger_tokens is not None:
                    extra_body["context_window_trigger_tokens"] = self.args.context_window_trigger_tokens
                if self.args.force_listen_count is not None:
                    extra_body["force_listen_count"] = self.args.force_listen_count
                await client.configure(
                    MODEL,
                    ref_audio=self.ref_audio,
                    session_id=self.session_id,
                    extra_body=extra_body or None,
                    timeout_s=self.args.timeout_s,
                )
                admitted = True
                await ready_queue.put(self.uid)
                ready_reported = True
                await start_event.wait()
                formal_started_at = self.formal_epoch_s + self.phase_s
                stream_started_at = formal_started_at - self.context_age_units * UNIT_MS / 1000
                total_units = self.context_age_units + self.args.duration_s
                total_chunks = total_units * CHUNKS_PER_UNIT
                for chunk_index in range(total_chunks):
                    jitter_ms = self.jitter_rng.uniform(
                        -self.args.arrival_jitter_ms,
                        self.args.arrival_jitter_ms,
                    )
                    deadline = stream_started_at + chunk_index * CHUNK_MS / 1000 + jitter_ms / 1000
                    await asyncio.sleep(max(0.0, deadline - time.monotonic()))
                    chunk = self._media_chunk(chunk_index)
                    event: dict[str, object] = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "input_audio_format": "pcm16",
                        "sample_rate_hz": PCM16_SAMPLE_RATE,
                        "duration_ms": CHUNK_MS,
                        "audio_end_ms": (chunk_index + 1) * CHUNK_MS,
                    }
                    if chunk_index % CHUNKS_PER_UNIT == 0:
                        unit_index = chunk_index // CHUNKS_PER_UNIT
                        event["video_frames"] = [self._media_frame(unit_index)]
                        event["max_slice_nums"] = self.args.max_slice_nums
                        if unit_index < self.context_age_units:
                            precondition_frames_sent += 1
                        else:
                            formal_frames_sent += 1
                    await client.send(event)
                    sent_at = time.monotonic()
                    if chunk_index % CHUNKS_PER_UNIT == 0:
                        unit_index = chunk_index // CHUNKS_PER_UNIT
                        if unit_index >= self.context_age_units:
                            # The video-bearing append makes this model unit
                            # runnable; the remaining audio chunks continue to
                            # arrive while the unit is being processed.
                            unit_ready_at.append(sent_at)
                    if (chunk_index + 1) % CHUNKS_PER_UNIT == 0:
                        unit_index = chunk_index // CHUNKS_PER_UNIT
                        if unit_index < self.context_age_units:
                            precondition_units_sent += 1
                        else:
                            formal_units_sent += 1
                    if chunk_index >= self.context_age_units * CHUNKS_PER_UNIT:
                        send_drift_ms.append((sent_at - deadline) * 1000)
                        arrival_jitter_ms.append(jitter_ms)

                # Integer model units need no semantic turn commit.  A fixed
                # grace period observes tail progress without waiting for an
                # arbitrary input/output count equality: native duplex may
                # create speech-continuation units on its own.
                await asyncio.sleep(self.args.post_stream_s)
                measurement_done_at = time.monotonic()
                await measurement_done_queue.put(self.uid)
                measurement_done_reported = True
                try:
                    await client.close_session(timeout_s=self.args.close_timeout_s)
                except TimeoutError as exc:
                    teardown_errors.append(str(exc))
        except Exception as exc:  # keep all user failures in the audit artifact
            target = teardown_errors if measurement_done_reported else errors
            target.append(f"{type(exc).__name__}: {exc}")
        finally:
            if not ready_reported:
                await ready_queue.put(self.uid)
            if not measurement_done_reported:
                measurement_done_at = time.monotonic()
                await measurement_done_queue.put(self.uid)

        formal_event_not_before_s = unit_ready_at[0] if unit_ready_at else formal_started_at
        completion_at = _unit_completion_times(client, not_before_s=formal_event_not_before_s)
        progress_gap_ms = [
            (current - previous) * 1000 for previous, current in zip(completion_at, completion_at[1:])
        ]
        audio = _audio_cadence(client, not_before_s=formal_event_not_before_s)
        server_errors, teardown_server_errors = _partition_server_errors(
            client,
            measurement_done_at_s=measurement_done_at,
        )
        event_types = Counter(
            str(event.get("type"))
            for event, received_at in zip(
                client.events.events,
                client.events.event_received_at_s,
                strict=True,
            )
            if received_at >= formal_event_not_before_s
        )
        return {
            "uid": self.uid,
            "session_id": self.session_id,
            "phase_s": round(self.phase_s, 3),
            "context_age_units": self.context_age_units,
            "formal_seq_start": self.context_age_units + 1,
            "formal_seq_end": self.context_age_units + self.args.duration_s,
            "media": str(self.media.path.resolve()),
            "media_offset_units": self.media_offset_units,
            "admitted": admitted,
            "wall_s": round(time.monotonic() - started_at, 3),
            "precondition_units_sent": precondition_units_sent,
            "units_sent": formal_units_sent,
            "precondition_frames_sent": precondition_frames_sent,
            "frames_sent": formal_frames_sent,
            "progress_events": len(completion_at),
            "first_unit_to_first_progress_ms": (
                round((completion_at[0] - unit_ready_at[0]) * 1000, 2)
                if completion_at and unit_ready_at
                else None
            ),
            "progress_gap_ms": [round(value, 2) for value in progress_gap_ms],
            "progress_gap_summary_ms": _summary(progress_gap_ms),
            "progress_stalls_over_1200ms": sum(value > 1200 for value in progress_gap_ms),
            "last_input_to_last_progress_ms": (
                round((completion_at[-1] - unit_ready_at[-1]) * 1000, 2)
                if completion_at and unit_ready_at
                else None
            ),
            "send_drift_ms": _summary(send_drift_ms),
            "scheduled_arrival_jitter_ms": _summary(arrival_jitter_ms),
            "listen_units": event_types["response.listen"],
            "audio_units": event_types["response.audio.delta"],
            "responses": event_types["response.created"],
            "audio": audio,
            "server_errors": server_errors,
            "errors": errors,
            "teardown_server_errors": teardown_server_errors,
            "teardown_errors": teardown_errors,
        }


def _unit_completion_times(
    client: RealtimeDuplexClient,
    *,
    not_before_s: float = -math.inf,
) -> list[float]:
    return [
        received_at
        for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=True)
        if received_at >= not_before_s
        and event.get("type") in {"response.listen", "response.audio.delta"}
    ]


def _audio_cadence(
    client: RealtimeDuplexClient,
    *,
    not_before_s: float = -math.inf,
) -> dict[str, Any]:
    arrivals: list[float] = []
    durations_ms: list[float] = []
    previous_cumulative_ms: dict[str, float] = {}
    for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=True):
        if received_at < not_before_s or event.get("type") != "response.audio.delta":
            continue
        response_id = client.events.response_id(event) or "unknown"
        metadata = event.get("metadata")
        cumulative = metadata.get("audio_duration_ms") if isinstance(metadata, dict) else None
        if isinstance(cumulative, int | float):
            previous = previous_cumulative_ms.get(response_id, 0.0)
            duration = float(cumulative) - previous
            previous_cumulative_ms[response_id] = float(cumulative)
        else:
            delta = event.get("delta") or event.get("audio")
            duration = (
                len(base64.b64decode(delta)) / (2 * client.events.output_sample_rate_hz) * 1000
                if isinstance(delta, str)
                else 0.0
            )
        arrivals.append(received_at)
        durations_ms.append(max(0.0, duration))

    playback_slack_ms = [
        (arrivals[index] - arrivals[index - 1]) * 1000 - durations_ms[index - 1]
        for index in range(1, len(arrivals))
    ]
    return {
        "chunks": len(arrivals),
        "chunk_duration_ms": _summary(durations_ms),
        "playback_slack_samples_ms": [round(value, 2) for value in playback_slack_ms],
        "playback_slack_ms": _summary(playback_slack_ms),
        "underruns_with_200ms_buffer": sum(value > 200 for value in playback_slack_ms),
    }


def _partition_server_errors(
    client: RealtimeDuplexClient,
    *,
    measurement_done_at_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime: list[dict[str, Any]] = []
    teardown: list[dict[str, Any]] = []
    for event, received_at in zip(
        client.events.events,
        client.events.event_received_at_s,
        strict=True,
    ):
        if event.get("type") != "error":
            continue
        target = runtime if received_at <= measurement_done_at_s else teardown
        target.append(event)
    return runtime, teardown


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    media_assets = []
    for raw_path in args.media:
        path = Path(raw_path)
        pcm16, frames, duration_s = _load_media(
            path,
            frame_max_side=args.frame_max_side,
        )
        media_assets.append(MediaAsset(path, pcm16, frames, duration_s))
    ref_audio = _ref_audio_data_url(Path(args.ref_audio))
    rng = random.Random(args.seed)
    phases = [rng.random() * args.phase_window_s for _ in range(args.users)]
    if args.context_age_max_units > 0 and args.users > 1:
        context_ages = [
            round(uid * args.context_age_max_units / (args.users - 1))
            for uid in range(args.users)
        ]
        rng.shuffle(context_ages)
    else:
        context_ages = [0] * args.users

    if args.workload_profile == "synchronized":
        media_positions = [(0, 0)] * args.users
    else:
        # Balance sources first, then disperse aligned offsets within each
        # source.  Flattening every (source, offset) pair before sampling can
        # accidentally assign a small run to only one source.
        media_indices = [uid % len(media_assets) for uid in range(args.users)]
        rng.shuffle(media_indices)
        offsets_by_media: dict[int, list[int]] = {}
        for media_index, media in enumerate(media_assets):
            if not media.frames:
                raise ValueError(f"{media.path} has no complete one-second AV units")
            offsets = list(range(len(media.frames)))
            rng.shuffle(offsets)
            offsets_by_media[media_index] = offsets
        source_counts: Counter[int] = Counter()
        media_positions = []
        for media_index in media_indices:
            offsets = offsets_by_media[media_index]
            offset = offsets[source_counts[media_index] % len(offsets)]
            source_counts[media_index] += 1
            media_positions.append((media_index, offset))

    users = []
    for uid in range(args.users):
        media_index, media_offset = media_positions[uid]
        media = media_assets[media_index]
        needed_units = context_ages[uid] + args.duration_s
        if not args.loop_media and media_offset + needed_units > len(media.frames):
            raise ValueError(
                f"{media.path} lacks {needed_units} units after offset {media_offset}; "
                "use --loop-media or a longer source"
            )
        users.append(
            UserSession(
                uid,
                args,
                media,
                ref_audio,
                phases[uid],
                context_ages[uid],
                media_offset,
                f"minicpm-cap-{args.seed}-{args.users}-{uid}",
            )
        )

    ready_queue: asyncio.Queue[int] = asyncio.Queue()
    measurement_done_queue: asyncio.Queue[int] = asyncio.Queue()
    start_event = asyncio.Event()
    tasks = [
        asyncio.create_task(user.run(ready_queue, measurement_done_queue, start_event))
        for user in users
    ]
    for _ in users:
        await asyncio.wait_for(ready_queue.get(), timeout=args.admission_timeout_s)

    precondition_lead_s = max(context_ages, default=0) * UNIT_MS / 1000
    formal_epoch_s = time.monotonic() + precondition_lead_s + 1.0
    for user in users:
        user.formal_epoch_s = formal_epoch_s
    start_event.set()
    await asyncio.sleep(max(0.0, formal_epoch_s - time.monotonic()))

    gpu_sampler = GPUSampler(args.gpus)
    gpu_task = asyncio.create_task(gpu_sampler.run())
    run_started_epoch_s = time.time()
    wall_started = time.monotonic()
    for _ in users:
        await measurement_done_queue.get()
    measurement_ended_epoch_s = time.time()
    measurement_wall_s = time.monotonic() - wall_started
    gpu_sampler.stop()
    await gpu_task
    results = await asyncio.gather(*tasks)
    teardown_ended_epoch_s = time.time()

    progress_gaps = [value for result in results for value in result["progress_gap_ms"]]
    playback_slack = [
        value for result in results for value in result["audio"]["playback_slack_samples_ms"]
    ]
    summary = {
        "config": {
            "users": args.users,
            "duration_s": args.duration_s,
            "workload_profile": args.workload_profile,
            "phase_window_s": args.phase_window_s,
            "arrival_jitter_ms": args.arrival_jitter_ms,
            "context_age_max_units": args.context_age_max_units,
            "seed": args.seed,
            "media": [str(media.path.resolve()) for media in media_assets],
            "media_duration_s": [round(media.duration_s, 3) for media in media_assets],
            "loop_media": args.loop_media,
            "audio_chunk_ms": CHUNK_MS,
            "video_fps": 1,
            "frame_max_side": args.frame_max_side,
            "max_slice_nums": args.max_slice_nums,
            "context_window_trigger_tokens": args.context_window_trigger_tokens or 36_000,
            "force_listen_count": args.force_listen_count,
            "close_timeout_s": args.close_timeout_s,
            "progress_stall_threshold_ms": 1200,
            "audio_startup_buffer_ms": 200,
            "frame_audit_required": True,
            "gpu_measurement_window": "formal input start through post-stream drain; teardown excluded",
        },
        "wall_s": round(measurement_wall_s, 3),
        "teardown_wall_s": round(teardown_ended_epoch_s - measurement_ended_epoch_s, 3),
        "started_epoch_s": run_started_epoch_s,
        "ended_epoch_s": measurement_ended_epoch_s,
        "teardown_ended_epoch_s": teardown_ended_epoch_s,
        "admitted": sum(result["admitted"] for result in results),
        "failed_users": sum(bool(result["errors"] or result["server_errors"]) for result in results),
        "teardown_failed_users": sum(
            bool(result["teardown_errors"] or result["teardown_server_errors"])
            for result in results
        ),
        "units_sent": sum(result["units_sent"] for result in results),
        "precondition_units_sent": sum(result["precondition_units_sent"] for result in results),
        "frames_sent": sum(result["frames_sent"] for result in results),
        "precondition_frames_sent": sum(result["precondition_frames_sent"] for result in results),
        "progress_events": sum(result["progress_events"] for result in results),
        "progress_gap_ms": _summary(progress_gaps),
        "progress_stalls_over_1200ms": sum(result["progress_stalls_over_1200ms"] for result in results),
        "sessions_with_progress_stall": sum(result["progress_stalls_over_1200ms"] > 0 for result in results),
        "audio_chunks": sum(result["audio"]["chunks"] for result in results),
        "audio_playback_slack_ms": _summary(playback_slack),
        "audio_underruns_with_200ms_buffer": sum(
            result["audio"]["underruns_with_200ms_buffer"] for result in results
        ),
        "gpu": gpu_sampler.report(),
        "users": results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8113/v1/realtime")
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--duration-s", type=int, default=12)
    parser.add_argument("--phase-window-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--close-timeout-s",
        type=float,
        default=30.0,
        help="bound teardown independently from the runtime request timeout",
    )
    parser.add_argument("--post-stream-s", type=float, default=3.0)
    parser.add_argument("--connect-stagger-s", type=float, default=0.5)
    parser.add_argument("--admission-timeout-s", type=float, default=90.0)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--media", nargs="+", required=True)
    parser.add_argument(
        "--workload-profile",
        choices=("production", "synchronized"),
        default="production",
        help="production disperses media offsets/context ages and adds bounded jitter; synchronized is a control",
    )
    parser.add_argument(
        "--context-age-max-units",
        type=int,
        help="maximum real AV units streamed before formal measurement (production default: 154 for long runs)",
    )
    parser.add_argument(
        "--arrival-jitter-ms",
        type=float,
        help="independent bounded jitter around each 200 ms send deadline (production default: 50 ms)",
    )
    parser.add_argument(
        "--loop-media",
        action="store_true",
        help="repeat the source AV only for long-session context rollover tests",
    )
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--frame-max-side", type=int, default=448)
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--context-window-trigger-tokens", type=int)
    parser.add_argument(
        "--force-listen-count",
        type=int,
        help="diagnostic control: force this many model units to stop at the listen decision",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.context_age_max_units is None:
        args.context_age_max_units = (
            154 if args.workload_profile == "production" and args.duration_s >= 180 else 0
        )
    if args.arrival_jitter_ms is None:
        args.arrival_jitter_ms = 50.0 if args.workload_profile == "production" else 0.0
    if args.workload_profile == "synchronized":
        args.context_age_max_units = 0
        args.arrival_jitter_ms = 0.0
    if args.users <= 0 or args.duration_s <= 0:
        parser.error("--users and --duration-s must be positive")
    if args.close_timeout_s <= 0:
        parser.error("--close-timeout-s must be positive")
    if args.context_age_max_units < 0:
        parser.error("--context-age-max-units must be non-negative")
    if not 0 <= args.arrival_jitter_ms < CHUNK_MS / 2:
        parser.error(f"--arrival-jitter-ms must be in [0, {CHUNK_MS / 2})")
    if args.frame_max_side < 0:
        parser.error("--frame-max-side must be non-negative (0 keeps source resolution)")
    if not 1 <= args.max_slice_nums <= 9:
        parser.error("--max-slice-nums must be in [1, 9]")
    if args.context_window_trigger_tokens is not None and args.context_window_trigger_tokens < 1024:
        parser.error("--context-window-trigger-tokens must be >= 1024")
    if args.force_listen_count is not None and args.force_listen_count < 0:
        parser.error("--force-listen-count must be non-negative")

    result = asyncio.run(_main(args))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    concise = {key: value for key, value in result.items() if key != "users"}
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
