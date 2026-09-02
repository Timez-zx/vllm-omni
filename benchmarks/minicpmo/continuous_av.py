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

from benchmarks.minicpmo.clean_server import collect_provenance  # noqa: E402
from vllm_omni.experimental.fullduplex.client import (  # noqa: E402
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    _event_stage_metrics,
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
        first_media_arrival_at: list[float] = []
        first_media_arrival_epoch_s: list[float] = []
        model_unit_ready_at: list[float] = []
        model_unit_ready_epoch_s: list[float] = []
        send_drift_ms: list[float] = []
        arrival_jitter_ms: list[float] = []
        admitted = False
        errors: list[str] = []
        teardown_errors: list[str] = []
        client = RealtimeDuplexClient(url, open_timeout_s=self.args.timeout_s)
        started_at = time.monotonic()
        formal_started_at = math.inf
        stream_finished_at = math.inf
        measurement_done_at = math.inf
        ready_reported = False
        measurement_done_reported = False
        precondition_units_sent = 0
        formal_units_sent = 0
        precondition_frames_sent = 0
        formal_frames_sent = 0
        drain_exit_reason = "input_failed"
        try:
            async with client:
                # Formal capacity runs need one fixed-size, physical D-request
                # completion record per model unit.  Full ``stage_metrics``
                # contain cumulative ITL arrays and would otherwise be copied
                # into every realtime output event, making client/IPC work
                # grow quadratically with session length.
                extra_body: dict[str, object] = {
                    "return_stage_metrics": False,
                    "return_completion_witness": True,
                }
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
                    unit_index = chunk_index // CHUNKS_PER_UNIT
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
                        # Stable client identity for this real one-second
                        # input.  Physical engine sequence numbers may also
                        # contain model-generated silence continuations and
                        # therefore cannot identify formal input by themselves.
                        "input_unit_index": unit_index + 1,
                    }
                    if chunk_index % CHUNKS_PER_UNIT == 0:
                        event["video_frames"] = [self._media_frame(unit_index)]
                        event["max_slice_nums"] = self.args.max_slice_nums
                        if unit_index < self.context_age_units:
                            precondition_frames_sent += 1
                        else:
                            formal_frames_sent += 1
                    await client.send(event)
                    sent_at = time.monotonic()
                    sent_epoch_s = time.time()
                    if chunk_index % CHUNKS_PER_UNIT == 0:
                        if unit_index >= self.context_age_units:
                            # User-visible input-start: the first 200 ms chunk
                            # for this unit also carries its camera frame.  The
                            # model unit is not runnable until the fifth audio
                            # chunk below completes the server-side 1 s buffer.
                            first_media_arrival_at.append(sent_at)
                            first_media_arrival_epoch_s.append(sent_epoch_s)
                    if (chunk_index + 1) % CHUNKS_PER_UNIT == 0:
                        if unit_index < self.context_age_units:
                            precondition_units_sent += 1
                        else:
                            formal_units_sent += 1
                            # Engine-work origin: after this fifth 200 ms
                            # chunk has been sent, the server has the complete
                            # one-second audio unit and may submit its P append.
                            model_unit_ready_at.append(sent_at)
                            model_unit_ready_epoch_s.append(sent_epoch_s)
                    if chunk_index >= self.context_age_units * CHUNKS_PER_UNIT:
                        send_drift_ms.append((sent_at - deadline) * 1000)
                        arrival_jitter_ms.append(jitter_ms)

                # Integer model units need no semantic turn commit. Drain on
                # exact real-input identities carried by the physical-D
                # completion witness; physical sequence also counts autonomous
                # continuations, and protocol speech may emit several events.
                stream_finished_at = time.monotonic()
                expected_input_unit_indices = set(
                    range(
                        self.context_age_units + 1,
                        self.context_age_units + self.args.duration_s + 1,
                    )
                )
                observed_input_unit_indices: set[int] = set()
                event_cursor = 0
                drain_deadline = stream_finished_at + self.args.post_stream_s
                while True:
                    current_events = client.events.events
                    for event in current_events[event_cursor:]:
                        identity = _pd_stage1_identity(event)
                        input_unit_index = _real_input_unit_index(identity)
                        if input_unit_index in expected_input_unit_indices:
                            observed_input_unit_indices.add(input_unit_index)
                    event_cursor = len(current_events)
                    if observed_input_unit_indices == expected_input_unit_indices:
                        drain_exit_reason = "exact_pd_terminal_set_complete"
                        break
                    remaining = drain_deadline - time.monotonic()
                    if remaining <= 0:
                        drain_exit_reason = "post_stream_timeout"
                        break
                    await asyncio.sleep(min(0.05, remaining))
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

        formal_event_not_before_s = (
            first_media_arrival_at[0]
            if first_media_arrival_at
            else formal_started_at
        )
        completion_at = _unit_completion_times(
            client,
            not_before_s=formal_event_not_before_s,
            not_after_s=measurement_done_at,
        )
        progress_gap_ms = [
            (current - previous) * 1000 for previous, current in zip(completion_at, completion_at[1:])
        ]
        audio = _audio_cadence(
            client,
            not_before_s=formal_event_not_before_s,
            not_after_s=measurement_done_at,
        )
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
            if formal_event_not_before_s <= received_at <= measurement_done_at
        )
        pd_completions = _pd_completion_records(
            client,
            sequence_start=self.context_age_units + 1,
            sequence_end=self.context_age_units + self.args.duration_s,
            unit_ready_at=first_media_arrival_at,
            unit_ready_epoch_s=first_media_arrival_epoch_s,
            model_unit_ready_at=model_unit_ready_at,
            model_unit_ready_epoch_s=model_unit_ready_epoch_s,
            not_after_s=measurement_done_at,
        )
        completed_input_unit_indices = {
            int(item["input_unit_index"]) for item in pd_completions
        }
        input_unit_counts = Counter(
            int(item["input_unit_index"]) for item in pd_completions
        )
        duplicate_input_unit_indices = sorted(
            unit_index
            for unit_index, count in input_unit_counts.items()
            if count != 1
        )
        expected_input_unit_indices = set(
            range(
                self.context_age_units + 1,
                self.context_age_units + self.args.duration_s + 1,
            )
        )
        return {
            "uid": self.uid,
            "session_id": self.session_id,
            "phase_s": round(self.phase_s, 6),
            "context_age_units": self.context_age_units,
            # Compatibility aliases. These are real input-unit indices, not
            # necessarily physical D sequences.
            "formal_seq_start": self.context_age_units + 1,
            "formal_seq_end": self.context_age_units + self.args.duration_s,
            "formal_input_unit_start": self.context_age_units + 1,
            "formal_input_unit_end": self.context_age_units + self.args.duration_s,
            "media": str(self.media.path.resolve()),
            "media_offset_units": self.media_offset_units,
            "admitted": admitted,
            "wall_s": round(time.monotonic() - started_at, 3),
            "precondition_units_sent": precondition_units_sent,
            "units_sent": formal_units_sent,
            "precondition_frames_sent": precondition_frames_sent,
            "frames_sent": formal_frames_sent,
            "input_unit_timings": [
                {
                    "sequence": self.context_age_units + index + 1,
                    "input_unit_index": self.context_age_units + index + 1,
                    "first_media_arrival_at_s": round(first_epoch, 6),
                    "model_unit_ready_at_s": round(ready_epoch, 6),
                    "input_aggregation_ms": round(
                        max(model_unit_ready_at[index] - first_media_arrival_at[index], 0.0)
                        * 1000.0,
                        3,
                    ),
                }
                for index, (first_epoch, ready_epoch) in enumerate(
                    zip(
                        first_media_arrival_epoch_s,
                        model_unit_ready_epoch_s,
                    )
                )
            ],
            "input_stream_complete": (
                formal_units_sent == self.args.duration_s
                and formal_frames_sent == self.args.duration_s
                and not errors
            ),
            "drain_observed_s": (
                round(max(measurement_done_at - stream_finished_at, 0.0), 3)
                if math.isfinite(stream_finished_at)
                else 0.0
            ),
            "drain_exit_reason": drain_exit_reason,
            "pd_completion_witness": {
                "source": "client-visible physical D completion witness",
                "expected": len(expected_input_unit_indices),
                "completed": len(pd_completions),
                "complete": (
                    completed_input_unit_indices == expected_input_unit_indices
                    and not duplicate_input_unit_indices
                    and len(pd_completions) == len(expected_input_unit_indices)
                ),
                "missing_input_unit_indices": sorted(
                    expected_input_unit_indices - completed_input_unit_indices
                ),
                "duplicate_input_unit_indices": duplicate_input_unit_indices,
                # Compatibility alias for older artifact readers.  This is an
                # input-unit identity, not necessarily the physical D seq.
                "missing_sequences": sorted(
                    expected_input_unit_indices - completed_input_unit_indices
                ),
                "input_video_frames": sum(
                    int(item.get("input_video_frames") or 0) for item in pd_completions
                ),
                "arrival_video_frames": sum(
                    int(item.get("arrival_video_frames") or 0) for item in pd_completions
                ),
                "vision_fallback_frames": sum(
                    int(item.get("vision_fallback_frames") or 0) for item in pd_completions
                ),
                "arrival_audio_units": sum(
                    int(item.get("arrival_audio_units") or 0) for item in pd_completions
                ),
                "audio_fallback_units": sum(
                    int(item.get("audio_fallback_units") or 0) for item in pd_completions
                ),
                "records": pd_completions,
            },
            "progress_events": len(completion_at),
            "first_unit_to_first_progress_ms": (
                round((completion_at[0] - first_media_arrival_at[0]) * 1000, 2)
                if completion_at and first_media_arrival_at
                else None
            ),
            "progress_gap_ms": [round(value, 2) for value in progress_gap_ms],
            "progress_gap_summary_ms": _summary(progress_gap_ms),
            "progress_stalls_over_1200ms": sum(value > 1200 for value in progress_gap_ms),
            "last_input_to_last_progress_ms": (
                round((completion_at[-1] - first_media_arrival_at[-1]) * 1000, 2)
                if completion_at and first_media_arrival_at
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
    not_after_s: float = math.inf,
) -> list[float]:
    return [
        received_at
        for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=True)
        if not_before_s <= received_at <= not_after_s
        and event.get("type") in {"response.listen", "response.audio.delta"}
    ]


def _pd_completion_records(
    client: RealtimeDuplexClient,
    *,
    sequence_start: int,
    sequence_end: int,
    unit_ready_at: list[float],
    unit_ready_epoch_s: list[float] | None = None,
    model_unit_ready_at: list[float] | None = None,
    model_unit_ready_epoch_s: list[float] | None = None,
    not_after_s: float = math.inf,
) -> list[dict[str, Any]]:
    """Recover exact finite-D completions with two input-time origins.

    ``unit_ready_at`` is retained as the legacy argument name, but represents
    the first-media/input-start timestamp.  New callers also provide the time
    when the fifth 200 ms audio chunk made the full one-second model unit
    runnable.  Missing model-ready timestamps leave the new ready-to-D fields
    absent so old callers remain readable without inventing a second origin.
    """
    by_physical_request: dict[str, dict[str, Any]] = {}
    for event, received_at in zip(
        client.events.events,
        client.events.event_received_at_s,
        strict=True,
    ):
        if received_at > not_after_s:
            continue
        identity = _pd_stage1_identity(event)
        if identity is None:
            continue
        sequence, request_id, stage1 = identity
        input_unit_index = _real_input_unit_index(identity)
        if input_unit_index is None or not sequence_start <= input_unit_index <= sequence_end:
            continue
        # Full stage metrics may be projected onto several protocol events.
        # Deduplicate only the exact physical request; two different physical
        # requests claiming one real input must remain visible so the caller's
        # exactly-once check can fail.
        if request_id in by_physical_request:
            continue
        ready_index = input_unit_index - sequence_start
        if not 0 <= ready_index < len(unit_ready_at):
            continue
        # ``num_tokens_in`` is accumulated across stage events in a logical
        # response.  The dedicated field is the prompt length of this exact
        # finite D engine request and must remain non-additive.
        prompt_tokens = int(stage1.get("engine_prompt_tokens", -1))
        cached_tokens = int(stage1.get("num_cached_tokens", -1))
        local_cached_tokens = int(stage1.get("local_cached_tokens", -1))
        external_cached_tokens = int(stage1.get("external_cached_tokens", -1))
        computed_tokens = int(stage1.get("computed_tokens", -1))
        stage_done_epoch = float(stage1.get("completed_epoch_s") or 0.0)
        has_wall_clock_completion = bool(
            stage_done_epoch > 0
            and unit_ready_epoch_s is not None
            and ready_index < len(unit_ready_epoch_s)
        )
        first_media_arrival_at = (
            unit_ready_epoch_s[ready_index]
            if has_wall_clock_completion and unit_ready_epoch_s is not None
            else unit_ready_at[ready_index]
        )
        if (
            has_wall_clock_completion
            and model_unit_ready_epoch_s is not None
            and ready_index < len(model_unit_ready_epoch_s)
        ):
            model_ready_at: float | None = model_unit_ready_epoch_s[ready_index]
        elif (
            not has_wall_clock_completion
            and model_unit_ready_at is not None
            and ready_index < len(model_unit_ready_at)
        ):
            model_ready_at = model_unit_ready_at[ready_index]
        else:
            model_ready_at = None
        done_at = stage_done_epoch if has_wall_clock_completion else received_at
        input_start_e2e_ms = round(
            max(done_at - first_media_arrival_at, 0.0) * 1000.0,
            3,
        )
        record = {
            "sequence": sequence,
            "input_unit_index": input_unit_index,
            "source": "real_input",
            "request_id": request_id,
            # ``ready_at_s`` and ``e2e_ms`` are retained for artifact
            # compatibility.  Their origin is now explicit: first media for
            # the unit, not completion of its one-second input buffer.
            "ready_at_s": first_media_arrival_at,
            "first_media_arrival_at_s": first_media_arrival_at,
            "done_at_s": done_at,
            "e2e_ms": input_start_e2e_ms,
            "input_start_e2e_ms": input_start_e2e_ms,
            "completion_clock": (
                "engine_stage_epoch" if has_wall_clock_completion else "client_receive_monotonic"
            ),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "local_cached_tokens": local_cached_tokens,
            "external_cached_tokens": external_cached_tokens,
            "computed_tokens": computed_tokens,
            "uncached_suffix_tokens": (
                prompt_tokens - cached_tokens
                if prompt_tokens >= 0 and cached_tokens >= 0
                else -1
            ),
            "d_service_ms": round(float(stage1.get("stage_gen_time_ms") or 0.0), 3),
            "batch_id": int(stage1.get("batch_id") or 0),
            "input_video_frames": int(stage1.get("input_video_frames") or 0),
            "arrival_video_frames": int(stage1.get("arrival_video_frames") or 0),
            "vision_fallback_frames": int(stage1.get("vision_fallback_frames") or 0),
            "arrival_audio_units": int(stage1.get("arrival_audio_units") or 0),
            "audio_fallback_units": int(stage1.get("audio_fallback_units") or 0),
            # These request-scoped connector scalars are copied verbatim.  In
            # particular, an unavailable write-submit clock remains ``-1``;
            # the client must not infer it from unrelated wall clocks.
            "kv_transfer_selected_blocks": stage1.get(
                "kv_transfer_selected_blocks", -1
            ),
            "kv_transfer_selected_tokens": stage1.get(
                "kv_transfer_selected_tokens", -1
            ),
            "kv_transfer_selected_bytes": stage1.get(
                "kv_transfer_selected_bytes", -1
            ),
            "kv_transfer_write_submit_to_d_ready_ms": stage1.get(
                "kv_transfer_write_submit_to_d_ready_ms", -1.0
            ),
        }
        if model_ready_at is not None:
            record.update(
                {
                    "model_unit_ready_at_s": model_ready_at,
                    "ready_to_d_ms": round(
                        max(done_at - model_ready_at, 0.0) * 1000.0,
                        3,
                    ),
                    "input_aggregation_ms": round(
                        max(model_ready_at - first_media_arrival_at, 0.0) * 1000.0,
                        3,
                    ),
                }
            )
        by_physical_request[request_id] = record
    return sorted(
        by_physical_request.values(),
        key=lambda record: (
            int(record["input_unit_index"]),
            int(record["sequence"]),
        ),
    )


def _real_input_unit_index(
    identity: tuple[int, str, dict[str, Any]] | None,
) -> int | None:
    """Return the stable real-input identity, excluding auto continuations."""
    if identity is None:
        return None
    sequence, _, stage1 = identity
    source = stage1.get("source")
    if source == "auto_continuation":
        return None
    if source not in (None, "real_input"):
        return None
    value = stage1.get("input_unit_index")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    # Old diagnostic artifacts predate the explicit input identity and did
    # not interleave continuation slots during their formal ranges.
    return sequence


def _pd_stage1_identity(
    event: dict[str, Any],
) -> tuple[int, str, dict[str, Any]] | None:
    witness = _event_completion_witness(event)
    if isinstance(witness, dict) and int(witness.get("stage_id") or -1) == 1:
        request_id = witness.get("engine_request_id")
        if isinstance(request_id, str):
            stage1 = {
                "engine_request_id": request_id,
                "engine_prompt_tokens": witness.get("prompt_tokens", -1),
                "num_cached_tokens": witness.get("cached_tokens", -1),
                "local_cached_tokens": witness.get("local_cached_tokens", -1),
                "external_cached_tokens": witness.get(
                    "external_cached_tokens", -1
                ),
                "computed_tokens": witness.get("computed_tokens", -1),
                "input_unit_index": witness.get("input_unit_index"),
                "source": witness.get("source"),
                "batch_id": witness.get("batch_id", 0),
                "submit_epoch_s": witness.get("submit_epoch_s", 0.0),
                "completed_epoch_s": witness.get("completed_epoch_s", 0.0),
                "stage_gen_time_ms": witness.get("service_ms", 0.0),
                "input_video_frames": witness.get("input_video_frames", 0),
                "arrival_video_frames": witness.get("arrival_video_frames", 0),
                "vision_fallback_frames": witness.get("vision_fallback_frames", 0),
                "arrival_audio_units": witness.get("arrival_audio_units", 0),
                "audio_fallback_units": witness.get("audio_fallback_units", 0),
                "kv_transfer_selected_blocks": witness.get(
                    "kv_transfer_selected_blocks", -1
                ),
                "kv_transfer_selected_tokens": witness.get(
                    "kv_transfer_selected_tokens", -1
                ),
                "kv_transfer_selected_bytes": witness.get(
                    "kv_transfer_selected_bytes", -1
                ),
                "kv_transfer_write_submit_to_d_ready_ms": witness.get(
                    "kv_transfer_write_submit_to_d_ready_ms", -1.0
                ),
            }
            sequence = _physical_d_sequence(request_id)
            declared_sequence = witness.get("physical_sequence")
            if (
                sequence is not None
                and (
                    declared_sequence is None
                    or (
                        isinstance(declared_sequence, int)
                        and not isinstance(declared_sequence, bool)
                        and declared_sequence == sequence
                    )
                )
            ):
                return sequence, request_id, stage1

    # Compatibility for diagnostic runs and old artifacts which explicitly
    # requested full response-level stage metrics.
    stage_metrics = _event_stage_metrics(event)
    stage1 = stage_metrics.get("1") if isinstance(stage_metrics, dict) else None
    if not isinstance(stage1, dict):
        return None
    request_id = stage1.get("engine_request_id") or stage1.get("request_id")
    if not isinstance(request_id, str):
        return None
    sequence = _physical_d_sequence(request_id)
    if sequence is None:
        return None
    return sequence, request_id, stage1


def _physical_d_sequence(request_id: str) -> int | None:
    _, separator, raw_sequence = request_id.rpartition("-")
    if not separator or len(raw_sequence) != 8:
        return None
    try:
        return int(raw_sequence, 16)
    except ValueError:
        return None


def _event_completion_witness(event: dict[str, Any]) -> dict[str, Any] | None:
    """Find the fixed-size physical completion record on any protocol shape."""
    candidates: list[object] = [event.get("vllm_omni")]
    # Compatibility with servers predating the explicit Realtime projection,
    # which wrapped this extension as ``duplex.response.model_unit.done``.
    nested_event = event.get("event")
    if isinstance(nested_event, dict):
        candidates.append(nested_event.get("vllm_omni"))
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata, metadata.get("vllm_omni")))
    response = event.get("response")
    if isinstance(response, dict):
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            candidates.extend((response_metadata, response_metadata.get("vllm_omni")))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        witness = candidate.get("completion_witness")
        if isinstance(witness, dict):
            return witness
    return None


def _audio_cadence(
    client: RealtimeDuplexClient,
    *,
    not_before_s: float = -math.inf,
    not_after_s: float = math.inf,
) -> dict[str, Any]:
    arrivals: list[float] = []
    durations_ms: list[float] = []
    previous_cumulative_ms: dict[str, float] = {}
    for event, received_at in zip(client.events.events, client.events.event_received_at_s, strict=True):
        if (
            not not_before_s <= received_at <= not_after_s
            or event.get("type") != "response.audio.delta"
        ):
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
    client_provenance = collect_provenance(REPO_ROOT, formal=True)
    formal_capacity_candidate = bool(
        args.duration_s >= 180
        and args.workload_profile == "production"
        and args.phase_window_s > 0
        and args.force_listen_count is None
        and not args.require_frame_audit
    )
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
    gpu_report = gpu_sampler.report()
    gpu_sampling_complete = all(
        isinstance(gpu_report.get(str(gpu_id)), dict)
        and int(gpu_report[str(gpu_id)].get("sample_count") or 0) > 0
        for gpu_id in args.gpus
    )

    progress_gaps = [value for result in results for value in result["progress_gap_ms"]]
    playback_slack = [
        value for result in results for value in result["audio"]["playback_slack_samples_ms"]
    ]
    pd_expected = sum(result["pd_completion_witness"]["expected"] for result in results)
    pd_completed = sum(result["pd_completion_witness"]["completed"] for result in results)
    witnessed_video_frames = sum(
        result["pd_completion_witness"]["input_video_frames"] for result in results
    )
    arrival_video_frames = sum(
        result["pd_completion_witness"]["arrival_video_frames"] for result in results
    )
    vision_fallback_frames = sum(
        result["pd_completion_witness"]["vision_fallback_frames"] for result in results
    )
    arrival_audio_units = sum(
        result["pd_completion_witness"]["arrival_audio_units"] for result in results
    )
    audio_fallback_units = sum(
        result["pd_completion_witness"]["audio_fallback_units"] for result in results
    )
    summary = {
        "config": {
            "measurement_classification": (
                "formal_capacity_candidate"
                if formal_capacity_candidate
                else "development_or_diagnostic"
            ),
            "formal_capacity_candidate": formal_capacity_candidate,
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
            "post_stream_s": args.post_stream_s,
            "progress_stall_threshold_ms": 1200,
            "audio_startup_buffer_ms": 200,
            # Fixed-size physical-D witnesses always make frame conservation a
            # formal validity requirement. This flag only requests the older,
            # perturbing server-trace audit for a diagnostic rerun.
            "frame_audit_required": True,
            # Every formal one-second input unit must use the arrival audio
            # sidecar; request-local audio reconstruction invalidates a run.
            "audio_sidecar_audit_required": True,
            "server_trace_frame_audit_requested": args.require_frame_audit,
            "gpu_measurement_window": "formal input start through post-stream drain; teardown excluded",
            "unit_timing_origins": {
                "first_media_arrival_at_s": (
                    "client finished sending the first video-bearing 200 ms chunk"
                ),
                "model_unit_ready_at_s": (
                    "client finished sending the fifth 200 ms chunk for the 1 s unit"
                ),
                "e2e_ms": "first media arrival to physical D completion",
                "ready_to_d_ms": "complete 1 s model unit to physical D completion",
            },
            "measurement_cutoff": (
                "all expected real input-unit identities have a physical-D "
                "completion witness, "
                "or post-stream timeout; analyze_rtf verifies the exact set"
            ),
        },
        "client_provenance": client_provenance,
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
        "input_stream_complete": all(result["input_stream_complete"] for result in results),
        "drain_observed_s": _summary([result["drain_observed_s"] for result in results]),
        "drain_exit_reasons": dict(
            Counter(result["drain_exit_reason"] for result in results)
        ),
        "pd_completion_witness": {
            "source": "client-visible physical D completion witness",
            "expected": pd_expected,
            "completed": pd_completed,
            "complete": bool(results)
            and all(result["pd_completion_witness"]["complete"] for result in results),
            "input_video_frames": witnessed_video_frames,
            "arrival_video_frames": arrival_video_frames,
            "vision_fallback_frames": vision_fallback_frames,
            "arrival_audio_units": arrival_audio_units,
            "audio_fallback_units": audio_fallback_units,
        },
        "progress_events": sum(result["progress_events"] for result in results),
        "progress_gap_ms": _summary(progress_gaps),
        "progress_stalls_over_1200ms": sum(result["progress_stalls_over_1200ms"] for result in results),
        "sessions_with_progress_stall": sum(result["progress_stalls_over_1200ms"] > 0 for result in results),
        "audio_chunks": sum(result["audio"]["chunks"] for result in results),
        "audio_playback_slack_ms": _summary(playback_slack),
        "audio_underruns_with_200ms_buffer": sum(
            result["audio"]["underruns_with_200ms_buffer"] for result in results
        ),
        "gpu_sampling_complete": gpu_sampling_complete,
        "gpu": gpu_report,
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
    parser.add_argument(
        "--post-stream-s",
        type=float,
        default=30.0,
        help=(
            "maximum exact input-unit drain time after the final input; timeout "
            "makes the measurement incomplete"
        ),
    )
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
    parser.add_argument(
        "--require-frame-audit",
        action="store_true",
        help=(
            "diagnostic rerun only: require per-frame server trace coverage; "
            "the formal server intentionally disables the tracing needed for it"
        ),
    )
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
        args.phase_window_s = 0.0
        args.context_age_max_units = 0
        args.arrival_jitter_ms = 0.0
    if args.users <= 0 or args.duration_s <= 0:
        parser.error("--users and --duration-s must be positive")
    if args.phase_window_s < 0:
        parser.error("--phase-window-s must be non-negative")
    if args.close_timeout_s <= 0:
        parser.error("--close-timeout-s must be positive")
    if args.post_stream_s < 0:
        parser.error("--post-stream-s must be non-negative")
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
