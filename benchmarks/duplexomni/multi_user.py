#!/usr/bin/env python3
"""Run wall-clock-paced multi-user DuplexOmni WebSocket sessions."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import numpy as np
import websockets
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from single_user import (  # noqa: E402
    CONTROL_KEYS,
    DEFAULT_SYSTEM_PROMPT,
    SAMPLE_RATE,
    SLOT_MS,
    SLOT_SAMPLES,
    _extract_response,
    _image_bytes,
    _load_audio,
    _parse_controls,
    _slot_has_video,
    _wav_stats,
)


@dataclass(frozen=True)
class SessionSpec:
    user: int
    session_id: str
    phase_ms: float
    prefill_only: bool = False


def _session_specs(
    users: int,
    seed: int,
    prefix: str,
    prefill_only_users: int = 0,
) -> list[SessionSpec]:
    """Assign each session one reproducible phase in [0, 480 ms)."""
    rng = random.Random(seed)
    probes = [
        SessionSpec(user=user, session_id=f"{prefix}-{user:03d}", phase_ms=rng.random() * SLOT_MS)
        for user in range(users)
    ]
    backgrounds = [
        SessionSpec(
            user=users + user,
            session_id=f"{prefix}-prefill-{user:03d}",
            phase_ms=rng.random() * SLOT_MS,
            prefill_only=True,
        )
        for user in range(prefill_only_users)
    ]
    return probes + backgrounds


def _scheduled_ms(spec: SessionSpec, slot: int) -> float:
    return spec.phase_ms + slot * SLOT_MS


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    e2e = [float(record["e2e_slot_latency_ms"]) for record in records]
    request = [float(record["request_latency_ms"]) for record in records]
    queue = [float(record["app_queue_ms"]) for record in records]
    thinker = [float(record["thinker_latency_ms"]) for record in records if "thinker_latency_ms" in record]
    misses = sum(bool(record["deadline_miss"]) for record in records)
    invalid_controls = sum(not bool(record.get("control_valid", True)) for record in records)

    def distribution(values: list[float]) -> dict[str, float | None]:
        return {
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
            "p99_ms": _percentile(values, 99),
            "max_ms": max(values) if values else None,
        }

    return {
        "e2e_slot_latency": distribution(e2e),
        "request_latency": distribution(request),
        "app_queue": distribution(queue),
        "thinker_latency": distribution(thinker),
        "deadline_ms": SLOT_MS,
        "deadline_misses": misses,
        "deadline_miss_rate": misses / len(records) if records else None,
        "invalid_controls": invalid_controls,
        "invalid_control_rate": invalid_controls / len(records) if records else None,
        "slots_submitted_before_predecessor_audio": sum(
            bool(record.get("submitted_before_predecessor_audio")) for record in records
        ),
    }


def _audio_pcm_slot(samples: np.ndarray, slot_index: int) -> bytes:
    start = slot_index * SLOT_SAMPLES
    chunk = samples[start : start + SLOT_SAMPLES]
    if chunk.size < SLOT_SAMPLES:
        chunk = np.pad(chunk, (0, SLOT_SAMPLES - chunk.size))
    return np.asarray(chunk, dtype="<i2").tobytes()


def _personalize_audio_pcm(pcm: bytes, user: int) -> bytes:
    """Prevent unrealistic cross-user media-cache hits with inaudible dither."""
    samples = np.frombuffer(pcm, dtype="<i2").copy()
    if not samples.size:
        return pcm
    stride = 251
    indices = np.arange((user * 17) % stride, samples.size, stride)
    values = samples[indices].astype(np.int32) + user % 7 + 1
    samples[indices] = np.clip(values, -32768, 32767).astype(np.int16)
    return samples.astype("<i2", copy=False).tobytes()


def _personalize_image_jpeg(image_jpeg: bytes, user: int) -> bytes:
    """Give each session a distinct decoded frame while preserving semantics."""
    with Image.open(io.BytesIO(image_jpeg)) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    color = ((37 * (user + 1)) % 256, (83 * (user + 1)) % 256, (149 * (user + 1)) % 256)
    draw.rectangle((image.width - 5, image.height - 5, image.width - 1, image.height - 1), fill=color)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _prepared_inputs(
    args: argparse.Namespace,
    source_audio: np.ndarray,
    user_variant: int | None = None,
) -> list[tuple[bytes, bytes | None]]:
    prepared: list[tuple[bytes, bytes | None]] = []
    for slot in range(args.slots):
        audio_pcm = _audio_pcm_slot(source_audio, slot)
        image_jpeg = (
            _image_bytes(args.image, slot)
            if _slot_has_video(slot, args.audio_only, args.video_frame_interval)
            else None
        )
        if user_variant is not None:
            audio_pcm = _personalize_audio_pcm(audio_pcm, user_variant)
            if image_jpeg is not None:
                image_jpeg = _personalize_image_jpeg(image_jpeg, user_variant)
        prepared.append((audio_pcm, image_jpeg))
    return prepared


def _websocket_uri(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    elif parsed.scheme == "https":
        scheme = "wss"
    else:
        scheme = "ws"
    return urlunparse((scheme, parsed.netloc, "/v1/video/chat/stream", "", "", ""))


async def _run_session(
    spec: SessionSpec,
    args: argparse.Namespace,
    prepared_inputs: list[tuple[bytes, bytes | None]],
    ready: asyncio.Future[None],
    benchmark_start_future: asyncio.Future[float],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    compressions: list[dict[str, int]] = []
    errors: list[str] = []
    scheduled_at: dict[int, float] = {}
    sent_at: dict[int, float] = {}
    starts: dict[int, dict[str, Any]] = {}
    audio_by_slot: dict[int, str] = {}
    frame_acks: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()

    try:
        async with websockets.connect(
            _websocket_uri(args.url),
            max_size=64 * 1024 * 1024,
            compression=None,
            open_timeout=args.timeout,
            close_timeout=5,
        ) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "session.config",
                        "session_id": spec.session_id,
                        "model": args.model,
                        "system_prompt": args.system_prompt,
                        "from_s2": args.from_s2,
                        "input_sample_rate": SAMPLE_RATE,
                        "max_inflight_per_session": args.max_inflight_per_session,
                        "slot_overlap": args.slot_overlap,
                        "context_window_trigger_tokens": args.context_trigger_tokens,
                        "max_tokens": args.max_tokens,
                        "enable_frame_filter": not args.disable_frame_filter,
                        "frame_filter_threshold": args.frame_filter_threshold,
                        "frame_filter_min_gap": args.frame_filter_min_gap,
                        "frame_filter_max_gap": args.frame_filter_max_gap,
                        "benchmark_prefill_only": spec.prefill_only,
                    },
                    ensure_ascii=False,
                )
            )
            handshake = json.loads(await websocket.recv())
            if handshake.get("type") != "session.ready":
                raise RuntimeError(f"session handshake failed: {handshake}")
            if not ready.done():
                ready.set_result(None)
            benchmark_start = await benchmark_start_future

            async def receive() -> None:
                async for raw in websocket:
                    event = json.loads(raw)
                    event_type = event.get("type")
                    slot = event.get("slot")
                    if event_type == "response.start" and isinstance(slot, int):
                        starts[slot] = event
                    elif event_type == "response.audio.delta" and isinstance(slot, int):
                        data = event.get("data")
                        if isinstance(data, str):
                            audio_by_slot[slot] = data
                    elif event_type == "session.history.compacted":
                        compressions.append(
                            {
                                "after_slot": int(event["after_slot"]),
                                "slots_before": int(event["slots_before"]),
                                "slots_after": int(event["slots_after"]),
                            }
                        )
                    elif event_type == "video.frame.ack":
                        frame_acks.append(
                            {
                                "frame_id": str(event.get("frame_id") or ""),
                                "accepted": bool(event.get("accepted")),
                            }
                        )
                    elif event_type == "response.done" and isinstance(slot, int):
                        ready_at = loop.time()
                        text = str(event.get("text") or "").strip()
                        audio_b64 = audio_by_slot.pop(slot, "")
                        response = {
                            "id": event.get("request_id"),
                            "choices": [{"message": {"content": text, "audio": {"data": audio_b64}}}],
                            "usage": event.get("usage") or {},
                            "metrics": event.get("metrics") or {},
                        }
                        if spec.prefill_only:
                            assistant_text = ""
                            output_wav = b""
                            codes = []
                            valid_turn = False
                            eos_emitted = False
                            controls: dict[str, Any] = {}
                            missing: list[str] = []
                        else:
                            assistant_text, output_wav, codes, valid_turn, eos_emitted = (
                                _extract_response(response)
                            )
                            controls = _parse_controls(assistant_text) or {}
                            missing = [key for key in CONTROL_KEYS if key not in controls]
                        usage = response["usage"]
                        prompt_details = usage.get("prompt_tokens_details") or {}
                        timing = event.get("timing") or {}
                        scheduled = scheduled_at[slot]
                        sent = sent_at[slot]
                        e2e_ms = (ready_at - scheduled) * 1000.0
                        start = starts.get(slot, {})
                        records.append(
                            {
                                "user": spec.user,
                                "session_id": spec.session_id,
                                "workload_role": "prefill_only" if spec.prefill_only else "probe",
                                "slot": slot,
                                "request_id": event.get("request_id"),
                                "phase_ms": spec.phase_ms,
                                "scheduled_input_ready_ms": (scheduled - benchmark_start) * 1000.0,
                                "submit_ms": (sent - benchmark_start) * 1000.0,
                                "response_ready_ms": (ready_at - benchmark_start) * 1000.0,
                                "client_send_queue_ms": max(0.0, (sent - scheduled) * 1000.0),
                                "app_queue_ms": float(timing.get("app_queue_ms") or 0.0),
                                "prompt_render_ms": float(timing.get("prompt_render_ms") or 0.0),
                                "thinker_latency_ms": float(timing.get("thinker_latency_ms") or 0.0),
                                "request_latency_ms": float(timing.get("request_latency_ms") or 0.0),
                                "server_e2e_slot_latency_ms": float(timing.get("e2e_slot_latency_ms") or 0.0),
                                "e2e_slot_latency_ms": e2e_ms,
                                "deadline_miss": e2e_ms > SLOT_MS,
                                "submitted_before_predecessor_audio": bool(
                                    start.get("overlapped_predecessor_audio")
                                ),
                                "cache_epoch": int(event.get("cache_epoch") or 0),
                                "epoch_slot": int(event.get("epoch_slot") or 0),
                                "cache_salt": f"{spec.session_id}:epoch-{int(event.get('cache_epoch') or 0)}",
                                "controls": controls,
                                "control_valid": not missing,
                                "missing_control_fields": missing,
                                "speaking": bool(controls.get("tts")),
                                "valid_turn": valid_turn,
                                "eos_emitted": eos_emitted,
                                "codec_shape": [len(codes), len(codes[0]) if codes else 0],
                                "audio": None if spec.prefill_only else _wav_stats(output_wav),
                                "usage": usage,
                                "cached_prompt_tokens": int(prompt_details.get("cached_tokens") or 0),
                                "metrics": response["metrics"],
                            }
                        )
                    elif event_type in {"error", "response.error"}:
                        errors.append(str(event.get("message") or event))
                    elif event_type == "session.done":
                        return

            receiver = asyncio.create_task(receive(), name=f"duplex-recv-{spec.user}")
            for slot, (audio_pcm, image_jpeg) in enumerate(prepared_inputs):
                scheduled = benchmark_start + _scheduled_ms(spec, slot) / 1000.0
                scheduled_at[slot] = scheduled
                remaining = scheduled - loop.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                if image_jpeg is not None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "video.frame",
                                "frame_id": f"{spec.session_id}-frame-{slot}",
                                "data": base64.b64encode(image_jpeg).decode("ascii"),
                            }
                        )
                    )
                sent_at[slot] = loop.time()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "slot": slot,
                            "final": slot == len(prepared_inputs) - 1,
                            "data": base64.b64encode(audio_pcm).decode("ascii"),
                        }
                    )
                )
            await websocket.send(json.dumps({"type": "session.finish"}))
            await asyncio.wait_for(receiver, timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001 - preserve other user sessions
        if not ready.done():
            ready.set_exception(exc)
        errors.append(f"{type(exc).__name__}: {exc}")

    records.sort(key=lambda record: int(record["slot"]))
    return {
        "user": spec.user,
        "session_id": spec.session_id,
        "workload_role": "prefill_only" if spec.prefill_only else "probe",
        "phase_ms": spec.phase_ms,
        "records": records,
        "compressions": compressions,
        "video_frames_sent": sum(image is not None for _, image in prepared_inputs),
        "video_frames_acked": len(frame_acks),
        "video_frames_accepted": sum(bool(ack["accepted"]) for ack in frame_acks),
        "error": "; ".join(errors) if errors else None,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_audio = _load_audio(args.audio)
    specs = _session_specs(
        args.users,
        args.seed,
        args.session_prefix,
        getattr(args, "prefill_only_users", 0),
    )
    prepared_inputs = {
        spec.user: _prepared_inputs(
            args,
            source_audio,
            None
            if args.shared_media_across_users
            else getattr(args, "media_seed", args.seed) * 1009 + spec.user,
        )
        for spec in specs
    }
    loop = asyncio.get_running_loop()
    benchmark_start_future: asyncio.Future[float] = loop.create_future()
    ready_futures = [loop.create_future() for _ in specs]
    tasks = [
        asyncio.create_task(
            _run_session(spec, args, prepared_inputs[spec.user], ready, benchmark_start_future),
            name=f"duplex-user-{spec.user}",
        )
        for spec, ready in zip(specs, ready_futures, strict=True)
    ]
    await asyncio.gather(*ready_futures)
    benchmark_start = loop.time() + args.start_lead_ms / 1000.0
    benchmark_start_future.set_result(benchmark_start)
    sessions = await asyncio.gather(*tasks)

    records = [record for session in sessions for record in session["records"]]
    probe_records = [record for record in records if record["workload_role"] == "probe"]
    prefill_records = [
        record for record in records if record["workload_role"] == "prefill_only"
    ]
    errors = [
        {"user": session["user"], "session_id": session["session_id"], "error": session["error"]}
        for session in sessions
        if session["error"] is not None
    ]
    manifest = {
        "schema_version": 2,
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "transport": "server-owned-websocket-session",
        "application_state_owner": "server",
        "engine_request_lifecycle": "finite-per-slot",
        "users": args.users,
        "probe_users": args.users,
        "prefill_only_users": getattr(args, "prefill_only_users", 0),
        "total_sessions": len(specs),
        "slots_per_user": args.slots,
        "slot_ms": SLOT_MS,
        "seed": args.seed,
        "media_seed": getattr(args, "media_seed", args.seed),
        "benchmark_start_monotonic_s": benchmark_start,
        "phase_policy": "per-session uniform [0, 480 ms)",
        "slot_pipeline": "thinker_next_overlaps_talker_current" if args.slot_overlap else "fully_serial",
        "max_inflight_per_session": args.max_inflight_per_session if args.slot_overlap else 1,
        "input_modalities": ["audio"] if args.audio_only else ["audio", "image"],
        "cross_user_media_identity": "shared" if args.shared_media_across_users else "deterministically_distinct",
        "video_frame_interval_slots": None if args.audio_only else args.video_frame_interval,
        "video_frames": {
            "sent": sum(int(session["video_frames_sent"]) for session in sessions),
            "acked": sum(int(session["video_frames_acked"]) for session in sessions),
            "accepted": sum(int(session["video_frames_accepted"]) for session in sessions),
        },
        "context_policy": {
            "trigger_tokens": args.context_trigger_tokens,
            "retained_slots": 1,
            "normal_operation": "append-only",
        },
        "summary": _latency_summary(probe_records),
        "prefill_only_summary": _latency_summary(prefill_records),
        "sessions": sessions,
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2), flush=True)
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_audio = repo_root / "tests/assets/minicpmo_4_5/response_required_16k.wav"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="DuplexOmni")
    parser.add_argument("--label", choices=("bf16", "fp8"), default="fp8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument(
        "--prefill-only-users",
        type=int,
        default=0,
        help="Additional real-media sessions that stop after Thinker prefill.",
    )
    parser.add_argument("--slots", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--media-seed",
        type=int,
        help="Media personalization seed; defaults to --seed.",
    )
    parser.add_argument("--session-prefix", default="duplexomni-load")
    parser.add_argument("--start-lead-ms", type=float, default=1000.0)
    parser.add_argument("--audio", type=Path, default=default_audio)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument(
        "--shared-media-across-users",
        action="store_true",
        help="Reuse byte-identical media across users (cache-control A/B only).",
    )
    parser.add_argument("--video-frame-interval", type=int, default=1)
    parser.add_argument("--from-s2", default="")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--context-trigger-tokens", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=999)
    parser.add_argument(
        "--slot-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Thinker(t+1) after Thinker(t), without waiting for Talker/Code2Wav(t).",
    )
    parser.add_argument("--max-inflight-per-session", type=int, default=4)
    parser.add_argument("--disable-frame-filter", action="store_true")
    parser.add_argument("--frame-filter-threshold", type=float, default=0.95)
    parser.add_argument("--frame-filter-min-gap", type=int, default=0)
    parser.add_argument("--frame-filter-max-gap", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.media_seed is None:
        args.media_seed = args.seed
    if args.users < 0:
        parser.error("--users cannot be negative")
    if args.prefill_only_users < 0:
        parser.error("--prefill-only-users cannot be negative")
    if args.users + args.prefill_only_users < 1:
        parser.error("at least one probe or prefill-only user is required")
    if args.slots < 1:
        parser.error("--slots must be positive")
    if args.start_lead_ms < 0:
        parser.error("--start-lead-ms cannot be negative")
    if args.context_trigger_tokens < 1024:
        parser.error("--context-trigger-tokens must be at least 1024")
    if args.video_frame_interval < 1:
        parser.error("--video-frame-interval must be positive")
    if args.max_inflight_per_session < 1:
        parser.error("--max-inflight-per-session must be positive")
    if not 0.0 <= args.frame_filter_threshold <= 1.0:
        parser.error("--frame-filter-threshold must be in [0, 1]")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image file not found: {args.image}")
    return args


def main() -> int:
    args = parse_args()
    manifest = asyncio.run(run(args))
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
