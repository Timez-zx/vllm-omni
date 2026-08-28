#!/usr/bin/env python3
"""Run one server-owned DuplexOmni session with finite per-slot requests."""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
import io
import json
import math
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SAMPLE_RATE = 24000
SLOT_MS = 480
SLOT_SAMPLES = SAMPLE_RATE * SLOT_MS // 1000
EXPECTED_CODEC_SHAPE = (16, 6)
CONTROL_KEYS = ("asr", "tts", "tts_control", "system2_control")
DEFAULT_SYSTEM_PROMPT = (
    "你是视频通话助手\n你的助手风格是：控场型。\n你的开场方式要求：先短答不展开，耐心听用户说完所有选项再回答。"
)


def _load_audio(path: Path | None) -> np.ndarray:
    if path is None:
        # Non-silent deterministic fallback; a real speech WAV is preferred.
        t = np.arange(SLOT_SAMPLES * 2, dtype=np.float64) / SAMPLE_RATE
        return (0.04 * 32767 * np.sin(2 * math.pi * 220 * t)).astype(np.int16)
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())
    if width != 2:
        raise ValueError(f"input WAV must be 16-bit PCM, got sample width {width}")
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if source_rate != SAMPLE_RATE and samples.size:
        target_len = max(1, round(samples.size * SAMPLE_RATE / source_rate))
        samples = np.interp(
            np.linspace(0.0, 1.0, target_len, endpoint=False),
            np.linspace(0.0, 1.0, samples.size, endpoint=False),
            samples.astype(np.float64),
        ).astype(np.int16)
    return samples


def _image_bytes(path: Path | None, slot_index: int) -> bytes:
    if path is not None:
        with Image.open(path) as source:
            image = source.convert("RGB")
    else:
        image = Image.new("RGB", (640, 360), (238, 238, 238))
        draw = ImageDraw.Draw(image)
        x = 40 + (slot_index * 97) % 360
        draw.rectangle((x, 70, x + 180, 290), fill=(48, 125, 210))
        draw.text((24, 20), f"DuplexOmni slot {slot_index}", fill=(20, 20, 20))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def _slot_has_video(slot_index: int, audio_only: bool, video_frame_interval: int) -> bool:
    return not audio_only and slot_index % video_frame_interval == 0


def _parse_controls(text: str) -> dict[str, Any] | None:
    clean = text.replace("```json", "").replace("```", "").strip()
    if "{" in clean:
        clean = clean[clean.index("{") :]
    if "}" in clean:
        clean = clean[: clean.rindex("}") + 1]
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(clean)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
    return None


def _extract_response(response: dict[str, Any]) -> tuple[str, bytes, list[list[int]], bool, bool]:
    text = ""
    audio = b""
    for choice in response.get("choices", []):
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            text = content.strip()
        audio_obj = message.get("audio")
        if isinstance(audio_obj, dict) and isinstance(audio_obj.get("data"), str):
            audio = base64.b64decode(audio_obj["data"])
    metrics = response.get("metrics") or {}
    duplex_metrics = metrics.get("duplexomni") or {}
    raw_codes = duplex_metrics.get("codec_codes")
    if not isinstance(raw_codes, list):
        raise RuntimeError("response did not expose DuplexOmni codec history")
    codes = [[int(item) for item in row] for row in raw_codes]
    if (len(codes), len(codes[0]) if codes else 0) != EXPECTED_CODEC_SHAPE:
        raise RuntimeError(
            f"expected codec shape {EXPECTED_CODEC_SHAPE}, got {(len(codes), len(codes[0]) if codes else 0)}"
        )
    if any(item < 0 or item >= 2048 for row in codes for item in row):
        raise RuntimeError("codec response contains an id outside [0, 2048)")
    valid_turn = duplex_metrics.get("valid_turn")
    eos_emitted = duplex_metrics.get("eos_emitted")
    if not isinstance(valid_turn, bool) or not isinstance(eos_emitted, bool):
        raise RuntimeError("response did not expose DuplexOmni EOS validation")
    if not text:
        raise RuntimeError("Thinker returned no structured text")
    if not audio:
        raise RuntimeError("Code2Wav returned no waveform")
    return text, audio, codes, valid_turn, eos_emitted


def _wav_stats(raw: bytes) -> dict[str, float | int]:
    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        pcm = wav_file.readframes(frames)
    if width != 2:
        raise RuntimeError(f"output WAV is not 16-bit PCM: width={width}")
    values = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    rms = float(np.sqrt(np.mean(np.square(values))) / 32768.0) if values.size else 0.0
    peak = float(np.max(np.abs(values)) / 32768.0) if values.size else 0.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_ms": 1000.0 * frames / sample_rate,
        "rms": rms,
        "peak": peak,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the same server-owned session path as the capacity benchmark."""
    from multi_user import run as run_multi_user

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_prefix = args.session_id or f"duplexomni-{uuid.uuid4().hex[:12]}"
    multi_args = argparse.Namespace(
        url=args.url,
        model=args.model,
        label=args.label,
        output=output_dir,
        users=1,
        prefill_only_users=0,
        slots=args.slots,
        seed=0,
        media_seed=0,
        session_prefix=session_prefix,
        start_lead_ms=100.0,
        audio=args.audio,
        image=args.image,
        audio_only=args.audio_only,
        shared_media_across_users=False,
        video_frame_interval=args.video_frame_interval,
        from_s2=args.from_s2,
        system_prompt=args.system_prompt,
        context_trigger_tokens=args.context_trigger_tokens,
        max_tokens=args.max_tokens,
        slot_overlap=True,
        max_inflight_per_session=4,
        disable_frame_filter=False,
        frame_filter_threshold=0.95,
        frame_filter_min_gap=0,
        frame_filter_max_gap=4,
        timeout=args.timeout,
    )
    capacity_manifest = asyncio.run(run_multi_user(multi_args))
    if capacity_manifest["errors"]:
        raise RuntimeError(capacity_manifest["errors"])
    session = capacity_manifest["sessions"][0]
    records = session["records"]

    manifest = {
        "schema_version": 2,
        "label": args.label,
        "model": args.model,
        "url": args.url,
        "transport": "server-owned-websocket-session",
        "application_state_owner": "server",
        "engine_request_lifecycle": "finite-per-slot",
        "session_id": session["session_id"],
        "slot_ms": SLOT_MS,
        "input_modalities": ["audio"] if args.audio_only else ["audio", "image"],
        "video_frame_interval_slots": None if args.audio_only else args.video_frame_interval,
        "video_frames": capacity_manifest["video_frames"],
        "context_policy": {
            "trigger_tokens": args.context_trigger_tokens,
            "retained_slots": 1,
            "normal_operation": "append-only",
        },
        "compressions": session["compressions"],
        "slots": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_audio = repo_root / "tests/assets/minicpmo_4_5/response_required_16k.wav"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8092")
    parser.add_argument("--model", default="DuplexOmni")
    parser.add_argument("--label", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--audio", type=Path, default=default_audio)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--video-frame-interval", type=int, default=1)
    parser.add_argument("--from-s2", default="")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--session-id")
    parser.add_argument("--context-trigger-tokens", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=999)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.slots < 1:
        parser.error("--slots must be positive")
    if args.context_trigger_tokens < 1:
        parser.error("--context-trigger-tokens must be positive")
    if args.video_frame_interval < 1:
        parser.error("--video-frame-interval must be positive")
    if args.audio is not None and not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    if args.image is not None and not args.image.is_file():
        parser.error(f"image file not found: {args.image}")
    return args


if __name__ == "__main__":
    run(parse_args())
