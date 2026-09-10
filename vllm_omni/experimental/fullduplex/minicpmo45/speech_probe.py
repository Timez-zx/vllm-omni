# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in speech correctness capture, never a capacity/latency profiler.

Set VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR to a run-owned directory. JSONL is
written per process; VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS=1 additionally
saves the actual TTS condition before in-place model execution. Enabling this
probe introduces CPU/GPU synchronization and file I/O. Disabled callers must
guard tensor inspection with ENABLED, so normal serving performs neither.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

DIRECTORY = os.environ.get("VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR", "")
ENABLED = bool(DIRECTORY)
SAVE_TENSORS = os.environ.get("VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS", "0") == "1"
_LOCK = threading.Lock()
_SEQUENCE = itertools.count()


def values(value: Any) -> list[Any]:
    """Materialize a diagnostic tensor only inside an enabled caller."""
    if value is None:
        return []
    if hasattr(value, "detach"):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def identity(info: dict[str, Any]) -> dict[str, Any]:
    def scalar(value: Any) -> Any:
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if hasattr(value, "numel") and value.numel() == 1:
            return value.detach().cpu().item()
        return value

    duplex = info.get("duplex") or {}
    meta = info.get("meta") or {}
    return {
        "request_id": str(info.get("request_id", "")),
        "session_id": scalar(duplex.get("session_id")),
        "incarnation": scalar(duplex.get("incarnation")),
        "epoch": scalar(duplex.get("epoch")),
        "model_turn_id": scalar(duplex.get("turn_id")),
        "turn_start": scalar(meta.get("turn_start")),
        "turn_end": scalar(meta.get("turn_end")),
        "segment_text": scalar(meta.get("native_duplex_segment_text")),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        cpu = value.detach().cpu()
        return cpu.item() if cpu.numel() == 1 else cpu.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported speech probe field type: {type(value).__name__}")


def emit(event: str, **fields: Any) -> int | None:
    if not ENABLED:
        return None
    with _LOCK:
        seq = next(_SEQUENCE)
        directory = Path(DIRECTORY)
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event, "pid": os.getpid(), "seq": seq,
            "wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
            **_json_value(fields),
        }
        with (directory / f"speech-{os.getpid()}.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return seq


def new_batch_id() -> int | None:
    """Process-local correlation ID; no tensor inspection or file write."""
    if not ENABLED:
        return None
    with _LOCK:
        return next(_SEQUENCE)


def dump_condition(**tensors: Any) -> str | None:
    if not ENABLED or not SAVE_TENSORS:
        return None
    import torch

    with _LOCK:
        directory = Path(DIRECTORY)
        directory.mkdir(parents=True, exist_ok=True)
        name = f"condition-{os.getpid()}-{next(_SEQUENCE)}.pt"
        # Clone BEFORE the model: its residual path can mutate inputs_embeds.
        captured = {key: value.detach().cpu().clone() for key, value in tensors.items()}
        torch.save(captured, directory / name)
    return name


def audio_summary(audio: Any) -> dict[str, Any]:
    import torch

    samples = audio.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    return {
        "audio_samples": samples.numel(),
        "audio_f32_sha256": sha256(samples.numpy().tobytes()).hexdigest(),
        "audio_finite": bool(torch.isfinite(samples).all()),
    }
