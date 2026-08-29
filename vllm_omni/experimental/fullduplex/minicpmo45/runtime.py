# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from base64 import b64decode
from binascii import Error as BinasciiError
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

_DUPLEX_CHUNK_SAMPLES = 16000
_DUPLEX_SAMPLES_PER_AUDIO_TOKEN = 1600
# Every source image or HD crop contributes one wrapper pair and 64 resampler
# rows, matching the official MiniCPMO45 duplex streaming_prefill contract.
_DUPLEX_VISION_TOKENS_PER_BLOCK = 66
_DUPLEX_VISION_SCALE_RESOLUTION = 448
_DEFAULT_CONTEXT_WINDOW_TRIGGER_TOKENS = 36_000
_MAX_CONTEXT_WINDOW_STATES = 4_096


def _duplex_max_slice_nums(payload: object, frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    raw = payload.get("max_slice_nums") if isinstance(payload, dict) else None
    if isinstance(raw, int) and not isinstance(raw, bool):
        value = max(1, raw)
        return [value] * frame_count
    if isinstance(raw, list) and len(raw) == frame_count:
        values: list[int] = []
        for item in raw:
            if not isinstance(item, int) or isinstance(item, bool):
                return [1] * frame_count
            values.append(max(1, item))
        return values
    return [1] * frame_count


def _duplex_image_block_count(width: int, height: int, max_slice_nums: int) -> int:
    """Match MiniCPMVImageProcessor.get_sliced_grid without materializing crops."""
    if width <= 0 or height <= 0 or max_slice_nums <= 1:
        return 1
    log_ratio = math.log(width / height)
    area_ratio = width * height / (_DUPLEX_VISION_SCALE_RESOLUTION**2)
    multiple = min(math.ceil(area_ratio), max_slice_nums)
    if multiple <= 1:
        return 1
    candidate_grids: list[tuple[int, int]] = []
    for split_count in (multiple - 1, multiple, multiple + 1):
        if split_count == 1 or split_count > max_slice_nums:
            continue
        for columns in range(1, split_count + 1):
            if split_count % columns == 0:
                candidate_grids.append((columns, split_count // columns))
    if not candidate_grids:
        return 1
    columns, rows = min(
        candidate_grids,
        key=lambda grid: abs(log_ratio - math.log(grid[0] / grid[1])),
    )
    return 1 + columns * rows


def duplex_vision_block_counts(payload: object) -> list[int]:
    if not isinstance(payload, dict):
        return []
    raw_frames = payload.get("video_frames")
    if not isinstance(raw_frames, list):
        return []
    frames = [
        frame
        for frame in raw_frames
        if isinstance(frame, str) and frame
    ]
    max_slices = _duplex_max_slice_nums(payload, len(frames))
    counts: list[int] = []
    for frame, max_slice_nums in zip(frames, max_slices, strict=True):
        try:
            raw = b64decode(frame, validate=True)
            with Image.open(BytesIO(raw)) as image:
                width, height = image.size
        except Exception:
            counts.append(1)
            continue
        counts.append(_duplex_image_block_count(width, height, max_slice_nums))
    return counts


def _duplex_vision_token_count(payload: object) -> int:
    return sum(duplex_vision_block_counts(payload)) * _DUPLEX_VISION_TOKENS_PER_BLOCK


def _duplex_pcm_sample_count(payload: object) -> int | None:
    if not isinstance(payload, dict):
        return None
    audio = payload.get("audio") or payload.get("data")
    if payload.get("format") != "pcm_f32le" or not isinstance(audio, str):
        return None
    try:
        raw = b64decode(audio, validate=True)
    except (BinasciiError, ValueError):
        return None
    return len(raw) // 4


def duplex_payload_is_exact_chunks(payload: object) -> bool:
    sample_count = _duplex_pcm_sample_count(payload)
    return bool(sample_count) and sample_count % _DUPLEX_CHUNK_SAMPLES == 0


def duplex_first_append_unit_count(payload: object) -> int | None:
    sample_count = _duplex_pcm_sample_count(payload)
    if not sample_count or sample_count % _DUPLEX_CHUNK_SAMPLES != 0:
        return None
    return max(1, sample_count // _DUPLEX_CHUNK_SAMPLES - 1)


def duplex_scheduler_token_budget(payload: object, *, default: int = 64) -> int:
    vision_tokens = _duplex_vision_token_count(payload)
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None:
        return max(1, int(default)) + vision_tokens
    sample_count = max(1, sample_count)
    if sample_count % _DUPLEX_CHUNK_SAMPLES == 0:
        units = sample_count // _DUPLEX_CHUNK_SAMPLES
        return units * (2 + _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN) + vision_tokens
    return max(16, min(768, sample_count // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)) + vision_tokens


def duplex_retained_unit_token_budget(payload: object) -> int:
    """Return the exact prompt rows for the latest complete model unit.

    A steady-state append also carries the previous unit's terminator and
    ``</unit>``.  Those two rows are supplied by the next append after a
    rollover, so the retained slot contains only ``<unit>``, its audio rows,
    and the camera blocks attached to that unit.
    """
    sample_count = _duplex_pcm_sample_count(payload)
    if sample_count is None or sample_count < _DUPLEX_CHUNK_SAMPLES:
        return 0
    audio_rows = _DUPLEX_CHUNK_SAMPLES // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN
    vision_counts = duplex_vision_block_counts(payload)
    vision_rows = (vision_counts[-1] if vision_counts else 0) * _DUPLEX_VISION_TOKENS_PER_BLOCK
    return 1 + audio_rows + vision_rows


def duplex_first_append_context_reserve(runtime_config: object) -> int:
    if not isinstance(runtime_config, dict):
        return 48
    exact = runtime_config.get("duplex_first_append_context_tokens")
    if isinstance(exact, int) and exact >= 0:
        return exact
    reserve = 48
    ref = runtime_config.get("ref_audio_data")
    if isinstance(ref, str) and ref:
        try:
            raw = b64decode(ref, validate=True)
        except (BinasciiError, ValueError):
            raw = b""
        if raw:
            reserve += max(0, (len(raw) // 4) // _DUPLEX_SAMPLES_PER_AUDIO_TOKEN + 8)
    return reserve


def _duplex_force_listen_count(extra_body: object) -> int:
    raw = extra_body.get("force_listen_count") if isinstance(extra_body, dict) else None
    try:
        return 0 if raw is None else max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def build_duplex_data_plane_prompt(
    *,
    request_id: str,
    fence: DuplexFence,
    session_config: dict[str, Any],
    runtime_config: dict[str, Any],
    seq: int,
    turn_seq: int,
    mode: DuplexInputMode,
    payload: object,
    final: bool,
    replace_streaming_prompt: bool = False,
    retained_unit_tokens: int = 0,
    context_generation: int = 0,
) -> dict[str, Any]:
    token_budget = duplex_scheduler_token_budget(payload)
    if seq <= 1:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve
        first_units = duplex_first_append_unit_count(payload)
        if first_units is not None:
            token_budget = context_reserve + first_units * 12 - 1 + _duplex_vision_token_count(payload)
    if seq > 1 and duplex_payload_is_exact_chunks(payload):
        token_budget += 1
    if final and duplex_payload_is_exact_chunks(payload):
        token_budget += 12
    context_reserve = 0
    if replace_streaming_prompt:
        context_reserve = duplex_first_append_context_reserve(runtime_config)
        token_budget += context_reserve + max(0, int(retained_unit_tokens))
    extra_body = session_config.get("extra_body")
    raw_token_id = runtime_config.get("duplex_scheduler_token_id")
    try:
        token_id = max(0, int(raw_token_id))
    except (TypeError, ValueError):
        token_id = 0
    force_listen_count = _duplex_force_listen_count(extra_body)
    if (
        force_listen_count > 0
        and turn_seq <= force_listen_count
        and isinstance(payload, dict)
        and payload.get("force_listen") is not True
    ):
        payload = {**payload, "force_listen": True}
    model_intermediate_buffer: dict[str, Any] = {
        "request_id": request_id,
        "global_request_id": [fence.session_id],
        "duplex": {
            "fence": fence,
            "session_id": fence.session_id,
            "incarnation": fence.incarnation,
            "epoch": fence.epoch,
            "seq": seq,
            "turn_id": fence.turn_id,
            "response_seq": fence.response_seq,
            "turn_seq": turn_seq,
            "mode": mode.value,
            "payload": payload,
            "final": final,
            "data_plane": True,
            "session_config": dict(session_config),
            "runtime_config": dict(runtime_config),
            "scheduler_token_budget": token_budget,
            "scheduler_token_id": token_id,
            "context_generation": context_generation,
        },
    }
    if replace_streaming_prompt:
        model_intermediate_buffer["meta"] = {
            "replace_streaming_prompt": True,
            "retain_streaming_output_tokens": True,
            "retained_output_insert_offset": context_reserve + max(0, int(retained_unit_tokens)),
            "streaming_cache_salt": (
                f"minicpmo45:{fence.session_id}:{fence.incarnation}:context-{context_generation}"
            ),
        }
    return {
        "prompt_token_ids": [token_id] * token_budget,
        "model_intermediate_buffer": model_intermediate_buffer,
    }


@dataclass
class _ContextWindowState:
    epoch: int
    last_seq: int = 0
    estimated_tokens: int = 0
    retained_unit_tokens: int = 0
    context_generation: int = 0
    last_replace: bool = False


def _coerce_int(value: object) -> int | None:
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1)
            if value.numel() == 0:
                return None
            value = value[0].item()
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [token_id for item in value if (token_id := _coerce_int(item)) is not None]


def _first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def _multimodal_output(output: object, completion: object | None) -> dict[str, Any]:
    metadata = getattr(output, "multimodal_output", None)
    if isinstance(metadata, dict):
        return metadata
    metadata = getattr(completion, "multimodal_output", None) if completion is not None else None
    return metadata if isinstance(metadata, dict) else {}


def _special_token_ids(metadata: dict[str, Any]) -> dict[str, int]:
    sources: list[object] = [metadata.get("special_token_ids"), metadata.get("meta")]
    sources.append(
        {
            key.removeprefix("meta."): value
            for key, value in metadata.items()
            if isinstance(key, str) and key.startswith("meta.")
        }
    )
    token_ids: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            token_id = _coerce_int(value)
            if isinstance(key, str) and token_id is not None and token_id >= 0:
                token_ids[key] = token_id
    return token_ids


def _completion_token_ids(completion: object | None) -> list[int]:
    if completion is None:
        return []
    for attribute in ("token_ids", "cumulative_token_ids"):
        token_ids = _coerce_int_list(getattr(completion, attribute, None))
        if token_ids:
            return token_ids
    return []


def _stage_config_value(runtime_config: dict[str, Any], key: str, stage_id: int) -> object | None:
    raw = runtime_config.get(key)
    if isinstance(raw, dict):
        value = raw.get(stage_id)
        return raw.get(str(stage_id)) if value is None else value
    if isinstance(raw, (list, tuple)) and stage_id < len(raw):
        return raw[stage_id]
    return None


class MiniCPMO45DuplexRuntimeExtension:
    def __init__(self) -> None:
        # The extension is process-scoped while session IDs are unbounded.
        # The server admits far fewer than this many live sessions, so an LRU
        # cap removes closed-session bookkeeping without evicting live state.
        self._context_windows: OrderedDict[tuple[str, int], _ContextWindowState] = OrderedDict()

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        configured: list[object] = []
        for stage_id, default in enumerate(defaults):
            max_tokens = _coerce_int(_stage_config_value(runtime_config, "duplex_stage_max_tokens", stage_id))
            raw_overrides = _stage_config_value(runtime_config, "duplex_stage_sampling_params", stage_id)
            overrides = dict(raw_overrides) if isinstance(raw_overrides, dict) else {}
            if not isinstance(default, SamplingParams) or (not overrides and (max_tokens is None or max_tokens <= 0)):
                configured.append(default)
                continue
            params = default.clone()
            if max_tokens is not None and max_tokens > 0:
                params.max_tokens = max_tokens
            for name, value in overrides.items():
                if not hasattr(params, name):
                    continue
                setattr(params, name, value)
                if name == "stop_token_ids":
                    all_stop_token_ids = getattr(params, "_all_stop_token_ids", None)
                    if isinstance(all_stop_token_ids, set):
                        all_stop_token_ids.update(int(token_id) for token_id in value)
            configured.append(params)
        return tuple(configured)

    def plan_append(
        self,
        *,
        request_id: str,
        fence: DuplexFence,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        del sampling_params
        key = (fence.session_id, fence.incarnation)
        window = self._context_windows.get(key)
        if window is None or window.epoch != fence.epoch or seq < window.last_seq:
            window = _ContextWindowState(epoch=fence.epoch)
            self._context_windows[key] = window
        self._context_windows.move_to_end(key)
        while len(self._context_windows) > _MAX_CONTEXT_WINDOW_STATES:
            self._context_windows.popitem(last=False)

        raw_budget = duplex_scheduler_token_budget(payload)
        if seq <= 1:
            raw_budget += duplex_first_append_context_reserve(runtime_config)
            first_units = duplex_first_append_unit_count(payload)
            if first_units is not None:
                raw_budget = (
                    duplex_first_append_context_reserve(runtime_config)
                    + first_units * 12
                    - 1
                    + _duplex_vision_token_count(payload)
                )
        elif duplex_payload_is_exact_chunks(payload):
            raw_budget += 1
        if final and duplex_payload_is_exact_chunks(payload):
            raw_budget += 12

        max_output_tokens = _coerce_int(_stage_config_value(runtime_config, "duplex_stage_max_tokens", 0))
        max_output_tokens = max(1, max_output_tokens or 20)
        raw_trigger = runtime_config.get("duplex_context_window_trigger_tokens")
        try:
            trigger_tokens = int(raw_trigger)
        except (TypeError, ValueError):
            trigger_tokens = _DEFAULT_CONTEXT_WINDOW_TRIGGER_TOKENS
        trigger_tokens = max(1024, trigger_tokens)

        # A retried operation reuses the same seq. Return the same decision and
        # do not advance the estimate twice.
        replace_streaming_prompt = window.last_replace if seq == window.last_seq else False
        retained_unit_tokens = window.retained_unit_tokens if replace_streaming_prompt else 0
        if seq > window.last_seq:
            replace_streaming_prompt = bool(
                window.last_seq > 0
                and window.retained_unit_tokens > 0
                and window.estimated_tokens >= trigger_tokens
            )
            retained_unit_tokens = window.retained_unit_tokens if replace_streaming_prompt else 0
            if replace_streaming_prompt:
                window.context_generation += 1
                window.estimated_tokens = (
                    duplex_first_append_context_reserve(runtime_config)
                    + retained_unit_tokens
                    + max_output_tokens
                )
            window.estimated_tokens += raw_budget + max_output_tokens
            window.retained_unit_tokens = duplex_retained_unit_token_budget(payload)
            window.last_seq = seq
            window.last_replace = replace_streaming_prompt

        if replace_streaming_prompt and isinstance(payload, dict):
            payload = {
                **payload,
                "duplex_context_rollover": True,
                "duplex_context_generation": window.context_generation,
            }
        return DuplexAppendPlan(
            prompt=build_duplex_data_plane_prompt(
                request_id=request_id,
                fence=fence,
                session_config=session_config,
                runtime_config=runtime_config,
                seq=seq,
                turn_seq=turn_seq,
                mode=mode,
                payload=payload,
                final=final,
                replace_streaming_prompt=replace_streaming_prompt,
                retained_unit_tokens=retained_unit_tokens,
                context_generation=window.context_generation,
            )
        )

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        if stage_id >= final_stage_id or not segment_finished:
            return None

        completion = _first_completion(output)
        output_metadata = _multimodal_output(output, completion)
        special_token_ids = _special_token_ids(segment_output_metadata)
        special_token_ids.update(_special_token_ids(output_metadata))
        listen_id = special_token_ids.get("listen_token_id")
        if listen_id is None:
            return None

        stop_reason = getattr(completion, "stop_reason", None) if completion is not None else None
        token_ids = _completion_token_ids(completion) or list(segment_token_ids)
        if _coerce_int(stop_reason) != listen_id and (not token_ids or token_ids[-1] != listen_id):
            return None

        metadata = dict(output_metadata)
        for key, value in special_token_ids.items():
            metadata.setdefault(f"meta.{key}", value)
        metadata.update(
            {
                "duplex_direct_response": True,
                "duplex_native_decision": "listen",
                "model_listen": True,
                "listen_source": "model_listen",
            }
        )
        return DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata=metadata,
        )


__all__ = [
    "MiniCPMO45DuplexRuntimeExtension",
    "build_duplex_data_plane_prompt",
    "duplex_first_append_context_reserve",
    "duplex_first_append_unit_count",
    "duplex_payload_is_exact_chunks",
    "duplex_retained_unit_token_budget",
    "duplex_scheduler_token_budget",
    "duplex_vision_block_counts",
]
