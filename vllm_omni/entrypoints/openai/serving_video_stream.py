# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen-Omni streaming video WebSocket handler.

Accepts video frames incrementally via WebSocket, buffers them, and
generates text + optional audio responses using the Qwen3-Omni multi-stage
pipeline (thinker -> talker -> code2wav).

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "..."}  # base64 JPEG/PNG frame
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import replace
from typing import Any

from vllm_omni.entrypoints.openai.video_stream_base import (
    _BAD_FRAME,
    _DEFAULT_CONFIG_TIMEOUT,
    _DEFAULT_IDLE_TIMEOUT,
    StreamingVideoSessionConfig,
    VideoStreamTurnTrigger,
    _TurnMediaEvent,
)
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler as OmniStreamingVideoHandlerBase,
)

__all__ = [
    "QwenOmniStreamingVideoHandler",
    "StreamingVideoSessionConfig",
    "create_streaming_video_handler",
]


class QwenOmniStreamingVideoHandler(OmniStreamingVideoHandlerBase):
    """Qwen-Omni pipeline: manual ``video.query`` trigger and image_pil prompts."""

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        return False

    def supports_incremental_canonical_prompt(self) -> bool:
        # Qwen's ChatML template renders complete system/user/assistant
        # messages as append-only blocks. This lets the application retain
        # processed historical media while every engine request remains
        # finite and carries the complete canonical prompt.
        return hasattr(self._chat_service, "renderer")

    @staticmethod
    def _audio_content(audio_bytes: bytes, *, streaming_approximation: bool) -> dict[str, Any]:
        prefix = "audio-stream-approx-v1" if streaming_approximation else "audio"
        return {
            "type": "input_audio",
            "input_audio": {
                "data": QwenOmniStreamingVideoHandler._pcm_to_wav_b64(audio_bytes),
                "format": "wav",
            },
            "uuid": f"{prefix}:{hashlib.sha256(audio_bytes).hexdigest()}",
        }

    @staticmethod
    def _frame_content(
        frame_b64: str,
        prewarmed: dict[str, tuple[Any, str]],
    ) -> dict[str, Any] | None:
        cached = prewarmed.get(frame_b64)
        if cached is _BAD_FRAME:
            return None
        if cached is not None:
            pil, pil_uuid = cached
            if pil is not None:
                return {"type": "image_pil", "image_pil": pil, "uuid": pil_uuid}
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                "uuid": pil_uuid,
            }
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        }

    def _normalize_engine_prompt_for_messages(
        self,
        config: StreamingVideoSessionConfig,
        messages: list[dict[str, Any]],
        prompt: dict[str, Any],
    ) -> dict[str, Any]:
        """Join adjacent immutable chunks into one logical audio token span.

        Qwen still encodes each one-second item independently, which is the
        documented approximation. Removing only the internal audio-end and
        audio-start token pair keeps the decoder-side stream and MRoPE
        positions contiguous while preserving per-chunk multimodal hashes.
        """
        if not config.enable_audio_arrival_prefill_approximation:
            return prompt

        boundary_after_audio_item: list[int] = []
        audio_items = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            previous_was_audio = False
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else None
                is_audio = part_type in {"input_audio", "audio", "audio_url"}
                if is_audio:
                    if previous_was_audio:
                        boundary_after_audio_item.append(audio_items - 1)
                    audio_items += 1
                previous_was_audio = is_audio

        if not boundary_after_audio_item:
            return prompt
        placeholders = prompt.get("mm_placeholders", {}).get("audio", [])
        if len(placeholders) != audio_items:
            raise ValueError(
                "streaming audio placeholder count does not match the rendered audio items"
            )

        removed_positions: list[int] = []
        for left_index in boundary_after_audio_item:
            left = placeholders[left_index]
            right = placeholders[left_index + 1]
            boundary = left.offset + left.length
            if right.offset != boundary + 2:
                raise ValueError(
                    "Qwen chat template no longer has exactly two internal audio boundary tokens"
                )
            removed_positions.extend((boundary, boundary + 1))
        removed_positions.sort()
        removed = set(removed_positions)

        normalized = dict(prompt)
        normalized["prompt_token_ids"] = [
            token
            for index, token in enumerate(prompt["prompt_token_ids"])
            if index not in removed
        ]
        assistant_mask = prompt.get("assistant_tokens_mask")
        if isinstance(assistant_mask, list):
            normalized["assistant_tokens_mask"] = [
                value for index, value in enumerate(assistant_mask) if index not in removed
            ]
        normalized["mm_placeholders"] = {
            modality: [
                replace(item, offset=item.offset - bisect_left(removed_positions, item.offset))
                for item in modality_placeholders
            ]
            for modality, modality_placeholders in prompt["mm_placeholders"].items()
        }
        return normalized

    def build_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        media_events: list[_TurnMediaEvent] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prewarmed = prewarmed_frames or {}
        user_content: list[dict] = []
        if config.enable_audio_arrival_prefill_approximation and media_events is not None:
            # Seal audio only at one-second boundaries and retain the actual
            # accepted-media order. Thus every arrival prompt is an append-only
            # extension even while audio and video arrive concurrently.
            for event in media_events:
                if event.modality == "image" and isinstance(event.payload, str):
                    item = self._frame_content(event.payload, prewarmed)
                    if item is not None:
                        user_content.append(item)
                elif event.modality == "audio" and isinstance(event.payload, bytes):
                    user_content.append(
                        self._audio_content(event.payload, streaming_approximation=True)
                    )
        else:
            # The similarity/freshness filter is the sole frame-selection policy.
            # Every accepted frame remains in this turn in arrival order.
            for frame_b64 in frame_buffer:
                item = self._frame_content(frame_b64, prewarmed)
                if item is not None:
                    user_content.append(item)

        if len(audio_buffer) > 0 and not (
            config.enable_audio_arrival_prefill_approximation and media_events is not None
        ):
            audio_bytes = bytes(audio_buffer)
            user_content.append(self._audio_content(audio_bytes, streaming_approximation=False))

        if query_text:
            user_content.append({"type": "text", "text": query_text})

        user_message: dict[str, Any] = {"role": "user", "content": user_content}

        messages = self._history_prefix_messages(config)

        # The application owns the canonical conversation. Reuse the exact
        # accepted multimodal messages on every finite engine request so vLLM
        # can match both token hashes and multimodal hashes in its prefix cache.
        # A cache miss changes cost, never prompt semantics.
        messages.extend(message_history)

        messages.append(user_message)

        return messages, user_message

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        message_history.append(user_message)
        message_history.append({"role": "assistant", "content": response_text})

    _build_messages = build_engine_prompt


def create_streaming_video_handler(
    chat_service: Any,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
    config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    engine_client: Any | None = None,
) -> OmniStreamingVideoHandlerBase:
    """Create the handler for ``/v1/video/chat/stream``.

    The transport and application/engine boundary are shared, while each
    model adapter owns its actual turn/slot semantics.
    """
    if getattr(engine_client, "pipeline_model_type", None) in {
        "duplexomni",
        "duplexomni_pd",
    }:
        from vllm_omni.entrypoints.openai.serving_duplexomni_stream import (
            DuplexOmniStreamingVideoHandler,
        )

        return DuplexOmniStreamingVideoHandler(
            chat_service=chat_service,
            idle_timeout=idle_timeout,
            config_timeout=config_timeout,
            engine_client=engine_client,
        )
    return QwenOmniStreamingVideoHandler(
        chat_service=chat_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
        engine_client=engine_client,
    )
