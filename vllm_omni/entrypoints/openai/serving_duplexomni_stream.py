# SPDX-License-Identifier: Apache-2.0
"""Server-owned live AV sessions for DuplexOmni.

The application owns slot assembly, dialogue history, context compaction and
per-session backpressure.  Each model invocation is still an ordinary finite
Chat Completions request; engine prefix/KV state remains a disposable cache.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.logger import init_logger

from vllm_omni.engine.duplexomni_pipeline import (
    PIPELINE_BASE_TURNS,
    PIPELINE_EPOCH,
    PIPELINE_FINAL,
    PIPELINE_SESSION_ID,
    PIPELINE_SLOT,
)
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.openai.video_stream_base import (
    OmniStreamingVideoHandler,
    _downscale_frame_bytes,
)

logger = init_logger(__name__)

SLOT_MS = 480
INPUT_SAMPLE_RATE = 24000
SLOT_PCM_BYTES = INPUT_SAMPLE_RATE * SLOT_MS // 1000 * 2
_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_BYTES = 10 * 1024 * 1024
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MESSAGE_QUEUE = 256


class DuplexOmniSessionConfig(BaseModel):
    """Application policy for one DuplexOmni WebSocket session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str = "DuplexOmni"
    system_prompt: str = (
        "你是视频通话助手\n你的助手风格是：控场型。\n"
        "你的开场方式要求：先短答不展开，耐心听用户说完所有选项再回答。"
    )
    from_s2: str = ""
    input_sample_rate: int = INPUT_SAMPLE_RATE
    max_inflight_per_session: int = Field(default=4, ge=1, le=32)
    slot_overlap: bool = True
    # A warmed one-user AV run first missed the 480 ms slot deadline near
    # 7.2k Thinker tokens.  Compact at 6k to retain measured headroom instead
    # of waiting for the model's much larger correctness-only context limit.
    context_window_trigger_tokens: int = Field(default=6144, ge=1024)
    max_tokens: int = Field(default=999, ge=1)
    enable_frame_filter: bool = True
    frame_filter_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    frame_filter_min_gap: int = Field(default=0, ge=0)
    frame_filter_max_gap: int = Field(default=4, ge=0)
    max_frame_width: int | None = Field(default=640, ge=32, le=8192)
    max_frame_height: int | None = Field(default=360, ge=32, le=8192)
    frame_jpeg_quality: int = Field(default=90, ge=1, le=100)
    # Diagnostic-only control used by the interference benchmark.  The
    # request materializes Thinker KV and then stops; it never reaches the
    # Talker or Code2Wav stages.
    benchmark_prefill_only: bool = False

    @model_validator(mode="after")
    def _validate_session(self) -> DuplexOmniSessionConfig:
        if self.input_sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError(f"DuplexOmni requires {INPUT_SAMPLE_RATE} Hz PCM input")
        if (self.max_frame_width is None) != (self.max_frame_height is None):
            raise ValueError("max_frame_width and max_frame_height must be set together")
        return self


@dataclass
class _CompletedTurn:
    user_message: dict[str, Any]
    assistant_text: str
    codec_codes: list[list[int]] = field(default_factory=list)
    valid_turn: bool = False


@dataclass
class _SlotInput:
    index: int
    audio_pcm: bytes
    image_jpeg: bytes | None
    input_ready: float
    final: bool
    client_slot: int | None


@dataclass
class _RunningSlot:
    slot_input: _SlotInput
    epoch: int
    epoch_slot: int
    turn: _CompletedTurn
    thinker_ready: asyncio.Future[str]
    completion: asyncio.Task[None] | None = None
    prompt_tokens: int = 0
    prompt_render_ms: float = 0.0


@dataclass
class _DuplexCanonicalPromptState:
    """Processed append-only blocks for one application session."""

    prefix_block: dict[str, Any]
    turn_blocks: list[dict[str, Any]]
    generation_suffix: tuple[int, ...]


@dataclass(frozen=True)
class _DuplexCanonicalTicket:
    user_block: dict[str, Any]
    history_turns: int


def _wav_b64(pcm: bytes) -> str:
    """Wrap one PCM16 mono slot in the minimal WAV container expected by HF."""
    import io
    import wave

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(INPUT_SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _user_message(slot: _SlotInput, from_s2: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "{'audio_input': '"},
        {
            "type": "input_audio",
            "input_audio": {"data": _wav_b64(slot.audio_pcm), "format": "wav"},
            "uuid": f"duplex-audio:{hashlib.sha256(slot.audio_pcm).hexdigest()}",
        },
    ]
    if slot.image_jpeg is None:
        content.append({"type": "text", "text": f"', 'from_s2': '{from_s2}'}}"})
    else:
        image_b64 = base64.b64encode(slot.image_jpeg).decode("ascii")
        content.extend(
            [
                {"type": "text", "text": f"', 'from_s2': '{from_s2}', 'input_video': '"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    "uuid": f"duplex-image:{hashlib.sha256(slot.image_jpeg).hexdigest()}",
                },
                {"type": "text", "text": "'}"},
            ]
        )
    return {"role": "user", "content": content}


def _messages(
    config: DuplexOmniSessionConfig,
    history: list[_CompletedTurn],
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": config.system_prompt}]
    for turn in history:
        messages.extend(
            [
                turn.user_message,
                {"role": "assistant", "content": turn.assistant_text},
            ]
        )
    messages.append(current)
    return messages


def _talker_history(turns: list[_CompletedTurn]) -> dict[str, Any]:
    valid = [(index, turn) for index, turn in enumerate(turns) if turn.valid_turn]
    return {
        "codes": {"ref": [turn.codec_codes for _, turn in valid]},
        "ids": {"duplex_history_indices": [index for index, _ in valid]},
    }


def _pipeline_information(
    base_turns: list[_CompletedTurn],
    *,
    session_id: str,
    epoch: int,
    epoch_slot: int,
    final: bool,
) -> dict[str, Any]:
    info = _talker_history(base_turns)
    info["meta"] = {
        PIPELINE_SESSION_ID: session_id,
        PIPELINE_EPOCH: epoch,
        PIPELINE_SLOT: epoch_slot,
        PIPELINE_FINAL: final,
        PIPELINE_BASE_TURNS: len(base_turns),
    }
    return info


def _merge_metrics(target: dict[str, Any], incoming: Any) -> None:
    if not isinstance(incoming, dict):
        return
    stage_metrics = incoming.get("stage_metrics")
    if isinstance(stage_metrics, dict):
        target.setdefault("stage_metrics", {}).update(stage_metrics)
    for key, value in incoming.items():
        if key != "stage_metrics":
            target[key] = value


def _duplex_result(metrics: Mapping[str, Any]) -> tuple[list[list[int]], bool, bool]:
    raw = metrics.get("duplexomni")
    if not isinstance(raw, Mapping):
        return [], False, False
    raw_codes = raw.get("codec_codes")
    codes = (
        [[int(token) for token in row] for row in raw_codes]
        if isinstance(raw_codes, list) and all(isinstance(row, list) for row in raw_codes)
        else []
    )
    return codes, bool(raw.get("valid_turn")), bool(raw.get("eos_emitted"))


class DuplexOmniStreamingVideoHandler(OmniStreamingVideoHandler):
    """Own one finite-request DuplexOmni application session per WebSocket."""

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        super().__init__(
            chat_service=chat_service,
            idle_timeout=idle_timeout,
            config_timeout=config_timeout,
            engine_client=engine_client,
        )

    async def _receive_config(self, websocket: WebSocket) -> DuplexOmniSessionConfig | None:
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._config_timeout)
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None
        if not isinstance(message, dict) or message.get("type") != "session.config":
            await self._send_error(websocket, "Expected session.config")
            return None
        try:
            return DuplexOmniSessionConfig.model_validate(
                {key: value for key, value in message.items() if key != "type"}
            )
        except ValidationError as exc:
            await self._send_error(websocket, f"Invalid session config: {exc}")
            return None

    @staticmethod
    def _chat_request(
        config: DuplexOmniSessionConfig,
        *,
        messages: list[dict[str, Any]],
        cache_salt: str | None = None,
        additional_information: dict[str, Any] | None = None,
        add_generation_prompt: bool = True,
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=config.model,
            messages=messages,
            modalities=["text", "audio"],
            audio={"voice": "Ethan", "format": "wav"},
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
            max_tokens=config.max_tokens,
            cache_salt=cache_salt,
            additional_information=additional_information,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=False,
            add_special_tokens=False,
            return_stage_metrics=True,
        )

    def _supports_canonical_prompt(self) -> bool:
        return hasattr(self._chat_service, "renderer") and callable(
            getattr(self._chat_service, "create_chat_completion_from_engine_prompt", None)
        )

    async def _render_canonical_block(
        self,
        config: DuplexOmniSessionConfig,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> dict[str, Any]:
        request = self._chat_request(
            config,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )
        prompt = await self._preprocess_to_engine_prompt(request)
        if not self._is_mergeable_engine_prompt(prompt):
            raise ValueError("DuplexOmni renderer did not return a mergeable token prompt")
        return prompt

    async def _initialize_canonical_prompt(
        self,
        config: DuplexOmniSessionConfig,
        history: list[_CompletedTurn],
    ) -> _DuplexCanonicalPromptState:
        if config.system_prompt:
            prefix_block = await self._render_canonical_block(
                config,
                [{"role": "system", "content": config.system_prompt}],
                add_generation_prompt=False,
            )
        else:
            prefix_block = {"type": "token", "prompt_token_ids": []}
        probe = [{"role": "user", "content": "canonical-boundary-probe"}]
        probe_without_generation = await self._render_canonical_block(
            config,
            probe,
            add_generation_prompt=False,
        )
        probe_with_generation = await self._render_canonical_block(
            config,
            probe,
            add_generation_prompt=True,
        )
        probe_prefix = probe_without_generation["prompt_token_ids"]
        probe_full = probe_with_generation["prompt_token_ids"]
        if probe_full[: len(probe_prefix)] != probe_prefix:
            raise ValueError("DuplexOmni chat template is not append-only")
        generation_suffix = tuple(probe_full[len(probe_prefix) :])
        if not generation_suffix:
            raise ValueError("DuplexOmni chat template has no generation suffix")

        turn_blocks: list[dict[str, Any]] = []
        for turn in history:
            user_block = await self._render_canonical_block(
                config,
                [turn.user_message],
                add_generation_prompt=False,
            )
            assistant_block = await self._render_canonical_block(
                config,
                [{"role": "assistant", "content": turn.assistant_text}],
                add_generation_prompt=False,
            )
            turn_blocks.append(self._merge_engine_prompt_blocks([user_block, assistant_block]))
        return _DuplexCanonicalPromptState(
            prefix_block=prefix_block,
            turn_blocks=turn_blocks,
            generation_suffix=generation_suffix,
        )

    async def _render_canonical_prompt(
        self,
        config: DuplexOmniSessionConfig,
        state: _DuplexCanonicalPromptState,
        history: list[_CompletedTurn],
        current_user_message: dict[str, Any],
        *,
        cache_salt: str,
        additional_information: dict[str, Any],
    ) -> tuple[dict[str, Any], _DuplexCanonicalTicket]:
        if len(state.turn_blocks) != len(history):
            raise ValueError(
                "DuplexOmni canonical history is not committed: "
                f"blocks={len(state.turn_blocks)}, turns={len(history)}"
            )
        current_block = await self._render_canonical_block(
            config,
            [current_user_message],
            add_generation_prompt=True,
        )
        user_block = self._strip_generation_suffix(current_block, state.generation_suffix)
        prompt = self._merge_engine_prompt_blocks(
            [state.prefix_block, *state.turn_blocks, current_block]
        )
        # Preserve renderer-owned fields such as the requested speaker, then
        # attach the authoritative application pipeline state.
        prompt_info = dict(current_block.get("additional_information") or {})
        prompt_info.update(additional_information)
        prompt["additional_information"] = prompt_info
        prompt["cache_salt"] = cache_salt
        return prompt, _DuplexCanonicalTicket(
            user_block=user_block,
            history_turns=len(history),
        )

    async def _commit_canonical_turn(
        self,
        config: DuplexOmniSessionConfig,
        state: _DuplexCanonicalPromptState,
        ticket: _DuplexCanonicalTicket,
        turn: _CompletedTurn,
    ) -> None:
        if len(state.turn_blocks) != ticket.history_turns:
            raise ValueError("DuplexOmni canonical turn committed out of order")
        assistant_block = await self._render_canonical_block(
            config,
            [{"role": "assistant", "content": turn.assistant_text}],
            add_generation_prompt=False,
        )
        state.turn_blocks.append(
            self._merge_engine_prompt_blocks([ticket.user_block, assistant_block])
        )

    async def handle_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        config = await self._receive_config(websocket)
        if config is None:
            return
        session_id = config.session_id or f"duplexomni-{uuid.uuid4().hex[:12]}"
        send_lock = asyncio.Lock()

        async def emit(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        await emit(
            {
                "type": "session.ready",
                "session_id": session_id,
                "slot_ms": SLOT_MS,
                "input_sample_rate": INPUT_SAMPLE_RATE,
            }
        )

        inputs: asyncio.Queue[_SlotInput | None] = asyncio.Queue(maxsize=_MAX_MESSAGE_QUEUE)
        audio_buffer = bytearray()
        pending_image: bytes | None = None
        input_final = False
        next_slot = 0
        frame_filter = (
            FrameSimilarityFilter(threshold=config.frame_filter_threshold)
            if config.enable_frame_filter
            else None
        )
        frames_since_retained = 0

        history: list[_CompletedTurn] = []
        epoch = 0
        epoch_slot = 0
        base_turns = 0
        last_prompt_tokens = 0
        active: deque[_RunningSlot] = deque()
        scheduler_failed: BaseException | None = None
        canonical_enabled = self._supports_canonical_prompt()
        canonical_state: _DuplexCanonicalPromptState | None = None

        async def finish_running(item: _RunningSlot) -> None:
            nonlocal last_prompt_tokens
            if item.completion is None:
                return
            await item.completion
            last_prompt_tokens = max(last_prompt_tokens, item.prompt_tokens)

        async def drain_finished() -> None:
            while active and active[0].completion is not None and active[0].completion.done():
                item = active.popleft()
                await finish_running(item)

        async def drain_all() -> None:
            while active:
                await finish_running(active.popleft())

        async def run_slot(
            running: _RunningSlot,
            request: ChatCompletionRequest,
            submitted: float,
            engine_prompt: dict[str, Any] | None,
            canonical_ticket: _DuplexCanonicalTicket | None,
        ) -> None:
            nonlocal canonical_state
            text_parts: list[str] = []
            metrics: dict[str, Any] = {}
            usage: dict[str, Any] = {}
            request_id: str | None = None
            thinker_at: float | None = None
            audio_chunks = 0

            async def mark_thinker_ready() -> None:
                nonlocal thinker_at, canonical_state
                if thinker_at is not None:
                    return
                thinker_at = time.monotonic()
                running.turn.assistant_text = "".join(text_parts).strip()
                if canonical_ticket is not None and canonical_state is not None:
                    try:
                        await self._commit_canonical_turn(
                            config,
                            canonical_state,
                            canonical_ticket,
                            running.turn,
                        )
                    except Exception:
                        canonical_state = None
                        logger.warning(
                            "DuplexOmni canonical commit failed; rebuilding on the next slot",
                            exc_info=True,
                        )
                if not running.thinker_ready.done():
                    running.thinker_ready.set_result(running.turn.assistant_text)
                await emit(
                    {
                        "type": "response.thinker.done",
                        "slot": running.slot_input.index,
                        "text": running.turn.assistant_text,
                        "latency_ms": (thinker_at - submitted) * 1000.0,
                    }
                )

            try:
                if config.benchmark_prefill_only:
                    if self._engine_client is None:
                        raise RuntimeError("prefill-only benchmark requires an engine client")
                    if engine_prompt is None:
                        engine_prompt = await self._preprocess_to_engine_prompt(request)
                    engine_prompt = dict(engine_prompt)
                    engine_prompt["prefill_only"] = True
                    prompt_token_ids = engine_prompt.get("prompt_token_ids")
                    if isinstance(prompt_token_ids, list):
                        running.prompt_tokens = len(prompt_token_ids)
                    request_id = (
                        f"duplex-prefill-{session_id}-{running.slot_input.index}-"
                        f"{uuid.uuid4().hex[:8]}"
                    )
                    outputs = self._engine_client.generate(
                        prompt=engine_prompt,
                        request_id=request_id,
                        output_modalities=["text"],
                    )
                    async for output in outputs:
                        output_request_id = getattr(output, "request_id", None)
                        if isinstance(output_request_id, str):
                            request_id = output_request_id
                    completed = time.monotonic()
                    await mark_thinker_ready()
                    await emit(
                        {
                            "type": "response.done",
                            "slot": running.slot_input.index,
                            "client_slot": running.slot_input.client_slot,
                            "request_id": request_id,
                            "text": "",
                            "audio_chunks": 0,
                            "valid_turn": False,
                            "eos_emitted": False,
                            "codec_shape": [0, 0],
                            "cache_epoch": running.epoch,
                            "epoch_slot": running.epoch_slot,
                            "usage": {"prompt_tokens": running.prompt_tokens},
                            "metrics": {"benchmark_prefill_only": True},
                            "timing": {
                                "app_queue_ms": (submitted - running.slot_input.input_ready) * 1000.0,
                                "prompt_render_ms": running.prompt_render_ms,
                                "thinker_latency_ms": (thinker_at - submitted) * 1000.0,
                                "request_latency_ms": (completed - submitted) * 1000.0,
                                "e2e_slot_latency_ms": (
                                    completed - running.slot_input.input_ready
                                )
                                * 1000.0,
                            },
                        }
                    )
                    return
                if engine_prompt is not None:
                    result = await self._chat_service.create_chat_completion_from_engine_prompt(
                        request,
                        engine_prompt,
                        raw_request=None,
                    )
                else:
                    result = await self._chat_service.create_chat_completion(request, raw_request=None)
                if isinstance(result, ErrorResponse):
                    message = result.error.message if result.error is not None else "Unknown model error"
                    raise RuntimeError(message)
                if not hasattr(result, "__aiter__"):
                    raise RuntimeError("DuplexOmni session requires a streaming Chat Completion response")
                generator: AsyncGenerator[str, None] = result
                async for chunk in generator:
                    if not isinstance(chunk, str):
                        continue
                    for line in chunk.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if not body or body == "[DONE]":
                            continue
                        event = json.loads(body)
                        if isinstance(event.get("id"), str):
                            request_id = event["id"]
                        modality = event.get("modality")
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if not isinstance(content, str) or not content:
                                continue
                            if modality == "audio":
                                audio_chunks += 1
                                await emit(
                                    {
                                        "type": "response.audio.delta",
                                        "slot": running.slot_input.index,
                                        "data": content,
                                        "format": "wav",
                                    }
                                )
                            elif modality == "text":
                                text_parts.append(content)
                                await emit(
                                    {
                                        "type": "response.text.delta",
                                        "slot": running.slot_input.index,
                                        "delta": content,
                                    }
                                )
                        incoming_metrics = event.get("metrics")
                        _merge_metrics(metrics, incoming_metrics)
                        if isinstance(event.get("usage"), dict):
                            usage = event["usage"]
                        if (
                            thinker_at is None
                            and modality == "text"
                            and isinstance(incoming_metrics, dict)
                            and incoming_metrics.get("stage_id") == 0
                        ):
                            await mark_thinker_ready()

                completed = time.monotonic()
                if thinker_at is None:
                    await mark_thinker_ready()

                codes, valid_turn, eos_emitted = _duplex_result(metrics)
                running.turn.codec_codes = codes
                running.turn.valid_turn = valid_turn
                running.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                await emit(
                    {
                        "type": "response.done",
                        "slot": running.slot_input.index,
                        "client_slot": running.slot_input.client_slot,
                        "request_id": request_id,
                        "text": running.turn.assistant_text,
                        "audio_chunks": audio_chunks,
                        "valid_turn": valid_turn,
                        "eos_emitted": eos_emitted,
                        "codec_shape": [len(codes), len(codes[0]) if codes else 0],
                        "cache_epoch": running.epoch,
                        "epoch_slot": running.epoch_slot,
                        "usage": usage,
                        "metrics": metrics,
                        "timing": {
                            "app_queue_ms": (submitted - running.slot_input.input_ready) * 1000.0,
                            "prompt_render_ms": running.prompt_render_ms,
                            "thinker_latency_ms": (thinker_at - submitted) * 1000.0,
                            "request_latency_ms": (completed - submitted) * 1000.0,
                            "e2e_slot_latency_ms": (completed - running.slot_input.input_ready) * 1000.0,
                        },
                    }
                )
            except BaseException as exc:
                if not running.thinker_ready.done():
                    running.thinker_ready.set_exception(exc)
                await emit(
                    {
                        "type": "response.error",
                        "slot": running.slot_input.index,
                        "message": str(exc),
                    }
                )
                raise

        async def scheduler() -> None:
            nonlocal epoch, epoch_slot, base_turns, last_prompt_tokens, scheduler_failed, canonical_state
            previous: _RunningSlot | None = None
            try:
                while True:
                    slot_input = await inputs.get()
                    if slot_input is None:
                        break
                    if previous is not None:
                        await previous.thinker_ready
                    await drain_finished()
                    if not config.slot_overlap:
                        await drain_all()
                    while len(active) >= config.max_inflight_per_session:
                        await finish_running(active.popleft())
                        await drain_finished()

                    if last_prompt_tokens >= config.context_window_trigger_tokens and len(history) > 1:
                        await drain_all()
                        before = len(history)
                        history[:] = history[-1:]
                        if canonical_state is not None:
                            if len(canonical_state.turn_blocks) != before:
                                canonical_state = None
                            else:
                                canonical_state.turn_blocks[:] = canonical_state.turn_blocks[-1:]
                        epoch += 1
                        epoch_slot = 0
                        base_turns = len(history)
                        last_prompt_tokens = 0
                        await emit(
                            {
                                "type": "session.history.compacted",
                                "after_slot": slot_input.index - 1,
                                "slots_before": before,
                                "slots_after": len(history),
                                "cache_epoch": epoch,
                            }
                        )

                    user_message = _user_message(slot_input, config.from_s2)
                    turn = _CompletedTurn(user_message=user_message, assistant_text="")
                    loop = asyncio.get_running_loop()
                    running = _RunningSlot(
                        slot_input=slot_input,
                        epoch=epoch,
                        epoch_slot=epoch_slot,
                        turn=turn,
                        thinker_ready=loop.create_future(),
                    )
                    cache_salt = f"{session_id}:epoch-{epoch}"
                    additional_information = _pipeline_information(
                        history[:base_turns],
                        session_id=session_id,
                        epoch=epoch,
                        epoch_slot=epoch_slot,
                        final=slot_input.final,
                    )
                    messages = _messages(config, history, user_message)
                    submitted = time.monotonic()
                    render_started = time.monotonic()
                    engine_prompt: dict[str, Any] | None = None
                    canonical_ticket: _DuplexCanonicalTicket | None = None
                    if canonical_enabled:
                        try:
                            if canonical_state is None:
                                canonical_state = await self._initialize_canonical_prompt(config, history)
                            engine_prompt, canonical_ticket = await self._render_canonical_prompt(
                                config,
                                canonical_state,
                                history,
                                user_message,
                                cache_salt=cache_salt,
                                additional_information=additional_information,
                            )
                        except Exception:
                            canonical_state = None
                            logger.warning(
                                "DuplexOmni canonical prompt unavailable; using one full render",
                                exc_info=True,
                            )
                    running.prompt_render_ms = (time.monotonic() - render_started) * 1000.0
                    request = self._chat_request(
                        config,
                        messages=messages,
                        cache_salt=cache_salt,
                        additional_information=additional_information,
                    )
                    await emit(
                        {
                            "type": "response.start",
                            "slot": slot_input.index,
                            "client_slot": slot_input.client_slot,
                            "cache_epoch": epoch,
                            "epoch_slot": epoch_slot,
                            "app_queue_ms": (submitted - slot_input.input_ready) * 1000.0,
                            "overlapped_predecessor_audio": bool(
                                active and active[-1].completion is not None and not active[-1].completion.done()
                            ),
                        }
                    )
                    task = asyncio.create_task(
                        run_slot(
                            running,
                            request,
                            submitted,
                            engine_prompt,
                            canonical_ticket,
                        ),
                        name=f"duplexomni-{session_id}-{slot_input.index}",
                    )
                    running.completion = task
                    active.append(running)
                    history.append(turn)
                    previous = running
                    epoch_slot += 1
                if previous is not None:
                    await previous.thinker_ready
                await drain_all()
            except BaseException as exc:
                scheduler_failed = exc
                for running in active:
                    if running.completion is not None and not running.completion.done():
                        running.completion.cancel()
                if active:
                    await asyncio.gather(
                        *(running.completion for running in active if running.completion is not None),
                        return_exceptions=True,
                    )

        scheduler_task = asyncio.create_task(scheduler(), name=f"duplexomni-session-{session_id}")

        async def seal_audio(*, final: bool, client_slot: int | None) -> list[int]:
            nonlocal next_slot, pending_image
            sealed: list[int] = []
            while len(audio_buffer) >= SLOT_PCM_BYTES:
                pcm = bytes(audio_buffer[:SLOT_PCM_BYTES])
                del audio_buffer[:SLOT_PCM_BYTES]
                is_final = final and not audio_buffer
                slot = _SlotInput(
                    index=next_slot,
                    audio_pcm=pcm,
                    image_jpeg=pending_image,
                    input_ready=time.monotonic(),
                    final=is_final,
                    client_slot=client_slot if len(sealed) == 0 else None,
                )
                pending_image = None
                await inputs.put(slot)
                sealed.append(next_slot)
                next_slot += 1
            if final and audio_buffer:
                pcm = bytes(audio_buffer) + bytes(SLOT_PCM_BYTES - len(audio_buffer))
                audio_buffer.clear()
                slot = _SlotInput(
                    index=next_slot,
                    audio_pcm=pcm,
                    image_jpeg=pending_image,
                    input_ready=time.monotonic(),
                    final=True,
                    client_slot=client_slot if not sealed else None,
                )
                pending_image = None
                await inputs.put(slot)
                sealed.append(next_slot)
                next_slot += 1
            return sealed

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=self._idle_timeout)
                except asyncio.TimeoutError:
                    await emit({"type": "error", "message": "Idle timeout"})
                    break
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await emit({"type": "error", "message": "Invalid JSON"})
                    continue
                if not isinstance(message, dict):
                    await emit({"type": "error", "message": "Messages must be JSON objects"})
                    continue
                message_type = message.get("type")

                if message_type == "video.frame":
                    data = message.get("data")
                    if not isinstance(data, str):
                        await emit({"type": "error", "message": "video.frame requires base64 data"})
                        continue
                    try:
                        frame = base64.b64decode(data, validate=True)
                    except (ValueError, binascii.Error):
                        await emit({"type": "error", "message": "Invalid image data"})
                        continue
                    if len(frame) > _MAX_FRAME_BYTES:
                        await emit({"type": "error", "message": "Frame too large"})
                        continue
                    if config.max_frame_width is not None and config.max_frame_height is not None:
                        try:
                            resized = await asyncio.to_thread(
                                _downscale_frame_bytes,
                                frame,
                                config.max_frame_width,
                                config.max_frame_height,
                                config.frame_jpeg_quality,
                            )
                            if resized is not None:
                                frame = resized
                        except Exception:
                            await emit({"type": "error", "message": "Invalid image data"})
                            continue
                    accepted = True
                    if frame_filter is not None:
                        try:
                            different = frame_filter.should_retain(frame)
                            below_min = (
                                config.frame_filter_min_gap > 0
                                and frames_since_retained < config.frame_filter_min_gap
                            )
                            force_fresh = (
                                config.frame_filter_max_gap > 0
                                and frames_since_retained >= config.frame_filter_max_gap
                            )
                            accepted = (different and not below_min) or force_fresh
                        except Exception:
                            await emit({"type": "error", "message": "Invalid image data"})
                            continue
                    if accepted:
                        pending_image = frame
                        frames_since_retained = 0
                    else:
                        frames_since_retained += 1
                    await emit(
                        {
                            "type": "video.frame.ack",
                            "frame_id": message.get("frame_id"),
                            "accepted": accepted,
                            "pending_for_next_slot": pending_image is not None,
                        }
                    )

                elif message_type == "audio.chunk":
                    if input_final:
                        await emit({"type": "error", "message": "Audio input is already final"})
                        continue
                    data = message.get("data")
                    if not isinstance(data, str):
                        await emit({"type": "error", "message": "audio.chunk requires base64 PCM16 data"})
                        continue
                    try:
                        pcm = base64.b64decode(data, validate=True)
                    except (ValueError, binascii.Error):
                        await emit({"type": "error", "message": "Invalid audio data"})
                        continue
                    if len(pcm) % 2:
                        await emit({"type": "error", "message": "PCM16 audio must contain complete samples"})
                        continue
                    if len(audio_buffer) + len(pcm) > _MAX_AUDIO_BUFFER_BYTES:
                        await emit({"type": "error", "message": "Audio buffer overflow"})
                        continue
                    audio_buffer.extend(pcm)
                    final = bool(message.get("final", False))
                    input_final = final
                    client_slot = message.get("slot")
                    if not isinstance(client_slot, int):
                        client_slot = None
                    sealed = await seal_audio(final=final, client_slot=client_slot)
                    await emit(
                        {
                            "type": "audio.chunk.ack",
                            "sealed_slots": sealed,
                            "buffered_bytes": len(audio_buffer),
                            "final": final,
                        }
                    )

                elif message_type in {"session.finish", "video.done"}:
                    if not input_final:
                        input_final = True
                        await seal_audio(final=True, client_slot=None)
                    break

                elif message_type == "ping":
                    await emit({"type": "pong"})
                else:
                    await emit({"type": "error", "message": f"Unknown type: {message_type}"})
        except WebSocketDisconnect:
            logger.info("DuplexOmni session %s disconnected", session_id)
        finally:
            await inputs.put(None)
            await scheduler_task
            if scheduler_failed is None:
                try:
                    await emit(
                        {
                            "type": "session.done",
                            "session_id": session_id,
                            "slots": next_slot,
                            "context_compactions": epoch,
                        }
                    )
                except Exception:
                    pass
            else:
                try:
                    await emit({"type": "error", "message": f"Session pipeline failed: {scheduler_failed}"})
                except Exception:
                    pass

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass


__all__ = [
    "DuplexOmniSessionConfig",
    "DuplexOmniStreamingVideoHandler",
    "INPUT_SAMPLE_RATE",
    "SLOT_MS",
    "SLOT_PCM_BYTES",
]
