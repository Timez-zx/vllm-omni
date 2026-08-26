# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base WebSocket handler for streaming video input understanding.

Shared session loop, frame/audio buffering, EVS pre-filter, prewarm,
interrupt handling, and engine ``generate()`` streaming. Pipeline-specific
behavior (trigger rules, prompt shape, history) is supplied by subclasses
via :class:`VideoStreamPipelineHooks`.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}         # Session config (sent once)
        {"type": "video.frame", "data": "...", "frame_id": "...", "pts_ms": 0}
        {"type": "audio.chunk", "data": "..."}  # base64 PCM16 16kHz mono
        {"type": "video.query", "text": "..."}  # Submit query about buffered frames
        {"type": "video.done"}                  # End of session

    Server -> Client:
        {"type": "video.frame.ack", ...}          # when frame_id is provided
        {"type": "video.frames.consumed", ...}    # after first engine output
        {"type": "response.start"}
        {"type": "response.text.delta", "delta": "..."}
        {"type": "response.text.done", "text": "..."}
        {"type": "response.audio.delta", "data": "...", "format": "wav"}
        {"type": "response.audio.done"}
        {"type": "session.done"}
        {"type": "error", "message": "..."}
"""

import asyncio
import base64
import copy
import hashlib
import io
import json
import os
import threading
import time as _time
import uuid
import wave
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai import media_pipeline, video_stream_envs
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_LOG_HANDOFF_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "0") not in ("0", "", "false", "False")
_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MSG_QUEUE = 200
_CODEC_FRAME_SAMPLES = 1920  # CausalConv leading-edge artifact length
_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_INPUT_SAMPLE_WIDTH = 2
_AUDIO_ARRIVAL_CHUNK_MS = 1000
_AUDIO_ARRIVAL_CHUNK_BYTES = _AUDIO_INPUT_SAMPLE_RATE * _AUDIO_INPUT_SAMPLE_WIDTH * _AUDIO_ARRIVAL_CHUNK_MS // 1000
_BAD_FRAME = object()
_THINKER_LINEAGE_TICKET_KEY = "_vllm_omni_app_thinker_lineage_ticket"
_MM_RENDER_OWNER: ContextVar[str] = ContextVar("vllm_omni_mm_render_owner", default="-")

# Keep JPEG decode/resize work outside the WebSocket process's GIL. This is
# application plumbing, not engine state, and prevents media preprocessing
# from becoming the apparent serving tail at higher concurrency.
if media_pipeline.enabled():
    threading.Thread(target=media_pipeline.prewarm_pool, daemon=True).start()


def _decode_frame_bytes(raw_bytes: bytes) -> Any:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _downscale_frame_bytes(
    raw_bytes: bytes,
    max_width: int,
    max_height: int,
    jpeg_quality: int,
) -> bytes | None:
    """Downscale one frame without upscaling or changing an already-small frame."""
    image = _decode_frame_bytes(raw_bytes)
    width, height = image.size
    if width <= max_width and height <= max_height:
        return None
    scale = min(max_width / width, max_height / height)
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def _prompt_token_count(prompt: Any) -> int:
    if isinstance(prompt, Mapping):
        token_ids = prompt.get("prompt_token_ids")
    else:
        token_ids = getattr(prompt, "prompt_token_ids", None)
    return len(token_ids or ())


def _common_token_prefix(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _attach_thinker_lineage(
    config: "StreamingVideoSessionConfig",
    engine_prompt: Any,
) -> "_ThinkerLineageTicket | None":
    """Attach an optimization hint while retaining the complete prompt."""
    if not isinstance(engine_prompt, dict):
        return None
    raw_ids = engine_prompt.get("prompt_token_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(token, int) for token in raw_ids):
        return None

    prompt_ids = list(raw_ids)
    prefix_tokens = _common_token_prefix(config._thinker_lineage_token_ids, prompt_ids)
    parent_revision = config._thinker_lineage_revision
    revision = parent_revision + 1
    engine_prompt["kv_lineage_id"] = config._thinker_lineage_id
    engine_prompt["kv_lineage_parent_revision"] = parent_revision
    engine_prompt["kv_lineage_revision"] = revision
    engine_prompt["kv_lineage_prefix_tokens"] = prefix_tokens
    return _ThinkerLineageTicket(
        lineage_id=config._thinker_lineage_id,
        parent_revision=parent_revision,
        revision=revision,
        prompt_token_ids=tuple(prompt_ids),
        prefix_tokens=prefix_tokens,
    )


def _commit_thinker_lineage(
    config: "StreamingVideoSessionConfig",
    ticket: "_ThinkerLineageTicket | None",
    generated_token_ids: list[int] | None = None,
) -> bool:
    if (
        ticket is None
        or ticket.lineage_id != config._thinker_lineage_id
        or ticket.parent_revision != config._thinker_lineage_revision
    ):
        return False
    config._thinker_lineage_revision = ticket.revision
    config._thinker_lineage_token_ids = [
        *ticket.prompt_token_ids,
        *(generated_token_ids or ()),
    ]
    return True


def _reset_thinker_lineage(config: "StreamingVideoSessionConfig") -> None:
    """Start a new cache lineage after canonical history is rewritten."""
    config._thinker_lineage_id = f"thinker-session:{uuid.uuid4().hex}"
    config._thinker_lineage_revision = 0
    config._thinker_lineage_token_ids = []


def _pop_thinker_lineage_ticket(engine_prompt: Any) -> "_ThinkerLineageTicket | None":
    if not isinstance(engine_prompt, dict):
        return None
    ticket = engine_prompt.pop(_THINKER_LINEAGE_TICKET_KEY, None)
    return ticket if isinstance(ticket, _ThinkerLineageTicket) else None


@runtime_checkable
class VideoStreamPipelineHooks(Protocol):
    """Pipeline-specific hooks for streaming video handlers."""

    def should_trigger_turn(self, trigger: "VideoStreamTurnTrigger") -> bool:
        """Return True to auto-start a turn after a new frame (no ``video.query``)."""
        ...

    def build_engine_prompt(
        self,
        config: "StreamingVideoSessionConfig",
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        media_events: list["_TurnMediaEvent"] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build OpenAI-style messages and the current user message."""
        ...

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        """Update session state after a successful turn."""
        ...


@dataclass(frozen=True)
class VideoStreamTurnTrigger:
    """Snapshot passed to :meth:`OmniStreamingVideoHandler.should_trigger_turn`."""

    frame_count: int
    is_generating: bool
    config: "StreamingVideoSessionConfig"


@dataclass(frozen=True)
class _TurnMediaEvent:
    """One immutable, append-only media item in the current user turn."""

    modality: str
    payload: str | bytes


@dataclass(frozen=True)
class _ThinkerLineageTicket:
    lineage_id: str
    parent_revision: int
    revision: int
    prompt_token_ids: tuple[int, ...]
    prefix_tokens: int


@dataclass
class _CanonicalPromptState:
    """Application-owned, processed canonical history for one live session.

    Blocks are ordinary renderer outputs, not engine requests or KV handles.
    Keeping them here avoids decoding and processing every historical media
    item again when the next finite request is constructed.
    """

    signature: tuple[Any, ...]
    prefix_block: dict[str, Any]
    turn_blocks: list[dict[str, Any]]
    turn_message_ids: list[tuple[int, int]]
    generation_suffix: tuple[int, ...]


@dataclass(frozen=True)
class _CanonicalRenderTicket:
    user_block: dict[str, Any]
    history_turns: int


_CANONICAL_RENDER_TICKET_KEY = "_vllm_omni_app_canonical_render_ticket"


def _record_render_diag(
    diag: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    """Accumulate one frontend prompt-construction wall-time component."""
    if diag is not None:
        diag[key] = diag.get(key, 0.0) + (_time.monotonic() - started) * 1000.0


class StreamingVideoSessionConfig(BaseModel):
    """Application session policy; engine requests remain finite per turn."""

    model_config = ConfigDict(extra="forbid")

    # Stable only for this application-session object.  It isolates Talker KV
    # lineages without making the engine request itself session-aware.
    _talker_cache_salt: str = PrivateAttr(default_factory=lambda: f"video-session:{uuid.uuid4().hex}")
    _thinker_lineage_id: str = PrivateAttr(default_factory=lambda: f"thinker-session:{uuid.uuid4().hex}")
    _thinker_lineage_revision: int = PrivateAttr(default=0)
    _thinker_lineage_token_ids: list[int] = PrivateAttr(default_factory=list)
    _canonical_prompt_state: _CanonicalPromptState | None = PrivateAttr(default=None)
    _history_compaction_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = None
    modalities: list[str] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    enable_video_arrival_prefill: bool = Field(
        default=True,
        description=(
            "Materialize the cumulative accepted-frame prefix with silent, "
            "Thinker-only finite requests as frames arrive."
        ),
    )
    enable_audio_arrival_prefill_approximation: bool = Field(
        default=False,
        description=(
            "Approximate streaming audio with immutable one-second chunks. "
            "Arrival requests only populate Thinker caches; only video.query "
            "may decode a response or invoke Talker. This is not bit-equivalent "
            "to Qwen's full-utterance audio encoder."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt.",
    )
    use_audio_in_video: bool = Field(
        default=True,
        description="Interleave audio chunks with video frames when audio input is present.",
    )
    sampling_params_list: list[dict[str, Any]] | None = Field(
        default=None,
        description="Per-stage sampling params [thinker, talker, code2wav].",
    )
    thinker_max_response_tokens: int = Field(
        default=256,
        ge=1,
        description=(
            "Hard cap for a normal Thinker response. Live voice turns must be "
            "bounded even when the model ignores the short-answer system prompt."
        ),
    )
    enable_frame_filter: bool = Field(
        default=True,
        description="EVS pixel-similarity pre-filter to drop near-duplicate frames.",
    )
    frame_filter_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="EVS similarity threshold (higher = keep more frames).",
    )
    frame_filter_min_gap: int = Field(
        default=0,
        ge=0,
        description="Minimum arrivals between retained frames; zero disables the floor.",
    )
    frame_filter_max_gap: int = Field(
        default=4,
        ge=0,
        description="Force retention after this many filtered frames; zero disables it.",
    )
    max_frame_width: int | None = Field(default=640, ge=32, le=8192)
    max_frame_height: int | None = Field(default=352, ge=32, le=8192)
    frame_jpeg_quality: int = Field(default=90, ge=1, le=100)
    context_window_trigger_tokens: int = Field(
        default=49152,
        ge=1024,
        description=(
            "When a rendered prompt reaches this size, drop the oldest complete "
            "turns and retain only the configured recent turns plus the current turn."
        ),
    )
    context_window_retained_turns: int = Field(
        default=2,
        ge=0,
        description=(
            "Newest completed audio/text turns retained at each context compaction. "
            "Historical images are discarded; the in-flight turn keeps only its "
            "newest accepted image."
        ),
    )

    @model_validator(mode="after")
    def _validate_context_window(self) -> "StreamingVideoSessionConfig":
        if (self.max_frame_width is None) != (self.max_frame_height is None):
            raise ValueError("max_frame_width and max_frame_height must be set together")
        return self


class OmniStreamingVideoHandler:
    """Base handler for WebSocket streaming video sessions.

    Subclasses implement :class:`VideoStreamPipelineHooks` to customize turn
    triggering, prompt construction, and history updates.
    """

    def should_trigger_turn(self, trigger: VideoStreamTurnTrigger) -> bool:
        """Auto-trigger after ``video.frame`` when True (default: never)."""
        return False

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
        raise NotImplementedError

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        raise NotImplementedError

    def supports_incremental_canonical_prompt(self) -> bool:
        """Whether processed chat-message blocks may be concatenated safely."""
        return False

    def _history_prefix_messages(
        self,
        config: StreamingVideoSessionConfig,
    ) -> list[dict[str, Any]]:
        """Return the stable system prefix owned by the application."""
        if not config.system_prompt:
            return []
        return [{"role": "system", "content": config.system_prompt}]

    def _normalize_engine_prompt_for_messages(
        self,
        config: StreamingVideoSessionConfig,
        messages: list[dict[str, Any]],
        prompt: dict[str, Any],
    ) -> dict[str, Any]:
        """Pipeline hook for semantics-preserving processed-prompt rewrites."""
        del config, messages
        return prompt

    def create_message_history(self, config: StreamingVideoSessionConfig) -> Any:
        """Per-session conversation state (default: empty OpenAI-style list)."""
        return []

    def on_frame_buffered(
        self,
        raw_bytes: bytes,
        frame_b64: str,
        message_history: Any,
        config: StreamingVideoSessionConfig,
    ) -> None:
        """Hook after a frame is accepted into the session buffer."""
        del raw_bytes, frame_b64, message_history, config

    def __init__(
        self,
        chat_service: Any,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
        engine_client: Any | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout
        self._engine_client = engine_client
        self._install_mm_processor_diag()

    def _install_mm_processor_diag(self) -> None:
        """Split frontend MM executor queueing from processor execution."""
        if not _LOG_HANDOFF_DIAG:
            return
        renderer = getattr(self._chat_service, "renderer", None)
        if renderer is None or getattr(renderer, "_omni_mm_processor_diag_installed", False):
            return

        from vllm.multimodal.processing import TimingContext

        original = renderer._process_multimodal
        executor = renderer._mm_executor
        timing_local = threading.local()

        class _ThreadLocalTimingRegistry:
            def get(self, _request_id: str) -> TimingContext:
                context = TimingContext(enabled=True)
                timing_local.context = context
                return context

            @staticmethod
            def pop() -> dict[str, float]:
                context = getattr(timing_local, "context", None)
                if context is None:
                    return {}
                del timing_local.context
                return context.get_stats_dict()

        registry = _ThreadLocalTimingRegistry()
        renderer._mm_timing_registry = registry
        next_call_id = 0

        def _process_multimodal_async(*args, **kwargs):
            nonlocal next_call_id
            next_call_id += 1
            call_id = next_call_id
            owner = _MM_RENDER_OWNER.get()
            submitted = _time.monotonic()
            loop = asyncio.get_event_loop()

            def _run():
                started = _time.monotonic()
                error = False
                try:
                    return original(*args, **kwargs)
                except BaseException:
                    error = True
                    raise
                finally:
                    finished = _time.monotonic()
                    raw_stats = registry.pop()
                    stage_ms = {
                        key.replace("_secs", "_ms"): value * 1000.0
                        for key, value in raw_stats.items()
                    }
                    mm_data = args[1] if len(args) > 1 and isinstance(args[1], Mapping) else {}
                    counts = {
                        modality: len(items) if hasattr(items, "__len__") else 1
                        for modality, items in mm_data.items()
                    }
                    logger.info(
                        "[MM-PROCESSOR-DIAG] id=%d owner=%s thread=%d "
                        "queue_ms=%.3f exec_ms=%.3f error=%s counts=%s stages=%s",
                        call_id,
                        owner,
                        threading.get_ident(),
                        (started - submitted) * 1000.0,
                        (finished - started) * 1000.0,
                        error,
                        json.dumps(counts, sort_keys=True, separators=(",", ":")),
                        json.dumps(stage_ms, sort_keys=True, separators=(",", ":")),
                    )

            return loop.run_in_executor(executor, _run)

        renderer._process_multimodal_async = _process_multimodal_async
        renderer._omni_mm_processor_diag_installed = True

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            frame_metadata: list[dict[str, Any]] = []
            # Per-frame PIL cache + uuid for mm_hash reuse. Aligned with frame_buffer by index.
            frame_pil_cache: dict[str, tuple[Any, str] | object] = {}  # b64 -> (PIL.Image, uuid) or _BAD_FRAME
            frame_filter = (
                FrameSimilarityFilter(threshold=config.frame_filter_threshold) if config.enable_frame_filter else None
            )
            frames_since_retained = 0
            audio_buffer = bytearray()  # raw PCM16 16kHz mono
            media_events: list[_TurnMediaEvent] = []
            audio_sealed_bytes = 0
            message_history: Any = self.create_message_history(config)
            active_request_id: str | None = None
            prev_request_id: str | None = None  # abort target iff prev was interrupted
            prev_was_interrupted: bool = False
            interrupt_event = asyncio.Event()
            prewarm_tasks: set[asyncio.Task[Any]] = set()
            query_task: asyncio.Task[Any] | None = None
            arrival_prefill_task: asyncio.Task[Any] | None = None
            arrival_prefill_dirty = False

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)

            async def _reader() -> None:
                """Receive WebSocket messages and enqueue them."""
                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(
                                websocket.receive_text(),
                                timeout=self._idle_timeout,
                            )
                        except asyncio.TimeoutError:
                            await self._send_error(websocket, "Idle timeout")
                            await msg_queue.put(None)
                            return

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            await self._send_error(websocket, "Invalid JSON")
                            continue

                        if not isinstance(msg, dict):
                            await self._send_error(websocket, "Messages must be JSON objects")
                            continue

                        msg_type = str(msg.get("type", ""))
                        if msg_type.startswith("_internal."):
                            await self._send_error(websocket, f"Unknown type: {msg_type}")
                            continue
                        if msg_type == "video.frame":
                            msg["_receiver_received_ts_ms"] = _time.monotonic() * 1000

                        await msg_queue.put(msg)
                        if msg.get("type") == "video.done":
                            return
                except WebSocketDisconnect:
                    await msg_queue.put(None)
                except Exception:
                    await msg_queue.put(None)
                    raise

            async def _cancel_active_query(*, abort_now: bool = False) -> None:
                """Signal soft interrupt for the active query."""
                nonlocal active_request_id, prev_was_interrupted, query_task
                if active_request_id is not None:
                    interrupt_event.set()
                    prev_was_interrupted = True
                    logger.info("Interrupt signaled for %s", active_request_id)
                    if abort_now and self._engine_client:
                        try:
                            await self._engine_client.abort(active_request_id)
                        except Exception:
                            logger.debug("Abort failed for %s", active_request_id, exc_info=True)
                    if query_task is not None and not query_task.done():
                        query_task.cancel()
                        await asyncio.gather(query_task, return_exceptions=True)
                    query_task = None

            def _schedule_arrival_prefill() -> None:
                """Coalesce append-only media into serial finite warm-up requests."""
                nonlocal arrival_prefill_task, arrival_prefill_dirty
                video_ready = config.enable_video_arrival_prefill and bool(frame_buffer)
                audio_ready = config.enable_audio_arrival_prefill_approximation and audio_sealed_bytes > 0
                if self._engine_client is None or active_request_id is not None or not (video_ready or audio_ready):
                    return
                arrival_prefill_dirty = True
                if arrival_prefill_task is not None and not arrival_prefill_task.done():
                    return

                async def _run() -> None:
                    nonlocal arrival_prefill_dirty
                    while arrival_prefill_dirty and active_request_id is None:
                        arrival_prefill_dirty = False
                        frames = list(frame_buffer)
                        audio = bytearray(audio_buffer[:audio_sealed_bytes])
                        events = list(media_events)
                        history = list(message_history)
                        cached = {frame: frame_pil_cache[frame] for frame in frames if frame in frame_pil_cache}
                        request_id = f"video-warm-{uuid.uuid4().hex[:12]}"
                        kwargs: dict[str, Any] = {}
                        if config.enable_audio_arrival_prefill_approximation:
                            kwargs["audio_buffer"] = audio
                            kwargs["media_events"] = events
                        await self._process_video_arrival_prefill(
                            config,
                            frames,
                            history,
                            request_id,
                            cached,
                            **kwargs,
                        )

                arrival_prefill_task = asyncio.create_task(
                    _run(),
                    name=f"arrival-prefill-{config.session_id or 'anonymous'}",
                )

            async def _finish_arrival_prefill(request_id: str) -> None:
                """Drain this session's one admitted arrival before its query."""
                nonlocal arrival_prefill_task, arrival_prefill_dirty
                started = _time.monotonic()
                arrival_prefill_dirty = False
                task = arrival_prefill_task
                warmup_was_running = task is not None and not task.done()
                if warmup_was_running:
                    await asyncio.gather(task, return_exceptions=True)
                arrival_prefill_task = None
                logger.info(
                    "[query-serial] session=%s request=%s warmup_running=%s wait_ms=%.1f",
                    config.session_id or "-",
                    request_id,
                    warmup_was_running,
                    (_time.monotonic() - started) * 1000.0,
                )

            async def _start_query_turn(*, query_text: str) -> None:
                """Schedule a new inference turn from the current buffers."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task
                nonlocal audio_sealed_bytes

                await _cancel_active_query()

                if not frame_buffer and not audio_buffer and not query_text:
                    await self._send_error(websocket, "No input buffered")
                    return

                request_id = f"video-{uuid.uuid4().hex[:12]}"
                await _finish_arrival_prefill(request_id)

                if prev_was_interrupted and prev_request_id and self._engine_client:
                    try:
                        await self._engine_client.abort(prev_request_id)
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)
                prev_was_interrupted = False

                active_request_id = request_id
                interrupt_event.clear()
                query_frames = list(frame_buffer)
                query_frame_metadata = list(frame_metadata)
                # A frame belongs to exactly one user turn. New arrivals during
                # generation accumulate independently for the following turn;
                # completed turns remain in the application-owned history.
                frame_buffer.clear()
                frame_metadata.clear()
                query_audio_buffer = bytearray(audio_buffer)
                audio_buffer.clear()
                query_media_events = list(media_events)
                if config.enable_audio_arrival_prefill_approximation:
                    tail = bytes(query_audio_buffer[audio_sealed_bytes:])
                    if tail:
                        query_media_events.append(_TurnMediaEvent("audio", tail))
                media_events.clear()
                audio_sealed_bytes = 0
                query_prewarmed_frames = {
                    frame: frame_pil_cache.pop(frame) for frame in query_frames if frame in frame_pil_cache
                }

                async def _run_query() -> None:
                    nonlocal active_request_id, prev_request_id
                    try:
                        process_kwargs: dict[str, Any] = {}
                        if any(metadata.get("frame_id") for metadata in query_frame_metadata):
                            process_kwargs["frame_metadata"] = query_frame_metadata
                        if config.enable_audio_arrival_prefill_approximation:
                            process_kwargs["media_events"] = query_media_events
                        await self._process_query(
                            websocket,
                            config,
                            query_frames,
                            query_audio_buffer,
                            message_history,
                            query_text,
                            request_id,
                            interrupt_event,
                            query_prewarmed_frames,
                            **process_kwargs,
                        )
                    finally:
                        if active_request_id == request_id:
                            prev_request_id = request_id
                            active_request_id = None
                        # Frames received while this response was generated now
                        # have a completed assistant turn in front of them and
                        # can start the next cache lineage immediately.
                        _schedule_arrival_prefill()

                query_task = asyncio.create_task(_run_query())

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task
                nonlocal frames_since_retained
                nonlocal audio_sealed_bytes

                while True:
                    msg = await msg_queue.get()
                    if msg is None:
                        await _cancel_active_query(abort_now=True)
                        return

                    msg_type = msg.get("type")

                    if msg_type == "_internal.frame_decode_failed":
                        frame_data = msg.get("b64", "")
                        removed = frame_data in frame_buffer
                        if removed:
                            retained_indices = [
                                index for index, frame in enumerate(frame_buffer) if frame != frame_data
                            ]
                            frame_buffer[:] = [frame_buffer[index] for index in retained_indices]
                            frame_metadata[:] = [frame_metadata[index] for index in retained_indices]
                            media_events[:] = [
                                event
                                for event in media_events
                                if not (event.modality == "image" and event.payload == frame_data)
                            ]
                        if frame_pil_cache.get(frame_data) is _BAD_FRAME:
                            frame_pil_cache.pop(frame_data, None)
                        if removed:
                            await self._send_error(websocket, "Frame decode failed")
                            _schedule_arrival_prefill()

                    elif msg_type == "video.frame":
                        frame_data = msg.get("data", "")
                        if not frame_data:
                            continue
                        if len(frame_data) > _MAX_FRAME_SIZE:
                            await self._send_error(websocket, "Frame too large")
                            continue
                        try:
                            raw_bytes = base64.b64decode(frame_data, validate=True)
                        except Exception:
                            await self._send_error(websocket, "Invalid image data")
                            continue
                        media_result = None
                        if media_pipeline.enabled():
                            try:
                                media_result = await media_pipeline.process_frame(
                                    raw_bytes,
                                    config.max_frame_width or 0,
                                    config.max_frame_height or 0,
                                    config.frame_jpeg_quality,
                                )
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                            if media_result.shrunk_jpeg is not None:
                                raw_bytes = media_result.shrunk_jpeg
                                frame_data = base64.b64encode(raw_bytes).decode("ascii")
                        elif config.max_frame_width and config.max_frame_height:
                            try:
                                resized = await asyncio.to_thread(
                                    _downscale_frame_bytes,
                                    raw_bytes,
                                    config.max_frame_width,
                                    config.max_frame_height,
                                    config.frame_jpeg_quality,
                                )
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                            if resized is not None:
                                raw_bytes = resized
                                frame_data = base64.b64encode(resized).decode("ascii")
                        if frame_filter is not None:
                            try:
                                different = (
                                    frame_filter.should_retain_thumb(media_result.thumb)
                                    if media_result is not None
                                    else frame_filter.should_retain(raw_bytes)
                                )
                                below_min_gap = (
                                    config.frame_filter_min_gap > 0
                                    and frames_since_retained < config.frame_filter_min_gap
                                )
                                force_fresh = (
                                    config.frame_filter_max_gap > 0
                                    and frames_since_retained >= config.frame_filter_max_gap
                                )
                                if (not different or below_min_gap) and not force_fresh:
                                    frames_since_retained += 1
                                    await self._send_frame_ack(
                                        websocket,
                                        msg,
                                        accepted=False,
                                        buffered_frames=len(frame_buffer),
                                        reason="filtered",
                                    )
                                    continue
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        frames_since_retained = 0
                        mm_uuid = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()
                        frame_buffer.append(frame_data)
                        if config.enable_audio_arrival_prefill_approximation:
                            media_events.append(_TurnMediaEvent("image", frame_data))
                        frame_metadata.append(
                            {
                                "frame_id": msg.get("frame_id"),
                                "pts_ms": msg.get("pts_ms"),
                                "source_pts_ms": msg.get("source_pts_ms"),
                                "quality_profile": msg.get("quality_profile"),
                                "capture_ts_ms": msg.get("capture_ts_ms"),
                                "receiver_received_ts_ms": msg.get("_receiver_received_ts_ms"),
                            }
                        )
                        self.on_frame_buffered(raw_bytes, frame_data, message_history, config)
                        await self._send_frame_ack(
                            websocket,
                            msg,
                            accepted=True,
                            buffered_frames=len(frame_buffer),
                        )
                        # Reuse the worker's single decode for prompt construction.
                        if media_result is not None:
                            frame_pil_cache[frame_data] = (
                                Image.frombytes("RGB", media_result.size, media_result.rgb),
                                mm_uuid,
                            )
                        elif frame_data not in frame_pil_cache:
                            # Publish the stable UUID before the asynchronous PIL
                            # decode completes. A warm-up may render immediately;
                            # image_url and image_pil must hash as the same item.
                            frame_pil_cache[frame_data] = (None, mm_uuid)

                            async def _prewarm(b64: str, b: bytes, u: str) -> None:
                                try:
                                    pil = await asyncio.to_thread(_decode_frame_bytes, b)
                                    frame_pil_cache[b64] = (pil, u)
                                except Exception:
                                    frame_pil_cache[b64] = _BAD_FRAME
                                    logger.warning("prewarm decode failed for frame (len=%d)", len(b))
                                    try:
                                        msg_queue.put_nowait({"type": "_internal.frame_decode_failed", "b64": b64})
                                    except asyncio.QueueFull:
                                        logger.warning(
                                            "frame decode failure event dropped because message queue is full"
                                        )
                                finally:
                                    # The query may have consumed the frame while
                                    # decoding was in flight. In that case its prompt
                                    # used the same base64 content directly and this
                                    # late render-cache entry must not accumulate.
                                    if b64 not in frame_buffer:
                                        frame_pil_cache.pop(b64, None)

                            task = asyncio.create_task(_prewarm(frame_data, raw_bytes, mm_uuid))
                            prewarm_tasks.add(task)
                            task.add_done_callback(prewarm_tasks.discard)

                        _schedule_arrival_prefill()

                        is_generating = active_request_id is not None or (
                            query_task is not None and not query_task.done()
                        )
                        if self.should_trigger_turn(
                            VideoStreamTurnTrigger(
                                frame_count=len(frame_buffer),
                                is_generating=is_generating,
                                config=config,
                            )
                        ):
                            await _start_query_turn(query_text="")

                    elif msg_type == "audio.chunk":
                        data_b64 = msg.get("data", "")
                        try:
                            pcm_bytes = base64.b64decode(data_b64)
                        except Exception:
                            continue
                        if len(audio_buffer) + len(pcm_bytes) > _MAX_AUDIO_BUFFER_BYTES:
                            await self._send_error(websocket, "Audio buffer overflow")
                            audio_buffer.clear()
                            audio_sealed_bytes = 0
                            media_events[:] = [event for event in media_events if event.modality != "audio"]
                            continue
                        audio_buffer.extend(pcm_bytes)
                        if config.enable_audio_arrival_prefill_approximation:
                            sealed_new_chunk = False
                            while len(audio_buffer) - audio_sealed_bytes >= _AUDIO_ARRIVAL_CHUNK_BYTES:
                                stop = audio_sealed_bytes + _AUDIO_ARRIVAL_CHUNK_BYTES
                                media_events.append(
                                    _TurnMediaEvent("audio", bytes(audio_buffer[audio_sealed_bytes:stop]))
                                )
                                audio_sealed_bytes = stop
                                sealed_new_chunk = True
                            if sealed_new_chunk:
                                _schedule_arrival_prefill()

                    elif msg_type == "video.query":
                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                decoded = base64.b64decode(audio_data_b64)
                                if len(audio_buffer) + len(decoded) <= _MAX_AUDIO_BUFFER_BYTES:
                                    audio_buffer.extend(decoded)
                                    if config.enable_audio_arrival_prefill_approximation:
                                        while len(audio_buffer) - audio_sealed_bytes >= _AUDIO_ARRIVAL_CHUNK_BYTES:
                                            stop = audio_sealed_bytes + _AUDIO_ARRIVAL_CHUNK_BYTES
                                            media_events.append(
                                                _TurnMediaEvent(
                                                    "audio",
                                                    bytes(audio_buffer[audio_sealed_bytes:stop]),
                                                )
                                            )
                                            audio_sealed_bytes = stop
                                else:
                                    await self._send_error(websocket, "Audio buffer overflow")
                                    audio_buffer.clear()
                                    audio_sealed_bytes = 0
                                    media_events[:] = [event for event in media_events if event.modality != "audio"]
                            except Exception:
                                pass

                        await _start_query_turn(query_text=query_text)

                    elif msg_type == "video.done":
                        if query_task is not None and not query_task.done():
                            await asyncio.gather(query_task, return_exceptions=True)
                            query_task = None
                        await websocket.send_json({"type": "session.done"})
                        return

                    elif msg_type == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass

                    else:
                        await self._send_error(websocket, f"Unknown type: {msg_type}")

            reader_task = asyncio.create_task(_reader())
            try:
                await _processor()
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
                for t in list(prewarm_tasks):
                    t.cancel()
                if prewarm_tasks:
                    await asyncio.gather(*prewarm_tasks, return_exceptions=True)
                if arrival_prefill_task is not None and not arrival_prefill_task.done():
                    arrival_prefill_task.cancel()
                    await asyncio.gather(arrival_prefill_task, return_exceptions=True)
                if query_task is not None and not query_task.done():
                    await _cancel_active_query(abort_now=True)
        except WebSocketDisconnect:
            logger.info("Streaming video: client disconnected")
        except Exception as e:
            logger.exception("Streaming video session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(self, websocket: WebSocket) -> StreamingVideoSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(),
                timeout=self._config_timeout,
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            await self._send_error(
                websocket,
                f"Expected session.config, got: {msg.get('type') if isinstance(msg, dict) else type(msg).__name__}",
            )
            return None

        config_data = {k: v for k, v in msg.items() if k != "type"}
        alias_map = {
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data:
                if new_key not in config_data:
                    config_data[new_key] = config_data[old_key]
                config_data.pop(old_key)

        try:
            config = StreamingVideoSessionConfig(**config_data)
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        return config

    async def _process_query(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, tuple[Any, str]],
        frame_metadata: list[dict[str, Any]] | None = None,
        media_events: list[_TurnMediaEvent] | None = None,
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        engine_kwargs: dict[str, Any] = {}
        if frame_metadata:
            engine_kwargs["frame_metadata"] = frame_metadata
        if media_events is not None:
            engine_kwargs["media_events"] = media_events
        await self._process_query_engine(
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
            **engine_kwargs,
        )

    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

    def _sampling_params_for_request(
        self,
        config: StreamingVideoSessionConfig,
        *,
        thinker_max_tokens: int | None = None,
        deterministic_thinker: bool = False,
    ) -> list[Any] | None:
        """Build request-local stage params without mutating deploy defaults."""
        if config.sampling_params_list:
            converter = getattr(self._chat_service, "_to_sampling_params_list", None)
            if converter is not None:
                params = list(converter(config.sampling_params_list))
            else:
                from vllm import SamplingParams

                params = [SamplingParams(**value) for value in config.sampling_params_list]
        else:
            defaults = getattr(self._engine_client, "default_sampling_params_list", None)
            params = copy.deepcopy(list(defaults)) if defaults else []

        # This WebSocket endpoint always consumes an output stream, including
        # when wire-level audio buffering is enabled. Passing an explicit copy
        # of the deploy defaults bypasses AsyncOmni's implicit DELTA coercion;
        # leaving the defaults as CUMULATIVE makes every Code2Wav event replay
        # all prior waveform chunks. Force per-stage engine outputs to DELTA so
        # each response.audio.delta contains only newly generated samples.
        params = coerce_param_message_types(params, is_streaming=True)

        effective_thinker_max_tokens = (
            thinker_max_tokens if thinker_max_tokens is not None else config.thinker_max_response_tokens
        )
        if effective_thinker_max_tokens is not None:
            if not params:
                from vllm import SamplingParams

                params = [SamplingParams()]
            stage_metadata = getattr(self._engine_client, "_stage_meta_list", None)
            thinker_stage_ids = [
                index
                for index, metadata in enumerate(stage_metadata or ())
                if getattr(metadata, "model_stage", None) == "thinker"
                or (getattr(metadata, "final_output", False) and getattr(metadata, "final_output_type", None) == "text")
            ]
            # A non-P/D topology has one Thinker at stage 0.  A P/D topology
            # has both a prefill-only and a decode-only Thinker; cap both so
            # request-local limits apply to the actual text generator too.
            if not thinker_stage_ids:
                thinker_stage_ids = [0]
            for stage_id in thinker_stage_ids:
                if stage_id >= len(params):
                    continue
                params[stage_id].max_tokens = effective_thinker_max_tokens
                params[stage_id].min_tokens = 0
                if deterministic_thinker:
                    params[stage_id].temperature = 0.0
        return params or None

    def _chat_request(
        self,
        config: StreamingVideoSessionConfig,
        messages: list[dict[str, Any]],
        *,
        output_modalities: list[str],
        add_generation_prompt: bool,
    ) -> Any:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": output_modalities,
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        # Keep multimodal hashing identical between image-only warm-ups and
        # the final request that appends audio. A changed processor kwarg is a
        # different cache key even when the media UUID stays the same.
        if config.use_audio_in_video:
            request_kwargs["mm_processor_kwargs"] = {"use_audio_in_video": True}
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list
        return ChatCompletionRequest(**request_kwargs)

    async def _render_message_block(
        self,
        config: StreamingVideoSessionConfig,
        messages: list[dict[str, Any]],
        *,
        output_modalities: list[str],
        add_generation_prompt: bool,
        diag: dict[str, float] | None = None,
        diag_label: str = "block",
    ) -> dict[str, Any]:
        started = _time.monotonic()
        request = self._chat_request(
            config,
            messages,
            output_modalities=output_modalities,
            add_generation_prompt=add_generation_prompt,
        )
        _record_render_diag(diag, f"{diag_label}_request_ms", started)

        started = _time.monotonic()
        prompt = await self._preprocess_to_engine_prompt(request)
        _record_render_diag(diag, f"{diag_label}_preprocess_ms", started)
        if not self._is_mergeable_engine_prompt(prompt):
            raise ValueError("renderer did not return a mergeable token prompt")
        started = _time.monotonic()
        prompt = self._normalize_engine_prompt_for_messages(config, messages, prompt)
        _record_render_diag(diag, f"{diag_label}_normalize_ms", started)
        return prompt

    @staticmethod
    def _is_mergeable_engine_prompt(prompt: Any) -> bool:
        if not isinstance(prompt, dict) or prompt.get("type") not in {"token", "multimodal"}:
            return False
        token_ids = prompt.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not all(isinstance(token, int) for token in token_ids):
            return False
        if prompt.get("type") == "multimodal":
            return all(key in prompt for key in ("mm_kwargs", "mm_hashes", "mm_placeholders"))
        return True

    @staticmethod
    def _merge_engine_prompt_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate already processed message blocks into one engine input."""
        from vllm.multimodal.inputs import MultiModalKwargsItems

        prompt_token_ids: list[int] = []
        kwargs_by_modality: dict[str, list[Any]] = {}
        hashes_by_modality: dict[str, list[str]] = {}
        placeholders_by_modality: dict[str, list[Any]] = {}
        arrival_time: float | None = None

        for block in blocks:
            if not OmniStreamingVideoHandler._is_mergeable_engine_prompt(block):
                raise ValueError("cannot merge an unsupported engine prompt block")
            offset = len(prompt_token_ids)
            prompt_token_ids.extend(block["prompt_token_ids"])
            if isinstance(block.get("arrival_time"), (int, float)):
                arrival_time = float(block["arrival_time"])
            if block.get("type") != "multimodal":
                continue
            for modality, items in block["mm_kwargs"].items():
                kwargs_by_modality.setdefault(modality, []).extend(items)
            for modality, hashes in block["mm_hashes"].items():
                hashes_by_modality.setdefault(modality, []).extend(hashes)
            for modality, placeholders in block["mm_placeholders"].items():
                target = placeholders_by_modality.setdefault(modality, [])
                target.extend(replace(item, offset=item.offset + offset) for item in placeholders)

        if kwargs_by_modality or hashes_by_modality or placeholders_by_modality:
            merged: dict[str, Any] = {
                "type": "multimodal",
                "prompt_token_ids": prompt_token_ids,
                "mm_kwargs": MultiModalKwargsItems(kwargs_by_modality),
                "mm_hashes": hashes_by_modality,
                "mm_placeholders": placeholders_by_modality,
            }
        else:
            merged = {"type": "token", "prompt_token_ids": prompt_token_ids}
        if arrival_time is not None:
            merged["arrival_time"] = arrival_time
        return merged

    @staticmethod
    def _strip_generation_suffix(
        block: dict[str, Any],
        generation_suffix: tuple[int, ...],
    ) -> dict[str, Any]:
        token_ids = block["prompt_token_ids"]
        suffix_len = len(generation_suffix)
        if suffix_len <= 0 or tuple(token_ids[-suffix_len:]) != generation_suffix:
            raise ValueError("current turn does not end in the canonical generation suffix")
        user_len = len(token_ids) - suffix_len
        for placeholders in block.get("mm_placeholders", {}).values():
            if any(item.offset + item.length > user_len for item in placeholders):
                raise ValueError("generation suffix overlaps a multimodal placeholder")
        user_block = dict(block)
        user_block["prompt_token_ids"] = list(token_ids[:user_len])
        assistant_mask = user_block.get("assistant_tokens_mask")
        if isinstance(assistant_mask, list):
            user_block["assistant_tokens_mask"] = assistant_mask[:user_len]
        return user_block

    @staticmethod
    def _canonical_signature(config: StreamingVideoSessionConfig) -> tuple[Any, ...]:
        return (
            config.model or "default",
            config.system_prompt,
            bool(config.use_audio_in_video),
            bool(config.enable_audio_arrival_prefill_approximation),
        )

    async def _initialize_canonical_prompt_state(
        self,
        config: StreamingVideoSessionConfig,
        message_history: list[dict[str, Any]],
        *,
        output_modalities: list[str],
        diag: dict[str, float] | None = None,
    ) -> _CanonicalPromptState:
        prefix_messages = self._history_prefix_messages(config)
        if prefix_messages:
            prefix_block = await self._render_message_block(
                config,
                prefix_messages,
                output_modalities=output_modalities,
                add_generation_prompt=False,
                diag=diag,
                diag_label="canonical_init",
            )
        else:
            prefix_block = {"type": "token", "prompt_token_ids": []}

        probe = [{"role": "user", "content": "canonical-boundary-probe"}]
        probe_without_generation = await self._render_message_block(
            config,
            probe,
            output_modalities=output_modalities,
            add_generation_prompt=False,
            diag=diag,
            diag_label="canonical_init",
        )
        probe_with_generation = await self._render_message_block(
            config,
            probe,
            output_modalities=output_modalities,
            add_generation_prompt=True,
            diag=diag,
            diag_label="canonical_init",
        )
        probe_prefix = probe_without_generation["prompt_token_ids"]
        probe_full = probe_with_generation["prompt_token_ids"]
        if probe_full[: len(probe_prefix)] != probe_prefix:
            raise ValueError("chat template is not append-only at a generation boundary")
        generation_suffix = tuple(probe_full[len(probe_prefix) :])
        if not generation_suffix:
            raise ValueError("chat template produced an empty generation suffix")

        if len(message_history) % 2:
            raise ValueError("canonical history must contain complete user/assistant pairs")
        turn_blocks: list[dict[str, Any]] = []
        turn_message_ids: list[tuple[int, int]] = []
        for index in range(0, len(message_history), 2):
            user_message = message_history[index]
            assistant_message = message_history[index + 1]
            user_block = await self._render_message_block(
                config,
                [user_message],
                output_modalities=output_modalities,
                add_generation_prompt=False,
                diag=diag,
                diag_label="canonical_init",
            )
            assistant_block = await self._render_message_block(
                config,
                [assistant_message],
                output_modalities=output_modalities,
                add_generation_prompt=False,
                diag=diag,
                diag_label="canonical_init",
            )
            turn_blocks.append(self._merge_engine_prompt_blocks([user_block, assistant_block]))
            turn_message_ids.append((id(user_message), id(assistant_message)))

        state = _CanonicalPromptState(
            signature=self._canonical_signature(config),
            prefix_block=prefix_block,
            turn_blocks=turn_blocks,
            turn_message_ids=turn_message_ids,
            generation_suffix=generation_suffix,
        )
        config._canonical_prompt_state = state
        return state

    @staticmethod
    def _canonical_history_offset(
        state: _CanonicalPromptState,
        message_history: list[dict[str, Any]],
    ) -> int | None:
        if len(message_history) % 2:
            return None
        pairs = [
            (id(message_history[index]), id(message_history[index + 1])) for index in range(0, len(message_history), 2)
        ]
        offset = len(state.turn_message_ids) - len(pairs)
        if offset < 0 or state.turn_message_ids[offset:] != pairs:
            return None
        return offset

    async def _render_incremental_canonical_prompt(
        self,
        config: StreamingVideoSessionConfig,
        message_history: list[dict[str, Any]],
        current_user_message: dict[str, Any],
        *,
        output_modalities: list[str],
        diag: dict[str, float] | None = None,
    ) -> tuple[dict[str, Any], _CanonicalRenderTicket]:
        started = _time.monotonic()
        state = config._canonical_prompt_state
        if state is None or state.signature != self._canonical_signature(config):
            state = await self._initialize_canonical_prompt_state(
                config,
                message_history,
                output_modalities=output_modalities,
                diag=diag,
            )
        history_offset = self._canonical_history_offset(state, message_history)
        if history_offset is None:
            # External mutation or a failed prior commit: rebuild once from the
            # application-owned messages, then resume incremental operation.
            config._canonical_prompt_state = None
            state = await self._initialize_canonical_prompt_state(
                config,
                message_history,
                output_modalities=output_modalities,
                diag=diag,
            )
            history_offset = 0
        _record_render_diag(diag, "canonical_state_ms", started)

        current_block = await self._render_message_block(
            config,
            [current_user_message],
            output_modalities=output_modalities,
            add_generation_prompt=True,
            diag=diag,
            diag_label="current",
        )
        started = _time.monotonic()
        user_block = self._strip_generation_suffix(current_block, state.generation_suffix)
        _record_render_diag(diag, "strip_suffix_ms", started)
        started = _time.monotonic()
        prompt = self._merge_engine_prompt_blocks(
            [state.prefix_block, *state.turn_blocks[history_offset:], current_block]
        )
        _record_render_diag(diag, "merge_blocks_ms", started)
        return prompt, _CanonicalRenderTicket(
            user_block=user_block,
            history_turns=len(message_history) // 2,
        )

    @staticmethod
    def _pop_canonical_render_ticket(engine_prompt: Any) -> _CanonicalRenderTicket | None:
        if not isinstance(engine_prompt, dict):
            return None
        ticket = engine_prompt.pop(_CANONICAL_RENDER_TICKET_KEY, None)
        return ticket if isinstance(ticket, _CanonicalRenderTicket) else None

    async def _commit_canonical_turn(
        self,
        config: StreamingVideoSessionConfig,
        ticket: _CanonicalRenderTicket | None,
        message_history: list[dict[str, Any]],
        *,
        output_modalities: list[str],
    ) -> None:
        state = config._canonical_prompt_state
        if (
            ticket is None
            or state is None
            or len(state.turn_blocks) != ticket.history_turns
            or len(message_history) != 2 * (ticket.history_turns + 1)
        ):
            config._canonical_prompt_state = None
            return

        user_message = message_history[-2]
        assistant_message = message_history[-1]
        try:
            assistant_block = await self._render_message_block(
                config,
                [assistant_message],
                output_modalities=output_modalities,
                add_generation_prompt=False,
            )
            state.turn_blocks.append(self._merge_engine_prompt_blocks([ticket.user_block, assistant_block]))
            state.turn_message_ids.append((id(user_message), id(assistant_message)))
            logger.info(
                "[canonical-prompt] session=%s committed_turns=%d cached_tokens=%d",
                config.session_id or "-",
                len(state.turn_blocks),
                sum(len(block["prompt_token_ids"]) for block in state.turn_blocks),
            )
        except Exception:
            config._canonical_prompt_state = None
            logger.warning(
                "Failed to commit incremental canonical history; next turn will rebuild",
                exc_info=True,
            )

    @staticmethod
    def _compact_canonical_prompt_state(
        config: StreamingVideoSessionConfig,
        dropped_turns: int,
    ) -> None:
        state = config._canonical_prompt_state
        if state is None or dropped_turns <= 0:
            return
        if dropped_turns > len(state.turn_blocks):
            config._canonical_prompt_state = None
            return
        del state.turn_blocks[:dropped_turns]
        del state.turn_message_ids[:dropped_turns]

    async def _render_engine_prompt(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        *,
        output_modalities: list[str],
        media_events: list[_TurnMediaEvent] | None = None,
        diag: dict[str, float] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        started = _time.monotonic()
        build_args = (
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )
        if media_events is None:
            messages, current_user_message = self.build_engine_prompt(*build_args)
        else:
            messages, current_user_message = self.build_engine_prompt(*build_args, media_events)
        _record_render_diag(diag, "build_messages_ms", started)
        if self.supports_incremental_canonical_prompt():
            try:
                engine_prompt, ticket = await self._render_incremental_canonical_prompt(
                    config,
                    message_history,
                    current_user_message,
                    output_modalities=output_modalities,
                    diag=diag,
                )
                engine_prompt[_CANONICAL_RENDER_TICKET_KEY] = ticket
            except Exception:
                config._canonical_prompt_state = None
                logger.warning(
                    "Incremental canonical prompt unavailable; falling back to full render",
                    exc_info=True,
                )
                engine_prompt = await self._preprocess_to_engine_prompt(
                    self._chat_request(
                        config,
                        messages,
                        output_modalities=output_modalities,
                        add_generation_prompt=True,
                    )
                )
                engine_prompt = self._normalize_engine_prompt_for_messages(config, messages, engine_prompt)
        else:
            engine_prompt = await self._preprocess_to_engine_prompt(
                self._chat_request(
                    config,
                    messages,
                    output_modalities=output_modalities,
                    add_generation_prompt=True,
                )
            )
            engine_prompt = self._normalize_engine_prompt_for_messages(config, messages, engine_prompt)
        if isinstance(engine_prompt, dict):
            engine_prompt["talker_cache_salt"] = config._talker_cache_salt
        return engine_prompt, current_user_message

    async def _process_video_arrival_prefill(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        message_history: list[dict[str, Any]],
        request_id: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        audio_buffer: bytearray | None = None,
        media_events: list[_TurnMediaEvent] | None = None,
    ) -> bool:
        """Materialize one cumulative media prefix without rewriting history."""
        arrival_audio = audio_buffer or bytearray()
        if self._engine_client is None or (not frame_buffer and not arrival_audio):
            return False
        diag_start = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
        try:
            # Arrival requests are disposable cache hints. Serialize their
            # prompt snapshot with the session's occasional history rewrite.
            owner_token = _MM_RENDER_OWNER.set(
                f"arrival:{config.session_id or '-'}:{request_id}"
            )
            try:
                async with config._history_compaction_lock:
                    engine_prompt, _ = await self._render_engine_prompt(
                        config,
                        frame_buffer,
                        arrival_audio,
                        message_history,
                        "",
                        prewarmed_frames,
                        output_modalities=["text"],
                        media_events=media_events,
                    )
                    self._pop_canonical_render_ticket(engine_prompt)
                    prompt_tokens = _prompt_token_count(engine_prompt)
                    if prompt_tokens >= config.context_window_trigger_tokens:
                        # The completed-turn path normally compacts during playback.
                        # If it did not, skip this optional warm-up and let the final
                        # request perform the correctness-preserving hard-limit rewrite.
                        logger.info(
                            "[arrival-prefill] skip-context-limit session=%s request=%s "
                            "prompt_tokens=%d trigger_tokens=%d",
                            config.session_id or "-",
                            request_id,
                            prompt_tokens,
                            config.context_window_trigger_tokens,
                        )
                        return False
                    lineage_ticket = _attach_thinker_lineage(config, engine_prompt)
            finally:
                _MM_RENDER_OWNER.reset(owner_token)
            diag_rendered = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
            if isinstance(engine_prompt, dict):
                engine_prompt["prefill_only"] = True
            logger.info(
                "[arrival-prefill] session=%s request=%s frames=%d audio_bytes=%d "
                "audio_chunks=%d prompt_tokens=%d "
                "lineage_prefix_tokens=%d",
                config.session_id or "-",
                request_id,
                len(frame_buffer),
                len(arrival_audio),
                sum(event.modality == "audio" for event in (media_events or ())),
                prompt_tokens,
                lineage_ticket.prefix_tokens if lineage_ticket is not None else 0,
            )
            outputs = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                sampling_params_list=self._sampling_params_for_request(
                    config,
                    thinker_max_tokens=1,
                ),
                output_modalities=["text"],
            )
            # The engine finishes after prompt prefill without committing the
            # sampled next token. final_stage_id=0 keeps Talker/Code2Wav idle.
            async for _ in outputs:
                pass
            diag_engine_done = _time.monotonic() if _LOG_HANDOFF_DIAG else 0.0
            _commit_thinker_lineage(config, lineage_ticket)
            if _LOG_HANDOFF_DIAG:
                logger.info(
                    "[ARRIVAL-DIAG] session=%s request=%s prompt_tokens=%d "
                    "render_ms=%.3f engine_ms=%.3f total_ms=%.3f",
                    config.session_id or "-",
                    request_id,
                    prompt_tokens,
                    (diag_rendered - diag_start) * 1000.0,
                    (diag_engine_done - diag_rendered) * 1000.0,
                    (diag_engine_done - diag_start) * 1000.0,
                )
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            # Cache warming is optional for correctness. The query path still
            # submits the complete canonical prompt on any failure or miss.
            logger.warning(
                "[arrival-prefill] failed session=%s request=%s",
                config.session_id or "-",
                request_id,
                exc_info=True,
            )
            return False
    async def _render_engine_prompt_with_compaction(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        *,
        output_modalities: list[str],
        media_events: list[_TurnMediaEvent] | None = None,
        attach_lineage: bool = False,
        diag: dict[str, float] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Serialize each application's context-rewrite transaction."""
        started = _time.monotonic()
        async with config._history_compaction_lock:
            _record_render_diag(diag, "history_lock_wait_ms", started)
            engine_prompt, user_message = await self._render_engine_prompt_with_compaction_locked(
                config,
                frame_buffer,
                audio_buffer,
                message_history,
                query_text,
                prewarmed_frames,
                output_modalities=output_modalities,
                media_events=media_events,
                diag=diag,
            )
            if attach_lineage:
                started = _time.monotonic()
                ticket = _attach_thinker_lineage(config, engine_prompt)
                _record_render_diag(diag, "attach_lineage_ms", started)
                if ticket is not None and isinstance(engine_prompt, dict):
                    engine_prompt[_THINKER_LINEAGE_TICKET_KEY] = ticket
            return engine_prompt, user_message

    async def _render_engine_prompt_with_compaction_locked(
        self,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        *,
        output_modalities: list[str],
        media_events: list[_TurnMediaEvent] | None = None,
        diag: dict[str, float] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Render one canonical prompt and compact history at turn boundaries.

        Arrival warm-ups and final response requests must make the same lineage
        decision. Otherwise a warm-up can exceed ``max_model_len`` using stale
        history even though the final request would compact that history and
        remain valid.
        """

        def _without_images(message: dict[str, Any]) -> dict[str, Any]:
            """Copy one message while removing canonical historical images."""
            content = message.get("content")
            if message.get("role") != "user" or not isinstance(content, list):
                return message
            retained = [
                part
                for part in content
                if not (
                    isinstance(part, dict)
                    and part.get("type") in {"image", "image_pil", "image_url", "input_image"}
                )
            ]
            if len(retained) == len(content):
                return message
            compacted = dict(message)
            compacted["content"] = retained
            return compacted

        def _audio_text_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [_without_images(message) for message in history]

        def _latest_image_events() -> list[_TurnMediaEvent] | None:
            if media_events is None:
                return None
            latest_image_index = next(
                (
                    index
                    for index in range(len(media_events) - 1, -1, -1)
                    if media_events[index].modality == "image"
                ),
                None,
            )
            return [
                event
                for index, event in enumerate(media_events)
                if event.modality != "image" or index == latest_image_index
            ]

        async def _render(
            history: list[dict[str, Any]],
            *,
            compact_media: bool = False,
        ) -> tuple[Any, dict[str, Any]]:
            return await self._render_engine_prompt(
                config,
                frame_buffer[-1:] if compact_media else frame_buffer,
                audio_buffer,
                history,
                query_text,
                prewarmed_frames,
                output_modalities=output_modalities,
                media_events=_latest_image_events() if compact_media else media_events,
                diag=diag,
            )

        trigger_tokens = config.context_window_trigger_tokens
        engine_prompt, user_message = await _render(message_history)
        prompt_tokens = _prompt_token_count(engine_prompt)
        if prompt_tokens < trigger_tokens:
            return engine_prompt, user_message

        before_turns = len(message_history) // 2
        dropped_turns = max(0, before_turns - config.context_window_retained_turns)
        renders = 1
        compacted_history = _audio_text_history(message_history[2 * dropped_turns :])
        engine_prompt, user_message = await _render(compacted_history, compact_media=True)
        prompt_tokens = _prompt_token_count(engine_prompt)
        renders += 1

        # A single unusually large retained audio turn may still exceed the
        # threshold. Drop additional oldest complete turns one at a time. The
        # current in-flight turn is never removed; its image set is already
        # reduced to the newest accepted frame.
        while prompt_tokens >= trigger_tokens and dropped_turns < before_turns:
            dropped_turns += 1
            compacted_history = _audio_text_history(message_history[2 * dropped_turns :])
            engine_prompt, user_message = await _render(compacted_history, compact_media=True)
            prompt_tokens = _prompt_token_count(engine_prompt)
            renders += 1

        # Commit the periodic application-owned rewrite only after the retained
        # prompt renders successfully. Engine prefix/KV entries remain caches.
        message_history[:] = compacted_history
        # Retained turns changed content as well as position, so their old
        # processed blocks cannot be reused even when no complete turn was
        # dropped.
        config._canonical_prompt_state = None
        _reset_thinker_lineage(config)
        logger.info(
            "[session-history] compact session=%s mode=audio-text-history-latest-image "
            "turns=%d->%d prompt_tokens=%d trigger_tokens=%d retained_turns=%d renders=%d",
            config.session_id or "-",
            before_turns,
            len(message_history) // 2,
            prompt_tokens,
            trigger_tokens,
            config.context_window_retained_turns,
            renders,
        )
        return engine_prompt, user_message

    async def _process_query_engine(
        self,
        websocket: WebSocket,
        config: StreamingVideoSessionConfig,
        frame_buffer: list[str],
        audio_buffer: bytearray,
        message_history: list[dict[str, Any]],
        query_text: str,
        request_id: str,
        interrupt_event: asyncio.Event,
        prewarmed_frames: dict[str, tuple[Any, str]],
        frame_metadata: list[dict[str, Any]] | None = None,
        media_events: list[_TurnMediaEvent] | None = None,
    ) -> None:
        """Direct engine_client.generate() path for async_chunk audio."""
        request_start = _time.monotonic()
        render_diag: dict[str, float] | None = {} if _LOG_HANDOFF_DIAG else None
        owner_token = _MM_RENDER_OWNER.set(f"query:{config.session_id or '-'}:{request_id}")
        try:
            engine_prompt, user_message = await self._render_engine_prompt_with_compaction(
                config,
                frame_buffer,
                audio_buffer,
                message_history,
                query_text,
                prewarmed_frames,
                output_modalities=config.modalities,
                media_events=media_events,
                attach_lineage=True,
                diag=render_diag,
            )
            canonical_ticket = self._pop_canonical_render_ticket(engine_prompt)
            lineage_ticket = _pop_thinker_lineage_ticket(engine_prompt)
        except Exception as e:
            await self._send_error(websocket, f"Prompt preprocessing failed: {e}")
            return
        finally:
            _MM_RENDER_OWNER.reset(owner_token)
        render_done = _time.monotonic()
        if render_diag is not None:
            logger.info(
                "[RENDER-DIAG] session=%s request=%s total_ms=%.3f components=%s",
                config.session_id or "-",
                request_id,
                (render_done - request_start) * 1000.0,
                json.dumps(render_diag, sort_keys=True, separators=(",", ":")),
            )
        decoded_ready_ts_ms = _time.monotonic() * 1000
        selected_metadata = list(frame_metadata or [])
        model_selected_ts_ms = _time.monotonic() * 1000

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        audio_chunk_count = 0
        # Number of per-step tensors in OmniRequestOutput.audio_data already
        # drained. Used by the fast path to skip already-emitted history.
        audio_chunks_drained = 0
        previous_text = ""
        interrupted = False
        frames_consumed_sent = False
        t_start = _time.monotonic()
        t_first_text = None
        t_first_audio = None

        # Wire-level async-chunk switch. "off" means
        # buffer all deltas server-side and flush once at the end; the engine
        # pipeline still overlaps internally.
        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        audio_tail_tensors: list[Any] = []
        thinker_output_token_ids: list[int] = []

        try:
            logger.info(
                "[finite-request] session=%s request=%s turn=%d prompt_tokens=%d "
                "history_messages=%d frames=%d audio_bytes=%d",
                config.session_id or "-",
                request_id,
                len(message_history) // 2,
                _prompt_token_count(engine_prompt),
                len(message_history),
                len(frame_buffer),
                len(audio_buffer),
            )
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                sampling_params_list=self._sampling_params_for_request(config),
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                if isinstance(output, OmniRequestOutput) and output.final_output_type == "text":
                    request_output = getattr(output, "request_output", None)
                    stage_outputs = getattr(request_output, "outputs", None)
                    if isinstance(stage_outputs, list) and stage_outputs:
                        cumulative_ids = getattr(stage_outputs[0], "cumulative_token_ids", None)
                        if isinstance(cumulative_ids, list) and all(isinstance(token, int) for token in cumulative_ids):
                            thinker_output_token_ids = list(cumulative_ids)
                # Soft interrupt: drain without sending
                if interrupt_event.is_set():
                    if not interrupted:
                        logger.info("Generation interrupted — draining")
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

                if not frames_consumed_sent and frame_metadata:
                    await websocket.send_json(
                        {
                            "type": "video.frames.consumed",
                            "request_id": request_id,
                            "model_selected_ts_ms": model_selected_ts_ms,
                            "frame_ids": [
                                metadata["frame_id"]
                                for metadata in selected_metadata
                                if isinstance(metadata.get("frame_id"), str)
                            ],
                            "frames": [
                                {
                                    "frame_id": metadata.get("frame_id"),
                                    "pts_ms": metadata.get("pts_ms"),
                                    "source_pts_ms": metadata.get("source_pts_ms"),
                                    "quality_profile": metadata.get("quality_profile"),
                                    "receiver_received_ts_ms": metadata.get("receiver_received_ts_ms"),
                                    "decoded_ready_ts_ms": decoded_ready_ts_ms,
                                }
                                for metadata in selected_metadata
                            ],
                            "latest_pts_ms": selected_metadata[-1].get("pts_ms") if selected_metadata else None,
                        }
                    )
                    frames_consumed_sent = True

                out_type = getattr(output, "final_output_type", "text")

                if out_type == "audio":
                    if streaming and not text_done_sent:
                        full_text = "".join(text_parts)
                        await websocket.send_json({"type": "response.text.done", "text": full_text})
                        text_done_sent = True

                    if t_first_audio is None:
                        t_first_audio = _time.monotonic()
                    audio_chunk_count += 1
                    if streaming:
                        b64, audio_chunks_drained = self._extract_audio_delta_b64(
                            output,
                            audio_chunks_drained,
                        )
                        if b64:
                            await websocket.send_json(
                                {
                                    "type": "response.audio.delta",
                                    "data": b64,
                                    "format": "wav",
                                }
                            )
                    else:
                        audio_data = self._get_audio_data(output)
                        if audio_data is not None:
                            if isinstance(audio_data, list):
                                audio_tail_tensors = list(audio_data)
                            else:
                                audio_tail_tensors = [audio_data]
                else:
                    delta_text, previous_text = self._extract_text_delta(
                        output,
                        previous_text,
                    )
                    if delta_text:
                        if t_first_text is None:
                            t_first_text = _time.monotonic()
                        text_parts.append(delta_text)
                        if streaming:
                            await websocket.send_json({"type": "response.text.delta", "delta": delta_text})

            if not text_done_sent:
                full_text = "".join(text_parts)
                await websocket.send_json({"type": "response.text.done", "text": full_text})
                text_done_sent = True

            if not streaming and audio_tail_tensors:
                try:
                    coalesced = (
                        audio_tail_tensors[0] if len(audio_tail_tensors) == 1 else torch.cat(audio_tail_tensors, dim=-1)
                    )
                    tail_np = self._tensor_to_1d_np(coalesced)
                    b64, _ = self._encode_tail(
                        tail_np,
                        0,
                        new_drained=len(audio_tail_tensors),
                        is_first=True,
                    )
                    if b64:
                        await websocket.send_json(
                            {
                                "type": "response.audio.delta",
                                "data": b64,
                                "format": "wav",
                            }
                        )
                except Exception:
                    logger.exception("Failed to coalesce off-path audio")

            if audio_chunk_count > 0:
                await websocket.send_json({"type": "response.audio.done"})

            response_text = "".join(text_parts)
            if not interrupted:
                _commit_thinker_lineage(
                    config,
                    lineage_ticket,
                    thinker_output_token_ids,
                )
            else:
                _reset_thinker_lineage(config)
            # Keep the completed turn append serialized with arrival/query
            # prompt construction for this application session.
            async with config._history_compaction_lock:
                self.on_turn_complete(message_history, user_message, response_text)
                if not interrupted:
                    await self._commit_canonical_turn(
                        config,
                        canonical_ticket,
                        message_history,
                        output_modalities=config.modalities,
                    )
                else:
                    config._canonical_prompt_state = None

            t_end = _time.monotonic()
            logger.info(
                "[TIMING] mode=%s total=%.2fs first_text=%.2fs first_audio=%.2fs audio_chunks=%d",
                async_chunk_mode,
                t_end - t_start,
                (t_first_text - t_start) if t_first_text else -1,
                (t_first_audio - t_start) if t_first_audio else -1,
                audio_chunk_count,
            )
            logger.info(
                "[QUERY-BREAKDOWN] session=%s request=%s render_ms=%.1f "
                "engine_to_first_text_ms=%.1f engine_to_first_audio_ms=%.1f",
                config.session_id or "-",
                request_id,
                (render_done - request_start) * 1000.0,
                ((t_first_text - t_start) * 1000.0) if t_first_text else -1.0,
                ((t_first_audio - t_start) * 1000.0) if t_first_audio else -1.0,
            )

        except Exception:
            logger.exception("Engine query failed")
            await self._send_error(websocket, "Query processing failed")

        if not text_done_sent:
            full_text = "".join(text_parts)
            await websocket.send_json({"type": "response.text.done", "text": full_text})

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pcm_to_wav_b64(pcm_data: bytes, sample_rate: int = 16000) -> str:
        """Wrap raw PCM16 mono in a WAV container and return base64."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return base64.b64encode(buf.getvalue()).decode()

    @classmethod
    def _extract_audio_delta_b64(
        cls,
        result: OmniRequestOutput,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Return (base64 WAV of new samples, updated chunks_drained).

        `chunks_drained` is the number of per-step tensors in
        ``audio_data`` that have already been emitted. Each engine step appends
        one tensor, so new samples are ``audio_data[chunks_drained:]`` — no
        matter how many steps accumulated between reads (handles backpressure
        cleanly, unlike a simple ``audio_data[-1]``).

        Two paths, selected at runtime by ``VLLM_VIDEO_AUDIO_DELTA_MODE``:
          * fast — only D2H the new tail. Per-call cost ∝ new chunks.
          * slow — full cat + D2H each call. Per-call cost ∝ total history.
                   Retained for A/B; remove once downstream callers confirm.
        """
        audio_data = cls._get_audio_data(result)
        if audio_data is None:
            return None, chunks_drained

        if video_stream_envs.VLLM_VIDEO_AUDIO_DELTA_MODE == "slow":
            return cls._delta_slow(audio_data, chunks_drained)
        return cls._delta_fast(audio_data, chunks_drained)

    @staticmethod
    def _get_audio_data(result: OmniRequestOutput):
        """Navigate OmniRequestOutput → multimodal_output['audio']. None on miss."""
        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return None
        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return None
        mm_output = getattr(outputs[0], "multimodal_output", None)
        if not isinstance(mm_output, Mapping):
            return None
        return mm_output.get("audio")

    @classmethod
    def _delta_fast(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Emit only tensors appended since the last call."""
        # DELTA output drains audio after each snapshot, so every bare tensor is
        # a new granule. Only cumulative lists use chunks_drained as an index.
        if not isinstance(audio_data, list):
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(
                tail_np,
                chunks_drained,
                new_drained=chunks_drained + 1,
                is_first=chunks_drained == 0,
            )

        n = len(audio_data)
        if n <= chunks_drained:
            return None, chunks_drained

        new_chunks = audio_data[chunks_drained:]
        tail = new_chunks[0] if len(new_chunks) == 1 else torch.cat(new_chunks, dim=-1)
        tail_np = cls._tensor_to_1d_np(tail)
        return cls._encode_tail(tail_np, chunks_drained, new_drained=n, is_first=(chunks_drained == 0))

    @classmethod
    def _delta_slow(
        cls,
        audio_data,
        chunks_drained: int,
    ) -> tuple[str | None, int]:
        """Pre-fix behaviour: concat everything each call and slice on CPU."""
        if isinstance(audio_data, list):
            if not audio_data:
                return None, chunks_drained
            audio_tensor = torch.cat(audio_data, dim=-1)
            new_drained = len(audio_data)
        else:
            audio_tensor = audio_data
            full_np = cls._tensor_to_1d_np(audio_tensor)
            if full_np is None:
                return None, chunks_drained
            return cls._encode_tail(
                full_np,
                chunks_drained,
                new_drained=chunks_drained + 1,
                is_first=chunks_drained == 0,
            )

        full_np = cls._tensor_to_1d_np(audio_tensor)
        if full_np is None:
            return None, chunks_drained
        # chunks_drained doesn't map directly to sample offset without tracking
        # per-chunk lengths, so we re-derive: replay the tail that corresponds
        # to chunks appended since last call by slicing off the part produced
        # by the already-drained prefix. For slow path this is intentionally
        # wasteful — the point is to reproduce the pre-fix hot loop.
        if chunks_drained == 0:
            tail_np = full_np
        else:
            # Recover prefix length by re-concatenating the already-drained
            # prefix tensors (cost intentionally identical to the baseline
            # implementation this was lifted from).
            if isinstance(audio_data, list) and chunks_drained < len(audio_data):
                prefix_len = sum(int(t.shape[-1]) for t in audio_data[:chunks_drained])
                tail_np = full_np[prefix_len:]
            else:
                tail_np = full_np[0:0]
        return cls._encode_tail(tail_np, chunks_drained, new_drained=new_drained, is_first=(chunks_drained == 0))

    @classmethod
    def _encode_tail(
        cls,
        tail_np,
        old_drained: int,
        *,
        new_drained: int,
        is_first: bool,
    ) -> tuple[str | None, int]:
        """Strip the CausalConv leading artifact on first emit, then b64-encode."""
        if tail_np is None or len(tail_np) == 0:
            return None, new_drained
        if is_first and len(tail_np) > _CODEC_FRAME_SAMPLES * 2:
            tail_np = tail_np[_CODEC_FRAME_SAMPLES:]
        if len(tail_np) == 0:
            return None, new_drained
        try:
            return cls._encode_audio_wav_b64(tail_np), new_drained
        except Exception:
            logger.exception("Failed to encode audio delta WAV")
            return None, old_drained

    @staticmethod
    def _tensor_to_1d_np(t):
        """Tensor → flat float32 numpy on CPU. None on failure."""
        if t is None or not hasattr(t, "float"):
            return None
        arr = t.float().detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.flatten()
        return arr

    @staticmethod
    def _encode_audio_wav_b64(audio_np) -> str:
        """Encode numpy float32 audio to base64 WAV (24kHz)."""
        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        audio_obj = CreateAudio(
            audio_tensor=audio_np,
            sample_rate=24000,
            response_format="wav",
            speed=1.0,
            base64_encode=True,
        )
        mixin = AudioMixin()
        resp = mixin.create_audio(audio_obj)
        return resp.audio_data

    @staticmethod
    def _extract_text_delta(
        result: OmniRequestOutput,
        previous_text: str,
    ) -> tuple[str, str]:
        """Extract incremental text delta from OmniRequestOutput."""
        if result.final_output_type != "text":
            return "", previous_text

        request_output = getattr(result, "request_output", None)
        if request_output is None:
            return "", previous_text

        outputs = getattr(request_output, "outputs", None)
        if not isinstance(outputs, list) or not outputs:
            return "", previous_text

        text = getattr(outputs[0], "text", None)
        if not isinstance(text, str) or not text:
            return "", previous_text

        if text.startswith(previous_text):
            return text[len(previous_text) :], text
        return text, text

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    async def _preprocess_to_engine_prompt(self, request) -> Any:
        """Use the chat handler's preprocessing to build an engine prompt."""
        handler = self._chat_service
        renderer = handler.renderer

        _conversation, engine_prompts = await handler._preprocess_chat(
            request,
            request.messages,
            default_template=getattr(request, "chat_template", None) or handler.chat_template,
            default_template_content_format=handler.chat_template_content_format,
            renderer=renderer,
            add_generation_prompt=request.add_generation_prompt,
            continue_final_message=request.continue_final_message,
            add_special_tokens=request.add_special_tokens,
        )
        return engine_prompts[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass

    @staticmethod
    async def _send_frame_ack(
        websocket: WebSocket,
        message: Mapping[str, Any],
        *,
        accepted: bool,
        buffered_frames: int,
        reason: str | None = None,
        dropped_frame_id: str | None = None,
    ) -> None:
        frame_id = message.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            return
        ack: dict[str, Any] = {
            "type": "video.frame.ack",
            "frame_id": frame_id,
            "pts_ms": message.get("pts_ms"),
            "capture_ts_ms": message.get("capture_ts_ms"),
            "accepted": accepted,
            "buffered_frames": buffered_frames,
            "server_receive_ts_ms": message.get("_receiver_received_ts_ms", _time.monotonic() * 1000),
        }
        if reason is not None:
            ack["reason"] = reason
        if dropped_frame_id is not None:
            ack["dropped_frame_id"] = dropped_frame_id
        await websocket.send_json(ack)
