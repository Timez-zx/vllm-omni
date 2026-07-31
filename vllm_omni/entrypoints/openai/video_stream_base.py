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

import asyncio
import base64
import hashlib
import io
import json
import os
import time as _time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai import video_stream_envs
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter
from vllm_omni.entrypoints.openai.video_stream_context import (
    text_only_message,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame
_MAX_BUFFER_FRAMES = 64
_MAX_AUDIO_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_MSG_QUEUE = 200
_CODEC_FRAME_SAMPLES = 1920  # CausalConv leading-edge artifact length
_BAD_FRAME = object()


def _decode_frame_bytes(raw_bytes: bytes) -> Any:
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def _downscale_frame_bytes(
    raw_bytes: bytes,
    max_width: int,
    max_height: int,
    jpeg_quality: int,
) -> bytes | None:
    """Shrink a frame to fit within max_width x max_height, preserving aspect ratio.

    Returns re-encoded JPEG bytes, or None if the frame already fits (so the caller can
    keep the original bytes untouched and avoid a needless re-encode).

    WHY THIS BELONGS ON THE SERVER. For a vision-language model the cost of a frame is the
    number of tokens it becomes, and that is set by its pixel dimensions: Qwen3-Omni emits
    (W/32) * (H/32) tokens per frame, so 1280x704 is 880 tokens and 640x352 is 220 --
    exactly 4x less for a halved edge. With a 16-frame prompt that is the difference between
    ~14,100 and ~3,500 video tokens, and prompt length is what the per-turn cost tracks.
    Leaving this to the client means every client has to know the model's patch geometry and
    get it right, and one client sending full-resolution frames degrades latency for
    everyone sharing the server.

    Downscale only, never upscale: a client that sends small frames should not have them
    interpolated up into more tokens than it asked for.
    """
    img = _decode_frame_bytes(raw_bytes)
    w, h = img.size
    if w <= max_width and h <= max_height:
        return None
    scale = min(max_width / float(w), max_height / float(h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    # BILINEAR rather than LANCZOS: this is on the per-frame arrival path at the client's
    # frame rate, and the result is fed to a vision encoder that will downsample it again.
    resized = img.resize(new_size, Image.BILINEAR)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()



# ======================================================================================
# SESSION-SCOPED REQUESTS: one engine request per websocket session, not one per turn.
# Enabled per session via StreamingVideoSessionConfig.session_scoped_request.
#
# WHY. Today `_start_query_turn` mints a fresh request id every turn
# (`request_id = f"video-{uuid4().hex[:12]}"`) and calls generate() once per turn. Two
# costs follow, and the second is much the larger:
#
#   1. The thinker re-prefills the whole prompt. Mitigated by its prefix cache, which is
#      why a cached token costs 1.26 ms/1k against 79.2 for a new one.
#   2. The connector re-ships the WHOLE prompt's per-position embeddings AND last-layer
#      hidden states from stage 0 to stage 1, every turn. Two [L, 2048] bf16 tensors, so
#      8 KB per prompt position: 29 MB at 16 frames, 291 MB at 160. Measured at ~225
#      MB/s, that is 1.3 s of pure memory copy at turn 60, and it is 57-77% of the
#      talker's entire cost. It is invisible in the logs because rx_transfer_bytes and
#      rx_decode_time_ms are hardcoded to 0 in engine/stage_pool.py.
#
# Both collapse under one condition. In stage_input_processors/qwen3_omni.py the full
# payload is shipped only when `chunk_id == 0`; otherwise, if `request.resumable`, it
# calls `_construct_thinker2talker_streaming_input_async_chunk`, which ships
#     new_prompt_len = thinker_emb.shape[0]                 # only THIS forward's rows
#     ids.prompt     = request.prompt_token_ids[-new_prompt_len:]
# i.e. delta-sized tensors AND delta-sized token ids, so the talker's placeholder prompt
# also becomes delta-sized with no further change. `chunk_id` comes from
# `put_req_chunk[external_req_id]`, which is never reset per segment -- with one request
# id for the whole session it therefore keeps incrementing, so every turn after the first
# takes the delta branch.
#
# The engine side needs nothing new: `_update_request_as_session` extends the live
# request's prompt in place, leaves num_computed_tokens untouched, rebases the new
# multimodal features' offsets, and never frees the KV, so this is genuine incremental
# prefill rather than a prefix-cache re-hit.
#
# HOW. `generate()` already accepts an AsyncGenerator of StreamingInput
# (async_omni.py:398); that branch submits the first chunk with resumable=True and every
# later chunk through add_streaming_update_async. So the entrypoint has to become:
# one native async generator per session feeding per-turn deltas, one long-lived output
# loop, and per-turn websocket events driven off segment boundaries instead of off the
# end of the loop.
#
# WHAT IS NOT DONE HERE, stated plainly. Frames are still turned into tokens at QUERY
# time, not on arrival. Doing it on arrival would move the remaining work off the
# critical path, but it needs a frames-only chunk, and every chunk the engine accepts
# also runs the talker, which would emit audio nobody asked for. The deferred saving is
# small: at ~2.5 new frames per turn the delta is ~550 tokens, about 44 ms of prefill
# plus ~20 ms of copy, against the ~1,900 ms this change is aimed at.
#
# ======================================================================================

# Per-output diagnostic for the session output loop. Off during measurement arms: it logs
# once per engine output, which is hundreds of lines per turn.
#
# It exists because the FIRST bring-up attempt failed on exactly the question it answers.
# `output.finished` is never True under session mode, so the segment boundary was never
# detected: the turn never completed, the client timed out, and -- because the per-turn
# accumulator therefore never reset -- `drained` kept growing and turns 2+ emitted no audio
# at all. The cause is structural: orchestrator._route_output computes
#     request_finished = final_output_stage_ids.issubset(finished_final_output_stage_ids)
# with final_output_stage_ids == {0, 2}, while upstream vLLM force-clears `finished` on
# stage-0 outputs for streaming-input requests, so stage 0 never joins that set. Guessing
# the replacement signal through three layers of wrapping is how a wrong fix gets shipped,
# so this dumps what the outputs actually carry.
_LOG_SESSION_OUTPUTS = os.environ.get("VLLM_OMNI_LOG_SESSION_OUTPUTS", "0") not in ("0", "false", "False", "")

# <|im_end|>\n . Every delta after the first must start with these two tokens. The
# scheduler folds the previous segment's generated tokens into the prompt but drops the
# last sampled one, and for the thinker that dropped token is the EOS <|im_end|> -- so
# without this the previous assistant turn is left unterminated and the chat structure
# that _compute_talker_prompt_ids_length relies on is broken.
_IM_END_NEWLINE = [151645, 198]

# Codec tokens carried by one audio chunk, for the talker-token estimate in session mode.
# Recovered as 24.3 by solving  sum(placeholder) + k * sum(chunks) = 66664  at the point a
# stage-1 worker died writing a 66,664-token array into its 65,536-token buffer, and it agrees
# with the reference deployment's connector setting codec_chunk_frames: 25. A round 25 is used
# rather than the fitted value: the estimate exists to say how close the session is to a wall
# it must not hit, so erring high is the safe direction.
_TALKER_TOKENS_PER_AUDIO_CHUNK = 25


def _segment_finish_reason(output: Any) -> Any:
    """Per-SEGMENT finish marker, which is the only usable turn boundary in session mode.

    Measured on a 2-turn session with per-output logging (264 outputs):

        out.finished  == False on ALL 264            <- unusable
        ro.finished   == False on ALL 264            <- unusable
        finish_reason == "stop" on exactly 4:
            #141 text/stage0   #162 audio/stage2     <- turn 1 ends
            #250 text/stage0   #263 audio/stage2     <- turn 2 ends

    `output.finished` is the ORCHESTRATOR's aggregate, computed in _route_output as
    `final_output_stage_ids.issubset(finished_final_output_stage_ids)` with
    final_output_stage_ids == {0, 2}. Upstream vLLM force-clears `finished` on stage-0
    outputs for streaming-input requests, so stage 0 never joins that set and the
    aggregate can never become true. That is by design -- it is what keeps generate()
    alive across turns -- but it means the boundary has to come from the per-segment
    finish_reason instead, which the scheduler still populates on every segment stop.

    The audio one arrives last, so it is the one that closes a turn.
    """
    ro = getattr(output, "request_output", None)
    if ro is None:
        return None
    outs = getattr(ro, "outputs", None)
    if not outs:
        return None
    return getattr(outs[0], "finish_reason", None)


def _shift_mm_placeholders(engine_prompt: Any, shift: int) -> None:
    """Move every multimodal placeholder offset by `shift` tokens, in place.

    Needed because prepending <|im_end|>\\n to a rendered chunk moves every image and
    audio span. Written defensively: placeholders may be dataclasses with `.offset`, or
    plain dicts, depending on the vLLM version, and getting this wrong is silent -- the
    model would read image tokens at the wrong positions rather than raise.
    """
    if shift == 0 or not isinstance(engine_prompt, dict):
        return
    ph = engine_prompt.get("mm_placeholders")
    if not ph:
        return
    groups = ph.values() if isinstance(ph, dict) else [ph]
    for group in groups:
        for item in group if isinstance(group, (list, tuple)) else [group]:
            if isinstance(item, dict):
                if "offset" in item:
                    item["offset"] = int(item["offset"]) + shift
            elif hasattr(item, "offset"):
                try:
                    object.__setattr__(item, "offset", int(item.offset) + shift)
                except Exception:
                    logger.warning("[session] could not shift a placeholder offset")


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


class StreamingVideoSessionConfig(BaseModel):
    """Configuration sent as the first WebSocket message."""

    model: str | None = None
    modalities: list[str] = Field(
        default_factory=lambda: ["text", "audio"],
        description="Output modalities: 'text', 'audio', or both.",
    )
    num_frames: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Max frames to sample from buffer for the model.",
    )
    max_frames: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Max frames to keep in the buffer.",
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
    max_frame_width: int | None = Field(
        default=None,
        ge=32,
        le=8192,
        description=(
            "Downscale arriving frames to fit within this width, preserving aspect ratio. "
            "None disables it. A frame becomes (W/32)*(H/32) tokens, so halving each edge "
            "cuts a frame's prompt cost 4x; this is the cheapest lever on per-turn latency "
            "for a video stream. Downscale only -- smaller frames are never upscaled."
        ),
    )
    max_frame_height: int | None = Field(
        default=None,
        ge=32,
        le=8192,
        description="Companion to max_frame_width. Both must be set for downscaling to apply.",
    )
    frame_jpeg_quality: int = Field(
        default=90,
        ge=1,
        le=100,
        description="JPEG quality used when re-encoding a downscaled frame.",
    )
    frame_filter_max_gap: int = Field(
        default=0,
        ge=0,
        description=(
            "Retain a frame unconditionally once this many consecutive frames have been "
            "dropped by the similarity filter. 0 disables. Bounds BLINDNESS: the filter's "
            "metric is a whole-frame MSE on a 64x64 thumbnail, which barely moves when only "
            "a small region changes, so a screen share can go minutes without retaining "
            "anything even as its content changes completely."
        ),
    )
    frame_filter_min_gap: int = Field(
        default=0,
        ge=0,
        description=(
            "Refuse to retain again until this many frames have passed, however different "
            "they look. 0 disables. Bounds the BLOWUP: handheld camera motion makes almost "
            "every frame look new, so the filter retains most of them and the prompt grows "
            "without limit. Together the two bounds make video tokens per turn a property "
            "of the configuration rather than of what the camera happens to be pointed at."
        ),
    )
    session_scoped_request: bool = Field(
        default=False,
        description=(
            "Submit the whole session as ONE resumable engine request, feeding each turn as "
            "an incremental update, instead of a fresh request per turn. Each turn then "
            "prefills only its new frames, and the stage-0 -> stage-1 connector ships only "
            "the delta rather than re-copying the entire prompt's embeddings and hidden "
            "states every turn. Trades a per-turn blast radius for a per-session one: a "
            "failure ends the conversation rather than one turn, barge-in becomes "
            "drain-only because aborting would destroy the accumulated KV, and the "
            "session's context is bounded by max_model_len with no eviction. See the "
            "SESSION-SCOPED REQUESTS note at the top of this module."
        ),
    )
    session_talker_token_budget: int | None = Field(
        default=None,
        description=(
            "End the session cleanly once the TALKER's accumulated tokens are estimated to "
            "reach this many, instead of letting it hit the stage's max_model_len. Off by "
            "default: the running estimate is logged every turn either way, and enforcing a "
            "guessed number would cut sessions short. Set it to slightly under the stage-1 "
            "max_model_len of the deployment (65,536 in the reference config).\n\n"
            "WHY THIS EXISTS -- measured, not theoretical. The talker's per-turn prompt stays "
            "delta-sized, but the resumable request's stored token array does not: it grows "
            "every segment by the delta PLUS the audio codes the talker generated. Crossing "
            "max_model_len does not produce a clean error. Two things happen instead, both "
            "observed:\n"
            "  * the worker writes the prompt into a max_model_len-sized buffer and dies with "
            "`ValueError: could not broadcast input array from shape (66664,) into shape "
            "(65536,)`, taking the stage-1 engine core with it;\n"
            "  * or the scheduler's clamp `min(num_new_tokens, max_model_len - "
            "num_computed_tokens - num_sampled_tokens_per_step)` reaches 0 first and the "
            "running loop does `continue` -- upstream's own comment for that branch names "
            "'async scheduling and the request has reached max_model_len'. The request is then "
            "skipped on every pass forever: no crash, no log line, the client simply receives "
            "a turn's text and never its audio. Stage 1 runs the async scheduler, so this is "
            "the reachable path.\n\n"
            "Growth is well fitted by  sum(talker_placeholder) + 25 * sum(audio_chunks)  -- "
            "the 25 was recovered as 24.3 from a crash and matches the connector's configured "
            "codec_chunk_frames. So the session's LIFETIME is set by how much the model "
            "SPEAKS, not by how many turns it takes: measured runs reached 66% of the wall "
            "after 50 short-answer turns but 102% after 27 verbose ones."
        ),
    )


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
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raise NotImplementedError

    def on_turn_complete(
        self,
        message_history: list[dict[str, Any]],
        user_message: dict[str, Any],
        response_text: str,
    ) -> None:
        raise NotImplementedError

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

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return

            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            # Per-frame PIL cache + uuid for mm_hash reuse. Aligned with frame_buffer by index.
            frame_pil_cache: dict[str, tuple[Any, str] | object] = {}  # b64 -> (PIL.Image, uuid) or _BAD_FRAME
            frame_filter = (
                FrameSimilarityFilter(threshold=config.frame_filter_threshold) if config.enable_frame_filter else None
            )
            # Frames dropped by the similarity filter since the last retain. Drives
            # frame_filter_min_gap / frame_filter_max_gap.
            frames_since_retained = 0
            audio_buffer = bytearray()  # raw PCM16 16kHz mono
            message_history: Any = self.create_message_history(config)
            active_request_id: str | None = None
            prev_request_id: str | None = None  # abort target iff prev was interrupted
            prev_was_interrupted: bool = False
            interrupt_event = asyncio.Event()
            prewarm_tasks: set[asyncio.Task[Any]] = set()
            query_task: asyncio.Task[Any] | None = None

            msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_MSG_QUEUE)
            # ---------------------------------------------------------------- PA_SESSION
            sess: dict[str, Any] = {
                "queue": asyncio.Queue(maxsize=4),
                "gen_task": None,
                "turn_done": asyncio.Event(),
                "turn_idx": 0,
                "first_sent": False,
                "fatal": None,
            }
            session_request_id = f"video-sess-{uuid.uuid4().hex[:12]}"

            async def _chunk_stream():
                """Native async generator of per-turn deltas.

                MUST be a real `async def ... yield` generator. async_omni.py:398 branches
                on `isinstance(prompt, collections.abc.AsyncGenerator)`, whose
                __subclasshook__ requires asend/athrow/aclose as well as __aiter__ and
                __anext__, so a hand-rolled iterator class silently falls through to the
                one-shot per-turn path: no error raised, and the entire change becomes a
                no-op that looks like a null result.
                """
                from vllm.engine.protocol import StreamingInput

                while True:
                    item = await sess["queue"].get()
                    if item is None:
                        return
                    yield StreamingInput(prompt=item)

            def _new_turn_state() -> dict[str, Any]:
                return {
                    "text_parts": [], "prev_text": "", "text_done_sent": False,
                    "audio_chunks": 0, "drained": 0, "started": False,
                    "t0": _time.monotonic(), "t_first_text": None, "t_first_audio": None,
                }

            async def _session_output_loop() -> None:
                """ONE generate() for the whole session, demultiplexed back into turns.

                In per-turn mode the wire events response.audio.done / response.text.done
                are emitted AFTER the `async for` loop exits. Here the loop only exits at
                session end, so every per-turn event has to be driven off a segment
                boundary detected inside the loop instead.

                The boundary is `output.finished` on an audio output. That works because
                only stage 0 has a detokenizer: stages 1 and 2 are routed to
                `_process_mm_only_outputs`, which sets `finished` from the per-segment
                `finish_reason`, whereas upstream vLLM force-clears `finished` for
                streaming-input requests on stage 0. The same asymmetry is why per-request
                StageRequestStats keeps being emitted once per TURN for the talker and
                code2wav under session mode, and only stage 0's per-turn table is lost.
                """
                st = _new_turn_state()
                try:
                    result_gen = self._engine_client.generate(
                        prompt=_chunk_stream(),
                        request_id=session_request_id,
                        output_modalities=config.modalities,
                    )
                    async for output in result_gen:
                        if not isinstance(output, OmniRequestOutput):
                            continue
                        if _LOG_SESSION_OUTPUTS:
                            ro = getattr(output, "request_output", None)
                            co = None
                            if ro is not None:
                                outs = getattr(ro, "outputs", None)
                                co = outs[0] if outs else None
                            logger.info(
                                "[session-out] type=%s stage=%s out.finished=%s ro.finished=%s "
                                "finish_reason=%s ntok=%s audio_n=%s",
                                getattr(output, "final_output_type", "?"),
                                getattr(output, "stage_id", "?"),
                                getattr(output, "finished", None),
                                getattr(ro, "finished", None) if ro is not None else None,
                                getattr(co, "finish_reason", None) if co is not None else None,
                                len(getattr(co, "token_ids", ()) or ()) if co is not None else None,
                                (len(output.audio_data) if isinstance(getattr(output, "audio_data", None), list)
                                 else ("1" if getattr(output, "audio_data", None) is not None else "0")),
                            )
                        if interrupt_event.is_set():
                            continue
                        if not st["started"]:
                            await websocket.send_json({"type": "response.start"})
                            st["started"] = True

                        if getattr(output, "final_output_type", "text") == "audio":
                            if not st["text_done_sent"]:
                                await websocket.send_json(
                                    {"type": "response.text.done",
                                     "text": "".join(st["text_parts"])}
                                )
                                st["text_done_sent"] = True
                            if st["t_first_audio"] is None:
                                st["t_first_audio"] = _time.monotonic()
                            st["audio_chunks"] += 1
                            b64, st["drained"] = self._extract_audio_delta_b64(output, st["drained"])
                            if b64:
                                await websocket.send_json(
                                    {"type": "response.audio.delta", "data": b64, "format": "wav"}
                                )
                            if _segment_finish_reason(output) is not None:
                                await websocket.send_json({"type": "response.audio.done"})
                                logger.info(
                                    "[session] turn=%d done first_text=%.3fs "
                                    "first_audio=%.3fs audio_chunks=%d chars=%d",
                                    sess["turn_idx"],
                                    (st["t_first_text"] - st["t0"]) if st["t_first_text"] else -1.0,
                                    (st["t_first_audio"] - st["t0"]) if st["t_first_audio"] else -1.0,
                                    st["audio_chunks"], len("".join(st["text_parts"])),
                                )
                                # The audio the talker just generated is appended to its own
                                # accumulated token array, so it counts against the same
                                # max_model_len as the deltas do -- and it is the larger of
                                # the two terms on a talkative turn. Without this the estimate
                                # would track only the deltas and stay reassuringly small
                                # right up to the point where the stage dies.
                                sess["talker_tokens"] = (
                                    sess.get("talker_tokens", 0)
                                    + _TALKER_TOKENS_PER_AUDIO_CHUNK * st["audio_chunks"]
                                )
                                # message_history is deliberately NOT updated: under session
                                # mode the conversation lives in the engine request's KV, and
                                # a second copy in the entrypoint would be dead state that
                                # future readers would mistake for the source of truth.
                                st = _new_turn_state()
                                sess["turn_done"].set()
                        else:
                            delta, st["prev_text"] = self._extract_text_delta(output, st["prev_text"])
                            if delta:
                                if st["t_first_text"] is None:
                                    st["t_first_text"] = _time.monotonic()
                                st["text_parts"].append(delta)
                                await websocket.send_json(
                                    {"type": "response.text.delta", "delta": delta}
                                )
                            if _segment_finish_reason(output) is not None:
                                # The text segment ended. Audio normally closes the turn a
                                # little later; for a text-only session there would be no
                                # audio output at all, so close here instead.
                                if "audio" not in (config.modalities or []):
                                    if not st["text_done_sent"]:
                                        await websocket.send_json(
                                            {"type": "response.text.done",
                                             "text": "".join(st["text_parts"])}
                                        )
                                    st = _new_turn_state()
                                    sess["turn_done"].set()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.exception("[session] output loop failed")
                    sess["fatal"] = str(e)
                    sess["turn_done"].set()   # never leave a turn waiting forever

            async def _run_session_turn(*, query_text: str) -> None:
                """Serialise turns, and refuse to overlap two of them.

                Every query arrives as its own task, and nothing here stopped a second
                turn from starting while the first was still waiting for its segment
                boundary. A 30-turn run hit exactly that: turn 28's boundary never came,
                the client gave up after ~127s and sent the next query, and the second
                call re-entered this function with ``turn_idx`` still 28. It logged the
                same turn number twice, swept the 32 frames that had piled up during the
                stall into one 8,768-token delta, and shared ``turn_done`` and the segment
                accumulator with the call still in flight. Nothing raised.

                Two turns cannot both be served correctly out of per-session state shaped
                like this, so a query that arrives mid-turn fails the session loudly
                instead of producing a turn whose bookkeeping is already wrong. It also
                keeps the diagnosis honest: the overlap was a consequence of the stall,
                not its cause, and letting it through buried the real event under a turn
                index that appeared twice with two different frame counts.
                """
                if sess.get("turn_busy"):
                    logger.error(
                        "[session] turn=%d is still in flight and another query "
                        "arrived -- refusing to overlap turns",
                        sess["turn_idx"],
                    )
                    sess["fatal"] = "overlapping turn"
                    await self._send_error(websocket, "Overlapping turn")
                    return
                sess["turn_busy"] = True
                try:
                    await _run_session_turn_body(query_text=query_text)
                finally:
                    sess["turn_busy"] = False

            async def _run_session_turn_body(*, query_text: str) -> None:
                """Queue this turn's delta, then wait for its audio to finish.

                Strictly turn-by-turn: the client waits for response.audio.done before
                sending the next query, so pipelining would buy nothing here and would
                complicate the segment bookkeeping.
                """
                if sess["fatal"]:
                    await self._send_error(websocket, f"Session failed: {sess['fatal']}")
                    return

                # Consume the backlog rather than tracking a cursor into it.
                #
                # An integer "frames already submitted" cursor is wrong here, because the
                # max_frames guard evicts from the FRONT of frame_buffer. Once that fires,
                # every index shifts and the cursor silently points at the wrong frame:
                # frames get skipped or resubmitted, with no error and nothing in the logs.
                # In session mode the buffer's only job is to hold frames that have not been
                # submitted yet, so it can simply be drained -- which also means it stays a
                # handful of frames long and the eviction path never fires at all.
                new_frames = list(frame_buffer)
                chunk = await self._build_session_chunk(
                    config, new_frames, audio_buffer, query_text, frame_pil_cache,
                    is_first=not sess["first_sent"],
                )
                audio_buffer.clear()
                if chunk is None:
                    # Nothing to submit: keep the frames for the next turn rather than
                    # dropping them on the floor.
                    await self._send_error(websocket, "Nothing new to submit this turn")
                    return
                # Delete exactly the consumed prefix, not the whole buffer: frames may have
                # arrived while the chunk was being built, and those belong to the next turn.
                del frame_buffer[: len(new_frames)]
                for consumed in new_frames:
                    frame_pil_cache.pop(consumed, None)

                ids = (chunk.get("prompt_token_ids") or ()) if isinstance(chunk, dict) else ()
                ntok = len(ids)
                sess["cum_tokens"] = sess.get("cum_tokens", 0) + ntok
                # Session mode loses StageRequestStats entirely -- those tables are printed
                # when a request FINISHES, and a resumable session request never does. So
                # the two quantities the measurement needs have to be logged here instead:
                #
                #   cum   the thinker's accumulated prompt length, i.e. the x-axis. It is
                #         the running sum of the deltas, because nothing else reports it.
                #   tlen  what the talker's placeholder will be for this delta, from the
                #         SAME function the connector uses. This is the direct check that
                #         the delta-shipping branch is doing its job: tlen must stay small
                #         and roughly constant while cum grows.
                tlen = -1
                try:
                    from vllm_omni.distributed.omni_connectors.adapter import (
                        compute_talker_prompt_ids_length,
                    )
                    tlen = compute_talker_prompt_ids_length(list(ids))
                except Exception:
                    pass
                # Running estimate of the TALKER's accumulated tokens, which is what actually
                # bounds the session -- see session_talker_token_budget for the measurements.
                # Logged unconditionally because the wall is otherwise invisible: crossing it
                # either kills the stage-1 engine core on a numpy broadcast or makes the
                # scheduler skip the request forever with no output at all.
                if tlen > 0:
                    sess["talker_tokens"] = sess.get("talker_tokens", 0) + tlen
                budget = config.session_talker_token_budget
                logger.info(
                    "[session] turn=%d queue delta: %d new frames, %d tokens, "
                    "cum=%d, talker_placeholder=%d, talker_est=%d%s, first=%s",
                    sess["turn_idx"], len(new_frames), ntok,
                    sess["cum_tokens"], tlen, sess.get("talker_tokens", 0),
                    f"/{budget}" if budget else "", not sess["first_sent"],
                )
                if budget and sess.get("talker_tokens", 0) >= budget:
                    # Refuse the turn rather than submit one that may not come back. Ending
                    # here is a real limitation, not a fix: the conversation is over. The
                    # actual repair is to roll the session -- close this engine request and
                    # open a fresh one seeded with the text history, paying one cold turn to
                    # keep talking -- which is a larger change than a guard.
                    logger.error(
                        "[session] turn=%d REFUSED: the talker's accumulated tokens are "
                        "estimated at %d, at or past the configured budget of %d. Submitting "
                        "it risks max_model_len, which does not fail cleanly: it either kills "
                        "the stage-1 engine core or makes the scheduler skip the request "
                        "silently forever. Ending the session instead.",
                        sess["turn_idx"], sess.get("talker_tokens", 0), budget,
                    )
                    sess["fatal"] = "talker token budget exhausted"
                    await self._send_error(
                        websocket,
                        f"Session ended: the talker's context budget ({budget} tokens) is "
                        f"exhausted after {sess['turn_idx']} turns. Start a new session.",
                    )
                    return
                interrupt_event.clear()
                sess["turn_done"].clear()
                if sess["gen_task"] is None:
                    sess["gen_task"] = asyncio.create_task(_session_output_loop())
                sess["first_sent"] = True
                await sess["queue"].put(chunk)
                # Bounded wait. A lost segment boundary must surface as an error rather
                # than a hang: the first bring-up attempt used an unusable boundary signal
                # and the symptom was the client sitting in its own timeout with no server
                # log to explain it. The bound is generous because a static-content turn
                # can legitimately produce ~105 s of speech.
                try:
                    await asyncio.wait_for(sess["turn_done"].wait(), timeout=240.0)
                except asyncio.TimeoutError:
                    logger.error(
                        "[session] turn=%d boundary NEVER ARRIVED after 240s -- the "
                        "segment-finish signal is wrong for this configuration",
                        sess["turn_idx"],
                    )
                    sess["fatal"] = "turn boundary lost"
                    await self._send_error(websocket, "Turn boundary lost")
                    return
                sess["turn_idx"] += 1

            async def _close_session_request() -> None:
                """End the session by CANCELLING, deliberately not by the finish sentinel.

                The obvious wind-down is to let the chunk generator return, which makes
                async_omni send a terminal `resumable=False` update. That KILLS THE TALKER:
                the sentinel prompt is `TokensPrompt(prompt_token_ids=[0])`, so stage 1's
                placeholder length comes out 0 and its scheduler trips
                `assert num_new_tokens > 0`, taking down the engine core process. Observed
                on the second bring-up: three turns completed correctly and then the
                sentinel crashed stage 1.

                `handle_inputs` sends that sentinel only `if not cancelled`
                (async_omni.py), and generate()'s own CancelledError handler already
                cancels the input-stream task and calls `_abort_internal_requests`. So
                cancelling the outer task is both the clean path and the one that skips the
                sentinel. Abort at session end is safe -- unlike a mid-session abort, which
                would destroy the accumulated KV this whole mode exists to preserve.
                """
                task = sess["gen_task"]
                if task is None:
                    return
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                # NO second abort here. generate()'s own CancelledError handler already
                # calls _abort_internal_requests, and adding another abort only doubles it.
                #
                # KNOWN UPSTREAM FRAGILITY, measured: ending a resumable session request
                # takes down the stage-1 engine core process with
                # `assert num_new_tokens > 0` in its scheduler. Both ways of ending it do
                # it -- the terminal resumable=False sentinel (whose prompt is
                # TokensPrompt([0]), giving the talker a zero-length placeholder) and the
                # abort. It happens strictly AFTER the last turn has been delivered, so no
                # turn is affected, but the stage-1 process is gone afterwards, which means
                # a second session on the same server would fail. The experiment therefore
                # runs one server boot per session rather than four sessions per boot.
                # Fixing it properly belongs upstream, in stage 1's handling of a resumable
                # request that is ending.


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

            async def _start_query_turn(*, query_text: str) -> None:
                """Schedule a new inference turn from the current buffers."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task

                if config.session_scoped_request:
                    # Run as a task, as the per-turn path does, so the processor keeps
                    # draining msg_queue and frames arriving during the turn are still
                    # buffered. `active_request_id` intentionally stays None: that makes
                    # every abort site in this file a no-op, which is exactly the
                    # drain-only barge-in behaviour session mode needs. A mid-session
                    # abort would pop the engine request and destroy the accumulated KV
                    # that is the entire point of this mode.
                    query_task = asyncio.create_task(_run_session_turn(query_text=query_text))
                    return

                await _cancel_active_query()

                if not frame_buffer:
                    await self._send_error(websocket, "No frames buffered")
                    return

                if prev_was_interrupted and prev_request_id and self._engine_client:
                    try:
                        await self._engine_client.abort(prev_request_id)
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)
                prev_was_interrupted = False

                request_id = f"video-{uuid.uuid4().hex[:12]}"
                active_request_id = request_id
                interrupt_event.clear()
                query_frames = list(frame_buffer)
                query_audio_buffer = bytearray(audio_buffer)
                audio_buffer.clear()
                query_prewarmed_frames = dict(frame_pil_cache)

                async def _run_query() -> None:
                    nonlocal active_request_id, prev_request_id
                    try:
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
                        )
                    finally:
                        if active_request_id == request_id:
                            prev_request_id = request_id
                            active_request_id = None

                query_task = asyncio.create_task(_run_query())

            async def _processor() -> None:
                """Process enqueued messages."""
                nonlocal active_request_id, prev_request_id, prev_was_interrupted, query_task
                nonlocal frames_since_retained

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
                            frame_buffer[:] = [f for f in frame_buffer if f != frame_data]
                        if frame_pil_cache.get(frame_data) is _BAD_FRAME:
                            frame_pil_cache.pop(frame_data, None)
                        if removed:
                            await self._send_error(websocket, "Frame decode failed")

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

                        # Downscale BEFORE anything else looks at the frame, so that the
                        # similarity filter, the prewarm PIL cache (keyed on these bytes),
                        # the multimodal hash and the prompt all see one consistent version.
                        # Doing it later would leave the filter comparing full-resolution
                        # frames while the model reads reduced ones.
                        #
                        # In a thread, for the same reason the prewarm below decodes in one:
                        # this coroutine also forwards generated audio to the client, so CPU
                        # spent inline here lands in somebody's time-to-first-audio. Decoding,
                        # resizing and re-encoding a frame is strictly more work than the bare
                        # decode upstream already judged worth offloading. Awaiting cannot
                        # reorder frames -- this loop reads one message at a time, so the next
                        # frame is not picked up until this one has been buffered.
                        if config.max_frame_width and config.max_frame_height:
                            try:
                                shrunk = await asyncio.to_thread(
                                    _downscale_frame_bytes,
                                    raw_bytes,
                                    config.max_frame_width,
                                    config.max_frame_height,
                                    config.frame_jpeg_quality,
                                )
                            except Exception:
                                logger.debug("Frame downscale failed; keeping original", exc_info=True)
                                shrunk = None
                            if shrunk is not None:
                                raw_bytes = shrunk
                                frame_data = base64.b64encode(shrunk).decode("ascii")
                        if frame_filter is not None:
                            try:
                                # Bracket the GAP between retained frames, in frames. The
                                # similarity filter itself is untouched; these two bounds only
                                # constrain how often it is allowed to say yes or no.
                                #
                                # Counting frames rather than seconds keeps the behaviour
                                # independent of whatever rate the client happens to send at.
                                #
                                # MIN_GAP is checked first and short-circuits, so a burst of
                                # genuinely different frames cannot flood the prompt. MAX_GAP
                                # then forces a retain by clearing the filter's reference
                                # frame, which is what makes the next comparison succeed --
                                # calling should_retain() with a stale reference is exactly
                                # how a static-looking stream stays invisible.
                                frames_since_retained += 1
                                if config.frame_filter_min_gap and frames_since_retained < config.frame_filter_min_gap:
                                    continue
                                if config.frame_filter_max_gap and frames_since_retained >= config.frame_filter_max_gap:
                                    frame_filter.force_next_retain()
                                if not frame_filter.should_retain(raw_bytes):
                                    continue
                                frames_since_retained = 0
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        max_buf = config.max_frames
                        if len(frame_buffer) >= max_buf:
                            dropped = frame_buffer.pop(0)
                            frame_pil_cache.pop(dropped, None)
                        frame_buffer.append(frame_data)
                        self.on_frame_buffered(raw_bytes, frame_data, message_history, config)
                        # Prewarm: decode PIL off the event loop so query-time chat_template
                        # can skip base64+Image.open. uuid=md5 lets mm_cache dedupe identical frames.
                        if frame_data not in frame_pil_cache:
                            mm_uuid = hashlib.md5(raw_bytes, usedforsecurity=False).hexdigest()

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

                            task = asyncio.create_task(_prewarm(frame_data, raw_bytes, mm_uuid))
                            prewarm_tasks.add(task)
                            task.add_done_callback(prewarm_tasks.discard)

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
                            continue
                        audio_buffer.extend(pcm_bytes)

                    elif msg_type == "video.query":
                        query_text = msg.get("text", "")
                        audio_data_b64 = msg.get("audio_data")
                        if audio_data_b64:
                            try:
                                decoded = base64.b64decode(audio_data_b64)
                                if len(audio_buffer) + len(decoded) <= _MAX_AUDIO_BUFFER_BYTES:
                                    audio_buffer.extend(decoded)
                                else:
                                    await self._send_error(websocket, "Audio buffer overflow")
                                    audio_buffer.clear()
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
                if config.session_scoped_request:
                    # Wind the session request down before anything else cancels tasks:
                    # the terminal resumable=False update is the only thing that finishes
                    # a resumable request, and it is sent when the chunk generator returns.
                    await _close_session_request()
                for t in list(prewarm_tasks):
                    t.cancel()
                if prewarm_tasks:
                    await asyncio.gather(*prewarm_tasks, return_exceptions=True)
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
            "num_sample_frames": "num_frames",
            "evs_enabled": "enable_frame_filter",
            "evs_threshold": "frame_filter_threshold",
        }
        for old_key, new_key in alias_map.items():
            if old_key in config_data and new_key not in config_data:
                config_data[new_key] = config_data[old_key]

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
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

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
        )


    # ------------------------------------------------------------------
    # PA_SESSION: build ONE per-turn delta
    # ------------------------------------------------------------------

    async def _build_session_chunk(
        self,
        config: StreamingVideoSessionConfig,
        new_frames: list[str],
        audio_buffer: bytearray,
        query_text: str,
        prewarmed_frames: dict[str, tuple[Any, str]],
        *,
        is_first: bool,
    ) -> Any:
        """Render one per-turn delta into an engine prompt, or None if it would be empty.

        The delta must be a SELF-CONTAINED chatml unit:

            <|im_start|>user\\n {frames}{audio}{text} <|im_end|>\\n<|im_start|>assistant\\n

        Both halves are load-bearing and neither fails loudly:

        * Without the `<|im_start|>user` header, `compute_talker_prompt_ids_length`
          (adapter.py) finds no im_start, returns 0, and the caller wraps it in
          `max(1, ...)` -- so the talker gets a ONE-token placeholder and the worker's
          `seg_len = min(span_len, req_embeds.shape[0])` silently keeps one row of
          conditioning and discards the rest. No exception, wrong audio.
        * Without the trailing assistant header the same function loses its +9 and
          mis-sizes the placeholder.

        `add_generation_prompt=True` produces exactly that shape, and the Qwen3-Omni chat
        template emits no default system block when messages[0] is not a system message,
        which is what lets chunks 2..N carry only a user block.

        Empty chunks are refused: the waiting-path scheduler asserts num_new_tokens > 0,
        so a turn contributing nothing would take down the engine core.
        """
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        prewarmed = prewarmed_frames or {}
        user_content: list[dict] = []
        for frame_b64 in new_frames:
            cached = prewarmed.get(frame_b64)
            if cached is _BAD_FRAME:
                continue
            if cached is not None:
                pil, pil_uuid = cached
                user_content.append({"type": "image_pil", "image_pil": pil, "uuid": pil_uuid})
            else:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
                    }
                )

        has_audio = len(audio_buffer) > 0
        if has_audio:
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": self._pcm_to_wav_b64(bytes(audio_buffer)),
                        "format": "wav",
                    },
                }
            )
        if query_text:
            user_content.append({"type": "text", "text": query_text})

        if not user_content:
            return None

        messages: list[dict[str, Any]] = []
        # The system block belongs to the session, so it is sent exactly once. Repeating
        # it per chunk would put it in the middle of the sequence, where
        # compute_talker_prompt_ids_length skips it and the model sees an instruction
        # block interleaved with the conversation.
        if is_first and config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": user_content})

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.use_audio_in_video and has_audio:
            request_kwargs["mm_processor_kwargs"] = {"use_audio_in_video": True}

        chat_request = ChatCompletionRequest(**request_kwargs)
        engine_prompt = await self._preprocess_to_engine_prompt(chat_request)

        if not is_first and isinstance(engine_prompt, dict):
            ids = engine_prompt.get("prompt_token_ids")
            if ids is not None:
                engine_prompt["prompt_token_ids"] = list(_IM_END_NEWLINE) + list(ids)
                _shift_mm_placeholders(engine_prompt, len(_IM_END_NEWLINE))

        return engine_prompt
    # ------------------------------------------------------------------
    # Engine-client path (async_chunk audio streaming)
    # ------------------------------------------------------------------

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
    ) -> None:
        """Direct engine_client.generate() path for async_chunk audio."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        messages, user_message = self.build_engine_prompt(
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            prewarmed_frames,
        )

        request_kwargs: dict[str, Any] = {
            "model": config.model or "default",
            "messages": messages,
            "stream": True,
            "modalities": config.modalities,
            "add_generation_prompt": True,
            "continue_final_message": False,
            "add_special_tokens": False,
        }
        if config.use_audio_in_video and len(audio_buffer) > 0:
            request_kwargs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
        if config.sampling_params_list:
            request_kwargs["sampling_params_list"] = config.sampling_params_list

        try:
            chat_request = ChatCompletionRequest(**request_kwargs)
        except Exception as e:
            await self._send_error(websocket, f"Failed to build request: {e}")
            return

        try:
            engine_prompt = await self._preprocess_to_engine_prompt(chat_request)
        except Exception as e:
            await self._send_error(websocket, f"Preprocess failed: {e}")
            return

        await websocket.send_json({"type": "response.start"})
        text_parts: list[str] = []
        text_done_sent = False
        audio_chunk_count = 0
        # Number of per-step tensors in OmniRequestOutput.audio_data already
        # drained. Used by the fast path to skip already-emitted history.
        audio_chunks_drained = 0
        previous_text = ""
        interrupted = False
        t_start = _time.monotonic()
        t_first_text = None
        t_first_audio = None

        # Wire-level async-chunk switch. "off" means
        # buffer all deltas server-side and flush once at the end; the engine
        # pipeline still overlaps internally.
        async_chunk_mode = video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK
        streaming = async_chunk_mode == "on"
        audio_tail_tensors: list[Any] = []

        try:
            result_gen = self._engine_client.generate(
                prompt=engine_prompt,
                request_id=request_id,
                output_modalities=config.modalities,
            )

            async for output in result_gen:
                # Soft interrupt: drain without sending
                if interrupt_event.is_set():
                    if not interrupted:
                        logger.info("Generation interrupted — draining")
                        interrupted = True
                    continue

                if not isinstance(output, OmniRequestOutput):
                    continue

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
            self.on_turn_complete(message_history, user_message, response_text)

            t_end = _time.monotonic()
            logger.info(
                "[TIMING] mode=%s total=%.2fs first_text=%.2fs first_audio=%.2fs audio_chunks=%d",
                async_chunk_mode,
                t_end - t_start,
                (t_first_text - t_start) if t_first_text else -1,
                (t_first_audio - t_start) if t_first_audio else -1,
                audio_chunk_count,
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
        # Single tensor: output_processor hands us one tensor before it becomes a
        # list (see output_processor.py:89). Treat it as chunk #0.
        if not isinstance(audio_data, list):
            if chunks_drained >= 1:
                return None, chunks_drained
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(tail_np, chunks_drained, new_drained=1, is_first=True)

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
            new_drained = 1

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

    _text_only_message = staticmethod(text_only_message)

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass
