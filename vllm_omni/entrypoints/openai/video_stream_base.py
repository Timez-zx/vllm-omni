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
import hashlib
import io
import json
import math
import os
import time as _time
import uuid
import wave
from collections import deque
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
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    PREFILL_ONLY_KEY as _TTS_PREFILL_ONLY_KEY,
)
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

# Marker for an append that must be prefilled but never answered. Kept identical to
# tts_utils.PREFILL_ONLY_KEY, and asserted equal at import so the two cannot drift: the
# failure mode of a mismatch is the model speaking unasked, which is loud but points at
# the wrong place.
_PREFILL_ONLY_KEY = "vllm_omni_prefill_only"

assert _PREFILL_ONLY_KEY == _TTS_PREFILL_ONLY_KEY, (
    "prefill-only marker key drifted between the entrypoint and the stage processors; the "
    "engine would stop recognising it and the talker would speak unasked"
)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_MAX_FRAME_SIZE = 10 * 1024 * 1024  # 10MB per frame

# [Tick engine WP7] Frame-tick mailbox. When enabled, an arriving frame is NOT
# appended to the engine immediately: it waits in frame_buffer (which already
# is the mailbox -- refused frames stay there today) and a per-session flush
# fires at the next edge of the global tick grid. Every session quantizes the
# same CLOCK_MONOTONIC to the same grid -- the exact scheme the engine-side
# pacer uses (temporal_pacing.py) -- so the sessions' flushes align WITHOUT
# any shared registry, and their frame appends reach stage 0 inside one tick
# window where the vision encoder batches them in one forward
# (_execute_mm_encoder batches everything co-scheduled in a pass). Greedy
# arrival phase is random per user, so without this the encoder runs batch=1
# per frame: 48 users x 480 ms = ~100 scattered encoder calls/s.
# Gated separately from VLLM_OMNI_TEMPORAL_TICK_MS because the M arm sets the
# tick env too; this is a T-family treatment and must be opt-in.
from vllm_omni.core.sched.temporal_pacing import live_env, live_env_on as _live_env_on

_FRAME_TICK_S: float = 0.0
if _live_env_on("VLLM_OMNI_TEMPORAL_FRAME_TICK"):  # live-vllm: default ON
    _FRAME_TICK_S = max(
        0.0, float(live_env("VLLM_OMNI_TEMPORAL_TICK_MS") or 0.0)
    ) / 1000.0
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

# Seed-budget cost of one retained frame, for the rolling-window trim. Measured 226
# tokens/frame at 640x352 (boot_shadow20c.log: 9-frame chunks = 2,03x tokens); 250 errs
# high so the first trim guess lands under budget and the rebuild loop rarely runs.
_SEED_TOKENS_PER_FRAME = 250

# Floor for the pool-guarded talker roll threshold. Below this, rolling every
# few turns would cost more than it saves (a roll turn pays ~3.7x TTFA), so a
# session that cannot be granted even this much of the stage-1 KV pool is
# refused at admission instead of being admitted into a preemption storm:
# at 64 sessions on a 116,384-token pool, every talker request was preempted
# exactly once and the recompute storm pushed p99 to 57 s.
_TALKER_ROLL_FLOOR = 2048
# Transcript retention when compression carries frames. The blocking-roll default
# (2 * session_roll_history_turns = 16 entries) binds far below a 32k rolling window
# (~25 turns of frames+text), so compression raises the floor to 48 turns of entries...
_COMPRESSION_TRANSCRIPT_KEEP = 96
# ...but strips the FRAMES off entries older than the newest 40 user turns: only the
# window (plus margin) can ever render them, and 40 turns x ~5 frames x ~40 KB of JPEG
# is a few MB per session where unbounded retention would grow forever.
_FRAME_KEEP_TURNS = 40
# Process-wide cap on concurrent shadow warm-ups. A warming seed holds up to
# target_tokens of KV on top of the live requests. The original 2 was sized for
# the 13-user/49k-trigger regime (731,904-token pool - 13 x 49,152 = 92,928,
# which fits TWO 32k seeds and not three). Under per-N-scaled triggers
# (trigger = 0.75 * pool / N, target = trigger / 2) the steady state uses about
# half the pool and a seed is only ~2-4k tokens, so six warm-ups cost < 8% of
# the pool -- and 2 was the bottleneck that turned synchronized trigger
# crossings into blocking-roll waves (39-63% of compressions degraded at u48).
# The permit covers warm-up only (seed submit -> ready); a hook that finds the
# limit busy skips silently and re-fires at the next hook, because cum_tokens
# keeps growing until a swap resets it.
_MAX_CONCURRENT_SHADOW_WARMUPS = 6


def _summarise_audio_payload(audio_data: Any) -> str:
    """Describe the audio payload the delta extractor will actually read.

    Shapes only, no device-to-host copy, so this is safe to call on every output.

    The point is to make the payload's CONTRACT visible. `_extract_audio_delta_b64`
    assumes a cumulative list that grows by one tensor per step; if the list is drained
    after each snapshot instead, it stays length 1 and the extractor emits only the very
    first granule. Those two cases are indistinguishable from a single output and obvious
    from a sequence of them, so the length is logged per output.
    """
    if audio_data is None:
        return "none"
    if isinstance(audio_data, list):
        lens = [int(getattr(t, "shape", (0,))[-1] or 0) for t in audio_data]
        return f"list(n={len(lens)},samples={sum(lens)},each={lens[:6]})"
    n = int(getattr(audio_data, "shape", (0,))[-1] or 0)
    return f"tensor(samples={n})"


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


def _mm_placeholder_span(engine_prompt: Any) -> tuple[int, int] | None:
    """First and last token index covered by multimodal placeholders, or None."""
    if not isinstance(engine_prompt, dict):
        return None
    ph = engine_prompt.get("mm_placeholders")
    if not ph:
        return None
    lo, hi = None, None
    groups = ph.values() if isinstance(ph, dict) else [ph]
    for group in groups:
        for item in group if isinstance(group, (list, tuple)) else [group]:
            off = item.get("offset") if isinstance(item, dict) else getattr(item, "offset", None)
            length = item.get("length") if isinstance(item, dict) else getattr(item, "length", None)
            if off is None:
                continue
            off = int(off)
            end = off + int(length or 1)
            lo = off if lo is None else min(lo, off)
            hi = end if hi is None else max(hi, end)
    return None if lo is None else (lo, hi)


def _strip_chatml_scaffolding(engine_prompt: Any) -> bool:
    """Reduce a rendered delta to just its multimodal run, headers removed. In place.

    THE fix for the prefill-on-arrival crash, and the only variant that matches the model's
    own structure instead of working around it.

    A frames-on-arrival append rendered as a normal chunk is a COMPLETE chatml turn --
    `<|im_start|>user … <|im_end|><|im_start|>assistant` -- i.e. a user turn nobody answers.
    `_compute_talker_prompt_ids_length` walks im_start boundaries and adds its +9 only for
    the LAST one, so an unanswered turn in the middle shifts the talker's placeholder span,
    and a shifted span feeds text-vocabulary ids into a ~4k-row codec embedding:
    `indexSelectSmallIndex: srcIndex < srcSelectDimSize`, stage 1 dead, engine gone.

    Both obvious alternatives were tried and measured to fail identically: withholding the
    append from the talker (three gates confirmed firing) and letting it through untouched.
    The crash is not about where the append is stopped; it is that the append is a TURN.

    So the frames extend the user's in-progress utterance instead. Keeping only the
    multimodal run means the eventual query's delta closes the same user block, and the
    token sequence the model sees is the same one it would have seen without this feature --
    only the timing of the prefill differs, which was the entire point.

    One token of margin each side keeps Qwen's `<|vision_start|>` / `<|vision_end|>`
    markers, which wrap the placeholder run; without them the frame is not a frame.
    Returns False if the shape is not what was expected, and the caller then declines the
    append rather than sending something malformed.
    """
    if not isinstance(engine_prompt, dict):
        return False
    ids = engine_prompt.get("prompt_token_ids")
    span = _mm_placeholder_span(engine_prompt)
    if not ids or span is None:
        return False
    lo, hi = span
    lo = max(0, lo - 1)
    hi = min(len(ids), hi + 1)
    if hi <= lo:
        return False
    engine_prompt["prompt_token_ids"] = list(ids[lo:hi])
    _shift_mm_placeholders(engine_prompt, -lo)
    return True


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
        description=(
            "Max frames to keep in the buffer; the oldest is evicted when it is full.\n\n"
            "In session mode this is a LATENCY cap rather than a memory one, and the "
            "live-agent configs set it to 8. The buffer only holds frames whose "
            "on-arrival prefill was REFUSED -- a turn is in flight (the frame would "
            "land between a turn's chunk and its answer), or a compression shadow is "
            "warming (the frame would land in KV that is about to be discarded) -- and "
            "the next turn submits the WHOLE buffer as one chunk. Measured cost of a "
            "turn is 348 ms + 59 ms per frame single-user, so an unbounded buffer turns "
            "a stalled moment into a 13-19 frame sweep: 876 ms single-user and ~4.7 s "
            "at 13 users, which is where the 13-user swap-window peak came from. "
            "Evicting the OLDEST is the right policy on a live feed: the newest frames "
            "are the ones the answer is about."
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
    fresh_frame_on_query: bool = Field(
        default=True,
        description=(
            "Ride the newest arrived frame at the END of each query's delta, bypassing the "
            "similarity filter. Without it the frames adjacent to the question are the ones "
            "REFUSED during the previous answer -- the oldest in the delta -- and the model "
            "reads the frames nearest the question as 'now': measured with a digit clock, "
            "answers ran one query-gap stale at 3 fps (median 4 s) and the filter's "
            "frame-denominated min_gap added an 8-frame blind window on top. Costs at most "
            "one duplicate frame (~222 tokens) per turn."
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
    session_roll_at_talker_tokens: int | None = Field(
        default=None,
        description=(
            "ROLL the session when the talker's estimated tokens reach this, instead of ending "
            "it: close the engine request and open a fresh one seeded with the recent text "
            "transcript, so the conversation continues indefinitely. Set this BELOW "
            "session_talker_token_budget -- rolling keeps the session alive, the budget only "
            "stops it dying badly, so the roll should always get there first.\n\n"
            "The cost is one cold turn per roll: the new request has to prefill the seed, and "
            "the accumulated visual KV is gone. What survives is text. Frames already in the "
            "buffer are re-sent with the first post-roll turn, so the model still sees the "
            "present -- it loses the older visual detail, which is the documented tradeoff "
            "measured for text-only memory elsewhere in this study (recall of spoken content "
            "held at 8/8; fine visual detail did not survive)."
        ),
    )
    stage1_kv_pool_tokens: int | None = Field(
        default=None,
        description=(
            "Size of the SPEECH stage's shared KV pool in tokens (read it off the boot "
            "log's stage-1 'GPU KV cache size' line). When set, the talker roll threshold "
            "is lowered per turn to 0.75 * pool / active_sessions, so sessions roll before "
            "the shared pool fills -- the per-session threshold alone cannot see the pool: "
            "at 64 sessions on a 116,384-token pool every talker request was preempted and "
            "recompute storms pushed p99 to 57 s while bandwidth and compute sat idle. "
            "Same family as the pool-aware compression trigger: the wall is shared, the "
            "guard must divide by the number of tenants. A session whose share would fall "
            "below the roll floor is refused at admission (graceful 'at capacity' instead "
            "of a preemption storm). None = guard off."
        ),
    )
    session_roll_settle_s: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Seconds to wait after retiring the old engine request before submitting the first "
            "chunk of the new one.\n\n"
            "WORKAROUND, not a fix, and it is here because of a measured failure. The API "
            "server orders the teardown correctly -- the old request is aborted and the "
            "orchestrator confirms it before the new one is added -- but each stage is a "
            "SEPARATE OS PROCESS, so stage 1 can process 'add new' before it processes 'abort "
            "old', and the abort's cleanup touches adapter state. Observed: the first roll "
            "submitted its seeded chunk, stage 0 produced a complete 329-output answer, and "
            "stage 1 produced NOTHING at all, so the turn never closed. Costs one wait per "
            "roll, i.e. once every few thousand tokens of speech. The real fix belongs "
            "upstream, in making a stage's request teardown observable so this can be awaited "
            "instead of slept on."
        ),
    )
    session_roll_history_turns: int = Field(
        default=8,
        ge=0,
        description=(
            "How many recent turns of TEXT to carry across a roll. Bounded on purpose: the "
            "seed is prefilled into the new request, so an unbounded transcript would grow "
            "every roll until the seed alone approached the wall the roll exists to avoid. "
            "This is what makes the session unbounded in TIME while bounded in MEMORY."
        ),
    )
    context_compression_trigger_tokens: int | None = Field(
        default=None,
        description=(
            "Compress the session when the THINKER's accumulated context (the running sum "
            "of every delta's tokens, frames included) reaches this. Same knob Gemini Live "
            "calls trigger_tokens. The wall it guards is context growth itself: per-frame "
            "prefill cost grows linearly with accumulated context, and stage-0 "
            "max_model_len is a hard ceiling behind it -- neither of which the talker-side "
            "trigger above ever looks at.\n\n"
            "Default (None) resolves to 75% of stage-0 max_model_len at session start, "
            "and that line is UNIFORM -- one user or many (Xiao's call): compression has "
            "exactly one job, lifetime; the latency a thick history causes under load "
            "belongs to scheduling, not to this knob. The uniform line carries a hard "
            "capacity rule: concurrent sessions <= KV pool / trigger (this card: ~14, "
            "take 13 for margin). Measured both sides: 13 users x 48 turns completed a "
            "clean sawtooth with zero deaths; 16 users exhausted the pool at ~45k/user, "
            "one step SHORT of the trigger, and every session lost its turn signals. "
            "0 disables compression entirely.\n\n"
            "Compression is a SHADOW roll: a second engine request is pre-warmed in the "
            "background (system prompt + trimmed transcript, prefill-only) while the live "
            "request keeps serving; the swap at the next turn boundary is a pointer flip, "
            "so no turn pays the cold prefill. A shadow needs a max_num_seqs slot on every "
            "stage while the old request still holds its own, so deployments must leave "
            "headroom (max_num_seqs >= sessions + 1) or warm-ups starve silently and every "
            "compression falls back to the blocking roll."
        ),
    )
    context_compression_target_tokens: int = Field(
        default=16384,
        ge=256,
        description=(
            "Token budget for the ROLLING WINDOW carried across a compression (Gemini "
            "Live's target_tokens; its docs describe the same sliding-window shape: drop "
            "the oldest turns, keep the result starting at a user turn, system "
            "instructions always retained). Trimmed newest-first: recent turns keep their "
            "FRAMES verbatim (~226 tokens each, budgeted at 250) as long as the budget "
            "lasts, then older turns degrade to text-only, then drop entirely. So what "
            "the model keeps after a swap is a window of full multimodal recent history "
            "with a text floor under it -- the fine visual detail that text-only carry "
            "measurably loses (frame recall 0/8 vs 8/8 elsewhere in this study) now "
            "survives as far back as the window reaches. The budget is what keeps the "
            "seed from growing roll over roll until it approaches the wall the "
            "compression exists to avoid; it is also the KV a warming shadow holds ON TOP "
            "of the live requests, which is why concurrent warm-ups are capped "
            "process-wide (see context_compression_carry_frames for the kill-switch).\n\n"
            "16,384 rather than a full half-context, from a MEASURED queueing argument: "
            "at 13 users the sessions cross the uniform trigger within a couple of "
            "minutes of each other, and a 28.6k seed took a median 9.5 s to warm while "
            "the 11,141-token runway to the hard roll allows ~5.8 s per warm-up at two "
            "permits. Halving the window halves the seed and closes that deficit; the "
            "pool cannot fund a third permit (the 13-user run already peaked near 110% "
            "of the KV pool with two in flight). The price is the window's REACH: ~62 "
            "frames instead of ~125, so visual recall falls back to the text floor "
            "sooner."
        ),
    )
    context_compression_carry_frames: bool = Field(
        default=True,
        description=(
            "Carry recent frames inside the compression seed (the rolling window above). "
            "False restores the old text-only carry: seeds shrink from ~32k to a few "
            "hundred tokens, warm-ups get cheaper, and every swap forgets everything the "
            "camera ever showed. The A/B knob for measuring what visual carry costs."
        ),
    )
    context_compression_warmup_timeout_s: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "How long a shadow request may take to become ready before it is abandoned. "
            "Readiness is presence-based -- the seed segment's own stage-0 finish_reason "
            "-- but that signal is not contractually guaranteed in every configuration, "
            "so a bound turns a wedged warm-up into a fallback to the blocking roll "
            "instead of a hang."
        ),
    )
    prefill_frames_on_arrival: bool = Field(
        default=False,
        description=(
            "Turn each retained frame into tokens WHEN IT ARRIVES instead of when the query "
            "is submitted, so the vision encoder and the thinker's prefill for it run while "
            "the user is still speaking rather than on the critical path.\n\n"
            "Session mode already prefills incrementally -- each turn submits only its new "
            "frames, never the history -- so what this changes is the TIMING, not the amount. "
            "The saving is therefore modest at conservative frame rates (measured ~2.5 new "
            "frames per turn = ~550 tokens = ~44 ms of prefill plus ~20 ms of copy). The real "
            "reason to want it is that it DECOUPLES frame rate from time-to-first-audio: today "
            "every extra frame per turn is paid at query time, which is why frame_filter_min_gap "
            "is set as conservatively as it is.\n\n"
            "Requires the prefill-only append path: such a chunk must reach stage 0 and stop "
            "there, because anything that reaches the talker makes the model speak when nobody "
            "asked it to. Since section 25 the engine discards the append's sampled token at "
            "the scheduler and parks the segment with ZERO output -- nothing enters the "
            "context beyond the appended tokens, nothing ships downstream.\n\n"
            "Needs session_scoped_request; without a live request there is nothing to append to."
        ),
    )
    prefill_audio_on_arrival: bool = Field(
        default=False,
        description=(
            "[Tick engine WP5] Prefill the mic audio INCREMENTALLY while the user is "
            "still speaking, in whole-second chunks, instead of paying the whole "
            "utterance's audio prefill at query time. Reuses the prefill-only append "
            "path (see prefill_frames_on_arrival). Qwen3-Omni's audio encoder is "
            "chunk-structured (1 s conv chunks, 8 s attention blocks), so 8 s-aligned "
            "splits are bit-faithful to whole-utterance encoding; sub-8 s chunks keep "
            "the conv/positional grid exact but shrink the attention context to the "
            "chunk -- a measured-quality trade, keep audio_prefill_chunk_s=8 unless "
            "TTFA at long utterances matters more. Needs session_scoped_request."
        ),
    )
    audio_prefill_chunk_s: float = Field(
        default=8.0,
        description="Whole-second chunk size for prefill_audio_on_arrival; 8 = encoder-faithful, 1 = latency-optimal.",
    )
    audio_prefill_reserve_s: float = Field(
        default=1.0,
        description="Residual seconds always left in the buffer for the turn-time tail splice.",
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
        # One handler instance serves every WebSocket session in this process (see
        # api_server: app.state.openai_streaming_video), so this is the process-wide
        # brake on simultaneous seed prefills. Sized by pool arithmetic, not by GPU
        # compute: see _MAX_CONCURRENT_SHADOW_WARMUPS.
        self._shadow_warmup_sem = asyncio.Semaphore(_MAX_CONCURRENT_SHADOW_WARMUPS)
        # Live session count for the stage-1 pool guard. Maintained in exactly
        # ONE place (the handle_session wrapper below) so it cannot leak: a
        # hand-maintained counter with scattered updates once parked the whole
        # engine loop (workflow section 13).
        self._active_sessions = 0

    async def handle_session(self, websocket: WebSocket) -> None:
        """Count the session in, run it, count it out -- whatever happens."""
        self._active_sessions += 1
        try:
            await self._handle_session_inner(websocket)
        finally:
            self._active_sessions -= 1

    async def _handle_session_inner(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()

        try:
            config = await self._receive_config(websocket)
            if config is None:
                return
            # Log the fields that differ from defaults, once per session. This exists
            # because a crash investigation stalled on exactly this gap: an engine died
            # during a hand-driven browser session, and nothing on the server recorded
            # which features that session had enabled -- the client tab could have been
            # days old. Post-mortems need the config that was live, not the one shipped.
            try:
                non_default = {
                    k: v for k, v in config.model_dump().items()
                    if v != type(config).model_fields[k].default and k != "system_prompt"
                }
                logger.info("[session] config (non-default): %s", non_default)
            except Exception:
                logger.debug("session config logging failed", exc_info=True)

            # Stage-1 pool guard, admission half: a session whose share of the
            # speech stage's KV pool would be below the roll floor cannot be
            # served without risking a preemption storm for EVERYONE, so it is
            # refused here, gracefully, instead. self._active_sessions already
            # counts this session.
            if config.stage1_kv_pool_tokens and config.session_scoped_request:
                _share = int(0.75 * config.stage1_kv_pool_tokens
                             / max(1, self._active_sessions))
                if _share < _TALKER_ROLL_FLOOR:
                    logger.warning(
                        "[session] REFUSED at admission: stage-1 pool share %d < floor %d "
                        "(pool=%d, active=%d)", _share, _TALKER_ROLL_FLOOR,
                        config.stage1_kv_pool_tokens, self._active_sessions)
                    await self._send_error(
                        websocket,
                        "at capacity: the speech stage's KV pool cannot hold another "
                        "session without preempting existing ones")
                    return

            # [Tick engine WP6] bin-packing admission: with paced generation
            # every session costs a KNOWN per-tick quantum, so capacity is a
            # number, not a hope. VLLM_OMNI_ADMIT_MAX_SESSIONS is that number
            # (measured: the largest N whose per-tick work fits the tick at
            # the target utilization). Overload is REFUSED, never queued --
            # queueing a periodic stream is already an SLA violation.
            _cap = int(os.environ.get("VLLM_OMNI_ADMIT_MAX_SESSIONS", "0") or 0)
            if _cap > 0 and self._active_sessions > _cap:
                logger.warning(
                    "[session] REFUSED at admission: tick-capacity cap %d reached "
                    "(active=%d)", _cap, self._active_sessions)
                await self._send_error(
                    websocket,
                    f"at capacity: this instance is provisioned for {_cap} "
                    "concurrent realtime sessions")
                return

            # Resolve the compression trigger once per session. None anchors to the
            # MODEL: 75% of stage-0 max_model_len -- the single-user default only guards
            # the lifetime wall. The blocking-roll backstop must sit BELOW that wall:
            # 1.5x a 75% trigger would be PAST max_model_len and never fire, so it is
            # capped at 92% of the model's context.
            _mml = 0
            try:
                _mc = getattr(self._engine_client, "model_config", None)
                _mml = int(getattr(_mc, "max_model_len", 0) or 0)
            except Exception:
                _mml = 0
            if config.context_compression_trigger_tokens is None:
                compression_trigger = int(0.75 * _mml) if _mml else 0
            else:
                compression_trigger = max(0, config.context_compression_trigger_tokens)
            if compression_trigger:
                # De-synchronize the cohort. Sessions that start together and grow
                # at the same rate cross the SAME trigger in the SAME turn, and the
                # simultaneous swap-turn + seed prefills queue on stage-0's
                # per-step token budget -- measured as the entire residual TTFA
                # tail after the roll-waive fix (8/128 turns at 1.8-7.7 s, all in
                # the compression window, all thinker-first-token). A per-session
                # factor in [0.80, 1.00) spreads the crossings over ~2 turns of
                # growth. DOWNWARD only, so the wave-peak budget that makes
                # waiving pool-safe (hard 1.5t + seed 0.5t = 2t <= pool/N when
                # trigger = share/2) still holds for every session.
                _jit = 0.80 + 0.20 * (int(uuid.uuid4().hex[:8], 16) / 0xFFFFFFFF)
                compression_trigger = max(256, int(compression_trigger * _jit))
            compression_hard = 0
            if compression_trigger:
                compression_hard = int(1.5 * compression_trigger)
                if _mml:
                    compression_hard = min(compression_hard, int(0.92 * _mml))
                logger.info(
                    "[session] context compression armed: trigger=%d hard_roll=%d "
                    "(max_model_len=%d, explicit=%s)",
                    compression_trigger, compression_hard, _mml,
                    config.context_compression_trigger_tokens is not None,
                )

            frame_buffer: list[str] = []  # base64-encoded JPEG frames
            # Newest arrived frame, similarity-filter-agnostic: the query-time
            # sweep rides it at the delta's end (fresh_frame_on_query).
            latest_frame: list[str | None] = [None]
            frame_metadata: list[dict[str, Any]] = []
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

            def _new_request_ctx() -> dict[str, Any]:
                """Per-ENGINE-REQUEST state. A session normally owns exactly one, but a
                compression shadow briefly makes it two: the live request keeps serving
                while the replacement pre-warms, and the swap is a pointer flip of
                sess["active_ctx"]. Everything whose identity is the REQUEST (its input
                queue, its audio-attribution FIFO, its drain task) lives here; everything
                whose identity is the SESSION (transcript, turn flags, counters) stays in
                `sess`."""
                return {
                    "rid": f"video-sess-{uuid.uuid4().hex[:12]}",
                    "queue": asyncio.Queue(maxsize=4),
                    # One owner tag ("append" | "turn") per chunk submitted to this
                    # request, pushed in submission order. The engine runs one segment per
                    # chunk and cannot start segment k+1 before segment k's stop, so the
                    # k-th audio stop the output loop sees belongs to the k-th submitted
                    # chunk. That makes submission ORDER a structural identity for the
                    # audio stream -- the per-chunk field that outputs do not carry. It is
                    # per-request state: order across two requests means nothing.
                    "fifo": deque(),
                    "task": None,
                    # Shadow warm-up bookkeeping; inert on the live request.
                    "ready": False,
                    "ready_evt": asyncio.Event(),
                    "failed": None,
                    "junk_chunks": 0,
                }

            main_ctx = _new_request_ctx()
            sess: dict[str, Any] = {
                # Aliases of the ACTIVE ctx's objects, so every producer site (turn body,
                # arrival appends) keeps addressing sess["queue"] / sess["audio_seg_fifo"]
                # and a swap only has to repoint these two references.
                "queue": main_ctx["queue"],
                "audio_seg_fifo": main_ctx["fifo"],
                "active_ctx": main_ctx,
                "shadow": None,
                "shadow_failed_at": 0.0,
                "gen_task": None,
                "turn_done": asyncio.Event(),
                "turn_idx": 0,
                "first_sent": False,
                "fatal": None,
                # Seed for a roll, and the query awaiting its answer so the two can be paired
                # when the turn closes. See session_roll_at_talker_tokens.
                "transcript": [],
                "pending_query": None,
                # Frame b64s the model has seen since the last transcript append -- the
                # turn body's post-filter list plus any arrival appends -- so the output
                # loop can attach them to the user entry when the turn closes. Popped on
                # attach; attribution of frames that land mid-answer skews one turn early,
                # which recency-ordered memory does not care about.
                "pending_frames": [],
                # One shadow warm-up waiter per session at a time (the permit queue).
                "warmup_waiting": False,
                "rolls": 0,
                # Frames prefilled on arrival, and the tokens they cost. Counted because the
                # whole point is a latency saving that is otherwise invisible: the query-time
                # delta simply gets smaller, and nothing says why.
                "query_claimed": False,
                "arrival_appends": 0,
                "arrival_frames": 0,
                "arrival_tokens": 0,
            }
            session_request_id = main_ctx["rid"]

            async def _chunk_stream(ctx: dict[str, Any]):
                """Native async generator of per-turn deltas for ONE engine request.

                MUST be a real `async def ... yield` generator. async_omni.py:398 branches
                on `isinstance(prompt, collections.abc.AsyncGenerator)`, whose
                __subclasshook__ requires asend/athrow/aclose as well as __aiter__ and
                __anext__, so a hand-rolled iterator class silently falls through to the
                one-shot per-turn path: no error raised, and the entire change becomes a
                no-op that looks like a null result.
                """
                from vllm.engine.protocol import StreamingInput

                from vllm.sampling_params import RequestOutputKind, SamplingParams

                while True:
                    item = await ctx["queue"].get()
                    if item is None:
                        return
                    # A prefill-only append rides through as (prompt, max_tokens). Without a
                    # per-chunk cap it would generate a whole answer to a question nobody
                    # asked: the delta ends in the assistant header, so the model answers.
                    # The talker never sees it, so it would be silent -- and still burn a
                    # reply's worth of decode and leave that reply in the context.
                    if isinstance(item, tuple):
                        prompt, max_tokens = item[0], item[1]
                        is_seed = len(item) > 2 and item[2] == "seed"
                        if is_seed:
                            # A compression shadow's seed is a REAL micro-segment, NOT a
                            # prefill-only append, for two measured reasons:
                            #   * max_tokens=2, not 1: a segment whose ENDING forward
                            #     prefilled >1 row ships nothing to the talker (the
                            #     structural skip in qwen3_omni.py), and a tensor-less
                            #     chunk 0 kills stage 1 -- the swap turn's chunk-0 payload
                            #     then pairs full-prompt ids with delta-only embeds
                            #     (RuntimeError: tensor a (6) vs b (9)). With 2, the
                            #     ending forward is the one-row decode step, so chunk 0
                            #     ships the same full-prompt payload every session's
                            #     first chunk ships: the talker is born the tested way.
                            #   * no prefill-only marker, same reason: the marker's whole
                            #     effect is to suppress that payload.
                            # output_kind is explicit because a first chunk's params
                            # become the REQUEST's params, and the bare default is the
                            # non-streaming kind.
                            # [live-vllm P2] The seed prefills in silence -- nobody is
                            # waiting on it. Mark it "background" so the scheduler's
                            # express lane lets turn-opens and joins pass it; the marker
                            # dies with the first real chunk (the swap turn) engine-side.
                            from vllm_omni.core.sched.temporal_pacing import SLACK_CLASS_KEY
                            yield StreamingInput(
                                prompt=prompt,
                                sampling_params=SamplingParams(
                                    max_tokens=max_tokens,
                                    output_kind=RequestOutputKind.DELTA,
                                    extra_args={SLACK_CLASS_KEY: "background"},
                                ),
                            )
                            continue
                        # extra_args is the marker's channel: it rides on the per-chunk
                        # sampling params, which this path is already required to carry.
                        yield StreamingInput(
                            prompt=prompt,
                            sampling_params=SamplingParams(
                                max_tokens=max_tokens,
                                extra_args={_PREFILL_ONLY_KEY: "1"},
                            ),
                        )
                        continue
                    yield StreamingInput(prompt=item)

            def _new_turn_state() -> dict[str, Any]:
                return {
                    "text_parts": [], "prev_text": "", "text_done_sent": False,
                    "audio_chunks": 0, "drained": 0, "started": False,
                    "t0": _time.monotonic(), "t_first_text": None, "t_first_audio": None,
                }

            async def _session_output_loop(ctx: dict[str, Any]) -> None:
                """ONE generate() per engine request, demultiplexed back into turns.

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
                        prompt=_chunk_stream(ctx),
                        request_id=ctx["rid"],
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
                                "finish_reason=%s ntok=%s audio=%s",
                                getattr(output, "final_output_type", "?"),
                                getattr(output, "stage_id", "?"),
                                getattr(output, "finished", None),
                                getattr(ro, "finished", None) if ro is not None else None,
                                getattr(co, "finish_reason", None) if co is not None else None,
                                len(getattr(co, "token_ids", ()) or ()) if co is not None else None,
                                # `output.audio_data` used to be logged here and it read 0 on every
                                # audio output, including the ones that carried the reply. It is
                                # not the field the delta extractor uses. Reporting a different
                                # field from the one that matters is worse than reporting nothing:
                                # it says "no audio was produced" while audio is being produced.
                                _summarise_audio_payload(self._get_audio_data(output)),
                            )
                        # SHADOW / RETIRED PATH. While this loop's request is not the live
                        # one -- a compression shadow warming up before its swap, or the
                        # old request draining after it -- nothing here may touch the
                        # websocket or the shared turn state. The one useful signal is the
                        # seed segment's own finish_reason: its PRESENCE is what marks the
                        # shadow ready (presence-based on purpose -- the inert-guard
                        # lesson; an absence-based gate here would hang silently).
                        if sess.get("active_ctx") is not ctx:
                            if getattr(output, "final_output_type", "text") == "audio":
                                # The transfer adapter suppresses the seed's tensor-less
                                # first boundary, so normally no audio arrives at all.
                                # Count whatever does: it sizes the shadow talker's array
                                # at swap time.
                                ctx["junk_chunks"] += 1
                                if _segment_finish_reason(output) is not None and ctx["fifo"]:
                                    ctx["fifo"].popleft()
                            if _segment_finish_reason(output) is not None and not ctx["ready"]:
                                # Ready means the seed's AUDIO stop arrived, not merely
                                # the stage-0 text finish. The talker free-runs junk for
                                # a seed segment (median ~1 s, measured up to 36 s), and
                                # segments within one request are strictly serial -- a
                                # swap before that stop parks the first real turn's
                                # speech behind the junk. Measured as 8/640 turns with
                                # ttft ~100-200 ms but ttfa 6.5-26.5 s, all on swap
                                # clusters. Waiting costs nothing: the live request keeps
                                # serving while the junk drains in the background.
                                seed_drained = (
                                    "audio" not in (config.modalities or [])
                                    or getattr(output, "final_output_type", "text") == "audio"
                                )
                                if seed_drained:
                                    ctx["ready"] = True
                                    ctx["ready_evt"].set()
                                    logger.info(
                                        "[session] COMPRESS: shadow %s is ready "
                                        "(seed fully drained)",
                                        ctx["rid"],
                                    )
                            continue
                        # Attribute every AUDIO output to the chunk that caused it, by
                        # submission order. An append flows through the whole pipeline on
                        # purpose (withholding it from the talker desynchronised the stages
                        # and killed the engine), and max_tokens=1 caps only stage 0 -- the
                        # talker free-runs a few unprompted codec frames per append and ends
                        # them with a REAL audio finish_reason=stop. That stop arrives a
                        # median 1 s (max measured 36 s) after the append, so any flag read
                        # AT ARRIVAL TIME tells you what is in flight now, not who caused
                        # the output: measured closing a real turn at chars=2 while its
                        # actual reply streamed into the void afterwards -- 27.5 s of real
                        # speech swallowed in one session, every broken turn a collision of
                        # an append's late stop with the next turn's open window.
                        #
                        # `audio_seg_fifo` is order-based, which is structural here: chunks
                        # enter the engine through one queue, the engine finishes segment k
                        # before starting k+1 (chunk polling is gated on the previous
                        # segment's stop), and every submitted chunk yields exactly one
                        # stage-2 stop -- held 141/141 in the log INCLUDING the error path
                        # where the payload build fails and an empty stop still ships. So
                        # head-of-FIFO == owner of the audio stream right now.
                        #
                        # This is not the counter that failed before (see git history of
                        # this block): that one popped on the append's STAGE-0 finish, which
                        # is not reliably delivered, so it latched and swallowed a real
                        # turn's text. This one pops on the audio stream's OWN stop, which
                        # the engine guarantees per chunk. And it cannot fail silently: a
                        # stuck "append" head swallows the next turn's audio, turn_done
                        # never sets, and the 240 s bounded wait already fatals the session
                        # with "turn boundary lost".
                        owner = None
                        if getattr(output, "final_output_type", "text") == "audio":
                            fifo = ctx["fifo"]
                            owner = fifo[0] if fifo else None
                            if _segment_finish_reason(output) is not None and fifo:
                                fifo.popleft()
                                # Presence probe, not noise: pushes==pops at session end is
                                # the invariant check, and absence of a warning proves
                                # nothing (the inert-guard lesson).
                                logger.info(
                                    "[session] audio segment stop owner=%s fifo_left=%d",
                                    owner, len(fifo),
                                )
                        if interrupt_event.is_set():
                            continue
                        if (getattr(output, "final_output_type", "text") == "audio"
                                and owner != "turn"):
                            # owner == "append": positively identified junk -- the talker's
                            # unprompted frames and their stop. Dropping the stop HERE is
                            # the actual fix: it can no longer close a real turn.
                            # owner is None: audio nobody submitted a chunk for. Swallow,
                            # but say so loudly -- if this ever fires the one-stop-per-chunk
                            # invariant broke and attribution is shifted.
                            if owner is None:
                                logger.warning(
                                    "[session] UNOWNED audio output (finish=%s) dropped -- "
                                    "audio_seg_fifo is empty, attribution may be shifted",
                                    _segment_finish_reason(output),
                                )
                            sess["arrival_skipped"] = sess.get("arrival_skipped", 0) + 1
                            continue

                        # Outer fallback for TEXT outputs (an append's single stage-0 token,
                        # a closed turn's late text): while no turn is in flight, nothing
                        # textual belongs on the wire either. Sound only because append
                        # AUDIO is already filtered positively above -- this flag check
                        # alone lost the race for a year of debugging hours.
                        if (config.prefill_frames_on_arrival
                                and not sess.get("turn_busy")
                                and not sess.get("query_claimed")):
                            sess["arrival_skipped"] = sess.get("arrival_skipped", 0) + 1
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
                                # Prefer the submission stamp: st["t0"] is the
                                # previous turn's end, which under duplex
                                # feeding is minutes of think-time away.
                                _t0 = max(st["t0"], sess.get("turn_t0") or 0.0)
                                logger.info(
                                    "[session] turn=%d done first_text=%.3fs "
                                    "first_audio=%.3fs audio_chunks=%d chars=%d "
                                    "arrival_skipped=%d frames_dropped=%d",
                                    sess["turn_idx"],
                                    (st["t_first_text"] - _t0) if st["t_first_text"] else -1.0,
                                    (st["t_first_audio"] - _t0) if st["t_first_audio"] else -1.0,
                                    st["audio_chunks"], len("".join(st["text_parts"])),
                                    sess.get("arrival_skipped", 0),
                                    sess.get("frames_dropped", 0),
                                )
                                sess["arrival_skipped"] = 0
                                sess["frames_dropped"] = 0
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
                                # `message_history` is still deliberately NOT updated: under
                                # session mode the conversation lives in the engine request's
                                # KV, and a second copy claiming to be the conversation would
                                # be dead state that future readers mistake for the source of
                                # truth.
                                #
                                # `sess["transcript"]` is a different thing and exists for one
                                # purpose: it is the SEED for a roll. When the talker nears its
                                # max_model_len the engine request has to be replaced, and text
                                # is the only part of the context that can be carried into the
                                # new one -- the visual KV cannot. Kept only when rolling is
                                # enabled, and trimmed to the configured window, so it cannot
                                # quietly become an unbounded second history.
                                # Compression alone must also populate it: the gate used
                                # to be the talker-roll knob only, and a compression-only
                                # config silently rolled with an EMPTY seed every time.
                                if config.session_roll_at_talker_tokens or compression_trigger:
                                    text = "".join(st["text_parts"]).strip()
                                    q = sess.get("pending_query") or ""
                                    frames = sess.get("pending_frames") or []
                                    sess["pending_frames"] = []
                                    carry_frames = bool(
                                        compression_trigger
                                        and config.context_compression_carry_frames
                                    )
                                    if q or (frames and carry_frames):
                                        entry: dict[str, Any] = {"role": "user", "content": q}
                                        if frames and carry_frames:
                                            # Sibling key, not content: every text-only
                                            # consumer keeps reading a plain string, and
                                            # only the seed renderer opts into the frames.
                                            entry["frames"] = frames
                                        sess["transcript"].append(entry)
                                    if text:
                                        sess["transcript"].append(
                                            {"role": "assistant", "content": text}
                                        )
                                    keep = 2 * max(0, config.session_roll_history_turns)
                                    if carry_frames:
                                        # The roll default (16 entries) binds far below a
                                        # 32k window; the window renders nothing the
                                        # transcript no longer holds.
                                        keep = max(keep, _COMPRESSION_TRANSCRIPT_KEEP)
                                    if keep and len(sess["transcript"]) > keep:
                                        del sess["transcript"][:-keep]
                                    holders = [m for m in sess["transcript"] if "frames" in m]
                                    for m in holders[:-_FRAME_KEEP_TURNS]:
                                        # JPEG bytes beyond any window's reach: keep the
                                        # text, drop the images, bound the session's RSS.
                                        m.pop("frames", None)
                                st = _new_turn_state()
                                sess["turn_done"].set()
                        else:
                            if "audio" not in (config.modalities or []):
                                # Text-only sessions have no audio stops for the FIFO to
                                # pop on, so the SAME order-based attribution runs on the
                                # text stops instead: appends and turns enter the engine
                                # through one queue, and segment k finishes before k+1
                                # starts, so head-of-FIFO == owner of this text stream.
                                # An append-owned output (its one junk token and its
                                # stop) is swallowed whole -- without this, the junk stop
                                # closes a real turn and the turn's reply streams into
                                # the void, the exact race the audio branch already
                                # solved. This is what makes duplex feeding legal on
                                # text-only (talker-less) sessions.
                                _fifo = ctx["fifo"]
                                _owner_t = _fifo[0] if _fifo else None
                                if _segment_finish_reason(output) is not None and _fifo:
                                    _fifo.popleft()
                                    logger.info(
                                        "[session] text segment stop owner=%s fifo_left=%d",
                                        _owner_t, len(_fifo),
                                    )
                                if _owner_t != "turn":
                                    if _owner_t is None:
                                        logger.warning(
                                            "[session] UNOWNED text output dropped -- "
                                            "fifo empty, attribution may be shifted")
                                    sess["arrival_skipped"] = sess.get("arrival_skipped", 0) + 1
                                    continue
                            delta, st["prev_text"] = self._extract_text_delta(output, st["prev_text"])
                            # Stamp on the first text OUTPUT, not the first non-empty delta.
                            # The first stage-0 output of a turn often carries no new text (it
                            # advances internal state only), so keying on `delta` put first_text
                            # AFTER first_audio by ~17 ms and made an accounting health check
                            # report a turn as misattributed when nothing was misattributed.
                            # Measuring the wrong instant is not the same as attributing to the
                            # wrong turn, and conflating them cost a debugging cycle tonight.
                            if st["t_first_text"] is None:
                                st["t_first_text"] = _time.monotonic()
                                # [turnprobe] One line per turn, pairing with the
                                # recv probe in _run_session_turn_body: splits a
                                # client-measured TTFT into handler time
                                # (recv -> ADMIT) and engine time (ADMIT ->
                                # first text). Exists to name the serialization
                                # point behind the residual 3-11 s outlier
                                # turns that survived v3-v5.
                                logger.info(
                                    "[turnprobe] first-text rid=%s turn=%d",
                                    ctx.get("rid"), sess.get("turn_idx", -1),
                                )
                            if delta:
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
                    if sess.get("active_ctx") is ctx:
                        logger.exception("[session] output loop failed")
                        sess["fatal"] = str(e)
                        sess["turn_done"].set()   # never leave a turn waiting forever
                    else:
                        # A non-live request's failure abandons that request, never the
                        # live session: reusing the fatal path here would let a shadow
                        # warm-up hiccup (or a retired request's teardown noise) kill a
                        # perfectly healthy conversation.
                        logger.warning(
                            "[session] COMPRESS: non-live request %s loop ended: %s",
                            ctx["rid"], e,
                        )
                        ctx["failed"] = str(e)
                        ctx["ready_evt"].set()

            async def _prefill_frames_on_arrival(frames: list[str]) -> bool:
                """Append just-arrived frame(s) to the live request so stage 0 prefills them now.

                Immediate mode passes a single frame; frame-tick mode (WP7) passes
                everything the mailbox collected this tick as ONE chunk, so the
                frames' encoder work and KV prefill travel together.

                The append carries no query text, and `max_tokens=1` because the engine needs
                a nonzero cap to schedule the chunk. Since section 25 the scheduler discards
                the sampled token and parks the segment with ZERO output: nothing lands in the
                context beyond the frame's own tokens, nothing is emitted, nothing ships
                downstream -- the append is invisible everywhere but this stage's KV.

                Two conditions, both load-bearing:

                * Not while a turn is in flight. The queue is the same one the turn's delta
                  goes through, so slipping a frame in mid-turn would put an append between a
                  turn's chunk and its answer, and the segment boundary the output loop waits
                  on would belong to the wrong thing.
                * Only after the first real chunk. The first chunk carries the system prompt
                  and the roll seed; a frames-only append cannot go first without stealing
                  that position.

                Failure is deliberately soft: the frame stays in frame_buffer and the ordinary
                query-time path picks it up. A latency optimisation must never be able to lose
                a frame.
                """
                if (not sess["first_sent"] or sess.get("turn_busy")
                        or sess.get("query_claimed") or sess["fatal"]):
                    return False
                # While a compression shadow warms up, frames are HELD, not appended: an
                # append lands in the OLD request's KV, which dies at the swap, and the
                # arrival path would also remove the frame from frame_buffer -- the swap
                # turn would then be blind to the whole warm-up window. Refusal keeps the
                # frame buffered for the swap turn's ordinary query-time sweep.
                if sess.get("shadow") is not None:
                    return False
                # Text-only sessions get no arrival appends at all: the append's junk
                # audio segment is what the FIFO attributes, and without audio outputs
                # there is nothing to pop against -- the text-only close at the text
                # branch would keep the old race instead.
                if "audio" not in (config.modalities or []):
                    return False
                # The queue must be EMPTY. Appends share it with the turn deltas, so anything
                # still queued means a turn's chunk has not been consumed yet and an append
                # would land between that chunk and its answer. Measured on the proxy path --
                # the one the browser uses -- as `Overlapping turn` plus a turn reporting
                # `first_text=-1.000s chars=0 audio_chunks=1`, i.e. a fraction of a second of
                # the wrong turn's reply. turn_busy alone does not cover it: the flag clears
                # when the turn's body returns, while its chunk may still be in the queue.
                if sess["queue"].qsize() > 0:
                    return False
                chunk = await self._build_session_chunk(
                    config, frames, bytearray(), "", frame_pil_cache,
                    is_first=False,
                )
                if chunk is None or not isinstance(chunk, dict):
                    return False
                # Headerless: the frames join the user's current utterance rather than
                # opening a turn of their own. See _strip_chatml_scaffolding.
                if not _strip_chatml_scaffolding(chunk):
                    logger.warning(
                        "[session] prefill-on-arrival: could not reduce the delta to its "
                        "frame tokens; skipping the append (the frame stays buffered)"
                    )
                    return False
                # The marker the engine side reads to keep this append away from the talker.
                #
                # It has to be a real AdditionalInformationPayload, not a plain dict. A dict
                # here is accepted silently and then does not survive: the field is a msgspec
                # Struct, so it crosses the process boundary only in that shape, and the
                # engine-side interceptions simply never fire. That failure is invisible --
                # measured as the model speaking unasked, with no error anywhere and the
                # symptom (an extra audio segment) three components away from the cause.
                from vllm_omni.engine import (
                    AdditionalInformationEntry,
                    AdditionalInformationPayload,
                )

                entries = {}
                existing = chunk.get("additional_information")
                if isinstance(getattr(existing, "entries", None), dict):
                    entries.update(existing.entries)
                entries[_PREFILL_ONLY_KEY] = AdditionalInformationEntry(list_data=["1"])
                chunk["additional_information"] = AdditionalInformationPayload(entries=entries)
                try:
                    # max_tokens=1 is still the floor the ENGINE requires to schedule the
                    # chunk; the scheduler discards that token at sampling time and parks
                    # the segment with zero output (section 25) -- it never lands in the
                    # context and never leaves stage 0.
                    sess["queue"].put_nowait((chunk, 1))
                except asyncio.QueueFull:
                    # The engine is behind. Leave the frame where it is rather than blocking
                    # the receive loop, which also forwards generated audio.
                    return False
                # NO fifo entry for this append (section 25): the engine parks
                # a marked segment with zero output, so there is no downstream
                # stop to pop against -- an entry here would sit at the head
                # and swallow the next real turn (measured: 240 s "turn
                # boundary lost" on the first turn after any append).
                ntok = len(chunk.get("prompt_token_ids") or ())
                sess["arrival_appends"] += 1
                sess["arrival_frames"] += len(frames)
                sess["arrival_tokens"] += ntok
                sess["cum_tokens"] = sess.get("cum_tokens", 0) + ntok
                # Arrival-consumed frames never reach the turn body's new_frames list,
                # so the transcript would lose exactly the frames this optimisation
                # touches. Same pending list the turn body feeds, same turn-close drain.
                if compression_trigger and config.context_compression_carry_frames:
                    sess.setdefault("pending_frames", []).extend(frames)
                logger.info(
                    "[session] prefill-on-arrival: %d frame(s) -> %d tokens (appends=%d "
                    "frames=%d tokens=%d cum=%d)",
                    len(frames), ntok, sess["arrival_appends"], sess["arrival_frames"],
                    sess["arrival_tokens"], sess.get("cum_tokens", 0),
                )
                # Frames alone can carry the context across the compression trigger during
                # a long silence; without this hook the warm-up would only start at the
                # next turn and the swap would slip one turn further.
                if _warmup_due() and _shadow_allowed():
                    _launch_shadow_warmup("arrival")
                return True

            def _arm_frame_flush() -> None:
                """[WP7] Schedule ONE mailbox flush at the next global tick edge.

                Every session computes the edge as ceil(now/tick)*tick on the
                shared CLOCK_MONOTONIC -- the same quantization the engine-side
                pacer uses -- so flushes align across sessions with no shared
                registry, and the aligned appends are what let stage 0 batch
                the frames' encoder work. The flush is delivered through
                msg_queue (the `_internal.*` precedent) so frame_buffer keeps
                its single writer: _processor.

                One timer per session at a time; a flush that finds the turn
                busy simply leaves the frames buffered (same soft-failure
                contract as the immediate path) and the NEXT frame arrival
                re-arms -- no periodic retry churn.
                """
                if sess.get("frame_flush_armed"):
                    return
                sess["frame_flush_armed"] = True

                async def _wait_edge() -> None:
                    now = _time.monotonic()
                    delay = math.ceil(now / _FRAME_TICK_S) * _FRAME_TICK_S - now
                    # Landing exactly on an edge (or sub-ms before it) would
                    # flush a mailbox the current frame has not reached yet.
                    if delay < 0.002:
                        delay += _FRAME_TICK_S
                    await asyncio.sleep(delay)
                    try:
                        msg_queue.put_nowait({"type": "_internal.frame_flush"})
                    except asyncio.QueueFull:
                        # The loop is already saturated; the next arrival
                        # re-arms. Frames stay buffered -- never lost.
                        sess["frame_flush_armed"] = False

                task = asyncio.create_task(_wait_edge())
                prewarm_tasks.add(task)
                task.add_done_callback(prewarm_tasks.discard)

            async def _prefill_audio_on_arrival() -> bool:
                """[Tick engine WP5] prefill buffered mic audio while the user speaks.

                A whole-second-aligned prefix of audio_buffer becomes a
                prefill-only append (same engine path as frames-on-arrival:
                zero-output park, nothing reaches the talker). Consumption is
                in multiples of audio_prefill_chunk_s so every piece -- and
                the residual tail spliced at turn time -- stays on the audio
                encoder's 1 s conv grid; 8 s chunks additionally respect its
                8 s attention blocks (bit-faithful split). Failure is soft:
                the audio stays buffered and the turn-time path takes it all.
                """
                if (not sess["first_sent"] or sess.get("turn_busy")
                        or sess.get("query_claimed") or sess["fatal"]):
                    return False
                if sess.get("shadow") is not None:
                    return False
                if "audio" not in (config.modalities or []):
                    return False
                if sess["queue"].qsize() > 0:
                    return False
                bytes_per_s = 32000  # PCM16 mono 16 kHz
                chunk_s = max(1, int(config.audio_prefill_chunk_s or 8))
                reserve_s = max(0, int(math.ceil(config.audio_prefill_reserve_s or 0)))
                usable_s = len(audio_buffer) // bytes_per_s - reserve_s
                consume_s = (usable_s // chunk_s) * chunk_s
                if consume_s <= 0:
                    return False
                k = consume_s * bytes_per_s
                prefix = bytes(audio_buffer[:k])
                chunk = await self._build_session_chunk(
                    config, [], bytearray(prefix), "", frame_pil_cache,
                    is_first=False,
                )
                if chunk is None or not isinstance(chunk, dict):
                    return False
                if not _strip_chatml_scaffolding(chunk):
                    logger.warning(
                        "[session] audio prefill-on-arrival: could not reduce the delta "
                        "to its audio tokens; skipping (audio stays buffered)")
                    return False
                from vllm_omni.engine import (
                    AdditionalInformationEntry,
                    AdditionalInformationPayload,
                )

                entries = {}
                existing = chunk.get("additional_information")
                if isinstance(getattr(existing, "entries", None), dict):
                    entries.update(existing.entries)
                entries[_PREFILL_ONLY_KEY] = AdditionalInformationEntry(list_data=["1"])
                chunk["additional_information"] = AdditionalInformationPayload(entries=entries)
                try:
                    sess["queue"].put_nowait((chunk, 1))
                except asyncio.QueueFull:
                    return False
                # Only now is the prefix truly out of our hands: consume it so
                # the turn-time chunk carries just the residual tail.
                del audio_buffer[:k]
                ntok = len(chunk.get("prompt_token_ids") or ())
                sess["arrival_appends"] += 1
                sess["arrival_tokens"] += ntok
                sess["cum_tokens"] = sess.get("cum_tokens", 0) + ntok
                logger.info(
                    "[session] audio prefill-on-arrival: %ds -> %d tokens "
                    "(appends=%d tokens=%d cum=%d)",
                    consume_s, ntok, sess["arrival_appends"],
                    sess["arrival_tokens"], sess.get("cum_tokens", 0),
                )
                if _warmup_due() and _shadow_allowed():
                    _launch_shadow_warmup("arrival")
                return True

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
                    sess["query_claimed"] = False
                    await _run_session_turn_body(query_text=query_text)
                finally:
                    sess["turn_busy"] = False

            async def _run_session_turn_body(*, query_text: str) -> None:
                """Queue this turn's delta, then wait for its audio to finish.

                Strictly turn-by-turn: the client waits for response.audio.done before
                sending the next query, so pipelining would buy nothing here and would
                complicate the segment bookkeeping.
                """
                # [turnprobe] see the first-text probe for why.
                logger.info(
                    "[turnprobe] recv rid=%s turn=%d",
                    (sess.get("active_ctx") or {}).get("rid"), sess.get("turn_idx", -1),
                )
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
                # Compress/roll BEFORE building the chunk, so this turn is the new
                # request's first one and carries the seed (or the carry); after would
                # waste a turn. The ladder, in order:
                #   1. a READY shadow wins -- the swap is a pointer flip and upgrades
                #      BOTH triggers to the invisible path;
                #   2. a wall that cannot wait pays the blocking roll, exactly the old
                #      guarantee (the talker wall does not fail cleanly, so a shadow
                #      that is not ready yet must not be waited for);
                #   3. otherwise a due warm-up is started in the background and this
                #      turn is served on the live request, which still has margin --
                #      the warm-up thresholds sit below the walls on purpose.
                carry: list[dict[str, Any]] | None = None
                shadow = sess.get("shadow")
                if (shadow is not None and shadow["ctx"].get("ready")
                        and not shadow["ctx"].get("failed")):
                    carry = await _swap_to_shadow()
                elif _must_roll_now():
                    if _can_defer_roll():
                        # [roll-waive] Housekeeping never blocks the user. A
                        # compression-cap roll with no ready shadow is simply
                        # WAIVED: the turn is served on the live request (a
                        # thick context is slow-ish, not wrong) and the swap
                        # happens whenever a shadow lands, via the ready-shadow
                        # branch above. Deferring the blocking roll to the
                        # turn-end silence was tried first and made the wave
                        # WORSE (p99 64.5 s -> 99.9 s): the roll itself takes
                        # tens of seconds under a synchronized wave, think time
                        # is 2-6 s, so the NEXT turn inherited the remainder --
                        # and turn-end deferral stampeded 32 permit-less
                        # rebuilds at once. The only blocking roll left is the
                        # emergency ceiling in _can_defer_roll.
                        if _shadow_allowed():
                            _launch_shadow_warmup("hard-cap-waive")
                        logger.info(
                            "[session] roll waived: turn=%d served on live "
                            "request (cum=%d >= hard=%d); waiting for a shadow",
                            sess["turn_idx"], sess.get("cum_tokens", 0),
                            compression_hard,
                        )
                    else:
                        await _roll_session()
                        if sess["fatal"]:
                            await self._send_error(
                                websocket, f"Session failed: {sess['fatal']}"
                            )
                            return
                elif _warmup_due() and _shadow_allowed():
                    _launch_shadow_warmup("turn start")

                new_frames = list(frame_buffer)
                n_buffered = len(new_frames)
                # Freshness (digit-clock study): without this, the frames adjacent to
                # the question are the ones refused during the PREVIOUS answer -- the
                # oldest in the delta -- and the model reads the frames nearest the
                # question as "now". Ride the newest arrival at the delta's end.
                if (config.fresh_frame_on_query and latest_frame[0] is not None
                        and (not new_frames or new_frames[-1] != latest_frame[0])):
                    new_frames.append(latest_frame[0])
                # The blocking roll's seed is deliberately TEXT-ONLY even when frames are
                # retained: it prefills in the foreground of a turn the user is waiting
                # on, and recovery speed beats fidelity on the emergency path. The shadow
                # seed is where frames ride (they prefill in silence).
                seed = (
                    _transcript_chat_msgs(sess["transcript"], with_frames=False)
                    if not sess["first_sent"] else None
                )
                chunk = await self._build_session_chunk(
                    config, new_frames, audio_buffer, query_text, frame_pil_cache,
                    is_first=not sess["first_sent"],
                    seed_history=seed if seed is not None else carry,
                )
                audio_buffer.clear()
                if chunk is None:
                    # Nothing to submit: keep the frames for the next turn rather than
                    # dropping them on the floor.
                    await self._send_error(websocket, "Nothing new to submit this turn")
                    return
                # Delete exactly the consumed prefix, not the whole buffer: frames may have
                # arrived while the chunk was being built, and those belong to the next turn.
                # n_buffered, NOT len(new_frames): the fresh-frame rider was never in the
                # buffer, and counting it here would delete one frame that arrived mid-build.
                del frame_buffer[:n_buffered]
                if compression_trigger and config.context_compression_carry_frames:
                    # Post-filter list: a frame the chunk dropped as undecodable must not
                    # come back to poison a seed later. Captured before the cache pops
                    # below erase the _BAD_FRAME verdicts.
                    kept = [
                        f for f in new_frames
                        if frame_pil_cache.get(f) is not _BAD_FRAME
                    ]
                    if kept:
                        sess.setdefault("pending_frames", []).extend(kept)
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
                # The turn stopwatch. The output loop's per-turn state is born
                # when the PREVIOUS turn closes, which made its t0 a fine
                # anchor when turns were the only input events -- but under
                # duplex feeding there is always input in flight, and
                # first_text measured from the previous turn's end reads as
                # think-time + speak-window + latency (~6 s that alarmed a
                # whole debugging session while the client correctly saw
                # 150 ms). Anchor on the delta's submission instead.
                sess["turn_t0"] = _time.monotonic()
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
                # Held so the output loop can pair this query with its answer in the
                # transcript, which is the seed a roll carries into the next request.
                sess["pending_query"] = query_text
                if sess["gen_task"] is None:
                    ctx0 = sess["active_ctx"]
                    ctx0["task"] = asyncio.create_task(_session_output_loop(ctx0))
                    sess["gen_task"] = ctx0["task"]
                sess["first_sent"] = True
                # Tag BEFORE the awaited put: if the put suspends on a full queue no
                # output for this chunk can exist yet, and appends are refused while
                # turn_busy, so nothing can interleave a push between these two lines.
                # UNCONDITIONAL since text-only attribution landed: audio sessions pop
                # this on audio stops, text-only sessions pop it on text stops -- a
                # turn that never enters the FIFO is swallowed as unowned by whichever
                # branch is doing the attributing (949 unowned drops, 0/1280 turns in
                # the first thinker-only run, with the old audio-gated push).
                sess["audio_seg_fifo"].append("turn")
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
                # Start a due warm-up in the silence AFTER the turn, not during one: the
                # seed prefill competes for the GPU with whatever is decoding.
                if _warmup_due() and _shadow_allowed():
                    _launch_shadow_warmup("turn end")

            def _talker_roll_at() -> int | None:
                """Effective talker roll threshold: the per-session configured value,
                LOWERED to this session's share of the stage-1 KV pool when that pool
                is declared. The configured threshold guards a per-session wall
                (stage-1 max_model_len); the pool is shared by every session in the
                process, so at N sessions the honest budget is pool/N -- measured at
                64 sessions, the per-session threshold slept 30x above the shared
                wall while every talker request got preempted."""
                roll_at = config.session_roll_at_talker_tokens
                pool = config.stage1_kv_pool_tokens
                if not pool:
                    return roll_at
                share = max(_TALKER_ROLL_FLOOR,
                            int(0.75 * pool / max(1, self._active_sessions)))
                return min(roll_at, share) if roll_at else share

            def _warmup_due() -> bool:
                """Should a shadow start warming? Thresholds sit BELOW the walls so the
                shadow is normally ready before any wall forces a blocking roll.

                Warming earlier than the trigger was tried (0.6x, swap gated
                at the trigger) and REGRESSED the whole distribution (p50
                2.8 s, 107/128 slow turns): a parked-ready shadow holds an
                engine slot, and 32 sessions' long-lived shadows + 32 live
                requests exceeded max_num_seqs=56 -- short-lived shadows were
                themselves the slot-pressure valve. With text-only seeds
                (~200 tokens) an early warm buys nothing anyway."""
                if compression_trigger and sess.get("cum_tokens", 0) >= compression_trigger:
                    return True
                roll_at = _talker_roll_at()
                return bool(roll_at and sess.get("talker_tokens", 0) >= 0.85 * roll_at)

            def _can_defer_roll() -> bool:
                """May a hard-cap roll be waived for this turn?

                Waivable only when the pressure is the COMPRESSION cap: that
                cap is a scheduling convenience (1.5x trigger), not a wall.
                The emergency ceiling (2x trigger, capped at half the model
                context) bounds how far a session can ride the live request
                while its shadow warms; with working warm-ups the overshoot is
                about one turn, and only a session whose shadows keep failing
                ever reaches the ceiling and pays the old blocking roll. The
                TALKER wall keeps its blocking semantics -- it does not fail
                cleanly (preemption storms on the shared stage-1 pool), so a
                turn must never be served past it.
                """
                roll_at = _talker_roll_at()
                if roll_at and sess.get("talker_tokens", 0) >= roll_at:
                    return False
                emergency = 0
                if compression_trigger:
                    emergency = 2 * compression_trigger
                    if _mml:
                        emergency = min(emergency, int(0.5 * _mml))
                return bool(emergency and sess.get("cum_tokens", 0) < emergency)

            def _must_roll_now() -> bool:
                """A wall that cannot wait for a shadow. The talker trigger keeps its
                original blocking semantics -- its wall does not fail cleanly -- and the
                context side gets a hard fallback (1.5x trigger, capped below
                max_model_len) in case shadows keep failing."""
                roll_at = _talker_roll_at()
                if roll_at and sess.get("talker_tokens", 0) >= roll_at:
                    if roll_at != config.session_roll_at_talker_tokens:
                        # Rare (once per roll), so a log line is affordable -- and it is
                        # the only visible trace that the POOL, not the per-session
                        # wall, forced this roll.
                        logger.warning(
                            "[session] talker-pool guard rolls at %d (configured %s, "
                            "pool=%d, active=%d)", roll_at,
                            config.session_roll_at_talker_tokens,
                            config.stage1_kv_pool_tokens or 0, self._active_sessions)
                    return True
                return bool(compression_hard
                            and sess.get("cum_tokens", 0) >= compression_hard)

            def _shadow_allowed() -> bool:
                if sess.get("shadow") is not None or sess["fatal"]:
                    return False
                if not config.session_scoped_request:
                    return False
                # Cooldown after a failed warm-up, so a broken shadow path degrades to
                # the blocking roll instead of spinning warm-up attempts.
                return (_time.monotonic() - sess.get("shadow_failed_at", 0.0)) > 60.0

            def _launch_shadow_warmup(where: str) -> None:
                t = asyncio.create_task(_start_shadow_warmup(where))
                prewarm_tasks.add(t)
                t.add_done_callback(prewarm_tasks.discard)

            def _transcript_chat_msgs(
                msgs: list[dict[str, Any]], *, with_frames: bool
            ) -> list[dict[str, Any]]:
                """Project transcript entries into chat messages the request schema knows.

                Transcript entries keep frames under a SIBLING key so every text-only
                consumer stays untouched; this is the one place that key is honoured.
                with_frames=False is the text view (blocking roll, swap carry); True
                renders a user entry's frames as image parts ahead of its text -- the
                exact per-turn shape _build_session_chunk already emits for live frames.
                """
                out: list[dict[str, Any]] = []
                for m in msgs:
                    frames = m.get("frames") if with_frames else None
                    if frames:
                        content: list[dict[str, Any]] = [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b}"},
                            }
                            for b in frames
                        ]
                        text = str(m.get("content") or "")
                        if text:
                            content.append({"type": "text", "text": text})
                        out.append({"role": m["role"], "content": content})
                    else:
                        out.append(
                            {"role": m["role"], "content": m.get("content") or ""}
                        )
                return out

            def _trim_transcript_for_seed() -> list[dict[str, Any]]:
                """Newest turns that fit context_compression_target_tokens (Gemini's
                target_tokens), as a ROLLING WINDOW: an entry's frames are kept while
                the budget lasts (newest first, 250 tokens per frame, erring high of
                the measured 226), and the first entry whose frames do not fit drops
                them for itself AND everything older -- a latch, so a small old entry
                cannot out-keep a big recent one. Text keeps accumulating under the
                same budget until it too runs out; chars/3 is only the first guess,
                the built chunk's real token count is checked afterwards and the seed
                rebuilt smaller if the guess was badly off."""
                budget = config.context_compression_target_tokens
                frames_allowed = bool(config.context_compression_carry_frames)
                out: list[dict[str, Any]] = []
                total = 0
                for m in reversed(sess["transcript"]):
                    text_cost = max(1, len(str(m.get("content", ""))) // 3)
                    frames = m.get("frames") if frames_allowed else None
                    cost = text_cost + _SEED_TOKENS_PER_FRAME * len(frames or ())
                    if frames and out and total + cost > budget:
                        frames = None
                        frames_allowed = False
                        cost = text_cost
                    if out and total + cost > budget:
                        break
                    entry: dict[str, Any] = {"role": m["role"], "content": m.get("content", "")}
                    if frames:
                        entry["frames"] = list(frames)
                    out.append(entry)
                    total += cost
                out.reverse()
                return out

            async def _start_shadow_warmup(where: str) -> None:
                """Admission shell around the warm-up: the process-wide permit.

                A warming seed holds up to target_tokens of KV on top of every live
                request, and the pool slack at full capacity fits two such seeds, not
                thirteen -- so warm-ups queue at the door instead of stampeding when a
                whole cohort crosses the trigger together. Busy means SKIP, silently:
                the hook that skipped re-fires (cum_tokens only grows until a swap),
                and the skip must not stamp shadow_failed_at -- a busy period is not a
                broken shadow path, and the 60s cooldown would convert it into forced
                blocking rolls at the hard wall.
                """
                sem = self._shadow_warmup_sem
                if sess.get("shadow") is not None or sess["fatal"]:
                    return
                # QUEUE, one waiter per session, rather than skip-and-retry-at-the-next
                # hook. Skipping was measured to strand permits: the hooks that would
                # retry are frame arrivals and turn boundaries, and the congestion that
                # makes the queue deep is exactly what slows those to ~1/s, so 1-2
                # permits sat idle for ~10 s with 9 sessions waiting. asyncio.Semaphore
                # is FIFO, so waiting also makes the order fair -- first over the line,
                # first served. Waiting is safe HERE and would not be one frame later:
                # sess["shadow"] is still unset, so prefill-on-arrival keeps running.
                if sess.get("warmup_waiting"):
                    return
                sess["warmup_waiting"] = True
                queued = sem.locked()
                if queued:
                    logger.info(
                        "[session] COMPRESS: warm-up queued (%s): %d permits in use",
                        where, _MAX_CONCURRENT_SHADOW_WARMUPS,
                    )
                try:
                    await sem.acquire()
                except asyncio.CancelledError:
                    sess["warmup_waiting"] = False
                    raise
                try:
                    # Re-check after the wait: the session may have rolled, failed, or
                    # been served by an earlier hook's warm-up while queued.
                    if sess.get("shadow") is None and not sess["fatal"] and _warmup_due():
                        await _start_shadow_warmup_inner(where)
                finally:
                    sess["warmup_waiting"] = False
                    sem.release()

            async def _start_shadow_warmup_inner(where: str) -> None:
                """Pre-warm the replacement request while the live one keeps serving.

                This is what turns the roll from a user-visible pause into a background
                action: the seed (system prompt + the rolling window of the transcript,
                recent frames included) is prefilled into a brand-new resumable request,
                which then parks with its KV warm. The swap at the next turn boundary is
                a pointer flip.

                Two engine-side facts this leans on, both verified in-tree:
                  * the FIRST chunk of a resumable request honors per-chunk sampling
                    params, so max_tokens=1 caps the seed at one discarded token and the
                    request parks in WAITING_FOR_STREAMING_REQ with KV retained;
                  * the transfer adapter suppresses a prefill-only chunk-0 boundary, so
                    stage 1 first hears of this request from the first REAL turn, as a
                    normal full chunk 0 -- no tensor-less bring-up payload.
                """
                if sess.get("shadow") is not None or sess["fatal"]:
                    return
                watermark = len(sess["transcript"])
                seed_msgs = _trim_transcript_for_seed()
                if not seed_msgs and not config.system_prompt:
                    # Nothing to seed with. The blocking roll handles this case fine: its
                    # seed rides the first real turn, which is never empty.
                    return
                ctx = _new_request_ctx()
                sess["shadow"] = {"ctx": ctx, "watermark": watermark, "seed_ntok": 0}
                try:
                    chunk = await self._build_session_chunk(
                        config, [], bytearray(), "", frame_pil_cache,
                        is_first=True,
                        seed_history=_transcript_chat_msgs(seed_msgs, with_frames=True),
                        seed_only=True,
                    )
                    if not isinstance(chunk, dict):
                        raise RuntimeError("seed chunk did not build")
                    ntok = len(chunk.get("prompt_token_ids") or ())
                    while (ntok > config.context_compression_target_tokens * 1.3
                           and len(seed_msgs) > 2):
                        seed_msgs = seed_msgs[2:]
                        chunk = await self._build_session_chunk(
                            config, [], bytearray(), "", frame_pil_cache,
                            is_first=True,
                            seed_history=_transcript_chat_msgs(
                                seed_msgs, with_frames=True
                            ),
                            seed_only=True,
                        )
                        if not isinstance(chunk, dict):
                            raise RuntimeError("seed chunk did not build")
                        ntok = len(chunk.get("prompt_token_ids") or ())
                    # Deliberately NOT marked prefill-only. The seed must ship a normal
                    # full chunk-0 payload so stage 1 brings the talker up the same way
                    # every session's first chunk does -- suppressing it was measured to
                    # kill stage 1 when the swap turn's chunk-0 payload paired full-prompt
                    # ids with delta-only embeds. The cost is a micro-segment: one stray
                    # text token in the seed context and a moment of junk audio that the
                    # warm-up drain discards.
                    sess["shadow"]["seed_ntok"] = ntok
                    # The talker never sees the seed chunk (its boundary is suppressed),
                    # but the first REAL turn ships as chunk 0, i.e. the FULL prompt --
                    # seed included -- so the shadow talker's array starts at the seed's
                    # placeholder length, not zero. The turn-time accounting only sees the
                    # delta, so this must be captured here or the wall guard undercounts
                    # from birth (in the unsafe direction).
                    seed_tlen = 0
                    try:
                        from vllm_omni.distributed.omni_connectors.adapter import (
                            compute_talker_prompt_ids_length,
                        )
                        seed_tlen = max(
                            0,
                            compute_talker_prompt_ids_length(
                                list(chunk.get("prompt_token_ids") or ())
                            ),
                        )
                    except Exception:
                        # Err high -- the estimate guards a wall -- but from the TEXT,
                        # not from ntok: the talker drops image rows (TALKER_TEXT_ONLY),
                        # so a mostly-image seed's ntok (~32k) would start the newborn
                        # talker a step from its roll line and re-roll every turn.
                        seed_tlen = max(
                            64,
                            sum(
                                (len(str(m.get("content", ""))) // 3) * 2
                                for m in seed_msgs
                            ),
                        )
                    sess["shadow"]["seed_tlen"] = seed_tlen
                    ctx["task"] = asyncio.create_task(_session_output_loop(ctx))
                    # The seed's junk audio needs an owner tag, exactly like an arrival
                    # append: its stage-2 stop can arrive AFTER the swap, on the by-then
                    # live loop, and an unowned stop would close the first real turn
                    # early. Tagged "append", it is positively identified junk on either
                    # side of the swap. No producer races this push: the shadow queue has
                    # exactly one writer until the swap.
                    if "audio" in (config.modalities or []):
                        ctx["fifo"].append("append")
                    await ctx["queue"].put((chunk, 2, "seed"))
                    seed_frames = sum(len(m.get("frames") or ()) for m in seed_msgs)
                    logger.info(
                        "[session] COMPRESS: warming shadow %s at %s (seed=%d msgs, "
                        "%d frames, %d tokens; cum=%d, talker_est=%d)",
                        ctx["rid"], where, len(seed_msgs), seed_frames, ntok,
                        sess.get("cum_tokens", 0), sess.get("talker_tokens", 0),
                    )
                    try:
                        await asyncio.wait_for(
                            ctx["ready_evt"].wait(),
                            timeout=config.context_compression_warmup_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        ctx["failed"] = "warm-up timeout"
                    if ctx.get("failed"):
                        raise RuntimeError(ctx["failed"])
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[session] COMPRESS: shadow warm-up abandoned (%s); the blocking "
                        "roll remains the fallback",
                        e,
                    )
                    task = ctx.get("task")
                    if task is not None and not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    if sess.get("shadow") and sess["shadow"]["ctx"] is ctx:
                        sess["shadow"] = None
                    sess["shadow_failed_at"] = _time.monotonic()

            async def _swap_to_shadow() -> list[dict[str, Any]]:
                """Point the session at the pre-warmed request; returns the carry.

                Runs at turn start with turn_busy held and the queue empty, which is what
                makes the flip race-free. The carry is the transcript delta since the seed
                was built (turns that closed while the shadow warmed); it rides THIS
                turn's chunk so the model never loses the newest exchange -- skipping it
                would be silent amnesia, invisible until a recall probe.
                """
                nonlocal session_request_id
                shadow = sess.pop("shadow")
                ctx = shadow["ctx"]
                old_ctx = sess["active_ctx"]
                old_task = old_ctx.get("task")
                old_rid = old_ctx["rid"]
                # Text view on purpose: the carry prefills on the swap turn's critical
                # path. Its turns' frames stay in the transcript, inside the NEXT seed's
                # window -- temporarily invisible to the model, never lost.
                carry = _transcript_chat_msgs(
                    sess["transcript"][shadow["watermark"]:], with_frames=False
                )
                if ctx["fifo"]:
                    # The seed's "append" tag is still pending: its audio stop has not
                    # arrived yet. KEEP it -- the live loop's append handling swallows the
                    # late stop by ownership, and this turn's "turn" tag queues behind it.
                    # Clearing it here would hand the seed's stop to the first real turn.
                    logger.info(
                        "[session] COMPRESS: seed audio stop still pending at swap "
                        "(fifo=%d) -- the append tag rides across",
                        len(ctx["fifo"]),
                    )
                sess["active_ctx"] = ctx
                sess["queue"] = ctx["queue"]
                sess["audio_seg_fifo"] = ctx["fifo"]
                sess["gen_task"] = ctx["task"]
                session_request_id = ctx["rid"]
                # True, not False: the shadow already consumed its is_first chunk (the
                # seed). Building the next chunk as first would inject a second system
                # block mid-request and skip the <|im_end|> shim.
                sess["first_sent"] = True
                sess["cum_tokens"] = shadow["seed_ntok"]
                sess["talker_tokens"] = (
                    shadow.get("seed_tlen", 0)
                    + _TALKER_TOKENS_PER_AUDIO_CHUNK * ctx.get("junk_chunks", 0)
                )
                sess["rolls"] = sess.get("rolls", 0) + 1
                this_turn = sess["turn_idx"]

                async def _retire_old() -> None:
                    # The old request stays parked until the swap turn has closed, so its
                    # abort's cross-process cleanup cannot race this turn's update -- the
                    # same overlap the blocking roll's settle sleep papers over, closed
                    # here by ordering instead of sleeping.
                    for _ in range(600):
                        if sess["turn_idx"] > this_turn or sess["fatal"]:
                            break
                        await asyncio.sleep(0.5)
                    if old_task is not None and not old_task.done():
                        old_task.cancel()
                        await asyncio.gather(old_task, return_exceptions=True)
                    logger.info("[session] COMPRESS: old request %s retired", old_rid)

                retire_task = asyncio.create_task(_retire_old())
                prewarm_tasks.add(retire_task)
                retire_task.add_done_callback(prewarm_tasks.discard)
                logger.info(
                    "[session] COMPRESS #%d at turn=%d: %s -> %s, seed=%d tokens, "
                    "carry=%d message(s); the swap is a pointer flip, this turn pays no "
                    "cold prefill.",
                    sess["rolls"], sess["turn_idx"], old_rid, ctx["rid"],
                    shadow["seed_ntok"], len(carry),
                )
                try:
                    await websocket.send_json(
                        {"type": "session.compressed", "turn": sess["turn_idx"],
                         "rolls": sess["rolls"], "carried_messages": len(carry)}
                    )
                except Exception:
                    # The client not understanding this event must not end the session.
                    logger.debug("[session] could not send session.compressed", exc_info=True)
                return carry

            async def _roll_session() -> None:
                """Replace the engine request, carrying the recent text across.

                The session's real limit is stage 1's `max_model_len`: the talker's stored
                token array grows every segment by the delta PLUS the audio codes it just
                generated, and crossing the limit does not fail cleanly -- it either kills the
                stage-1 engine core on a numpy broadcast or makes the scheduler skip the
                request silently forever. See session_talker_token_budget for the measurements.

                So the request is retired while it is still healthy and a fresh one takes over.
                What crosses is TEXT; the accumulated visual KV does not, which is the whole
                cost of the mechanism together with one cold prefill. Frames still in the
                buffer are submitted with the first post-roll turn, so the model is not blind
                to the present -- it has lost the older visual detail only.

                Cancelling rather than sending the terminal sentinel is the same choice
                `_close_session_request` documents. Doing it MID-session is only safe because
                of the fix in `1aed4032`: before that, ending a resumable request left an
                aborted entry in `skipped_waiting` that upstream's waiting loop picked up and
                asserted on, taking the stage down -- which is exactly what a roll would have
                triggered every single time.
                """
                nonlocal session_request_id
                prev_id = session_request_id
                prev_tokens = sess.get("talker_tokens", 0)

                # A live shadow is torn down first: the blocking roll replaces the request
                # NOW, and a half-warm shadow seeded from the pre-roll transcript would
                # otherwise swap in a context that no longer matches the conversation.
                stale_shadow = sess.pop("shadow", None)
                if stale_shadow is not None:
                    stask = stale_shadow["ctx"].get("task")
                    if stask is not None and not stask.done():
                        stask.cancel()
                        await asyncio.gather(stask, return_exceptions=True)

                task = sess["gen_task"]
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

                # A fresh ctx (queue, fifo, request id), not the old one drained: the
                # cancelled generator may have been suspended mid-item, and reusing the
                # queue would feed the new request a chunk built for the old one's
                # context. Race-free because the generator task was cancelled AND awaited
                # above, so no old-request output can arrive after this point to pop a
                # stale tag.
                ctx = _new_request_ctx()
                sess["active_ctx"] = ctx
                sess["queue"] = ctx["queue"]
                sess["audio_seg_fifo"] = ctx["fifo"]
                sess["gen_task"] = None
                sess["first_sent"] = False        # next chunk carries system + seed
                sess["cum_tokens"] = 0            # new request, new context
                sess["talker_tokens"] = 0
                sess["rolls"] = sess.get("rolls", 0) + 1
                session_request_id = ctx["rid"]

                # Let the stages finish retiring the old request before the new one arrives.
                # See session_roll_settle_s: the ordering is right on this side, but the stages
                # are separate processes and getting 'add new' before 'abort old' cost a whole
                # roll -- stage 0 answered in full and stage 1 produced nothing.
                if config.session_roll_settle_s > 0:
                    await asyncio.sleep(config.session_roll_settle_s)

                logger.info(
                    "[session] ROLL #%d at turn=%d: talker was at ~%d tokens, retiring "
                    "req=%s for req=%s, carrying %d transcript message(s). This turn pays a "
                    "cold prefill; the accumulated visual context is gone and the text is not.",
                    sess["rolls"], sess["turn_idx"], prev_tokens,
                    prev_id, session_request_id, len(sess["transcript"]),
                )
                try:
                    await websocket.send_json(
                        {"type": "session.rolled", "turn": sess["turn_idx"],
                         "rolls": sess["rolls"],
                         "carried_messages": len(sess["transcript"])}
                    )
                except Exception:
                    # The client not understanding this event must not end the session.
                    logger.debug("[session] could not send session.rolled", exc_info=True)

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
                # A warming shadow holds its own engine request; leaving it behind would
                # leak a parked request (and its max_num_seqs slot) past the session.
                shadow = sess.pop("shadow", None)
                if shadow is not None:
                    stask = shadow["ctx"].get("task")
                    if stask is not None and not stask.done():
                        stask.cancel()
                        await asyncio.gather(stask, return_exceptions=True)
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
                    # Claim the turn SYNCHRONOUSLY, before the task is scheduled.
                    #
                    # create_task only queues the coroutine: `turn_busy = True` inside
                    # _run_session_turn does not run until the event loop gets round to it, and
                    # the output loop can be woken in between. The arrival-append suppression
                    # keys on "no turn in flight", so anything of THIS turn's that lands in that
                    # window is swallowed -- measured as 2 bad turns in 16, one with chars=0 and
                    # first_text=-1 (its text eaten) and one with first_audio BEFORE first_text
                    # (impossible within a turn, so the audio belonged elsewhere).
                    #
                    # A separate flag rather than setting turn_busy here: turn_busy also gates
                    # the overlap refusal, and pre-setting it would make the very next query
                    # look like an overlap and be rejected.
                    sess["query_claimed"] = True
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
                query_frame_metadata = list(frame_metadata)
                query_audio_buffer = bytearray(audio_buffer)
                audio_buffer.clear()
                query_prewarmed_frames = dict(frame_pil_cache)

                async def _run_query() -> None:
                    nonlocal active_request_id, prev_request_id
                    try:
                        process_kwargs: dict[str, Any] = {}
                        if any(metadata.get("frame_id") for metadata in query_frame_metadata):
                            process_kwargs["frame_metadata"] = query_frame_metadata
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

                    if msg_type == "_internal.frame_flush":
                        # [WP7] Tick-edge mailbox flush. Runs inside _processor, so
                        # frame_buffer cannot change under the await below (this loop
                        # is its single writer; frames arriving meanwhile sit in
                        # msg_queue). Prefix deletion therefore removes exactly the
                        # flushed frames -- same shape as the turn body's drain.
                        sess["frame_flush_armed"] = False
                        if (frame_buffer and config.prefill_frames_on_arrival
                                and config.session_scoped_request):
                            flush = list(frame_buffer)
                            if await _prefill_frames_on_arrival(flush):
                                del frame_buffer[:len(flush)]
                                for _fb in flush:
                                    frame_pil_cache.pop(_fb, None)
                        continue

                    if msg_type == "_internal.frame_decode_failed":
                        frame_data = msg.get("b64", "")
                        removed = frame_data in frame_buffer
                        if removed:
                            retained_indices = [
                                index for index, frame in enumerate(frame_buffer) if frame != frame_data
                            ]
                            frame_buffer[:] = [frame_buffer[index] for index in retained_indices]
                            frame_metadata[:] = [frame_metadata[index] for index in retained_indices]
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
                        # Stash BEFORE the filter: a dropped frame is still the newest
                        # picture of the world, and fresh_frame_on_query needs exactly that.
                        latest_frame[0] = frame_data
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
                                    await self._send_frame_ack(
                                        websocket,
                                        msg,
                                        accepted=False,
                                        buffered_frames=len(frame_buffer),
                                        reason="filtered",
                                    )
                                    continue
                                frames_since_retained = 0
                            except Exception:
                                await self._send_error(websocket, "Invalid image data")
                                continue
                        max_buf = config.max_frames
                        dropped_frame_id: str | None = None
                        if len(frame_buffer) >= max_buf:
                            dropped = frame_buffer.pop(0)
                            dropped_metadata = frame_metadata.pop(0)
                            dropped_frame_id = dropped_metadata.get("frame_id")
                            frame_pil_cache.pop(dropped, None)
                            # Counted per turn because this is a LATENCY control, and a
                            # latency control that silently discards input is one nobody
                            # can audit: the buffer only fills when arrival prefill is
                            # refused, so this number is the size of the sweep that did
                            # NOT land on the next turn.
                            sess["frames_dropped"] = sess.get("frames_dropped", 0) + 1
                        frame_buffer.append(frame_data)
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
                        # Prefill this frame now rather than when the query arrives. Only if it
                        # is actually consumed does it leave frame_buffer -- see the helper: a
                        # refusal leaves the frame for the ordinary query-time path.
                        # [WP7] Under frame-tick, the frame WAITS in the mailbox instead and
                        # the flush at the next global tick edge appends everything at once.
                        if config.prefill_frames_on_arrival and config.session_scoped_request:
                            if _FRAME_TICK_S > 0:
                                _arm_frame_flush()
                            elif await _prefill_frames_on_arrival([frame_data]):
                                try:
                                    frame_buffer.remove(frame_data)
                                except ValueError:
                                    pass
                                frame_pil_cache.pop(frame_data, None)
                        await self._send_frame_ack(
                            websocket,
                            msg,
                            accepted=True,
                            buffered_frames=len(frame_buffer),
                            dropped_frame_id=dropped_frame_id,
                        )
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
                        # [Tick engine WP5] opportunistic incremental prefill of
                        # the buffered speech; soft-fails and leaves the buffer.
                        if (config.prefill_audio_on_arrival
                                and config.session_scoped_request):
                            try:
                                await _prefill_audio_on_arrival()
                            except Exception:
                                logger.debug("audio prefill-on-arrival failed",
                                             exc_info=True)

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
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Build prompt, run inference, stream text + audio response."""

        if self._engine_client is None:
            await self._send_error(websocket, "Streaming video requires an engine client")
            return

        engine_kwargs: dict[str, Any] = {}
        if frame_metadata:
            engine_kwargs["frame_metadata"] = frame_metadata
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
        seed_history: list[dict[str, Any]] | None = None,
        seed_only: bool = False,
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

        if not user_content and not (seed_only and (seed_history or config.system_prompt)):
            # Only a compression shadow's seed may build without user content: it has
            # hundreds of transcript tokens, so the waiting-scheduler's
            # num_new_tokens > 0 assert stays safe.
            return None

        messages: list[dict[str, Any]] = []
        # The system block belongs to the session, so it is sent exactly once. Repeating
        # it per chunk would put it in the middle of the sequence, where
        # compute_talker_prompt_ids_length skips it and the model sees an instruction
        # block interleaved with the conversation.
        if is_first and config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        # Carried text. On the FIRST chunk this is the roll/compression seed, placed
        # between the system block and this turn so the new request reads as one
        # conversation. On a LATER chunk it is the compression carry -- turns that closed
        # while a shadow warmed -- and the <|im_end|> shim below closes the previous
        # assistant turn before these render, so the chatml structure stays valid. In
        # both cases the messages are COMPLETED turns and precede the current user block,
        # which is what keeps the model answering the current question.
        if seed_history:
            messages.extend(seed_history)
        if user_content:
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
        frame_metadata: list[dict[str, Any]] | None = None,
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
        decoded_ready_ts_ms = _time.monotonic() * 1000
        selected_metadata = self._sample_frame_metadata(frame_metadata or [], config.num_frames)
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
        # A BARE TENSOR IS ALREADY THE NEW AUDIO, and this is where the reply used to be
        # thrown away. The old code read the single tensor as a transient first state that
        # would "become a list", so it emitted it once and answered None to every later
        # output -- `if chunks_drained >= 1: return None`.
        #
        # It never becomes a list. Any streaming request is coerced to
        # RequestOutputKind.DELTA (entrypoints/utils.py maybe_coerce_to_message_type), and
        # under DELTA the output processor calls drain_delta_payload() after every snapshot
        # (outputs/output_processor.py), which pops the audio key outright. So each output
        # carries only what stage 2 produced since the previous one -- a fresh granule, every
        # time, and nothing cumulative to index into.
        #
        # Measured on one 151-character reply, per stage-2 output:
        #     7,125 samples, then 48,000 x 6 (25 codec frames x 1920), then 32,640 with stop
        #     = 13.66 s produced.  0.22 s delivered.  Only the first granule survived.
        # Cumulative payloads cannot look like that -- the lengths would grow monotonically
        # and could never drop to 32,640 -- so the contract is per-step by measurement as
        # well as by construction.
        #
        # `chunks_drained` therefore just counts granules emitted, and its only remaining job
        # is to mark the first one, whose leading CausalConv frame has to be stripped. If a
        # future engine ever does hand a cumulative tensor here, the symptom is audio that
        # repeats and grows turn by turn.
        if not isinstance(audio_data, list):
            tail_np = cls._tensor_to_1d_np(audio_data)
            return cls._encode_tail(
                tail_np, chunks_drained,
                new_drained=chunks_drained + 1,
                is_first=(chunks_drained == 0),
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
            # Same correction as _delta_fast: a bare tensor is this step's new audio, not a
            # cumulative buffer. This arm had the bug too, via `tail_np = full_np[0:0]` once
            # chunks_drained reached 1 -- which is why the fast/slow A/B showed no difference
            # and wrongly cleared the delivery path. Two implementations of one wrong
            # assumption agree with each other, so agreement between them proved nothing.
            audio_tensor = audio_data
            new_drained = chunks_drained + 1
            full_np = cls._tensor_to_1d_np(audio_tensor)
            if full_np is None:
                return None, chunks_drained
            return cls._encode_tail(
                full_np, chunks_drained,
                new_drained=new_drained,
                is_first=(chunks_drained == 0),
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

    _text_only_message = staticmethod(text_only_message)

    async def _send_error(self, websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass

    @staticmethod
    def _sample_frame_metadata(
        frame_metadata: list[dict[str, Any]],
        num_frames: int,
    ) -> list[dict[str, Any]]:
        if len(frame_metadata) <= num_frames:
            return list(frame_metadata)
        stride = max(1, len(frame_metadata) // num_frames)
        indices = [index * stride for index in range(num_frames - 1)] + [len(frame_metadata) - 1]
        return [frame_metadata[index] for index in indices]

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
