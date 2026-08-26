# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the serving-layer streaming video WebSocket handler."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
from typing import Any

import pytest
from PIL import Image
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.openai import video_stream_base, video_stream_envs
from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
    StreamingVideoSessionConfig,
)
from vllm_omni.entrypoints.openai.video_stream_base import OmniStreamingVideoHandler
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_jpeg(r: int = 128, g: int = 128, b: int = 128) -> bytes:
    img = Image.new("RGB", (64, 64), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _text_result(text: str) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.text = text
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="text", request_output=request_output)


def _audio_result(audio_data: Any) -> OmniRequestOutput:
    class Output:
        pass

    class RequestOutput:
        pass

    output = Output()
    output.multimodal_output = {"audio": audio_data}
    request_output = RequestOutput()
    request_output.outputs = [output]
    return OmniRequestOutput(final_output_type="audio", request_output=request_output)


class MockWebSocket:
    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self._idx = 0
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        if self._idx >= len(self._messages):
            await asyncio.sleep(999)
        msg = self._messages[self._idx]
        self._idx += 1
        return msg

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)


class TimedWebSocket:
    def __init__(self):
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self.accepted = False
        self.sent: list[dict[str, Any]] = []

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        return await self._q.get()

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)

    def put(self, msg: dict[str, Any]):
        self._q.put_nowait(json.dumps(msg))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


def test_api_server_registers_video_stream_route():
    from vllm_omni.entrypoints.openai.api_server import router

    assert any(getattr(route, "path", None) == "/v1/video/chat/stream" for route in router.routes)


def test_video_stream_requests_delta_outputs_without_mutating_deploy_defaults():
    defaults = [SamplingParams(), SamplingParams(), SamplingParams()]

    class Engine:
        default_sampling_params_list = defaults

    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=Engine())
    params = handler._sampling_params_for_request(StreamingVideoSessionConfig(model="test"))

    assert params is not None
    assert params[0].max_tokens == 256
    assert all(param.output_kind == RequestOutputKind.DELTA for param in params)
    assert all(param.output_kind == RequestOutputKind.CUMULATIVE for param in defaults)


def test_request_local_thinker_limit_covers_both_pd_stages():
    class Stage:
        def __init__(self, model_stage: str):
            self.model_stage = model_stage

    class Engine:
        default_sampling_params_list = [SamplingParams(), SamplingParams(), SamplingParams()]
        _stage_meta_list = [Stage("thinker"), Stage("thinker"), Stage("talker")]

    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=Engine())
    params = handler._sampling_params_for_request(
        StreamingVideoSessionConfig(model="test"),
        thinker_max_tokens=128,
        deterministic_thinker=True,
    )

    assert params is not None
    assert [params[index].max_tokens for index in range(2)] == [128, 128]
    assert [params[index].temperature for index in range(2)] == [0.0, 0.0]
    assert params[2].max_tokens != 128


@pytest.mark.asyncio
async def test_receive_config_accepts_client_legacy_aliases():
    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "evs_enabled": False,
                    "evs_threshold": 0.87,
                }
            )
        ]
    )
    handler = OmniStreamingVideoHandler(chat_service=object())

    config = await handler._receive_config(ws)

    assert config is not None
    assert config.enable_frame_filter is False
    assert config.frame_filter_threshold == 0.87


@pytest.mark.asyncio
async def test_video_frame_ack_reports_receiver_buffer_state():
    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "enable_frame_filter": False,
                }
            ),
            json.dumps(
                {
                    "type": "video.frame",
                    "data": _b64(_make_jpeg()),
                    "frame_id": "frame-7",
                    "pts_ms": 700,
                    "capture_ts_ms": 1234.5,
                }
            ),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = QwenOmniStreamingVideoHandler(chat_service=object())

    await handler.handle_session(ws)

    ack = next(message for message in ws.sent if message.get("type") == "video.frame.ack")
    assert ack["frame_id"] == "frame-7"
    assert ack["pts_ms"] == 700
    assert ack["capture_ts_ms"] == 1234.5
    assert ack["accepted"] is True
    assert ack["buffered_frames"] == 1
    assert ack["server_receive_ts_ms"] > 0


@pytest.mark.asyncio
async def test_video_frames_consumed_is_emitted_after_engine_uses_frame_prompt():
    class OneOutputEngine:
        def generate(self, **_kwargs):
            async def _gen():
                await asyncio.sleep(0.05)
                yield _text_result("visible")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "with-video"}

    ws = MockWebSocket(
        [
            json.dumps(
                {
                    "type": "session.config",
                    "model": "test",
                    "modalities": ["text"],
                    "enable_frame_filter": False,
                }
            ),
            json.dumps(
                {
                    "type": "video.frame",
                    "data": _b64(_make_jpeg()),
                    "frame_id": "frame-9",
                    "pts_ms": 900,
                    "source_pts_ms": 880,
                    "quality_profile": "balanced",
                }
            ),
            json.dumps({"type": "video.query", "text": "describe"}),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = CapturingHandler(
        chat_service=object(),
        engine_client=OneOutputEngine(),
        idle_timeout=2.0,
    )

    await handler.handle_session(ws)

    consumed = next(message for message in ws.sent if message.get("type") == "video.frames.consumed")
    assert consumed["frame_ids"] == ["frame-9"]
    assert consumed["latest_pts_ms"] == 900
    assert consumed["request_id"].startswith("video-")
    assert consumed["model_selected_ts_ms"] > 0
    assert consumed["frames"] == [
        {
            "frame_id": "frame-9",
            "pts_ms": 900,
            "source_pts_ms": 880,
            "quality_profile": "balanced",
            "receiver_received_ts_ms": consumed["frames"][0]["receiver_received_ts_ms"],
            "decoded_ready_ts_ms": consumed["frames"][0]["decoded_ready_ts_ms"],
        }
    ]
    assert consumed["frames"][0]["receiver_received_ts_ms"] > 0
    assert consumed["frames"][0]["decoded_ready_ts_ms"] >= consumed["frames"][0]["receiver_received_ts_ms"]
    assert consumed["model_selected_ts_ms"] >= consumed["frames"][0]["decoded_ready_ts_ms"]
    assert ws.sent.index(consumed) < next(
        index for index, message in enumerate(ws.sent) if message.get("type") == "response.text.delta"
    )


@pytest.mark.asyncio
async def test_arrival_prefill_is_silent_thinker_only_one_token():
    calls: list[dict[str, Any]] = []

    class TextEngine:
        def generate(self, **kwargs):
            calls.append(kwargs)

            async def _gen():
                yield _text_result("discard me")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt_token_ids": list(range(64))}

    handler = CapturingHandler(chat_service=object(), engine_client=TextEngine())
    ok = await handler._process_video_arrival_prefill(
        StreamingVideoSessionConfig(model="test"),
        [_b64(_make_jpeg())],
        [],
        "video-warm-test",
        {},
    )

    assert ok is True
    assert len(calls) == 1
    assert calls[0]["output_modalities"] == ["text"]
    assert "priority" not in calls[0]
    assert calls[0]["sampling_params_list"][0].max_tokens == 1
    assert calls[0]["prompt"]["prefill_only"] is True


@pytest.mark.asyncio
async def test_audio_arrival_prefill_seals_one_second_and_leaves_query_tail():
    warm_snapshots: list[tuple[int, list[int]]] = []
    query_chunks: list[int] = []
    warm_done = asyncio.Event()
    query_done = asyncio.Event()

    class EmptyEngine:
        async def abort(self, _request_id):
            return None

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_video_arrival_prefill(
            self,
            _config,
            _frame_buffer,
            _message_history,
            _request_id,
            _prewarmed_frames,
            audio_buffer=None,
            media_events=None,
        ):
            warm_snapshots.append(
                (
                    len(audio_buffer or ()),
                    [len(event.payload) for event in (media_events or ()) if event.modality == "audio"],
                )
            )
            warm_done.set()
            return True

        async def _process_query(self, *args, media_events=None, **kwargs):
            del args, kwargs
            query_chunks.extend(len(event.payload) for event in (media_events or ()) if event.modality == "audio")
            query_done.set()

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "enable_video_arrival_prefill": False,
            "enable_audio_arrival_prefill_approximation": True,
        }
    )
    await asyncio.sleep(0)

    pcm_200ms = b"\x01\x02" * 3200
    for _ in range(4):
        ws.put({"type": "audio.chunk", "data": _b64(pcm_200ms)})
        await asyncio.sleep(0)
    assert warm_snapshots == []

    ws.put({"type": "audio.chunk", "data": _b64(pcm_200ms)})
    await asyncio.wait_for(warm_done.wait(), timeout=2.0)
    assert warm_snapshots == [(32000, [32000])]

    tail = b"\x03\x04" * 1600
    ws.put({"type": "audio.chunk", "data": _b64(tail)})
    ws.put({"type": "video.query", "text": ""})
    await asyncio.wait_for(query_done.wait(), timeout=2.0)
    assert query_chunks == [32000, 3200]

    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_arrival_prefill_requests_from_sessions_enter_engine_concurrently():
    started: list[str] = []
    all_started = asyncio.Event()
    release = asyncio.Event()

    class HoldingEngine:
        def generate(self, **kwargs):
            async def _gen():
                started.append(kwargs["request_id"])
                if len(started) == 3:
                    all_started.set()
                await release.wait()
                yield _text_result("discarded")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _render_engine_prompt(self, *args, **kwargs):
            return {"prompt_token_ids": list(range(64))}, {}

    handler = CapturingHandler(chat_service=object(), engine_client=HoldingEngine())
    tasks = [
        asyncio.create_task(
            handler._process_video_arrival_prefill(
                StreamingVideoSessionConfig(model="test", session_id=f"session-{index}"),
                ["frame"],
                [],
                f"warm-{index}",
                {},
            )
        )
        for index in range(3)
    ]

    await asyncio.wait_for(all_started.wait(), timeout=2.0)
    assert set(started) == {"warm-0", "warm-1", "warm-2"}
    release.set()
    assert await asyncio.gather(*tasks) == [True, True, True]


@pytest.mark.asyncio
async def test_session_arrival_prefill_coalesces_cumulative_snapshots_serially():
    snapshots: list[int] = []
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    latest_started = asyncio.Event()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_video_arrival_prefill(
            self,
            _config,
            frame_buffer,
            _message_history,
            _request_id,
            _prewarmed_frames,
        ):
            snapshots.append(len(frame_buffer))
            if len(snapshots) == 1:
                first_started.set()
                await release_first.wait()
            if len(snapshots) == 2:
                latest_started.set()
            return True

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=object(), idle_timeout=5.0)
    session = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text"],
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "frame_id": "frame-1", "data": _b64(_make_jpeg(1, 2, 3))})
    await asyncio.wait_for(first_started.wait(), timeout=2.0)
    ws.put({"type": "video.frame", "frame_id": "frame-2", "data": _b64(_make_jpeg(4, 5, 6))})
    ws.put({"type": "video.frame", "frame_id": "frame-3", "data": _b64(_make_jpeg(7, 8, 9))})
    for _ in range(100):
        if any(message.get("frame_id") == "frame-3" for message in ws.sent):
            break
        await asyncio.sleep(0.01)

    assert snapshots == [1]
    assert any(message.get("frame_id") == "frame-3" for message in ws.sent)
    release_first.set()
    await asyncio.wait_for(latest_started.wait(), timeout=2.0)
    assert snapshots == [1, 3]

    ws.put({"type": "video.done"})
    await asyncio.wait_for(session, timeout=2.0)


@pytest.mark.asyncio
async def test_query_waits_for_session_arrival_prefill():
    warmup_started = asyncio.Event()
    release_warmup = asyncio.Event()
    warmup_finished = asyncio.Event()
    query_started = asyncio.Event()

    class RecordingEngine:
        def __init__(self):
            self.aborted: list[str] = []

        async def abort(self, request_id):
            self.aborted.append(request_id)

    class WaitingHandler(QwenOmniStreamingVideoHandler):
        async def _process_video_arrival_prefill(self, *args, **kwargs):
            warmup_started.set()
            await release_warmup.wait()
            warmup_finished.set()
            return True

        async def _process_query(self, *args, **kwargs):
            query_started.set()

    ws = TimedWebSocket()
    engine = RecordingEngine()
    handler = WaitingHandler(chat_service=object(), engine_client=engine, idle_timeout=5.0)
    session = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text"],
            "enable_frame_filter": False,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.wait_for(warmup_started.wait(), timeout=2.0)

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0.05)
    assert not query_started.is_set()
    assert engine.aborted == []

    release_warmup.set()
    await asyncio.wait_for(warmup_finished.wait(), timeout=2.0)
    await asyncio.wait_for(query_started.wait(), timeout=2.0)
    assert engine.aborted == []

    ws.put({"type": "video.done"})
    await asyncio.wait_for(session, timeout=2.0)


@pytest.mark.asyncio
async def test_arrival_prefill_prompts_are_cumulative_and_final_appends_audio():
    rendered: list[Any] = []
    calls: list[dict[str, Any]] = []

    class TextEngine:
        def generate(self, **kwargs):
            calls.append(kwargs)

            async def _gen():
                yield _text_result("discarded-or-final")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered.append(request)
            return {"prompt_token_ids": list(range(64 * len(request.messages)))}

    first = _b64(_make_jpeg(1, 2, 3))
    second = _b64(_make_jpeg(4, 5, 6))
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"])
    handler = CapturingHandler(chat_service=object(), engine_client=TextEngine())

    assert await handler._process_video_arrival_prefill(config, [first], [], "video-warm-1", {})
    assert await handler._process_video_arrival_prefill(config, [first, second], [], "video-warm-2", {})
    await handler._process_query_engine(
        MockWebSocket(),
        config,
        [first, second],
        bytearray(b"\x00\x00"),
        [],
        "",
        "video-final",
        asyncio.Event(),
        {},
    )

    content = [request.messages[-1]["content"] for request in rendered]
    assert [part["type"] for part in content[0]] == ["image_url"]
    assert [part["type"] for part in content[1]] == ["image_url", "image_url"]
    assert [part["type"] for part in content[2]] == ["image_url", "image_url", "input_audio"]
    assert [call["output_modalities"] for call in calls] == [
        ["text"],
        ["text"],
        ["text", "audio"],
    ]
    assert all("priority" not in call for call in calls)
    assert calls[0]["sampling_params_list"][0].max_tokens == 1
    assert calls[1]["sampling_params_list"][0].max_tokens == 1
    cache_salts = [call["prompt"]["talker_cache_salt"] for call in calls]
    assert len(set(cache_salts)) == 1
    assert cache_salts[0].startswith("video-session:")


def test_talker_cache_salt_is_private_and_session_isolated():
    first = StreamingVideoSessionConfig(session_id="same-client-id")
    second = StreamingVideoSessionConfig(session_id="same-client-id")

    assert first._talker_cache_salt != second._talker_cache_salt
    assert "talker_cache_salt" not in first.model_dump()


def test_thinker_lineage_tracks_prefix_and_generated_tokens():
    config = StreamingVideoSessionConfig(session_id="lineage-test")
    first_prompt = {"prompt_token_ids": [1, 2, 3, 4]}
    first = video_stream_base._attach_thinker_lineage(config, first_prompt)

    assert first is not None
    assert first.prefix_tokens == 0
    assert video_stream_base._commit_thinker_lineage(config, first, [5, 6])

    second_prompt = {"prompt_token_ids": [1, 2, 3, 4, 5, 6, 7]}
    second = video_stream_base._attach_thinker_lineage(config, second_prompt)

    assert second is not None
    assert second.parent_revision == 1
    assert second.prefix_tokens == 6
    assert second_prompt["kv_lineage_prefix_tokens"] == 6


def test_thinker_lineage_advances_as_one_linear_chain():
    config = StreamingVideoSessionConfig(session_id="lineage-linear")
    parent = video_stream_base._attach_thinker_lineage(config, {"prompt_token_ids": [1, 2]})
    assert video_stream_base._commit_thinker_lineage(config, parent)

    first = video_stream_base._attach_thinker_lineage(config, {"prompt_token_ids": [1, 2, 3]})
    assert first is not None
    assert first.parent_revision == parent.revision
    assert first.revision == parent.revision + 1
    assert video_stream_base._commit_thinker_lineage(config, first)

    second = video_stream_base._attach_thinker_lineage(config, {"prompt_token_ids": [1, 2, 3, 4]})
    assert second is not None
    assert second.parent_revision == first.revision
    assert second.revision == first.revision + 1


def test_thinker_lineage_rejects_a_speculative_sibling():
    config = StreamingVideoSessionConfig(session_id="lineage-linear-only")
    first = video_stream_base._attach_thinker_lineage(config, {"prompt_token_ids": [1, 2, 3]})
    sibling = video_stream_base._attach_thinker_lineage(config, {"prompt_token_ids": [1, 2, 3, 4]})

    assert first is not None and sibling is not None
    assert first.parent_revision == sibling.parent_revision == 0
    assert first.revision == sibling.revision == 1
    assert video_stream_base._commit_thinker_lineage(config, first)
    assert not video_stream_base._commit_thinker_lineage(config, sibling)
    assert config._thinker_lineage_token_ids == [1, 2, 3]


def test_thinker_lineage_reset_invalidates_old_ticket():
    config = StreamingVideoSessionConfig(session_id="lineage-reset")
    prompt = {"prompt_token_ids": [1, 2, 3]}
    ticket = video_stream_base._attach_thinker_lineage(config, prompt)
    old_id = config._thinker_lineage_id

    video_stream_base._reset_thinker_lineage(config)

    assert config._thinker_lineage_id != old_id
    assert config._thinker_lineage_revision == 0
    assert not video_stream_base._commit_thinker_lineage(config, ticket)


def test_processed_canonical_blocks_merge_multimodal_offsets_and_items():
    from vllm.multimodal.inputs import (
        MultiModalKwargsItem,
        MultiModalKwargsItems,
        PlaceholderRange,
    )

    first = {
        "type": "multimodal",
        "prompt_token_ids": [1, 2, 3],
        "mm_kwargs": MultiModalKwargsItems({"image": [MultiModalKwargsItem.dummy()]}),
        "mm_hashes": {"image": ["image-a"]},
        "mm_placeholders": {"image": [PlaceholderRange(offset=1, length=1)]},
    }
    second = {
        "type": "multimodal",
        "prompt_token_ids": [4, 5, 6, 7],
        "mm_kwargs": MultiModalKwargsItems({"audio": [MultiModalKwargsItem.dummy()]}),
        "mm_hashes": {"audio": ["audio-b"]},
        "mm_placeholders": {"audio": [PlaceholderRange(offset=0, length=2)]},
    }

    merged = OmniStreamingVideoHandler._merge_engine_prompt_blocks([first, second])

    assert merged["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert merged["mm_hashes"] == {"image": ["image-a"], "audio": ["audio-b"]}
    assert merged["mm_placeholders"]["image"][0].offset == 1
    assert merged["mm_placeholders"]["audio"][0].offset == 3
    assert len(merged["mm_kwargs"]["image"]) == 1
    assert len(merged["mm_kwargs"]["audio"]) == 1


def test_audio_arrival_prompt_preserves_interleaved_media_order_and_stable_chunks():
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    first_frame = _b64(_make_jpeg(1, 2, 3))
    second_frame = _b64(_make_jpeg(4, 5, 6))
    first_audio = b"\x01\x02" * 16000
    second_audio = b"\x03\x04" * 3200
    events = [
        video_stream_base._TurnMediaEvent("image", first_frame),
        video_stream_base._TurnMediaEvent("audio", first_audio),
        video_stream_base._TurnMediaEvent("image", second_frame),
        video_stream_base._TurnMediaEvent("audio", second_audio),
    ]

    _, user = handler.build_engine_prompt(
        StreamingVideoSessionConfig(enable_audio_arrival_prefill_approximation=True),
        [first_frame, second_frame],
        bytearray(first_audio + second_audio),
        [],
        "question",
        {},
        events,
    )

    content = user["content"]
    assert [item["type"] for item in content] == [
        "image_url",
        "input_audio",
        "image_url",
        "input_audio",
        "text",
    ]
    assert content[1]["uuid"].startswith("audio-stream-approx-v1:")
    assert content[3]["uuid"].startswith("audio-stream-approx-v1:")
    assert content[1]["uuid"] != content[3]["uuid"]


def test_audio_arrival_prompt_removes_only_adjacent_internal_audio_boundaries():
    from vllm.multimodal.inputs import PlaceholderRange

    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    config = StreamingVideoSessionConfig(enable_audio_arrival_prefill_approximation=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio"},
                {"type": "input_audio"},
                {"type": "image_url"},
            ],
        }
    ]
    prompt = {
        "type": "multimodal",
        "prompt_token_ids": list(range(12)),
        "mm_placeholders": {
            "audio": [
                PlaceholderRange(offset=1, length=2),
                PlaceholderRange(offset=5, length=3),
            ],
            "image": [PlaceholderRange(offset=9, length=1)],
        },
    }

    normalized = handler._normalize_engine_prompt_for_messages(config, messages, prompt)

    assert normalized["prompt_token_ids"] == [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]
    assert [item.offset for item in normalized["mm_placeholders"]["audio"]] == [1, 3]
    assert normalized["mm_placeholders"]["image"][0].offset == 7


@pytest.mark.asyncio
async def test_incremental_canonical_prompt_never_rerenders_completed_history():
    rendered_roles: list[list[str]] = []
    generated_prompts: list[dict[str, Any]] = []

    class TextEngine:
        def generate(self, *, prompt, **_kwargs):
            generated_prompts.append(prompt)

            async def _gen():
                yield _text_result("answer")

            return _gen()

    class IncrementalHandler(QwenOmniStreamingVideoHandler):
        def supports_incremental_canonical_prompt(self):
            return True

        async def _preprocess_to_engine_prompt(self, request):
            roles = [message["role"] for message in request.messages]
            rendered_roles.append(roles)
            role_tokens = {"system": 10, "user": 20, "assistant": 30}
            token_ids = [role_tokens[role] for role in roles]
            if request.add_generation_prompt:
                token_ids.append(99)
            return {"type": "token", "prompt_token_ids": token_ids}

    config = StreamingVideoSessionConfig(
        model="test",
        modalities=["text"],
        system_prompt="system",
    )
    history: list[dict[str, Any]] = []
    handler = IncrementalHandler(chat_service=object(), engine_client=TextEngine())

    await handler._process_query_engine(
        MockWebSocket(),
        config,
        [],
        bytearray(),
        history,
        "first",
        "req-first",
        asyncio.Event(),
        {},
    )
    calls_after_first = len(rendered_roles)
    await handler._process_query_engine(
        MockWebSocket(),
        config,
        [],
        bytearray(),
        history,
        "second",
        "req-second",
        asyncio.Event(),
        {},
    )

    # Initialization renders only one-message blocks. The second turn renders
    # only its current user block and the short completed assistant block.
    assert all(len(roles) == 1 for roles in rendered_roles)
    assert rendered_roles[calls_after_first:] == [["user"], ["assistant"]]
    assert generated_prompts[0]["prompt_token_ids"] == [10, 20, 99]
    assert generated_prompts[1]["prompt_token_ids"] == [10, 20, 30, 20, 99]
    assert len(config._canonical_prompt_state.turn_blocks) == 2


@pytest.mark.asyncio
async def test_more_than_eight_filtered_frames_remain_in_one_turn():
    captured_frames: list[list[str]] = []

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(
            self,
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
            **kwargs,
        ):
            captured_frames.append(list(frame_buffer))

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "enable_frame_filter": False,
            "enable_video_arrival_prefill": False,
        }
    )
    await asyncio.sleep(0)
    for index in range(9):
        ws.put(
            {
                "type": "video.frame",
                "data": _b64(_make_jpeg(index, index + 1, index + 2)),
                "frame_id": f"frame-{index}",
            }
        )
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0.1)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert [len(frames) for frames in captured_frames] == [9]
    accepted = [message for message in ws.sent if message.get("type") == "video.frame.ack"]
    assert len(accepted) == 9
    assert all(message["accepted"] is True for message in accepted)
    assert not any("dropped_frame_id" in message for message in accepted)


@pytest.mark.asyncio
async def test_audio_in_video_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=True)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}


@pytest.mark.asyncio
async def test_audio_in_video_disabled_omits_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text", "audio"], use_audio_in_video=False)

    await handler._process_query_engine(
        ws,
        config,
        [_b64(_make_jpeg())],
        bytearray(b"\x00\x00"),
        [],
        "what is happening?",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs is None


@pytest.mark.asyncio
async def test_query_inline_audio_data_sets_mm_processor_kwargs():
    captured_requests = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            captured_requests.append(request)
            return {"prompt": "x"}

    ws = MockWebSocket(
        [
            json.dumps({"type": "session.config", "model": "test"}),
            json.dumps({"type": "video.frame", "data": _b64(_make_jpeg())}),
            json.dumps(
                {
                    "type": "video.query",
                    "text": "describe",
                    "audio_data": _b64(b"\x00\x00"),
                }
            ),
            json.dumps({"type": "video.done"}),
        ]
    )
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=2.0)

    await handler.handle_session(ws)

    assert captured_requests
    assert captured_requests[0].mm_processor_kwargs == {"use_audio_in_video": True}
    assert "session.done" in [m.get("type") for m in ws.sent]


def test_audio_delta_mode_is_read_by_serving_code_at_runtime(monkeypatch):
    handler = OmniStreamingVideoHandler(chat_service=object())
    result = _audio_result([object()])

    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_fast",
        classmethod(lambda cls, audio_data, chunks_drained: ("fast-path", chunks_drained)),
    )
    monkeypatch.setattr(
        OmniStreamingVideoHandler,
        "_delta_slow",
        classmethod(lambda cls, audio_data, chunks_drained: ("slow-path", chunks_drained)),
    )

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "fast")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "fast-path"

    monkeypatch.setenv("VLLM_VIDEO_AUDIO_DELTA_MODE", "slow")
    assert handler._extract_audio_delta_b64(result, 0)[0] == "slow-path"


def test_video_stream_envs_strip_and_warn_once_per_invalid_value(monkeypatch):
    warnings = []

    video_stream_envs._warned_invalid_envs.clear()
    try:
        monkeypatch.setattr(
            video_stream_envs.logger,
            "warning",
            lambda message, *args, **_kwargs: warnings.append((message, args)),
        )

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", " off ")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "off"
        assert not warnings

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 1

        monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "still_bad")
        assert video_stream_envs.VLLM_VIDEO_ASYNC_CHUNK == "on"
        assert len(warnings) == 2
    finally:
        video_stream_envs._warned_invalid_envs.clear()


@pytest.mark.asyncio
async def test_async_chunk_mode_is_read_by_engine_path_at_runtime(monkeypatch):
    class TextEngine:
        def generate(self, **_kwargs):
            async def _gen():
                yield _text_result("hello")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt": "x"}

    handler = CapturingHandler(chat_service=object(), engine_client=TextEngine())
    config = StreamingVideoSessionConfig(model="test", modalities=["text"])

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "on")
    ws_on = MockWebSocket()
    await handler._process_query_engine(
        ws_on,
        config,
        [_b64(_make_jpeg())],
        bytearray(),
        [],
        "describe",
        "req-on",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.delta", "delta": "hello"} in ws_on.sent

    monkeypatch.setenv("VLLM_VIDEO_ASYNC_CHUNK", "off")
    ws_off = MockWebSocket()
    await handler._process_query_engine(
        ws_off,
        config,
        [_b64(_make_jpeg())],
        bytearray(),
        [],
        "describe",
        "req-off",
        asyncio.Event(),
        {},
    )
    assert {"type": "response.text.done", "text": "hello"} in ws_off.sent
    assert not any(m.get("type") == "response.text.delta" for m in ws_off.sent)


@pytest.mark.asyncio
async def test_query_without_engine_client_sends_error():
    ws = MockWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), engine_client=None)

    await handler._process_query(
        ws,
        StreamingVideoSessionConfig(model="test"),
        [],
        bytearray(),
        [],
        "describe",
        "req-1",
        asyncio.Event(),
        {},
    )

    assert {"type": "error", "message": "Streaming video requires an engine client"} in ws.sent


@pytest.mark.asyncio
async def test_new_query_cancels_in_flight_query():
    query_started = asyncio.Event()
    query_cancelled = asyncio.Event()
    calls = 0

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                return
            query_started.set()
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                query_cancelled.set()
                raise

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.query", "text": "interrupt"})
    await asyncio.wait_for(query_cancelled.wait(), timeout=2.0)
    ws.put({"type": "video.done"})

    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_video_done_waits_for_in_flight_query():
    query_started = asyncio.Event()
    allow_finish = asyncio.Event()
    query_finished = asyncio.Event()

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()
            await allow_finish.wait()
            query_finished.set()

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    ws.put({"type": "video.done"})
    await asyncio.sleep(0.05)
    assert not task.done()
    assert not query_finished.is_set()

    allow_finish.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert query_finished.is_set()
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_frame_prewarm_does_not_block_following_query(monkeypatch):
    decode_started = threading.Event()
    release_decode = threading.Event()
    query_started = asyncio.Event()

    def blocked_decode(raw_bytes: bytes):
        decode_started.set()
        release_decode.wait(timeout=2.0)
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    class BlockingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(self, *args, **kwargs):
            query_started.set()

    monkeypatch.setattr(video_stream_base, "_decode_frame_bytes", blocked_decode)
    monkeypatch.setattr(video_stream_base.media_pipeline, "enabled", lambda: False)

    ws = TimedWebSocket()
    handler = BlockingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})

    for _ in range(100):
        if decode_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert decode_started.is_set()

    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.wait_for(query_started.wait(), timeout=2.0)

    release_decode.set()
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_client_cannot_send_internal_frame_decode_failed_message():
    captured_frames: list[list[str]] = []
    frame = _b64(_make_jpeg())

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query(
            self,
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
        ):
            captured_frames.append(list(frame_buffer))

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": frame})
    await asyncio.sleep(0)
    ws.put({"type": "_internal.frame_decode_failed", "b64": frame})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Unknown type: _internal.frame_decode_failed"} in ws.sent
    assert captured_frames == [[frame]]


@pytest.mark.asyncio
async def test_invalid_frame_is_rejected_before_query():
    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test", "enable_frame_filter": False})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(b"not-a-jpeg")})

    for _ in range(100):
        if any(m.get("message") == "Invalid image data" for m in ws.sent):
            break
        await asyncio.sleep(0.01)

    assert {"type": "error", "message": "Invalid image data"} in ws.sent

    ws.put({"type": "video.query", "text": ""})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "No input buffered"} in ws.sent


@pytest.mark.asyncio
async def test_frame_filter_error_sends_invalid_image(monkeypatch):
    def fail_should_retain(self, frame_jpeg):
        raise ValueError("decode failed")

    monkeypatch.setattr(video_stream_base.FrameSimilarityFilter, "should_retain", fail_should_retain)
    monkeypatch.setattr(video_stream_base.media_pipeline, "enabled", lambda: False)

    ws = TimedWebSocket()
    handler = OmniStreamingVideoHandler(chat_service=object(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Invalid image data"} in ws.sent
    assert "session.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_audio_buffer_overflow_clears_buffer_before_query(monkeypatch):
    captured_audio_lengths: list[int] = []

    class EmptyEngine:
        def generate(self, **_kwargs):
            async def _gen():
                if False:
                    yield None

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _process_query_engine(
            self,
            websocket,
            config,
            frame_buffer,
            audio_buffer,
            message_history,
            query_text,
            request_id,
            interrupt_event,
            prewarmed_frames,
        ):
            captured_audio_lengths.append(len(audio_buffer))

    monkeypatch.setattr(video_stream_base, "_MAX_AUDIO_BUFFER_BYTES", 4)

    ws = TimedWebSocket()
    handler = CapturingHandler(chat_service=object(), engine_client=EmptyEngine(), idle_timeout=5.0)
    task = asyncio.create_task(handler.handle_session(ws))

    ws.put({"type": "session.config", "model": "test"})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"1234")})
    await asyncio.sleep(0)
    ws.put({"type": "audio.chunk", "data": _b64(b"5")})
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg())})
    await asyncio.sleep(0)
    ws.put({"type": "video.query", "text": "describe"})
    await asyncio.sleep(0)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert {"type": "error", "message": "Audio buffer overflow"} in ws.sent
    assert captured_audio_lengths == [0]


def test_build_messages_replays_canonical_multimodal_history():
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    old_frame = _b64(_make_jpeg(1, 2, 3))
    current_frame = _b64(_make_jpeg(4, 5, 6))
    history = [
        {"role": "user", "content": [{"type": "text", "text": "old question"}]},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{old_frame}"}},
                {"type": "input_audio", "input_audio": {"data": "ignored", "format": "wav"}},
                {"type": "text", "text": "recent question"},
            ],
        },
        {"role": "assistant", "content": "recent answer"},
    ]

    messages, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test"),
        [current_frame],
        bytearray(),
        history,
        "current question",
        {},
    )

    assert messages[:-1] == history
    assert messages[2]["content"][0]["image_url"]["url"].endswith(old_frame)
    assert messages[2]["content"][1]["type"] == "input_audio"
    assert messages[-1] == user_message
    assert user_message["content"][-1] == {"type": "text", "text": "current question"}


def test_build_messages_does_not_expose_a_per_turn_history_limit():
    handler = QwenOmniStreamingVideoHandler(chat_service=object())
    history = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]

    messages, user_message = handler._build_messages(
        StreamingVideoSessionConfig(model="test"),
        [],
        bytearray(),
        history,
        "current question",
        {},
    )

    assert messages[:-1] == history
    assert messages[-1] == user_message


@pytest.mark.asyncio
async def test_context_compaction_drops_only_complete_oldest_turns():
    rendered_message_counts: list[int] = []
    generated_prompts: list[dict[str, Any]] = []

    class EmptyEngine:
        def generate(self, *, prompt, **_kwargs):
            generated_prompts.append(prompt)

            async def _gen():
                if False:
                    yield None

            return _gen()

    class CountingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_message_counts.append(len(request.messages))
            return {"prompt_token_ids": list(range(1100 * len(request.messages)))}

    history = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]
    handler = CountingHandler(chat_service=object(), engine_client=EmptyEngine())

    await handler._process_query_engine(
        MockWebSocket(),
        StreamingVideoSessionConfig(
            model="test",
            modalities=["text"],
            context_window_trigger_tokens=3000,
            context_window_retained_turns=1,
        ),
        [],
        bytearray(),
        history,
        "current question",
        "req-current",
        asyncio.Event(),
        {},
    )

    assert rendered_message_counts == [5, 3, 1]
    assert len(generated_prompts[0]["prompt_token_ids"]) == 1100
    # The completed current turn becomes the new application-owned lineage.
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1] == {"role": "assistant", "content": ""}


@pytest.mark.asyncio
async def test_arrival_prefill_does_not_rewrite_application_owned_history():
    rendered_message_counts: list[int] = []
    generated_prompts: list[dict[str, Any]] = []

    class EmptyEngine:
        def generate(self, *, prompt, **_kwargs):
            generated_prompts.append(prompt)

            async def _gen():
                if False:
                    yield None

            return _gen()

    class CountingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_message_counts.append(len(request.messages))
            return {"prompt_token_ids": list(range(1100 * len(request.messages)))}

    history = [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]
    handler = CountingHandler(chat_service=object(), engine_client=EmptyEngine())

    ok = await handler._process_video_arrival_prefill(
        StreamingVideoSessionConfig(
            model="test",
            modalities=["text"],
            context_window_trigger_tokens=10_000,
            context_window_retained_turns=2,
        ),
        [_b64(_make_jpeg())],
        history,
        "video-warm-current",
        {},
    )

    assert ok is True
    assert rendered_message_counts == [5]
    assert len(generated_prompts[0]["prompt_token_ids"]) == 5_500
    assert history == [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "question two"},
        {"role": "assistant", "content": "answer two"},
    ]


@pytest.mark.asyncio
async def test_arrival_prefill_skips_prompt_at_hard_context_limit():
    rendered_message_counts: list[int] = []
    engine_calls = 0

    class EmptyEngine:
        def generate(self, **_kwargs):
            nonlocal engine_calls
            engine_calls += 1

            async def _gen():
                if False:
                    yield None

            return _gen()

    class CountingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_message_counts.append(len(request.messages))
            return {"prompt_token_ids": list(range(2000 * len(request.messages)))}

    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    handler = CountingHandler(chat_service=object(), engine_client=EmptyEngine())

    ok = await handler._process_video_arrival_prefill(
        StreamingVideoSessionConfig(
            model="test",
            modalities=["text"],
            context_window_trigger_tokens=5_000,
            context_window_retained_turns=2,
        ),
        [_b64(_make_jpeg())],
        history,
        "video-warm-headroom",
        {},
    )

    assert ok is False
    assert rendered_message_counts == [3]
    assert engine_calls == 0
    assert history == [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]


@pytest.mark.asyncio
async def test_context_compaction_rebuilds_from_configured_recent_turns():
    rendered_message_counts: list[int] = []

    class CountingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_message_counts.append(len(request.messages))
            return {"prompt_token_ids": list(range(1000 * len(request.messages)))}

    history = []
    for turn in range(16):
        history.extend(
            [
                {"role": "user", "content": f"question {turn}"},
                {"role": "assistant", "content": f"answer {turn}"},
            ]
        )
    handler = CountingHandler(chat_service=object())

    prompt, _ = await handler._render_engine_prompt_with_compaction(
        StreamingVideoSessionConfig(
            model="test",
            context_window_trigger_tokens=20_000,
            context_window_retained_turns=2,
        ),
        [],
        bytearray(),
        history,
        "current",
        {},
        output_modalities=["text"],
    )

    assert len(prompt["prompt_token_ids"]) == 5_000
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "question 14"}
    assert history[-1] == {"role": "assistant", "content": "answer 15"}
    assert rendered_message_counts[0] == 33
    assert rendered_message_counts == [33, 5]


@pytest.mark.asyncio
async def test_context_compaction_keeps_audio_text_history_and_latest_current_image():
    rendered_messages: list[list[dict[str, Any]]] = []

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_messages.append(list(request.messages))
            return {"prompt_token_ids": list(range(1000 * len(request.messages)))}

    def historical_user(turn: int) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,old-{turn}"}},
                {
                    "type": "input_audio",
                    "input_audio": {"data": f"audio-{turn}", "format": "wav"},
                },
                {"type": "text", "text": f"question {turn}"},
            ],
        }

    history: list[dict[str, Any]] = []
    for turn in range(3):
        history.extend(
            [
                historical_user(turn),
                {"role": "assistant", "content": f"answer {turn}"},
            ]
        )

    current_frames = [_b64(_make_jpeg(1, 2, 3)), _b64(_make_jpeg(4, 5, 6))]
    handler = CapturingHandler(chat_service=object())
    prompt, current_user = await handler._render_engine_prompt_with_compaction(
        StreamingVideoSessionConfig(
            model="test",
            context_window_trigger_tokens=6_000,
            context_window_retained_turns=2,
        ),
        current_frames,
        bytearray(b"current audio"),
        history,
        "current question",
        {},
        output_modalities=["text"],
    )

    assert len(prompt["prompt_token_ids"]) == 5_000
    assert [len(messages) for messages in rendered_messages] == [7, 5]
    assert len(history) == 4
    assert history[0]["content"] == [
        {
            "type": "input_audio",
            "input_audio": {"data": "audio-1", "format": "wav"},
        },
        {"type": "text", "text": "question 1"},
    ]
    assert history[2]["content"] == [
        {
            "type": "input_audio",
            "input_audio": {"data": "audio-2", "format": "wav"},
        },
        {"type": "text", "text": "question 2"},
    ]
    assert history[1] == {"role": "assistant", "content": "answer 1"}
    assert history[3] == {"role": "assistant", "content": "answer 2"}

    current_types = [part["type"] for part in current_user["content"]]
    assert current_types == ["image_url", "input_audio", "text"]
    assert current_user["content"][0]["image_url"]["url"].endswith(current_frames[-1])
    assert current_user["content"][2] == {"type": "text", "text": "current question"}


@pytest.mark.asyncio
async def test_completed_turn_append_waits_for_context_rewrite_transaction():
    generation_started = asyncio.Event()
    release_generation = asyncio.Event()

    class PausingEngine:
        def generate(self, **_kwargs):
            async def _gen():
                generation_started.set()
                await release_generation.wait()
                yield _text_result("answer")

            return _gen()

    class CountingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            return {"prompt_token_ids": list(range(100 * len(request.messages)))}

    config = StreamingVideoSessionConfig(model="test", modalities=["text"])
    history: list[dict[str, Any]] = []
    websocket = MockWebSocket()
    handler = CountingHandler(chat_service=object(), engine_client=PausingEngine())
    query = asyncio.create_task(
        handler._process_query_engine(
            websocket,
            config,
            [],
            bytearray(),
            history,
            "question",
            "query-waits-for-history-lock",
            asyncio.Event(),
            {},
        )
    )
    await asyncio.wait_for(generation_started.wait(), timeout=2.0)
    await config._history_compaction_lock.acquire()
    release_generation.set()
    for _ in range(20):
        if any(message.get("type") == "response.text.done" for message in websocket.sent):
            break
        await asyncio.sleep(0)

    assert history == []
    assert not query.done()
    config._history_compaction_lock.release()
    await asyncio.wait_for(query, timeout=2.0)
    assert history == [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.asyncio
async def test_websocket_turns_use_distinct_finite_requests_and_replay_media():
    request_ids: list[str] = []
    rendered_requests: list[Any] = []

    class TextEngine:
        def generate(self, *, request_id, **_kwargs):
            request_ids.append(request_id)

            async def _gen():
                yield _text_result("answer")

            return _gen()

    class CapturingHandler(QwenOmniStreamingVideoHandler):
        async def _preprocess_to_engine_prompt(self, request):
            rendered_requests.append(request)
            return {"prompt_token_ids": list(range(32 * len(request.messages)))}

    async def wait_for_responses(ws: TimedWebSocket, count: int) -> None:
        for _ in range(200):
            if ws.sent_types().count("response.text.done") >= count:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"only {ws.sent_types().count('response.text.done')} responses")

    ws = TimedWebSocket()
    handler = CapturingHandler(
        chat_service=object(),
        engine_client=TextEngine(),
        idle_timeout=5.0,
    )
    task = asyncio.create_task(handler.handle_session(ws))
    ws.put(
        {
            "type": "session.config",
            "model": "test",
            "modalities": ["text"],
            "enable_frame_filter": False,
            "enable_video_arrival_prefill": False,
        }
    )
    await asyncio.sleep(0)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(1, 2, 3))})
    ws.put({"type": "video.query", "text": "first"})
    await wait_for_responses(ws, 1)
    ws.put({"type": "video.frame", "data": _b64(_make_jpeg(4, 5, 6))})
    ws.put({"type": "video.query", "text": "second"})
    await wait_for_responses(ws, 2)
    ws.put({"type": "video.done"})
    await asyncio.wait_for(task, timeout=2.0)

    assert len(request_ids) == 2
    assert request_ids[0] != request_ids[1]
    assert all(request_id.startswith("video-") for request_id in request_ids)
    assert len(rendered_requests[0].messages) == 1
    assert len(rendered_requests[1].messages) == 3
    first_turn = rendered_requests[1].messages[0]
    assert any(item["type"] in {"image_url", "image_pil"} for item in first_turn["content"])
