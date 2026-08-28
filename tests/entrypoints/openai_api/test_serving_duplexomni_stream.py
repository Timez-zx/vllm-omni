from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any

import pytest
from PIL import Image

from vllm_omni.engine.duplexomni_pipeline import (
    PIPELINE_BASE_TURNS,
    PIPELINE_EPOCH,
    PIPELINE_FINAL,
    PIPELINE_SESSION_ID,
    PIPELINE_SLOT,
)
from vllm_omni.entrypoints.openai.serving_duplexomni_stream import (
    SLOT_PCM_BYTES,
    DuplexOmniStreamingVideoHandler,
)
from vllm_omni.entrypoints.openai.serving_video_stream import (
    QwenOmniStreamingVideoHandler,
    create_streaming_video_handler,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _codes(value: int) -> list[list[int]]:
    return [[value] * 6 for _ in range(16)]


def _model_stream(
    text: str,
    *,
    value: int = 1,
    prompt_tokens: int = 100,
    wait_after_thinker: asyncio.Event | None = None,
):
    async def generate():
        yield _sse(
            {
                "id": f"request-{value}",
                "modality": "text",
                "choices": [{"delta": {"content": text}}],
                "metrics": {"stage_id": 0},
            }
        )
        if wait_after_thinker is not None:
            await wait_after_thinker.wait()
        yield _sse(
            {
                "id": f"request-{value}",
                "modality": "audio",
                "choices": [{"delta": {"content": base64.b64encode(b"wav").decode()}}],
                "metrics": {
                    "stage_id": 2,
                    "duplexomni": {
                        "codec_codes": _codes(value),
                        "valid_turn": True,
                        "eos_emitted": True,
                    },
                },
                "usage": {"prompt_tokens": prompt_tokens},
            }
        )
        yield "data: [DONE]\n\n"

    return generate()


class _Chat:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def create_chat_completion(self, request, raw_request=None):
        self.requests.append(request)
        return _model_stream(
            '{"asr":"hi","tts":"ok","tts_control":"","system2_control":"[WAIT]"}',
            value=len(self.requests),
        )


class _WebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages: asyncio.Queue[str] = asyncio.Queue()
        for message in messages:
            self.messages.put_nowait(json.dumps(message))
        self.sent: list[dict[str, Any]] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive_text(self) -> str:
        return await self.messages.get()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


def _pcm_message(slot: int, *, final: bool) -> dict[str, Any]:
    return {
        "type": "audio.chunk",
        "slot": slot,
        "final": final,
        "data": base64.b64encode(bytes(SLOT_PCM_BYTES)).decode(),
    }


def _jpeg() -> bytes:
    image = Image.new("RGB", (64, 64), (120, 30, 10))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


@pytest.mark.asyncio
async def test_session_owns_media_history_and_finite_request_metadata() -> None:
    chat = _Chat()
    websocket = _WebSocket(
        [
            {
                "type": "session.config",
                "session_id": "session-a",
                "model": "DuplexOmni",
                "enable_frame_filter": False,
            },
            {
                "type": "video.frame",
                "frame_id": "frame-0",
                "data": base64.b64encode(_jpeg()).decode(),
            },
            _pcm_message(0, final=True),
            {"type": "session.finish"},
        ]
    )

    await DuplexOmniStreamingVideoHandler(chat).handle_session(websocket)

    assert websocket.accepted
    assert [event["type"] for event in websocket.sent].count("response.done") == 1
    assert websocket.sent[-1]["type"] == "session.done"
    assert len(chat.requests) == 1
    request = chat.requests[0]
    assert request.cache_salt == "session-a:epoch-0"
    assert len(request.messages) == 2
    user_content = request.messages[-1]["content"]
    assert any(part["type"] == "input_audio" for part in user_content)
    assert any(part["type"] == "image_url" for part in user_content)
    meta = request.additional_information["meta"]
    assert meta == {
        PIPELINE_SESSION_ID: "session-a",
        PIPELINE_EPOCH: 0,
        PIPELINE_SLOT: 0,
        PIPELINE_FINAL: True,
        PIPELINE_BASE_TURNS: 0,
    }


@pytest.mark.asyncio
async def test_next_thinker_starts_before_predecessor_audio_finishes() -> None:
    release_first = asyncio.Event()
    second_started = asyncio.Event()

    class OverlapChat(_Chat):
        async def create_chat_completion(self, request, raw_request=None):
            self.requests.append(request)
            index = len(self.requests)
            if index == 2:
                second_started.set()
            return _model_stream(
                f'{{"asr":"{index}","tts":"ok","tts_control":"","system2_control":"[WAIT]"}}',
                value=index,
                wait_after_thinker=release_first if index == 1 else None,
            )

    chat = OverlapChat()
    websocket = _WebSocket(
        [
            {
                "type": "session.config",
                "session_id": "session-overlap",
                "enable_frame_filter": False,
                "max_inflight_per_session": 4,
            },
            _pcm_message(0, final=False),
            _pcm_message(1, final=True),
            {"type": "session.finish"},
        ]
    )
    task = asyncio.create_task(DuplexOmniStreamingVideoHandler(chat).handle_session(websocket))

    await asyncio.wait_for(second_started.wait(), timeout=2.0)
    assert not release_first.is_set()
    release_first.set()
    await asyncio.wait_for(task, timeout=2.0)

    starts = [event for event in websocket.sent if event["type"] == "response.start"]
    assert starts[1]["overlapped_predecessor_audio"] is True
    assert chat.requests[1].messages[-2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_context_compaction_is_server_owned_and_starts_a_new_epoch() -> None:
    class LongPromptChat(_Chat):
        async def create_chat_completion(self, request, raw_request=None):
            self.requests.append(request)
            return _model_stream(
                '{"asr":"hi","tts":"ok","tts_control":"","system2_control":"[WAIT]"}',
                value=len(self.requests),
                prompt_tokens=1024,
            )

    chat = LongPromptChat()
    websocket = _WebSocket(
        [
            {
                "type": "session.config",
                "session_id": "session-compact",
                "enable_frame_filter": False,
                "slot_overlap": False,
                "context_window_trigger_tokens": 1024,
            },
            _pcm_message(0, final=False),
            _pcm_message(1, final=False),
            _pcm_message(2, final=True),
            {"type": "session.finish"},
        ]
    )

    await DuplexOmniStreamingVideoHandler(chat).handle_session(websocket)

    compacted = [event for event in websocket.sent if event["type"] == "session.history.compacted"]
    assert compacted == [
        {
            "type": "session.history.compacted",
            "after_slot": 1,
            "slots_before": 2,
            "slots_after": 1,
            "cache_epoch": 1,
        }
    ]
    third = chat.requests[2]
    assert len(third.messages) == 4
    assert third.additional_information["meta"][PIPELINE_EPOCH] == 1
    assert third.additional_information["meta"][PIPELINE_BASE_TURNS] == 1


@pytest.mark.asyncio
async def test_canonical_prompt_processes_only_new_slot_blocks() -> None:
    class CanonicalChat(_Chat):
        renderer = object()

        def __init__(self) -> None:
            super().__init__()
            self.engine_prompts: list[dict[str, Any]] = []

        async def create_chat_completion_from_engine_prompt(
            self,
            request,
            engine_prompt,
            raw_request=None,
        ):
            self.requests.append(request)
            self.engine_prompts.append(engine_prompt)
            return _model_stream(
                '{"asr":"hi","tts":"ok","tts_control":"","system2_control":"[WAIT]"}',
                value=len(self.requests),
                prompt_tokens=len(engine_prompt["prompt_token_ids"]),
            )

    class CanonicalHandler(DuplexOmniStreamingVideoHandler):
        def __init__(self, chat) -> None:
            super().__init__(chat)
            self.rendered_roles: list[tuple[str, ...]] = []

        async def _render_canonical_block(
            self,
            config,
            messages,
            *,
            add_generation_prompt,
        ):
            del config
            roles = tuple(message["role"] for message in messages)
            self.rendered_roles.append(roles)
            role_tokens = {"system": 10, "user": 20, "assistant": 30}
            tokens = [role_tokens[message["role"]] for message in messages]
            if add_generation_prompt:
                tokens.append(99)
            return {"type": "token", "prompt_token_ids": tokens}

    chat = CanonicalChat()
    handler = CanonicalHandler(chat)
    websocket = _WebSocket(
        [
            {
                "type": "session.config",
                "session_id": "session-canonical",
                "enable_frame_filter": False,
                "slot_overlap": False,
            },
            _pcm_message(0, final=False),
            _pcm_message(1, final=True),
            {"type": "session.finish"},
        ]
    )

    await handler.handle_session(websocket)

    assert len(chat.engine_prompts) == 2
    assert chat.engine_prompts[0]["prompt_token_ids"] == [10, 20, 99]
    assert chat.engine_prompts[1]["prompt_token_ids"] == [10, 20, 30, 20, 99]
    # Prefix/probe setup is one-time. Historical user media is represented by
    # the committed block and is not sent back through the renderer.
    assert handler.rendered_roles.count(("system",)) == 1
    assert handler.rendered_roles.count(("assistant",)) == 2
    assert handler.rendered_roles.count(("user",)) == 4  # two probes + two slots
    assert chat.engine_prompts[1]["cache_salt"] == "session-canonical:epoch-0"
    assert chat.engine_prompts[1]["additional_information"]["meta"][PIPELINE_SLOT] == 1
    assert chat.engine_prompts[0]["kv_lineage_parent_revision"] == 0
    assert chat.engine_prompts[0]["kv_lineage_revision"] == 1
    assert chat.engine_prompts[0]["kv_lineage_prefix_tokens"] == 0
    assert chat.engine_prompts[1]["kv_lineage_parent_revision"] == 1
    assert chat.engine_prompts[1]["kv_lineage_revision"] == 2
    assert chat.engine_prompts[1]["kv_lineage_prefix_tokens"] == 2


def test_factory_selects_only_the_duplexomni_session_adapter() -> None:
    class Engine:
        pipeline_model_type = "duplexomni"

    duplex = create_streaming_video_handler(object(), engine_client=Engine())
    qwen = create_streaming_video_handler(object(), engine_client=object())

    assert isinstance(duplex, DuplexOmniStreamingVideoHandler)
    assert isinstance(qwen, QwenOmniStreamingVideoHandler)


def test_factory_selects_duplexomni_pd_adapter_and_decode_stage() -> None:
    class Engine:
        pipeline_model_type = "duplexomni_pd"

    handler = create_streaming_video_handler(object(), engine_client=Engine())

    assert isinstance(handler, DuplexOmniStreamingVideoHandler)
    assert handler._thinker_output_stage_id == 1
