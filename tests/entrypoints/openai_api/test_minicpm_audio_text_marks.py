"""Response-scoped alignment regressions through the real CPU output path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import MiniCPMO45DataPlaneContext
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.openai.realtime_session import NativeRealtimeSessionProtocol
from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler
from vllm_omni.outputs.output_processor import OmniRequestState
from vllm_omni.utils.mm_outputs import partition_payload_list

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class _ChatService:
    duplex_serving_adapter_path = (
        "vllm_omni.experimental.fullduplex.minicpmo45.serving_adapter.MiniCPMO45ServingRuntimeAdapter"
    )
    model_config = SimpleNamespace(model="test-model")
    engine_client = SimpleNamespace()

    def create_audio(self, audio_obj):
        return SimpleNamespace(audio_data=f"pcm-{audio_obj.audio_tensor.shape[0]}")


class _OutputPath:
    def __init__(self):
        self.handler = OmniDuplexSessionHandler(chat_service=_ChatService())
        self.session = DuplexSession(
            session_id="marks-test", config=DuplexSessionConfig(extra_body={"auto_response": True})
        )
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def emit(
        self, text, durations=(), *, turn=0, native=True, end=False, marks=None, delta_seq=None, cache_epoch=0
    ):
        mm = {"sr": 24000}
        if native:
            mm.update({"meta.duplex_epoch": 0, "meta.duplex_turn_id": turn, "meta.turn_end": end})
        if text:
            mm["meta.llm_output_text_utf8"] = np.frombuffer(text.encode(), dtype=np.uint8)
        if durations:
            mm["audio"] = [np.zeros(24 * duration, dtype=np.float32) for duration in durations]
        if marks is not None:
            mm["audio_text_marks"] = marks
        if delta_seq is not None:
            mm.update(
                {
                    "meta.llm_output_text_is_delta": True,
                    "meta.cache_epoch": cache_epoch,
                    "meta.chunk_seq": delta_seq,
                }
            )
        output = SimpleNamespace(
            request_id="duplex-marks-test-e0-stage0",
            finished=False,
            outputs=[SimpleNamespace(text=text, token_ids=[], multimodal_output={})],
            multimodal_output=mm,
        )
        session = self.session
        context = MiniCPMO45DataPlaneContext(
            epoch=session.epoch,
            turn_id=session.turn_id,
            active_response_turn_id=session.active_response_turn_id,
            active_response_id=session.active_response_id,
            auto_responds=True,
        )
        results = list(self.handler._minicpmo_data_plane.project_output(output, context=context))
        start = len(self.sent)
        for result in results:
            await self.handler._send_one_native_duplex_event(self.send, result, session=session)
        return [p for p in self.sent[start:] if p["type"] == "response.output_audio.delta"]


@pytest.mark.asyncio
async def test_marks_use_response_offsets_across_talker_emissions():
    path = _OutputPath()
    first = await path.emit("hello", [240])
    second = await path.emit("world", [1000])
    assert first[0]["audio_text_marks"] == [{"text_chars": 5, "audio_end_ms": 240}]
    assert second[0]["audio_text_marks"] == [{"text_chars": 10, "audio_end_ms": 1240}]
    # Fixing alignment must not change PCM chunking or transcript delivery.
    assert [p["audio"] for p in first + second] == ["pcm-5760", "pcm-24000"]
    assert [p["text"] for p in first + second] == ["hello", "world"]


@pytest.mark.asyncio
async def test_multi_chunk_group_marks_only_its_end_with_cumulative_offsets():
    path = _OutputPath()
    await path.emit("hello", [240])
    chunks = await path.emit("world", [400, 600])
    assert [p["text"] for p in chunks] == ["world", ""]
    assert [p["audio_duration_ms"] for p in chunks] == [640, 1240]
    assert "audio_text_marks" not in chunks[0]
    assert chunks[1]["audio_text_marks"] == [{"text_chars": 10, "audio_end_ms": 1240}]


@pytest.mark.asyncio
async def test_audio_only_continuation_does_not_restart_text_coordinates():
    path = _OutputPath()
    await path.emit("hello", [240])
    chunk = (await path.emit("", [1000]))[0]
    assert chunk["text"] == ""
    assert chunk["audio_text_marks"] == [{"text_chars": 5, "audio_end_ms": 1240}]


@pytest.mark.asyncio
async def test_delayed_legacy_text_marks_end_of_all_buffered_audio():
    path = _OutputPath()
    assert await path.emit("", [400], native=False) == []
    assert await path.emit("", [600], native=False) == []
    chunks = await path.emit("hello", native=False)
    assert [p["audio"] for p in chunks] == ["pcm-9600", "pcm-14400"]
    assert [p["text"] for p in chunks] == ["hello", ""]
    assert "audio_text_marks" not in chunks[0]
    assert chunks[1]["audio_text_marks"] == [{"text_chars": 5, "audio_end_ms": 1000}]


@pytest.mark.asyncio
async def test_new_response_starts_new_alignment_coordinates():
    path = _OutputPath()
    first = await path.emit("hello", [1000], end=True)
    second = await path.emit("ok", [200], turn=1)
    assert first[0]["response_id"] != second[0]["response_id"]
    assert second[0]["audio_text_marks"] == [{"text_chars": 2, "audio_end_ms": 200}]


@pytest.mark.asyncio
async def test_explicit_native_alignment_is_not_replaced_by_fallback():
    path = _OutputPath()
    native_marks = [{"text_chars": 2, "audio_end_ms": 100}, {"text_chars": 5, "audio_end_ms": 240}]
    chunks = await path.emit("hello", [100, 140], marks=native_marks)
    assert "audio_text_marks" not in chunks[0]
    assert chunks[1]["audio_text_marks"] == native_marks


@pytest.mark.asyncio
async def test_interleaved_sessions_have_independent_alignment_cursors():
    first, second = _OutputPath(), _OutputPath()
    await first.emit("hello", [240])
    await second.emit("x", [100])
    assert (await first.emit("world", [1000]))[0]["audio_text_marks"] == [{"text_chars": 10, "audio_end_ms": 1240}]
    assert (await second.emit("yz", [200]))[0]["audio_text_marks"] == [{"text_chars": 3, "audio_end_ms": 300}]


@pytest.mark.asyncio
async def test_distinct_native_payloads_preserve_repeated_and_prefix_text():
    path = _OutputPath()
    emitted = []
    for seq, text in enumerate(["呢，还没到", "呢，还没到呢，还没到", "呢，还没到", "呢，还没到呢，还"]):
        emitted.extend(await path.emit(text, [1000], delta_seq=seq))
    assert [p["text"] for p in emitted] == ["呢，还没到", "呢，还没到呢，还没到", "呢，还没到", "呢，还没到呢，还"]
    assert len([p for p in path.sent if p["type"] == "response.created"]) == 1


@pytest.mark.asyncio
async def test_replayed_native_payload_does_not_duplicate_text_or_pcm():
    path = _OutputPath()
    first = await path.emit("same", [1000], delta_seq=0)
    assert len(first) == 1
    assert await path.emit("same", [1000], delta_seq=0) == []
    newer = await path.emit("same", [1000], delta_seq=1)
    assert newer[0]["text"] == "same"
    assert newer[0]["audio_duration_ms"] == 2000
    assert await path.emit("old", [1000], delta_seq=0) == []


@pytest.mark.asyncio
async def test_chunk_sequence_may_restart_in_new_cache_epoch():
    path = _OutputPath()
    first = await path.emit("same", [1000], delta_seq=4, cache_epoch=0)
    second = await path.emit("same", [1000], delta_seq=0, cache_epoch=1)
    assert [p["text"] for p in first + second] == ["same", "same"]
    assert await path.emit("stale", [1000], delta_seq=5, cache_epoch=0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("had_audio", [False, True])
async def test_native_terminal_text_without_pcm_is_delivered_without_spoken_mark(had_audio):
    path = _OutputPath()
    if had_audio:
        await path.emit("hello", [1000], delta_seq=0)
    ending = await path.emit("tail", delta_seq=1, end=True)
    assert len(ending) == 1
    assert ending[0]["text"] == "tail"
    assert ending[0]["audio"] == ""
    assert "audio_text_marks" not in ending[0]
    assert "audio_duration_ms" not in ending[0]
    assert len([p for p in path.sent if p["type"] == "response.done"]) == 1
    assert await path.emit("tail", delta_seq=1, end=True) == []
    protocol = NativeRealtimeSessionProtocol(SimpleNamespace())
    wire = [event for payload in path.sent for event in protocol._from_duplex_event(payload)]
    done = [event for event in wire if event["type"] == "response.done"]
    assert len(done) == 1
    content = done[0]["response"]["output"][0]["content"][0]
    assert content["transcript"] == ("hello" if had_audio else "") + "tail"
    assert len([event for event in wire if event["type"] == "response.audio.delta"]) == int(had_audio)
    assert all(mark["text_chars"] <= (5 if had_audio else 0) for mark in content.get("audio_text_marks", []))


@pytest.mark.asyncio
async def test_native_nonterminal_text_without_pcm_keeps_response_open():
    path = _OutputPath()
    first = await path.emit("hello", delta_seq=0)
    assert first[0]["audio"] == ""
    assert "audio_text_marks" not in first[0]
    assert not any(p["type"] == "response.done" for p in path.sent)
    second = await path.emit("world", [1000], delta_seq=1, end=True)
    assert second[0]["response_id"] == first[0]["response_id"]
    assert second[0]["text"] == "world"
    assert len([p for p in path.sent if p["type"] == "response.done"]) == 1


@pytest.mark.asyncio
async def test_native_delta_requires_identity_instead_of_guessing_from_text():
    path = _OutputPath()
    assert await path.emit("hello", [1000], delta_seq=0, native=False) == []
    errors = [p for p in path.sent if p["type"] == "error"]
    assert errors[0]["code"] == "runtime_data_plane_invalid_chunk_identity"


class _WorkerOutputPath(_OutputPath):
    """Use the real generation worker partition and request accumulator.

    Previous regressions injected the producer metadata directly into the
    projector, missing the client-facing worker whitelist entirely.
    """

    def __init__(self):
        super().__init__()
        self.state = OmniRequestState(
            request_id="duplex-marks-test-e0-stage0",
            external_req_id="duplex-marks-test-e0-stage0",
            parent_req=None,
            request_index=0,
            lora_request=None,
            prompt=None,
            prompt_token_ids=[0],
            prompt_embeds=None,
            logprobs_processor=None,
            detokenizer=None,
            max_tokens_param=None,
            arrival_time=0.0,
            queue=None,
            log_stats=False,
            stream_interval=1,
            output_kind=RequestOutputKind.CUMULATIVE,
        )
        self.outputs_by_chunk = {}

    async def chunk(self, text, seq, *, end=False, cache_epoch=0, remove=(), pcm=True):
        flat = {
            "model_outputs": torch.zeros(2400 if pcm else 0, dtype=torch.float32),
            "sr": torch.tensor(24000, dtype=torch.int32),
            "meta.duplex_epoch": torch.tensor(0, dtype=torch.int32),
            "meta.duplex_turn_id": torch.tensor(0, dtype=torch.int32),
            "meta.cache_epoch": torch.tensor(cache_epoch, dtype=torch.int64),
            "meta.chunk_seq": torch.tensor(seq, dtype=torch.int64),
            "meta.llm_output_text_is_delta": torch.tensor(True),
            "meta.llm_output_text_utf8": torch.tensor(list(text.encode()), dtype=torch.uint8),
            "meta.tts_is_last_chunk": torch.tensor(end),
            "meta.turn_end": torch.tensor(end),
        }
        for key in remove:
            flat.pop(key)
        # This is the exact async_chunk worker branch, including one tensor
        # per request (not the model's batched list-of-tensors output).
        _intermediate, client = partition_payload_list([flat])
        key = (cache_epoch, seq)
        if key not in self.outputs_by_chunk:
            self.state.add_multimodal_tensor(client[0], mm_type="audio")
            self.outputs_by_chunk[key] = self.state.make_request_output([], None, None, None)
        output = self.outputs_by_chunk[key]
        context = MiniCPMO45DataPlaneContext(
            epoch=0,
            turn_id=0,
            auto_responds=True,
            require_native_audio_chunk_identity=True,
            active_response_id=self.session.active_response_id,
            active_response_turn_id=self.session.active_response_turn_id,
        )
        start = len(self.sent)
        for result in self.handler._minicpmo_data_plane.project_output(output, context=context):
            await self.handler._send_one_native_duplex_event(self.send, result, session=self.session)
        return self.sent[start:]


@pytest.mark.asyncio
async def test_worker_partition_accumulation_projector_handler_preserve_native_text():
    path = _WorkerOutputPath()
    texts = ["0", "0", "00", "呢，还没到", "呢，还没到呢，还没到"]
    for seq, text in enumerate(texts):
        await path.chunk(text, seq)
    assert await path.chunk(texts[-1], len(texts) - 1) == []
    # A new cache epoch may reuse seq=0 without dropping genuinely equal text.
    await path.chunk("0", 0, cache_epoch=1)
    await path.chunk("末尾", 1, cache_epoch=1, end=True, pcm=False)
    protocol = NativeRealtimeSessionProtocol(SimpleNamespace())
    wire = [event for payload in path.sent for event in protocol._from_duplex_event(payload)]
    expected = "".join(texts) + "0末尾"
    assert "".join(e["delta"] for e in wire if e["type"] == "response.audio_transcript.delta") == expected
    finals = [e for e in wire if e["type"] == "response.done"]
    assert len(finals) == 1
    content = finals[0]["response"]["output"][0]["content"][0]
    assert content["transcript"] == expected
    assert len([e for e in wire if e["type"] == "response.audio.delta"]) == 6
    assert content["audio_text_marks"][-1] == {"text_chars": len(expected) - 2, "audio_end_ms": 600}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    [
        ("meta.llm_output_text_is_delta",),
        ("meta.chunk_seq",),
        ("meta.cache_epoch",),
        ("meta.duplex_epoch",),
        ("meta.duplex_turn_id",),
        ("meta.llm_output_text_is_delta", "meta.chunk_seq", "meta.cache_epoch"),
    ],
)
async def test_native_serving_missing_transport_contract_is_explicit_error(missing):
    path = _WorkerOutputPath()
    events = await path.chunk("00", 0, remove=missing)
    assert not any(e["type"].startswith("response.") for e in events)
    assert [e["code"] for e in events if e["type"] == "error"] == ["runtime_data_plane_invalid_chunk_identity"]


def test_serving_adapter_requires_native_chunk_contract_only_for_auto_response():
    from vllm_omni.experimental.fullduplex.minicpmo45.serving_adapter import MiniCPMO45ServingRuntimeAdapter

    for auto in (False, True):
        context = MiniCPMO45ServingRuntimeAdapter.data_plane_context(
            epoch=0,
            turn_id=0,
            active_response_turn_id=None,
            active_response_id=None,
            auto_responds=auto,
            response_format="wav",
            speed=None,
            modalities=(),
        )
        assert context.require_native_audio_chunk_identity is auto
