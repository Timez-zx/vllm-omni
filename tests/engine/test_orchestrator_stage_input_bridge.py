# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock

import janus
import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

import vllm_omni.engine.orchestrator as orchestrator_module
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _extend_native_pd_feedback_budget,
    _OrchestratorDuplexStagePort,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexStageRequestContext,
    DuplexStageSubmission,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStageClient:
    def __init__(
        self,
        *,
        next_inputs: list[dict[str, Any]] | None = None,
        final_output: bool = False,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = "llm"
        self.final_output = final_output
        self.final_output_type = "text"
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = [0]
        self.is_comprehension = False
        self.model_stage = None
        self.custom_process_input_func = None
        self.next_inputs = list(next_inputs or [])
        self.add_request_calls: list[tuple[Any, ...]] = []
        self.decoded_source_tokens: str | None = None
        self._engine_core_outputs = queue.Queue()

    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def process_engine_inputs(self, _source_outputs, prompt=None, streaming_context=None):
        decoder = getattr(streaming_context, "source_token_decoder", None)
        if callable(decoder):
            self.decoded_source_tokens = decoder([11, 12], skip_special_tokens=True)
        return list(self.next_inputs)

    async def abort_requests_async(self, _request_ids: list[str]) -> None:
        return None

    def set_engine_outputs(self, _outputs) -> None:
        return None

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class FakeOutputProcessor:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def add_request(self, *args, **kwargs) -> None:
        return None


class FakeInputProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process_inputs(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            request_id=kwargs["request_id"],
            prompt_token_ids=[101, 102],
            prompt_embeds=None,
            external_req_id=None,
        )


class FakePrewarmPool:
    stage_type = "llm"

    def __init__(self, role: str) -> None:
        self.stage_vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=64,
                stage_connector_config={"extra": {"role": role}},
            )
        )
        self.submitted: list[Any] = []

    async def submit_initial(self, _request_id, _req_state, request, prompt_text=None):
        self.submitted.append(request)
        return 0

    def get_bound_replica_id(self, _request_id):
        return 0


def _duplex_stage_port_submission():
    stage_pools = []
    for stage_id in range(3):
        pool = SimpleNamespace(
            stage_client=SimpleNamespace(default_sampling_params=SamplingParams(max_tokens=1)),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            submit_initial=AsyncMock(return_value=stage_id + 10),
            submit_update=AsyncMock(return_value=stage_id + 20),
        )
        stage_pools.append(pool)
    request_states: dict[str, OrchestratorRequestState] = {}
    prewarm = AsyncMock()
    port = _OrchestratorDuplexStagePort(
        stage_pools=stage_pools,
        request_states=request_states,
        running_counter=None,
        cleanup_request_ids=AsyncMock(),
        async_chunk=True,
        prewarm_async_chunk_stages=prewarm,
    )
    context = DuplexStageRequestContext(
        request_id="req-duplex",
        session_id="session-duplex",
        fence=DuplexFence("session-duplex"),
        stage_id=0,
        final_stage_id=2,
        config_generation=0,
        sampling_params=tuple(SamplingParams(max_tokens=1) for _ in range(3)),
    )
    port.ensure_request(context)
    submission = DuplexStageSubmission(
        context=context,
        prompt={"prompt_token_ids": [1, 2]},
        already_submitted=False,
    )
    return port, stage_pools, request_states, prewarm, submission


def _request_output(request_id: str) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="transcript",
        token_ids=[11, 12],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
        metrics=None,
        lora_request=None,
    )


@pytest.mark.asyncio
async def test_forward_text_prompt_uses_target_stage_input_processor() -> None:
    class SourceTokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ":".join(str(token_id) for token_id in token_ids)

    stage0 = FakeStageClient(final_output=True)
    stage1 = FakeStageClient(
        final_output=True,
        next_inputs=[{"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}],
    )
    stage_pools = [
        StagePool(
            0,
            [stage0],
            output_processor=FakeOutputProcessor(tokenizer=SourceTokenizer()),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
        StagePool(
            1,
            [stage1],
            output_processor=FakeOutputProcessor(),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ),
    ]
    request_q = janus.Queue()
    output_q = janus.Queue()
    rpc_q = janus.Queue()
    orchestrator = Orchestrator(
        request_async_queue=request_q.async_q,
        output_async_queue=output_q.async_q,
        rpc_async_queue=rpc_q.async_q,
        stage_pools=stage_pools,
        async_chunk=False,
    )
    input_processor = FakeInputProcessor()
    orchestrator._stage_input_processors[1] = input_processor
    req_state = OrchestratorRequestState(
        request_id="req-text",
        prompt={"prompt": "original"},
        sampling_params_list=[SamplingParams(max_tokens=1), SamplingParams(max_tokens=1)],
        final_stage_id=1,
    )

    await orchestrator._forward_to_next_stage("req-text", 0, _request_output("req-text"), req_state)

    assert input_processor.calls
    assert input_processor.calls[0]["prompt"] == {"prompt": "hello", "multi_modal_data": {"video": ["frame"]}}
    assert stage1.decoded_source_tokens == "11:12"
    assert req_state.streaming.source_token_decoder is None
    assert stage1.add_request_calls
    submitted_request = stage1.add_request_calls[0][0]
    assert submitted_request.prompt_token_ids == [101, 102]
    assert submitted_request.external_req_id == "req-text"


@pytest.mark.asyncio
async def test_async_prewarm_skips_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("sender")
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._record_duplex_stage_submission = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-prewarm",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        duplex_identity=SimpleNamespace(),
    )

    await orchestrator._prewarm_async_chunk_stages(
        "req-prewarm",
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert stage1.submitted == []
    assert len(stage2.submitted) == 1
    assert 1 not in req_state.stage_submit_ts
    assert 2 in req_state.stage_submit_ts
    orchestrator._record_duplex_stage_submission.assert_called_once_with(
        2,
        "req-prewarm",
        0,
        req_state,
    )


@pytest.mark.asyncio
async def test_pd_only_diagnostic_does_not_prewarm_downstream_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator_module,
        "_MINICPMO_PD_ONLY_DIAGNOSTIC",
        True,
    )
    orchestrator = object.__new__(Orchestrator)
    stage0 = FakePrewarmPool("sender")
    stage1 = FakePrewarmPool("sender")
    stage2 = FakePrewarmPool("receiver")
    orchestrator.stage_pools = [stage0, stage1, stage2]
    orchestrator._pd_pair = (0, 1)
    orchestrator._is_duplex_session_request = lambda _state: True
    orchestrator._emit_tx_edge = lambda **_kwargs: None
    orchestrator._record_duplex_stage_submission = MagicMock()
    req_state = OrchestratorRequestState(
        request_id="req-pd-only-prewarm",
        prompt={"prompt_token_ids": [1, 2]},
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        duplex_identity=SimpleNamespace(),
    )

    await orchestrator._prewarm_async_chunk_stages(
        req_state.request_id,
        SimpleNamespace(prompt_token_ids=[1, 2], resumable=True),
        req_state,
    )

    assert stage1.submitted == []
    assert stage2.submitted == []
    assert req_state.stage_submit_ts == {}
    orchestrator._record_duplex_stage_submission.assert_not_called()


@pytest.mark.asyncio
async def test_duplex_prewarm_runs_after_first_stage0_submission() -> None:
    port, stage_pools, request_states, prewarm, submission = _duplex_stage_port_submission()

    result = await port.submit(submission)

    assert result.stage_id == 0
    stage_pools[0].submit_initial.assert_awaited_once()
    prewarm.assert_awaited_once_with("req-duplex", ANY, request_states["req-duplex"])


@pytest.mark.asyncio
async def test_native_pd_preregisters_physical_d_slot_during_p() -> None:
    port, stage_pools, request_states, _prewarm, submission = _duplex_stage_port_submission()
    schedule_early = MagicMock()
    port._pd_pair = (0, 1)
    port._schedule_pd_early_cache_sync = schedule_early
    submission = DuplexStageSubmission(
        context=submission.context,
        prompt={
            "prompt_token_ids": [1, 2],
            "model_intermediate_buffer": {"duplex": {"seq": 7}},
        },
        already_submitted=False,
    )

    await port.submit(submission)

    stage_pools[0].submit_initial.assert_awaited_once()
    schedule_early.assert_called_once_with(
        "req-duplex",
        request_states["req-duplex"],
        engine_request_id="req-duplex-00000007",
        prompt_token_ids=[1, 2],
    )
    assert "pd_duplex_predicted_prompt_token_ids" not in request_states["req-duplex"].streaming.bridge_states


@pytest.mark.asyncio
async def test_native_pd_next_submit_observes_background_route_error() -> None:
    port, stage_pools, request_states, _prewarm, submission = _duplex_stage_port_submission()
    port._pd_pair = (0, 1)
    state = request_states[submission.context.request_id]
    decode_ready = asyncio.Event()
    decode_ready.set()
    state.streaming.bridge_states.update(
        {
            "pd_duplex_decode_ready": decode_ready,
            "pd_duplex_prefill_raw_error": "RuntimeError: D admission failed",
        }
    )
    update = DuplexStageSubmission(
        context=submission.context,
        prompt={"prompt_token_ids": [3, 4]},
        already_submitted=True,
    )

    with pytest.raises(RuntimeError, match="previous native P/D route failed"):
        await port.submit(update)

    stage_pools[0].submit_update.assert_not_awaited()


def test_native_pd_prefix_prediction_tracks_append_and_rollover() -> None:
    bridge = {
        "pd_duplex_remote_prompt_token_ids": [1, 2],
        "pd_duplex_prefill_sample_token_ids": [9],
    }

    assert _OrchestratorDuplexStagePort._native_pd_prefix_prediction(
        {"prompt_token_ids": [3, 4]},
        bridge,
        already_submitted=True,
    ) == [1, 2, 3, 4]
    assert _OrchestratorDuplexStagePort._native_pd_prefix_prediction(
        {
            "prompt_token_ids": [5, 6, 7],
            "model_intermediate_buffer": {
                "meta": {
                    "replace_streaming_prompt": True,
                    "retain_streaming_output_tokens": True,
                    "retained_output_insert_offset": 1,
                }
            },
        },
        bridge,
        already_submitted=True,
    ) == [5, 9, 6, 7]


@pytest.mark.asyncio
async def test_native_pd_feedback_extends_stage0_scheduler_budget() -> None:
    port, stage_pools, request_states, _prewarm, submission = _duplex_stage_port_submission()
    port._pd_pair = (0, 1)
    state = request_states[submission.context.request_id]
    ready = asyncio.Event()
    ready.set()
    state.streaming.bridge_states["pd_duplex_decode_ready"] = ready
    state.streaming.bridge_states["pd_duplex_feedback_token_ids"] = [81, 82, 93]
    original_prompt = {
        "prompt_token_ids": [7, 7, 7],
        "model_intermediate_buffer": {
            "duplex": {
                "seq": 2,
                "scheduler_token_id": 7,
                "scheduler_token_budget": 3,
            }
        },
    }
    update = DuplexStageSubmission(
        context=submission.context,
        prompt=original_prompt,
        already_submitted=True,
    )

    await port.submit(update)

    submitted_request = stage_pools[0].submit_update.await_args.args[2]
    assert submitted_request.prompt_token_ids == [7, 7, 7, 7, 7]
    submitted_duplex = submitted_request.model_intermediate_buffer["duplex"]
    assert submitted_duplex["pd_feedback_token_ids"] == [81, 82, 93]
    assert submitted_duplex["scheduler_token_budget"] == 5
    assert original_prompt["prompt_token_ids"] == [7, 7, 7]
    assert "pd_feedback_token_ids" not in original_prompt["model_intermediate_buffer"]["duplex"]


@pytest.mark.parametrize(
    "rollover_marker",
    [
        {"meta": {"replace_streaming_prompt": True}},
        {"payload": {"duplex_context_rollover": True}},
    ],
)
def test_native_pd_feedback_expands_rollover_budget(rollover_marker: dict) -> None:
    model_buffer = {
        "duplex": {
            "scheduler_token_id": 7,
            "scheduler_token_budget": 3,
        }
    }
    if "meta" in rollover_marker:
        model_buffer["meta"] = rollover_marker["meta"]
    else:
        model_buffer["duplex"]["payload"] = rollover_marker["payload"]
    prompt = {
        "prompt_token_ids": [7, 7, 7],
        "model_intermediate_buffer": model_buffer,
    }

    _extend_native_pd_feedback_budget(prompt, [81, 82, 93])

    assert prompt["prompt_token_ids"] == [7, 7, 7, 7, 7]
    assert model_buffer["duplex"]["scheduler_token_budget"] == 5
    assert model_buffer["duplex"]["pd_feedback_token_ids"] == [81, 82, 93]


def test_native_pd_talker_metadata_is_compact() -> None:
    orchestrator = object.__new__(Orchestrator)
    metadata = {"duplex_prompt_token_ids": [999] * 32}
    completion = SimpleNamespace(
        token_ids=[21, 22, 9308],
        multimodal_output=SimpleNamespace(metadata=metadata),
    )
    output = SimpleNamespace(outputs=[completion])
    req_state = OrchestratorRequestState(request_id="req-compact")
    req_state.streaming.bridge_states["pd_decode_prompt"] = {
        "prompt_token_ids": [10, 20, 30, 40],
    }
    req_state.streaming.bridge_states["pd_duplex_special_token_ids"] = {
        "tts_bos_token_id": 9301,
    }

    orchestrator._ensure_native_duplex_pd_talker_metadata(output, req_state)

    assert "duplex_prompt_token_ids" not in metadata
    assert metadata["duplex_prompt_len"] == 4
    assert metadata["duplex_last_prompt_token_id"] == 40
    assert metadata["duplex_segment_token_ids"] == [21, 22, 9308]
    assert metadata["special_token_ids"] == {"tts_bos_token_id": 9301}


def test_native_pd_talker_metadata_uses_scalars_after_prompt_release() -> None:
    orchestrator = object.__new__(Orchestrator)
    metadata = {"duplex_prompt_token_ids": [999] * 32}
    completion = SimpleNamespace(
        token_ids=[21, 22, 9308],
        multimodal_output=SimpleNamespace(metadata=metadata),
    )
    output = SimpleNamespace(outputs=[completion])
    req_state = OrchestratorRequestState(request_id="req-compact-scalars")
    req_state.streaming.bridge_states.update(
        {
            "pd_decode_prompt_len": 4,
            "pd_decode_last_prompt_token_id": 40,
            "pd_duplex_special_token_ids": {"tts_bos_token_id": 9301},
        }
    )

    orchestrator._ensure_native_duplex_pd_talker_metadata(output, req_state)

    assert "duplex_prompt_token_ids" not in metadata
    assert metadata["duplex_prompt_len"] == 4
    assert metadata["duplex_last_prompt_token_id"] == 40
    assert metadata["duplex_segment_token_ids"] == [21, 22, 9308]
    assert metadata["special_token_ids"] == {"tts_bos_token_id": 9301}


@pytest.mark.asyncio
async def test_async_route_forwards_to_outgoing_only_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
    )
    stage0 = SimpleNamespace(final_output=False)
    stage1 = FakePrewarmPool("sender")
    orchestrator.stage_pools = [stage0, stage1]
    orchestrator._forward_to_next_stage = AsyncMock()
    req_state = OrchestratorRequestState(
        request_id="req-route",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(2)],
        final_stage_id=1,
    )
    req_state.stage_submit_ts[0] = 1.0
    output = SimpleNamespace(request_id="req-route", finished=True)

    await orchestrator._route_output(0, 0, output, req_state, None)

    orchestrator._forward_to_next_stage.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_segment_does_not_complete_final_output_stage() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = None
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator.stage_pools = [SimpleNamespace(final_output=True)]
    orchestrator.output_async_queue = asyncio.Queue()
    orchestrator._cleanup_request_ids = AsyncMock()

    req_state = OrchestratorRequestState(
        request_id="req-segment-final-output",
        sampling_params_list=[SamplingParams(max_tokens=1)],
        final_stage_id=0,
        final_output_stage_ids={0},
    )
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = True
    output = SimpleNamespace(
        request_id=req_state.request_id,
        finished=True,
    )

    await orchestrator._route_output(0, 0, output, req_state, None)

    assert req_state.finished_final_output_stage_ids == set()
    orchestrator._cleanup_request_ids.assert_not_awaited()
    routed = orchestrator.output_async_queue.get_nowait()
    assert routed.finished is False


@pytest.mark.asyncio
async def test_native_pd_decode_finish_keeps_talker_request_resumable() -> None:
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = (0, 1)
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator.stage_pools = [
        SimpleNamespace(final_output=False),
        SimpleNamespace(final_output=True),
        SimpleNamespace(final_output=False),
    ]
    orchestrator._ensure_native_duplex_pd_talker_metadata = MagicMock()
    orchestrator._is_duplex_session_request = lambda _state: True
    orchestrator._duplexomni_thinker_output_stage = lambda: 1
    orchestrator._duplex_output_decision = lambda *_args: None
    orchestrator._stage_receives_async_chunks = lambda _stage: False
    orchestrator._forward_to_next_stage = AsyncMock()

    req_state = OrchestratorRequestState(
        request_id="req-pd-segment",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        duplex_identity=SimpleNamespace(),
    )
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = True
    output = SimpleNamespace(request_id=req_state.request_id, finished=True)

    await orchestrator._route_output(1, 0, output, req_state, None)

    orchestrator._forward_to_next_stage.assert_awaited_once()
    assert orchestrator._forward_to_next_stage.await_args.kwargs["is_final_update"] is False


@pytest.mark.asyncio
async def test_pd_only_diagnostic_terminates_after_complete_d_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orchestrator_module,
        "_MINICPMO_PD_ONLY_DIAGNOSTIC",
        True,
    )
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = True
    orchestrator._pd_pair = (0, 1)
    orchestrator._cfg_tracker = SimpleNamespace(
        is_companion=lambda _request_id: False,
        has_companions=lambda _request_id: False,
        cleanup_parent=lambda _request_id: [],
    )
    orchestrator.stage_pools = [
        SimpleNamespace(final_output=False),
        SimpleNamespace(final_output=True),
        SimpleNamespace(final_output=False),
    ]
    orchestrator._ensure_native_duplex_pd_talker_metadata = MagicMock()
    orchestrator._is_duplex_session_request = lambda _state: True
    orchestrator._duplexomni_thinker_output_stage = lambda: 1
    orchestrator._duplex_output_decision = lambda *_args: None
    orchestrator._stage_receives_async_chunks = lambda _stage: False
    orchestrator._emit_duplex_direct_output = AsyncMock()
    orchestrator._forward_to_next_stage = AsyncMock()

    req_state = OrchestratorRequestState(
        request_id="req-pd-only-segment",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        duplex_identity=SimpleNamespace(),
    )
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = True
    output = SimpleNamespace(request_id=req_state.request_id, finished=True)

    await orchestrator._route_output(1, 0, output, req_state, None)

    orchestrator._emit_duplex_direct_output.assert_awaited_once()
    decision = orchestrator._emit_duplex_direct_output.await_args.args[3]
    assert decision.metadata["duplex_pd_only_diagnostic"] is True
    orchestrator._forward_to_next_stage.assert_not_awaited()
