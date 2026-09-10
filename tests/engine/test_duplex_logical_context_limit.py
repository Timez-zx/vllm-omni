"""Logical-limit isolation, independent of physical SWA and GPU execution."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_omni.engine.orchestrator import _OrchestratorDuplexStagePort
from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexContextLimitError,
    DuplexStageRequestContext,
    DuplexStageSubmission,
)
from vllm_omni.experimental.fullduplex.engine.duplex_control_client import DuplexControlRequestError
from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import DuplexControlPlane
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    DuplexFence,
    OpenDuplexSessionMessage,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexSession,
    DuplexSessionConfig,
    DuplexSessionState,
)
from vllm_omni.experimental.fullduplex.openai.runtime_bridge import NativeRuntimeBridgeMixin

LIMIT = 262144
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _port():
    params = [SamplingParams(max_tokens=n) for n in (1, 20, 8192, 1)]
    pools = [
        SimpleNamespace(
            stage_client=SimpleNamespace(default_sampling_params=p),
            stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=LIMIT)),
            submit_initial=AsyncMock(return_value=0),
            submit_update=AsyncMock(return_value=0),
        )
        for p in params
    ]
    states = {}
    port = _OrchestratorDuplexStagePort(
        stage_pools=pools,
        request_states=states,
        running_counter=None,
        cleanup_request_ids=AsyncMock(),
        async_chunk=False,
        prewarm_async_chunk_stages=AsyncMock(),
        pd_pair=(0, 1),
        schedule_pd_early_cache_sync=MagicMock(),
    )
    context = DuplexStageRequestContext(
        request_id="request-a",
        session_id="a",
        fence=DuplexFence("a"),
        stage_id=0,
        final_stage_id=3,
        config_generation=0,
        sampling_params=tuple(params),
    )
    port.ensure_request(context)
    return port, pools, states, context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "length,accepted",
    [
        (89263, True),
        (262123, True),
        (262124, False),
        (262144, False),
        (262255, False),
    ],
)
async def test_real_config_boundary_is_checked_before_any_engine_ingress(length, accepted):
    port, pools, states, context = _port()
    bridge = states[context.request_id].streaming.bridge_states
    ready = asyncio.Event()
    ready.set()
    bridge.update(pd_duplex_decode_ready=ready, pd_duplex_remote_prompt_token_ids=[0] * (length - 211))
    submission = DuplexStageSubmission(
        context=context,
        prompt={"prompt_token_ids": [1] * 211},
        already_submitted=True,
    )
    if accepted:
        await port.submit(submission)
        pools[0].submit_update.assert_awaited_once()
    else:
        with pytest.raises(DuplexContextLimitError) as caught:
            await port.submit(submission)
        assert caught.value.prompt_tokens == length
        assert caught.value.generation_tokens == 21
        pools[0].submit_update.assert_not_awaited()
        pools[0].submit_initial.assert_not_awaited()
        port._schedule_pd_early_cache_sync.assert_not_called()
        # No new unfinished P/D slot, no fake context reset, no changed KV prefix.
        assert bridge["pd_duplex_decode_ready"] is ready and ready.is_set()
        assert len(bridge["pd_duplex_remote_prompt_token_ids"]) == length - 211


@pytest.mark.asyncio
async def test_guard_counts_actual_D_feedback_before_submit():
    port, pools, states, context = _port()
    bridge = states[context.request_id].streaming.bridge_states
    ready = asyncio.Event()
    ready.set()
    bridge.update(
        pd_duplex_decode_ready=ready,
        pd_duplex_remote_prompt_token_ids=[0] * (LIMIT - 24),
        pd_duplex_feedback_token_ids=[81, 82, 93],
        pd_duplex_feedback_sampling_state=[1, 2],
    )
    prompt = {
        "prompt_token_ids": [0] * 3,
        "model_intermediate_buffer": {"duplex": {"seq": 2, "scheduler_token_id": 0, "scheduler_token_budget": 3}},
    }
    # Three media rows fit; two feedback rows make the new complete unit unsafe.
    with pytest.raises(DuplexContextLimitError) as caught:
        await port.submit(DuplexStageSubmission(context=context, prompt=prompt, already_submitted=True))
    assert caught.value.prompt_tokens == LIMIT - 19
    pools[0].submit_update.assert_not_awaited()
    assert bridge["pd_duplex_feedback_token_ids"] == [81, 82, 93]
    assert bridge["pd_duplex_feedback_sampling_state"] == [1, 2]
    # Rejection must not consume feedback and make a retry appear shorter.
    with pytest.raises(DuplexContextLimitError):
        await port.submit(DuplexStageSubmission(context=context, prompt=prompt, already_submitted=True))


def test_limit_uses_smaller_PD_stage_and_actual_generation_budget():
    port, pools, _, context = _port()
    pools[1].stage_vllm_config.model_config.max_model_len = 4096
    port._validate_native_pd_logical_context(context, 4075)
    with pytest.raises(DuplexContextLimitError):
        port._validate_native_pd_logical_context(context, 4076)
    context.sampling_params[1].max_tokens = 40
    with pytest.raises(DuplexContextLimitError) as caught:
        port._validate_native_pd_logical_context(context, 4075)
    assert caught.value.generation_tokens == 41


def test_previous_unprotected_CPU_buffer_failure_is_reproduced():
    # Actual upstream method and real configured buffer size; no CUDA allocation.
    batch = SimpleNamespace(
        _register_add_request=lambda r: 0,
        _req_ids=[],
        req_output_token_ids=[],
        spec_token_ids=[],
        req_id_to_index={},
        num_prompt_tokens=np.zeros(1, dtype=np.int64),
        token_ids_cpu=np.zeros((1, LIMIT), dtype=np.int32),
    )
    req = SimpleNamespace(
        req_id="too-long", prompt_token_ids=[1] * (LIMIT + 111), prompt_embeds=None, output_token_ids=[]
    )
    with pytest.raises(ValueError, match="could not broadcast"):
        InputBatch.add_request(batch, req)
    port, _, _, context = _port()
    with pytest.raises(DuplexContextLimitError):
        port._validate_native_pd_logical_context(context, len(req.prompt_token_ids))


@pytest.mark.asyncio
async def test_control_plane_returns_request_local_error_and_other_user_continues():
    port, pools, _, _ = _port()

    class Extension:
        def configure_sampling_params(self, *, runtime_config, defaults):
            return defaults

        def plan_append(self, *, fence, **kwargs):
            return DuplexAppendPlan(prompt={"prompt_token_ids": [1] * (LIMIT if fence.session_id == "bad" else 211)})

        def decide_output(self, **kwargs):
            return None

    sink = asyncio.Queue()
    plane = DuplexControlPlane(extension=Extension(), stage_port=port, result_sink=sink)
    for sid in ("bad", "good"):
        fence = DuplexFence(sid)
        await plane.handle(
            OpenDuplexSessionMessage(
                control_id=f"open-{sid}",
                fence=fence,
                session_id=sid,
                capabilities={"input_modes": ["append_audio_chunk"]},
            )
        )
        assert (await sink.get()).ok
        await plane.handle(
            AppendDuplexInputMessage(
                control_id=f"append-{sid}",
                fence=fence,
                session_id=sid,
                mode="append_audio_chunk",
                payload={},
                final=False,
            )
        )
        result = await sink.get()
        if sid == "bad":
            assert not result.ok and result.error.code == "context_limit_exceeded"
            assert result.error.retryable is False
            pools[0].submit_initial.assert_not_awaited()
        else:
            assert result.ok
    pools[0].submit_initial.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("raised", [True, False])
async def test_websocket_explicitly_closes_only_the_limited_session(raised):
    result = {
        "operation": "append",
        "ok": False,
        "error": {"code": "context_limit_exceeded", "message": "Logical limit reached", "retryable": False},
    }
    append = AsyncMock(side_effect=DuplexControlRequestError(result)) if raised else AsyncMock(return_value=result)
    close = AsyncMock(return_value={"ok": True})
    bridge = NativeRuntimeBridgeMixin()
    bridge._chat_service = SimpleNamespace(
        engine_client=SimpleNamespace(append_duplex_input_async=append, close_duplex_session_async=close)
    )
    session = DuplexSession("limited", DuplexSessionConfig(), DuplexCapabilities(supports_input_append=True))
    other = DuplexSession("other", DuplexSessionConfig())
    emit = AsyncMock()
    assert await bridge._append_runtime_input(session, {}, final=False, send_json=emit) == (False, False)
    assert session.state == DuplexSessionState.CLOSED
    assert other.state == DuplexSessionState.OPEN
    close.assert_awaited_once()
    assert close.await_args.args == ("limited",)
    assert close.await_args.kwargs["reason"] == "context_limit_exceeded"
    events = [call.args[0] for call in emit.await_args_list]
    assert [event["type"] for event in events] == ["error", "session.closed"]
    assert events[0]["code"] == "context_limit_exceeded" and events[0]["retryable"] is False
    assert all(event["session_id"] == "limited" for event in events)
