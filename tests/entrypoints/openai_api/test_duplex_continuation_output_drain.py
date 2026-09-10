"""Silence scheduling is background work, never an output-consumer barrier."""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.session import MiniCPMO45ServingSessionState
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionState
from vllm_omni.experimental.fullduplex.openai.runtime_bridge import NativeRuntimeBridgeMixin


def setup_bridge(scheduler):
    native = MiniCPMO45ServingSessionState(silence_continuation_scheduler=scheduler)
    session = SimpleNamespace(
        state=DuplexSessionState.OPEN, active_request_id="request",
        active_response_id="response", active_response_turn_id=0,
        turn_id=0, epoch=0, incarnation=0,
    )
    bridge = SimpleNamespace(
        _runtime_session_state=lambda _: native,
        _session_auto_responds=lambda _: True,
        _native_silence_unit_payload=lambda: {},
        _native_silence_continuation_is_stale=lambda *a, **kw: (
            kw["expected_epoch"] != session.epoch or kw["response_id"] != session.active_response_id
        ),
    )

    async def consume_output():
        await NativeRuntimeBridgeMixin._maybe_continue_native_response(
            bridge, None, session=session, expected_epoch=session.epoch,
        )

    return native, session, consume_output


@pytest.mark.asyncio
async def test_output_consumer_never_waits_for_silence_and_scheduling_is_bounded():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def scheduler(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        await release.wait()
        return True

    native, _, consume = setup_bridge(scheduler)
    try:
        await asyncio.wait_for(consume(), timeout=0.1)
        await asyncio.wait_for(entered.wait(), timeout=0.1)
        first = native.continuation_schedule_task
        for _ in range(10):
            await asyncio.wait_for(consume(), timeout=0.1)
        assert native.continuation_schedule_task is first and len(calls) == 1
        release.set()
        await first
        assert native.continuation_schedule_task is None
        assert native.continuation_units == 1
    finally:
        native.clear_continuation()


@pytest.mark.asyncio
async def test_retired_timer_cannot_overwrite_new_response_owner():
    releases, calls = {}, []

    async def scheduler(*args, **kwargs):
        owner = kwargs["owner_id"]
        calls.append(owner)
        releases[owner] = asyncio.Event()
        await releases[owner].wait()
        return True

    native, session, consume = setup_bridge(scheduler)
    try:
        await consume()
        await asyncio.sleep(0)
        old = native.continuation_schedule_task
        session.active_response_id = "new-response"
        await consume()
        new = native.continuation_schedule_task
        await asyncio.sleep(0)
        assert old.cancelled() and new is not old
        assert calls == ["response:response", "response:new-response"]
        releases["response:new-response"].set()
        await new
        assert native.continuation_owner_id == "response:new-response"
        assert native.continuation_units == 1
    finally:
        native.clear_continuation()


@pytest.mark.asyncio
async def test_clear_cancels_pending_schedule_without_cancelling_submitted_append():
    append_release = asyncio.Event()
    append = asyncio.create_task(append_release.wait())

    async def scheduler(*args, **kwargs):
        return await asyncio.shield(append)

    native, _, consume = setup_bridge(scheduler)
    native.pending_silence_task = append
    await consume()
    await asyncio.sleep(0)
    scheduling = native.continuation_schedule_task
    native.clear_continuation()
    with pytest.raises(asyncio.CancelledError):
        await scheduling
    assert not append.done()
    append_release.set()
    await append
    assert native.continuation_units == 0 and native.continuation_owner_id is None


@pytest.mark.asyncio
async def test_failed_background_schedule_is_observed_and_released():
    async def scheduler(*args, **kwargs):
        raise RuntimeError("injected scheduling failure")

    native, _, consume = setup_bridge(scheduler)
    await consume()
    task = native.continuation_schedule_task
    await task
    assert task.exception() is None
    assert native.continuation_schedule_task is None and native.continuation_units == 0


@pytest.mark.asyncio
async def test_stopping_output_stream_also_retires_its_pending_timer():
    gate = asyncio.Event()

    async def scheduler(*args, **kwargs):
        await gate.wait()
        return True

    native, session, consume = setup_bridge(scheduler)
    await consume()
    timer = native.continuation_schedule_task
    bridge = SimpleNamespace(_runtime_session_state=lambda _: native)
    await NativeRuntimeBridgeMixin._cancel_native_data_plane_stream(bridge, session)
    with pytest.raises(asyncio.CancelledError):
        await timer
    assert native.continuation_schedule_task is None
