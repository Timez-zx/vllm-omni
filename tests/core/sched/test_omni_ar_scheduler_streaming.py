# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

from time import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Imports must run in this order: vllm_omni applies patches to vllm.v1.request before
# Request / StreamingUpdate are bound in this module. Ruff isort would reorder them.
# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(*, stage_id: int = 0) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=stage_id))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.chunk_transfer_adapter = None
    sched.skipped_waiting = set()
    return sched


def _make_request() -> Request:
    return Request(
        request_id="req-ar-streaming-test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )


def _make_update(prompt_token_ids: list[int] | None = None) -> StreamingUpdate:
    return StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[10, 20] if prompt_token_ids is None else prompt_token_ids,
        max_tokens=32,
        arrival_time=200.0,
        sampling_params=SamplingParams(max_tokens=16),
    )


def test_wedge_probe_ignores_request_waiting_for_upstream_chunk() -> None:
    sched = _make_scheduler(stage_id=1)
    request = SimpleNamespace(
        output_token_ids=[],
        status=RequestStatus.WAITING_FOR_CHUNK,
        num_computed_tokens=10,
    )
    sched.requests = {"req-waiting-chunk": request}
    sched._had_requests = 1
    sched._empty_since = None
    sched._req_seen_t = {"req-waiting-chunk": time() - 60}
    sched._mute_reported = set()
    sched._progress_fingerprint = None
    sched._last_progress_t = time()
    sched._wedge_reported = False
    sched._log_request_table = MagicMock()

    sched._check_for_wedged_requests(SimpleNamespace(total_num_scheduled_tokens=0))

    sched._log_request_table.assert_not_called()
    assert "req-waiting-chunk" not in sched._req_seen_t


def _run_resumable_segment_stop(
    session: Request,
    *,
    session_finished: bool = False,
):
    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False

    def stop_request(request: Request, _token_ids: list[int]):
        request.status = RequestStatus.FINISHED_STOPPED
        return [42], True

    sched._update_request_with_output.side_effect = stop_request
    sched._handle_stopped_request.return_value = session_finished
    # vLLM 0.26 returns (kv_xfer_params, ec_xfer_params); an unconfigured
    # MagicMock iterates empty and fails to unpack at the call site.
    sched._free_request.return_value = (None, None)
    sched.chunk_transfer_adapter = None
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None

    return OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)


@pytest.mark.parametrize("outstanding_async_tokens", [0, 1, 2])
def test_resumable_segment_stop_reconciles_async_placeholders(
    outstanding_async_tokens: int,
) -> None:
    """A segment stop discards and rolls back only in-flight async tokens."""
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.append_output_token_ids([7, 8])
    session.num_computed_tokens = session.num_tokens + outstanding_async_tokens
    session.num_output_placeholders = outstanding_async_tokens
    session.spec_token_ids = [-1] * outstanding_async_tokens

    _run_resumable_segment_stop(session)

    assert session.async_tokens_to_discard == outstanding_async_tokens
    assert session.num_computed_tokens == session.num_tokens
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    assert session._output_token_ids == []


def test_resumable_session_terminal_is_not_marked_as_segment_boundary() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True

    outputs = _run_resumable_segment_stop(session, session_finished=True)

    output = outputs[session.client_index].outputs[0]
    assert output.finish_reason is not None
    assert output.is_segment_finished is False


def test_update_from_output_settles_in_flight_tokens() -> None:
    """vLLM 0.26: schedule() increments num_in_flight_tokens per scheduled
    token; update_from_output must decrement it symmetrically. If the
    decrement is dropped the counter grows monotonically and both readers
    (allocate_slots, _connector_finished) clamp
    max(0, num_computed_tokens - num_in_flight_tokens) to zero forever,
    silently freezing sliding-window block freeing.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.num_in_flight_tokens = 1  # as left by schedule() for this step

    _run_resumable_segment_stop(session)

    assert session.num_in_flight_tokens == 0


def test_running_decode_step_without_inter_stage_payload_does_not_raise() -> None:
    """A decode step that neither stops nor carries an inter-stage payload.

    ``finished`` is only assigned when the request stops, yet the async-chunk
    save condition reads it for every request, so this step used to raise
    ``UnboundLocalError: cannot access local variable 'finished'``.
    """
    session = _make_request()
    session.status = RequestStatus.RUNNING

    sched = MagicMock()
    sched.requests = {session.request_id: session}
    sched.perf_metrics = None
    sched.structured_output_manager.should_advance.return_value = False
    sched._update_request_with_output.return_value = ([42], False)
    sched._process_kv_transfer_trigger.return_value = False
    sched.chunk_transfer_adapter = MagicMock()
    sched.running = [session]
    sched.waiting_for_transfer_free = set()
    sched.transfer_triggered_requests = set()
    sched.active_kv_transfers = set()
    sched.pending_stop_after_extraction = set()
    sched.connector = None
    sched.kv_cache_manager.take_events.return_value = None
    sched.finished_req_ids_dict = {}
    sched.make_stats.return_value = None

    scheduler_output = MagicMock(spec=SchedulerOutput)
    scheduler_output.num_scheduled_tokens = {session.request_id: 1}
    scheduler_output.scheduled_spec_decode_tokens = {}
    scheduler_output.num_invalid_spec_tokens = 0

    model_runner_output = MagicMock(spec=ModelRunnerOutput)
    model_runner_output.sampled_token_ids = [[42]]
    model_runner_output.logprobs = None
    model_runner_output.prompt_logprobs_dict = {}
    model_runner_output.pooler_output = None
    model_runner_output.num_nans_in_logits = None
    model_runner_output.kv_connector_output = None
    model_runner_output.cudagraph_stats = None
    model_runner_output.req_id_to_index = {session.request_id: 0}
    model_runner_output.routed_experts = None
    model_runner_output.inter_stage_outputs = None

    OmniARScheduler.update_from_output(sched, scheduler_output, model_runner_output)

    # Nothing to hand downstream: no payload, no segment boundary, not finished.
    sched.chunk_transfer_adapter.save_async.assert_not_called()


def test_stage0_streaming_update_discards_outstanding_async_placeholder_token() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 1
    session.spec_token_ids = [-1]

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert session.async_tokens_to_discard == 1
    assert session.num_output_placeholders == 0
    assert session.spec_token_ids == []
    # The async placeholder makes token 9 unconfirmed, so only 7 and 8 are
    # carried into the next streaming prompt before the new chunk tokens.
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 7
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_stage0_streaming_update_keeps_all_computed_tokens_without_placeholder() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    session.num_output_placeholders = 0

    sched._update_request_as_session(session, _make_update([10, 20]))

    assert getattr(session, "async_tokens_to_discard", 0) == 0
    assert session.num_output_placeholders == 0
    assert session.prompt_token_ids == [1, 2, 3, 7, 8, 9, 10, 20]
    assert list(session._all_token_ids) == [1, 2, 3, 7, 8, 9, 10, 20]
    assert session._output_token_ids == []
    assert session.num_prompt_tokens == 8
    assert sched._new_prompt_len_snapshot[session.request_id] == 2


def test_explicit_streaming_payload_replaces_placeholder_prompt() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=False,
        segment_finished_requests=set(),
    )
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    update = _make_update([10, 20])
    update.additional_information = {
        "tts_token_ids": [10, 20],
        "meta": {"replace_streaming_prompt": True},
    }
    update.model_intermediate_buffer = {
        "ids": {"tts": [41, 42, 99]},
        "meta": {"turn_eos_token_id": 99},
    }

    sched._update_request_as_session(session, update)

    assert session.prompt_token_ids == [10, 20]
    assert session.additional_information == update.additional_information
    assert session.model_intermediate_buffer == {
        "ids": {"tts": [41, 42, 99]},
        "meta": {"turn_eos_token_id": 99},
    }
    assert session.status == RequestStatus.WAITING


def test_model_intermediate_streaming_payload_replaces_computed_prompt() -> None:
    sched = _make_scheduler(stage_id=1)
    sched.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=False,
        segment_finished_requests=set(),
    )
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.prompt_token_ids = [0] * 59
    session._all_token_ids.clear()
    session._all_token_ids.extend(session.prompt_token_ids)
    session.num_prompt_tokens = 59
    session.num_computed_tokens = 59
    update = _make_update([0] * 10)
    update.additional_information = None
    update.model_intermediate_buffer = {
        "ids": {"tts": list(range(8))},
        "hidden_states": {"tts": [[0.0]] * 8},
        "meta": {
            "replace_streaming_prompt": True,
            "next_stage_prompt_len": 10,
        },
    }

    sched._update_request_as_session(session, update)

    assert session.prompt_token_ids == [0] * 10
    assert list(session._all_token_ids) == [0] * 10
    assert session.num_prompt_tokens == 10
    assert session.num_computed_tokens == 0
    assert session.additional_information is None
    assert session.model_intermediate_buffer == update.model_intermediate_buffer
    assert session.status == RequestStatus.WAITING


def _make_clamp_scheduler(*, tracked_statuses: list[RequestStatus], counter: int, max_seqs: int = 4):
    """Just enough scheduler for `_clamp_streaming_parked_counter`.

    Tracked requests are given only a `status`, which is the only field the clamp reads --
    deliberately, since the point of counting over `self.requests` rather than over the
    queues is that it does not care where a request is being held.
    """
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=1))
    sched.num_waiting_for_streaming_input = counter
    sched.max_num_running_reqs = max_seqs
    sched.requests = {
        f"req-{i}": SimpleNamespace(status=status) for i, status in enumerate(tracked_statuses)
    }
    sched._counter_clamped_t = 0.0
    return sched


def test_clamp_repairs_the_leak_that_closes_admission():
    """The measured wedge: counter at max_num_seqs, nothing actually parked.

    Upstream's waiting loop breaks when `len(running) + counter >= max_num_running_reqs`, so
    a counter stuck at max_num_seqs admits nothing, forever, while `schedule()` keeps running
    and the stage keeps looking alive.
    """
    sched = _make_clamp_scheduler(tracked_statuses=[RequestStatus.WAITING], counter=4, max_seqs=4)
    sched._clamp_streaming_parked_counter()
    assert sched.num_waiting_for_streaming_input == 0


def test_clamp_counts_requests_the_queues_cannot_see():
    """Genuinely parked requests must keep their slots.

    This is the case that made an earlier queue-derived repair worse: the chunk transfer
    adapter holds a parked request out of both queues, so a queue-derived count reads 0 and
    the repair zeroed a counter upstream then decremented to -1. Counting over
    `self.requests` sees the request wherever it is held.
    """
    sched = _make_clamp_scheduler(
        tracked_statuses=[RequestStatus.WAITING_FOR_STREAMING_REQ] * 2,
        counter=2,
    )
    sched._clamp_streaming_parked_counter()
    assert sched.num_waiting_for_streaming_input == 2


def test_clamp_removes_only_the_excess():
    sched = _make_clamp_scheduler(
        tracked_statuses=[RequestStatus.WAITING_FOR_STREAMING_REQ, RequestStatus.RUNNING],
        counter=3,
    )
    sched._clamp_streaming_parked_counter()
    assert sched.num_waiting_for_streaming_input == 1


def test_clamp_never_invents_slots():
    """Too LOW means a park this scheduler never saw. Covering for it would hide it."""
    sched = _make_clamp_scheduler(
        tracked_statuses=[RequestStatus.WAITING_FOR_STREAMING_REQ] * 3,
        counter=1,
    )
    sched._clamp_streaming_parked_counter()
    assert sched.num_waiting_for_streaming_input == 1
