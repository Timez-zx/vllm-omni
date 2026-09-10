# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Omni AR streaming-session async placeholder handling."""

from __future__ import annotations

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
from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARScheduler,
    _compact_native_duplex_prompt_metadata,
)

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cache_only_import_keeps_last_media_kv_position():
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.connector = MagicMock()
    sched.kv_cache_manager = MagicMock()
    sched._connector_finished = MagicMock()
    req = SimpleNamespace(request_id="cache-only", num_computed_tokens=279, num_tokens=279,
                          pd_cache_sync_retain=True, kv_transfer_params=None)
    sched.complete_direct_pd_cache_sync(req)
    assert req.num_computed_tokens == 279
    sched.kv_cache_manager.cache_blocks.assert_called_once_with(req, 279)
    sched.kv_cache_manager.free.assert_not_called()


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


def test_preemption_is_visible_to_capacity_audit(mocker) -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    from benchmarks.minicpmo.analyze_rtf import PREEMPTION_OR_EVICTION

    sched = _make_scheduler(stage_id=2)
    sched.kv_cache_manager = MagicMock()
    sched.kv_cache_manager.block_pool.get_num_free_blocks.return_value = 0
    session = _make_request()
    session.num_computed_tokens = 2
    base = mocker.patch.object(Scheduler, "_preempt_request")
    warning = mocker.patch("vllm_omni.core.sched.omni_ar_scheduler.logger.warning")

    sched._preempt_request(session, 123.0)

    base.assert_called_once_with(session, 123.0)
    text = warning.call_args.args[0] % warning.call_args.args[1:]
    assert PREEMPTION_OR_EVICTION.search(text)
    assert "stage=2" in text and "free_blocks=0" in text


def test_native_duplex_output_compacts_prompt_snapshot() -> None:
    latent = object()
    original = {
        "latent": latent,
        "duplex_prompt_token_ids": [[10, 20, 30]],
        "meta": {"tts_bos_token_id": 99},
    }

    compact = _compact_native_duplex_prompt_metadata(
        original,
        current_segment_token_ids=[40, 41],
    )

    assert compact == {
        "latent": latent,
        "duplex_prompt_len": 3,
        "duplex_last_prompt_token_id": 30,
        "duplex_segment_token_ids": [40, 41],
        "meta": {"tts_bos_token_id": 99},
    }
    assert original["duplex_prompt_token_ids"] == [[10, 20, 30]]


def test_compact_tensor_prompt_metadata_survives_engine_wire_roundtrip():
    import torch
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs

    compact = _compact_native_duplex_prompt_metadata(
        {"duplex_prompt_token_ids": torch.tensor([[10, 20, 30]]), "latent": torch.zeros(2, 4)},
        current_segment_token_ids=[40, 41],
    )
    output = OmniEngineCoreOutputs(outputs=[OmniEngineCoreOutput(
        request_id="test", new_token_ids=[40], multimodal_output=compact,
    )])
    decoded = MsgpackDecoder(OmniEngineCoreOutputs).decode(MsgpackEncoder().encode(output))
    metadata = decoded.outputs[0].multimodal_output
    assert metadata["duplex_prompt_len"].item() == 3
    assert metadata["duplex_last_prompt_token_id"].item() == 30
    assert metadata["duplex_segment_token_ids"].tolist() == [40, 41]


def test_native_segment_snapshots_replace_instead_of_concatenating():
    import torch

    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.output_modality import TensorAccumulationStrategy

    first = MultimodalPayload.from_dict({
        "latent": torch.ones(1, 4),
        "duplex_prompt_len": torch.tensor(100),
        "duplex_segment_token_ids": torch.tensor([40]),
    })
    second = MultimodalPayload.from_dict({
        "latent": torch.ones(1, 4),
        "duplex_prompt_len": torch.tensor(100),
        "duplex_segment_token_ids": torch.tensor([40, 41]),
    })
    merged = first.merged_with(second)
    merged.consolidate_tensors(TensorAccumulationStrategy.CONCAT_DIM0)
    merged.consolidate_metadata()
    assert merged["duplex_prompt_len"].item() == 100
    assert merged["duplex_segment_token_ids"].tolist() == [40, 41]
    assert merged["latent"].shape == (2, 4)


def test_native_decode_end_refreshes_segment_without_prompt_snapshot():
    import torch

    payload = {"latent": torch.ones(1, 4)}
    compact = _compact_native_duplex_prompt_metadata(
        payload, current_segment_token_ids=[151706, 42, 151718], native_segment=True,
    )
    assert compact["duplex_segment_token_ids"].tolist() == [151706, 42, 151718]
    assert "duplex_prompt_len" not in compact
    assert "duplex_segment_token_ids" not in payload


def test_preempted_minicpmo_duplex_request_rebases_to_exact_compact_prompt() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.prompt_token_ids = [0] * 200
    session.num_prompt_tokens = 200
    session._all_token_ids.clear()
    session._all_token_ids.extend(session.prompt_token_ids)
    session.append_output_token_ids([91, 92])
    session.num_computed_tokens = 0
    session.num_preemptions = 1
    session.status = RequestStatus.PREEMPTED
    session.model_intermediate_buffer = {
        "duplex": {
            "data_plane": True,
            "session_id": "sid-rebase",
            "incarnation": 2,
            "epoch": 3,
            "seq": 4,
            "scheduler_token_id": 7,
            "scheduler_token_budget": 13,
            "compact_rebase_prefix_tokens": 59,
        }
    }

    assert sched._rebase_preempted_minicpmo_duplex_request(session) is True

    assert session.prompt_token_ids == [7] * 72
    assert session.num_prompt_tokens == 72
    assert session.num_computed_tokens == 0
    assert list(session.output_token_ids) == [91, 92]
    assert list(session.all_token_ids) == [*([7] * 72), 91, 92]
    assert session.cache_salt == "minicpmo45:sid-rebase:2:preempt-rebase-3-4-1"
    assert session._minicpmo_duplex_preemption_rebased is True


def test_preempted_minicpmo_duplex_rebase_rejects_async_inflight_tokens() -> None:
    sched = _make_scheduler(stage_id=0)
    session = _make_request()
    session.num_in_flight_tokens = 1
    session.model_intermediate_buffer = {
        "duplex": {
            "data_plane": True,
            "scheduler_token_budget": 13,
            "compact_rebase_prefix_tokens": 59,
        }
    }

    with pytest.raises(RuntimeError, match="tokens still in flight"):
        sched._rebase_preempted_minicpmo_duplex_request(session)


def test_preempted_minicpmo_duplex_rebase_is_stage0_only() -> None:
    sched = _make_scheduler(stage_id=1)
    session = _make_request()
    session.model_intermediate_buffer = {
        "duplex": {
            "data_plane": True,
            "scheduler_token_budget": 13,
            "compact_rebase_prefix_tokens": 59,
        }
    }

    assert sched._rebase_preempted_minicpmo_duplex_request(session) is False
    assert session.prompt_token_ids == [1, 2, 3]


def _run_resumable_segment_stop(
    session: Request,
    *,
    session_finished: bool = False,
    pd_segment: bool = False,
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
    sched.connector = MagicMock() if pd_segment else None
    if pd_segment:
        session.kv_transfer_params = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
        sched._connector_finished.return_value = (
            True,
            {
                "remote_request_id": session.request_id,
                "remote_num_tokens": session.num_computed_tokens,
            },
        )
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


def test_resumable_pd_segment_publishes_cumulative_prompt_identity() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.num_computed_tokens = session.num_prompt_tokens

    outputs = _run_resumable_segment_stop(session, pd_segment=True)

    output = outputs[session.client_index].outputs[0]
    assert output.is_segment_finished is True
    assert output.kv_transfer_params["remote_request_id"] == session.request_id
    assert output.kv_transfer_params["remote_prompt_token_ids"] == [1, 2, 3]
    assert output.kv_transfer_params["remote_prompt_token_offset"] == 0
    assert session._omni_pd_published_prompt_tokens == 3


def test_resumable_pd_prompt_echo_is_incremental_and_owned() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = True
    session.num_computed_tokens = session.num_prompt_tokens
    session._omni_pd_published_prompt_tokens = 2
    outputs = _run_resumable_segment_stop(session, pd_segment=True)
    params = outputs[session.client_index].outputs[0].kv_transfer_params
    assert params["remote_prompt_token_offset"] == 2
    assert params["remote_prompt_token_ids"] == [3]
    session.prompt_token_ids[-1] = 99
    assert params["remote_prompt_token_ids"] == [3]
    assert session._omni_pd_published_prompt_tokens == 3


def test_resumable_pd_rebase_discards_published_prompt_offset() -> None:
    sched = _make_scheduler()
    sched._replace_streaming_session = MagicMock()
    session = _make_request()
    session._omni_pd_published_prompt_tokens = 3
    update = _make_update([4, 5, 6])  # Equal length is still a different prefix.
    update.model_intermediate_buffer = {"meta": {"replace_streaming_prompt": True}}
    sched._update_request_as_session(session, update)
    assert session._omni_pd_published_prompt_tokens == 0
    sched._replace_streaming_session.assert_called_once_with(session, update)


def test_finite_pd_decode_segment_exposes_allocated_transfer_evidence() -> None:
    session = _make_request()
    session.status = RequestStatus.RUNNING
    session.resumable = False
    session.num_computed_tokens = session.num_prompt_tokens
    session.pd_transfer_evidence = {
        "kv_transfer_selected_blocks": 3,
        "kv_transfer_selected_tokens": 41,
        "kv_transfer_selected_bytes": 98_304,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }

    outputs = _run_resumable_segment_stop(session)

    output = outputs[session.client_index].outputs[0]
    assert output.is_segment_finished is True
    assert output.kv_transfer_params == {
        "kv_transfer_selected_blocks": 3,
        "kv_transfer_selected_tokens": 41,
        "kv_transfer_selected_bytes": 98_304,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }


def test_direct_pd_cache_sync_snapshots_connector_evidence() -> None:
    request = SimpleNamespace(
        kv_transfer_params={
            "do_remote_prefill": False,
            "remote_host": "control-only",
            "kv_transfer_selected_blocks": 2,
            "kv_transfer_selected_tokens": 17,
            "kv_transfer_selected_bytes": 24_576,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        }
    )

    OmniARScheduler._snapshot_pd_transfer_evidence(request)

    assert request.pd_transfer_evidence == {
        "kv_transfer_selected_blocks": 2,
        "kv_transfer_selected_tokens": 17,
        "kv_transfer_selected_bytes": 24_576,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }


def test_pd_send_completion_retains_live_resumable_request_blocks() -> None:
    scheduler = MagicMock()
    scheduler.connector = MagicMock()
    session = _make_request()
    session.resumable = True
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.requests = {session.request_id: session}
    connector_output = SimpleNamespace(
        finished_recving=None,
        finished_sending={session.request_id},
    )

    OmniARScheduler._update_from_kv_xfer_finished(scheduler, connector_output)

    scheduler.connector.update_connector_output.assert_called_once_with(
        connector_output
    )
    scheduler._free_blocks.assert_not_called()


def test_pd_completion_after_session_abort_is_ignored() -> None:
    scheduler = MagicMock()
    scheduler.connector = MagicMock()
    scheduler.requests = {}
    connector_output = SimpleNamespace(
        finished_recving={"aborted-d-request"},
        finished_sending={"aborted-p-request"},
    )

    OmniARScheduler._update_from_kv_xfer_finished(scheduler, connector_output)

    scheduler.connector.update_connector_output.assert_called_once_with(
        connector_output
    )
    scheduler._free_blocks.assert_not_called()


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


def test_context_rollover_retains_latest_segment_output_and_releases_old_kv() -> None:
    sched = _make_scheduler(stage_id=0)
    sched.kv_cache_manager = MagicMock()
    session = _make_request()
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session.append_output_token_ids([7, 8, 9])
    session.num_computed_tokens = 6
    update = _make_update([10, 20, 30, 40])
    update.model_intermediate_buffer = {
        "meta": {
            "replace_streaming_prompt": True,
            "retain_streaming_output_tokens": True,
            "retained_output_insert_offset": 2,
            "streaming_cache_salt": "context-1",
        }
    }

    sched._update_request_as_session(session, update)

    sched.kv_cache_manager.free.assert_called_once_with(session)
    assert session.prompt_token_ids == [10, 20, 7, 8, 9, 30, 40]
    assert session.num_computed_tokens == 0
    assert session.cache_salt == "context-1"
