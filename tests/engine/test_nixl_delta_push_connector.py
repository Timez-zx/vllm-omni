# SPDX-License-Identifier: Apache-2.0
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.v1.metrics.stats import PrefillStats

from vllm_omni.engine.nixl_delta_push_connector import (
    NixlDeltaPushConnectorScheduler,
    NixlDeltaPushConnectorWorker,
    _delta_registration_fields,
    _delta_transfer_evidence,
    _select_delta_source_blocks,
)
from vllm_omni.worker.gpu_ar_worker import GPUARWorker


def _progress_worker():
    worker = object.__new__(NixlDeltaPushConnectorWorker)
    worker.shutdown = lambda: None  # No actual registered native resources.
    worker._recving_metadata = {}
    worker._sending_transfers = {}
    worker._sending_transfers_lock = threading.Lock()
    worker._push_writer_stop = threading.Event()
    worker._push_writer_wake = threading.Event()
    worker._reg_send_inbox = queue.Queue()
    worker._finished_blocks_inbox = queue.Queue()
    worker._evict_finished_inbox = queue.Queue()
    worker._push_finished_blocks = {}
    worker._pending_d_registrations = {}
    worker._pending_completion_notifs = queue.Queue()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_new_notifs.return_value = {}
    return worker


@pytest.mark.parametrize("wait_for_completion", [False, True])
def test_direct_poll_only_waits_when_core_has_no_model_work(wait_for_completion):
    worker = _progress_worker()
    worker._ensure_direct_progress_state()
    worker._direct_cache_sync_req_ids = {"import"}
    worker._direct_cache_sync_finished = set()
    worker._partition_finished = MagicMock()
    with patch.object(worker._completion_notif_available, "wait", return_value=False) as wait:
        assert worker.poll_direct_cache_sync(wait_for_completion) == set()
        if wait_for_completion:
            wait.assert_called_once_with(.001)
        else:
            wait.assert_not_called()
    # A nonblocking miss leaves ownership intact; the later completion is
    # consumed once, not dropped when the Core resumes a decode step.
    assert worker._direct_cache_sync_req_ids == {"import"}
    worker._direct_cache_sync_finished.add("import")
    with patch.object(worker._completion_notif_available, "wait") as wait:
        assert worker.poll_direct_cache_sync(False) == {"import"}
        assert worker.poll_direct_cache_sync(False) == set()
        wait.assert_not_called()


def test_idle_direct_poll_retries_notification_in_same_core_step():
    worker = _progress_worker()
    worker._ensure_direct_progress_state()
    worker._direct_cache_sync_req_ids = {"import"}
    worker._direct_cache_sync_finished = set()
    calls = []
    def partition():
        calls.append(1)
        if len(calls) == 2:
            worker._direct_cache_sync_finished.add("import")
    worker._partition_finished = partition
    with patch.object(worker._completion_notif_available, "wait", return_value=True):
        assert worker.poll_direct_cache_sync() == {"import"}
    assert len(calls) == 2


@pytest.mark.parametrize('pending', [None, 'send', 'recv', 'unmatched'])
def test_writer_only_self_polls_with_pending_work(pending):
    worker = _progress_worker()
    if pending == 'send':
        worker._sending_transfers['r'] = [object()]
    elif pending == 'recv':
        worker._recving_metadata['r'] = object()
    elif pending == 'unmatched':
        worker._push_finished_blocks['r'] = [1]
    with patch.object(worker._push_writer_wake, 'wait',
                      side_effect=lambda timeout: worker._push_writer_stop.set()) as wait:
        worker._push_writer_loop()
    wait.assert_called_once_with(.001 if pending else None)


@pytest.mark.parametrize('pending', ['send', 'recv'])
def test_writer_completes_handoff_without_an_engine_step(pending):
    worker = _progress_worker()
    handle = object()
    if pending == 'send':
        worker._sending_transfers['r'] = [handle]
    else:
        worker._recving_metadata['r'] = object()
    calls = []
    def check(h):
        assert h is handle
        assert worker._sending_transfers_lock.locked()
        calls.append(1)
        return 'DONE' if len(calls) >= 3 else 'PROC'
    def poll():
        if pending == 'recv':
            calls.append(1)
        return {'peer': [b'r:1']} if len(calls) >= 3 else {}
    worker.nixl_wrapper.check_xfer_state.side_effect = check
    worker.nixl_wrapper.get_new_notifs.side_effect = poll
    thread = threading.Thread(target=worker._push_writer_loop)
    thread.start()
    try:
        assert worker._pending_completion_notifs.get(timeout=2) == b'r:1'
        assert len(calls) >= 3
    finally:
        worker._push_writer_stop.set()
        worker._push_writer_wake.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    worker.nixl_wrapper.get_xfer_telemetry.assert_not_called()
    if pending == 'send':
        # Only the Core may consume completion and release the original handle.
        assert worker._sending_transfers == {'r': [handle]}
        worker.xfer_stats = MagicMock()
        with worker._sending_transfers_lock:
            assert worker._pop_done_transfers(worker._sending_transfers) == {'r'}
            assert worker._pop_done_transfers(worker._sending_transfers) == set()
        worker.nixl_wrapper.release_xfer_handle.assert_called_once_with(handle)


def test_progress_leaves_failed_handle_for_native_core_cleanup():
    worker = _progress_worker()
    handle = object()
    worker._sending_transfers['r'] = [handle]
    worker.nixl_wrapper.check_xfer_state.return_value = 'ERR'
    worker._progress_pending_writes()
    assert worker._sending_transfers == {'r': [handle]}
    worker.nixl_wrapper.release_xfer_handle.assert_not_called()
    worker._handle_failed_transfer = MagicMock()
    worker._log_failure = MagicMock()
    assert worker._pop_done_transfers(worker._sending_transfers) == {'r'}
    worker._handle_failed_transfer.assert_called_once_with('r', handle)


def test_window_delta_uses_absolute_positions_with_evicted_prefix():
    assert _select_delta_source_blocks(
        ([0, 0, 31, 32, 33, 34],), ([71, 72],),
        source_block_offset=0, source_block_size=16, decode_block_size=16,
        source_block_indices=[[4, 5]],
    ) == ([33, 34],)
    # A cold D window miss copies the resident window, not the null prefix.
    assert _select_delta_source_blocks(
        ([0, 0, 31, 32, 33, 34],), ([71, 72, 73, 74],),
        source_block_offset=0, source_block_size=16, decode_block_size=16,
        source_block_indices=[[2, 3, 4, 5]],
    ) == ([31, 32, 33, 34],)


@pytest.mark.parametrize("indices", [[[1, 2]], [[3, 2]], [[2, 2]], [[2, 6]]])
def test_window_delta_fails_closed_on_invalid_or_freed_positions(indices):
    with pytest.raises(ValueError):
        _select_delta_source_blocks(
            ([0, 0, 31, 32, 33, 34],), ([71, 72],),
            source_block_offset=0, source_block_size=16, decode_block_size=16,
            source_block_indices=indices,
        )


def test_explicit_remote_boundary_full_local_hit_does_not_include_decode_token() -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    request = SimpleNamespace(
        request_id="req-00000001",
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_prompt_tokens": 912,
        },
    )

    assert scheduler.get_num_new_matched_tokens(request, 912) == (0, False)


def test_delta_registration_uses_aligned_local_prefix() -> None:
    assert _delta_registration_fields(
        remote_prompt_tokens=1123,
        external_tokens=211,
        block_size=16,
    ) == {
        "matched_prefix_tokens": 912,
        "source_block_offset": 57,
        "decode_block_size": 16,
        "remote_prompt_tokens": 1123,
    }


def test_delta_registration_rejects_unaligned_prefix() -> None:
    with pytest.raises(ValueError, match="block-aligned"):
        _delta_registration_fields(
            remote_prompt_tokens=912,
            external_tokens=1,
            block_size=16,
        )


def test_delta_transfer_evidence_distinguishes_tokens_blocks_and_bytes() -> None:
    assert _delta_transfer_evidence(
        ([1, 2], [3]),
        external_tokens=17,
        bytes_per_block_by_group=(100, 200),
    ) == {
        "kv_transfer_selected_blocks": 3,
        "kv_transfer_selected_tokens": 17,
        "kv_transfer_selected_bytes": 400,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }
    assert _delta_transfer_evidence(
        (),
        external_tokens=0,
        bytes_per_block_by_group=(100,),
    ) == {
        "kv_transfer_selected_blocks": 0,
        "kv_transfer_selected_tokens": 0,
        "kv_transfer_selected_bytes": 0,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }


def test_delta_transfer_evidence_is_request_scoped_and_returned_on_d_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer.0", "layer.1", "layer.2"],
                kv_cache_spec=SimpleNamespace(page_size_bytes=4096),
            )
        ]
    )
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=2)
    )
    scheduler._newly_finished_push_blocks = {}
    request = SimpleNamespace(
        request_id="req-evidence",
        kv_transfer_params={"do_remote_prefill": False},
    )
    registration = {"local_block_ids": ([10, 11, 12],)}

    scheduler._attach_transfer_evidence(
        request,
        registration,
        external_tokens=41,
    )

    expected = {
        "kv_transfer_selected_blocks": 3,
        "kv_transfer_selected_tokens": 41,
        "kv_transfer_selected_bytes": 3 * 4096 * 3 * 2,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }
    assert {name: registration[name] for name in expected} == expected
    assert {name: request.kv_transfer_params[name] for name in expected} == expected

    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "request_finished",
        lambda self, request, block_ids: (False, None),
    )
    delay_free, output_params = scheduler.request_finished(request, ())

    assert delay_free is False
    assert output_params == expected


def test_delta_registration_records_direct_cache_sync_prefill_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    scheduler.block_size = 16
    scheduler._has_mamba = False
    scheduler._push_pending_registrations = {}
    scheduler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer.0", "layer.1"],
                kv_cache_spec=SimpleNamespace(page_size_bytes=4096),
            )
        ]
    )
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1)
    )
    request = SimpleNamespace(
        request_id="req-00000002",
        prompt_token_ids=[0] * 1123,
        num_prompt_tokens=1123,
        num_computed_tokens=0,
        prefill_stats=PrefillStats(),
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_prompt_tokens": 1123,
        },
    )

    def fake_update_state_after_alloc(self, request, blocks, external_tokens):
        del blocks
        assert external_tokens == 211
        self._push_pending_registrations[request.request_id] = {
            "local_block_ids": ([*range(14)],),
        }
        request.kv_transfer_params["do_remote_prefill"] = False

    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "update_state_after_alloc",
        fake_update_state_after_alloc,
    )

    scheduler.update_state_after_alloc(request, blocks=object(), num_external_tokens=211)

    assert request.prefill_stats.num_prompt_tokens == 1123
    assert request.prefill_stats.num_local_cached_tokens == 912
    assert request.prefill_stats.num_external_cached_tokens == 211
    assert request.prefill_stats.num_cached_tokens == 1123
    assert request.prefill_stats.num_computed_tokens == 0
    assert request.kv_transfer_params["kv_transfer_selected_blocks"] == 14
    assert request.kv_transfer_params["kv_transfer_selected_tokens"] == 211
    assert request.kv_transfer_params["kv_transfer_selected_bytes"] == (
        14 * 4096 * 2
    )


def test_finished_blocks_bypass_next_batch_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    blocks = ([3, 4],)
    scheduler._newly_finished_push_blocks = {"req-a": blocks}
    scheduler._finished_request_blocks = {"req-a": blocks}
    scheduler._reqs_need_send = {"req-a": 123.5}
    scheduler._push_registration_deadlines = {}
    scheduler._push_pending_registrations = {}

    metadata = scheduler.take_immediate_push_metadata()

    assert metadata is not None
    assert metadata.push_finished_blocks == {"req-a": blocks}
    assert metadata.reqs_to_send == {"req-a": 123.5}
    assert metadata.reqs_in_batch == {"req-a"}
    assert scheduler._newly_finished_push_blocks == {}
    assert scheduler._reqs_need_send == {}
    # The scheduler lease remains until worker finished_sending is observed.
    assert scheduler._finished_request_blocks == {"req-a": blocks}
    assert scheduler.take_immediate_push_metadata() is None

    monkeypatch.setattr(
        NixlBaseConnectorScheduler,
        "build_connector_meta",
        lambda self, scheduler_output: NixlConnectorMetadata(),
    )
    next_batch = NixlPushConnectorScheduler.build_connector_meta(
        scheduler,
        SimpleNamespace(),
    )
    assert next_batch.push_finished_blocks == {}


def test_immediate_push_rejects_finished_blocks_without_a_lease() -> None:
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    blocks = ([3, 4],)
    scheduler._newly_finished_push_blocks = {"req-a": blocks}
    scheduler._finished_request_blocks = {"req-a": blocks}
    scheduler._reqs_need_send = {}

    with pytest.raises(RuntimeError, match="missing worker lease"):
        scheduler.take_immediate_push_metadata()

    # Validation occurs before either piece of scheduler state is consumed.
    assert scheduler._newly_finished_push_blocks == {"req-a": blocks}
    assert scheduler._finished_request_blocks == {"req-a": blocks}


def test_worker_routes_immediate_push_without_model_execution() -> None:
    worker = MagicMock(spec=NixlDeltaPushConnectorWorker)
    connector = SimpleNamespace(connector_worker=worker)
    ar_worker = GPUARWorker.__new__(GPUARWorker)
    metadata = NixlConnectorMetadata()

    with patch(
        "vllm_omni.worker.gpu_ar_worker.get_kv_transfer_group",
        return_value=connector,
    ):
        assert ar_worker.publish_pd_finished_blocks(metadata) is True

    worker.start_load_kv.assert_called_once_with(metadata)
