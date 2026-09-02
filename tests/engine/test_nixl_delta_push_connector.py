# SPDX-License-Identifier: Apache-2.0
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
    NixlDeltaPushConnectorWorker,
    NixlDeltaPushConnectorScheduler,
    _delta_registration_fields,
    _delta_transfer_evidence,
)
from vllm_omni.worker.gpu_ar_worker import GPUARWorker


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
