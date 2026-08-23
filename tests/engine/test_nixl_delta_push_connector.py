from threading import Lock
from types import SimpleNamespace

import pytest
import yaml
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

from vllm_omni.engine.nixl_delta_push_connector import (
    NixlDeltaPushConnectorScheduler,
    NixlDeltaPushConnectorWorker,
    _delta_registration_fields,
    _select_delta_source_blocks,
)


def test_delta_registration_fields_encode_block_offset():
    assert _delta_registration_fields(
        remote_prompt_tokens=160,
        external_tokens=32,
        block_size=16,
    ) == {
        "matched_prefix_tokens": 128,
        "source_block_offset": 8,
        "decode_block_size": 16,
        "remote_prompt_tokens": 160,
    }


def test_delta_registration_rejects_unaligned_prefix():
    with pytest.raises(ValueError, match="block-aligned"):
        _delta_registration_fields(
            remote_prompt_tokens=159,
            external_tokens=32,
            block_size=16,
        )


@pytest.mark.parametrize(
    ("offset", "destinations", "expected"),
    [
        (0, ([100, 101, 102],), ([0, 1, 2],)),
        (5, ([100, 101, 102],), ([5, 6, 7],)),
        (8, ([],), ([],)),
    ],
)
def test_select_delta_source_blocks(offset, destinations, expected):
    source = (list(range(10)),)
    selected = _select_delta_source_blocks(
        source,
        destinations,
        source_block_offset=offset,
        source_block_size=16,
        decode_block_size=16,
    )
    assert selected == expected


def test_select_delta_source_blocks_rejects_wrong_layout():
    with pytest.raises(ValueError, match="identical P/D block sizes"):
        _select_delta_source_blocks(
            ([0, 1],),
            ([10],),
            source_block_offset=1,
            source_block_size=16,
            decode_block_size=8,
        )
    with pytest.raises(ValueError, match="outside P's completed prompt"):
        _select_delta_source_blocks(
            ([0, 1],),
            ([10, 11],),
            source_block_offset=1,
            source_block_size=16,
            decode_block_size=16,
        )


def test_scheduler_attaches_delta_offset_to_registration(monkeypatch):
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    scheduler.block_size = 16
    scheduler._has_mamba = False
    scheduler._push_pending_registrations = {}

    def fake_parent_update(self, request, blocks, num_external_tokens):
        self._push_pending_registrations[request.request_id] = {"local_block_ids": ([90, 91],)}

    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "update_state_after_alloc",
        fake_parent_update,
    )
    request = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(160)),
        kv_transfer_params={"do_remote_prefill": True},
    )
    scheduler.update_state_after_alloc(request, blocks=None, num_external_tokens=32)

    registration = scheduler._push_pending_registrations["req-1"]
    assert registration["matched_prefix_tokens"] == 128
    assert registration["source_block_offset"] == 8


def test_scheduler_sends_empty_registration_on_full_d_hit(monkeypatch):
    scheduler = object.__new__(NixlDeltaPushConnectorScheduler)
    scheduler.block_size = 16
    scheduler._has_mamba = False
    scheduler._push_pending_registrations = {}
    scheduler.engine_id = "d-engine"
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 5601
    scheduler.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))

    monkeypatch.setattr(
        NixlPushConnectorScheduler,
        "update_state_after_alloc",
        lambda self, request, blocks, num_external_tokens: None,
    )
    request = SimpleNamespace(
        request_id="req-full-hit",
        prompt_token_ids=list(range(160)),
        kv_transfer_params={
            "do_remote_prefill": True,
            "remote_engine_id": "p-engine",
            "remote_host": "127.0.0.1",
            "remote_port": 5600,
            "tp_size": 1,
        },
    )
    scheduler.update_state_after_alloc(request, blocks=None, num_external_tokens=0)

    registration = scheduler._push_pending_registrations["req-full-hit"]
    assert registration["local_block_ids"] == ()
    assert registration["source_block_offset"] == 10
    assert registration["matched_prefix_tokens"] == 160


def test_worker_passes_only_selected_suffix_to_upstream(monkeypatch):
    captured = {}

    def fake_parent_push(self, request_id, local_block_ids, registration_data):
        captured["request_id"] = request_id
        captured["local_block_ids"] = local_block_ids

    monkeypatch.setattr(
        NixlPushConnectorWorker,
        "_do_start_push_kv",
        fake_parent_push,
    )
    worker = object.__new__(NixlDeltaPushConnectorWorker)
    # The test bypasses the real worker constructor; keep its destructor from
    # trying to stop queues and threads that were intentionally not created.
    worker.shutdown = lambda: None
    worker.block_size = 16
    worker._do_start_push_kv(
        "req-2",
        (list(range(10)),),
        {
            "local_block_ids": ([100, 101],),
            "source_block_offset": 7,
            "matched_prefix_tokens": 112,
            "decode_block_size": 16,
            "remote_prompt_tokens": 144,
        },
    )

    assert captured == {
        "request_id": "req-2",
        "local_block_ids": ([7, 8],),
    }


def test_worker_completes_without_write_on_full_d_hit():
    worker = object.__new__(NixlDeltaPushConnectorWorker)
    worker.shutdown = lambda: None
    worker.block_size = 16
    worker._sending_transfers = {}
    worker._sending_transfers_lock = Lock()
    worker._do_start_push_kv(
        "req-full-hit",
        (list(range(10)),),
        {
            "local_block_ids": (),
            "source_block_offset": 10,
            "matched_prefix_tokens": 160,
            "decode_block_size": 16,
            "remote_prompt_tokens": 160,
        },
    )

    assert worker._sending_transfers == {"req-full-hit": []}


def test_pd_deployment_enables_external_delta_connector():
    with open("benchmarks/thinker_talker/pd_deploy_4gpu.yaml") as f:
        deploy = yaml.safe_load(f)

    thinker_p, thinker_d = deploy["stages"][:2]
    for stage in (thinker_p, thinker_d):
        transfer = stage["engine_extras"]["kv_transfer_config"]
        assert transfer["kv_connector"] == "NixlDeltaPushConnector"
        assert transfer["kv_connector_module_path"] == ("vllm_omni.engine.nixl_delta_push_connector")
        assert stage["enable_prefix_caching"] is True
