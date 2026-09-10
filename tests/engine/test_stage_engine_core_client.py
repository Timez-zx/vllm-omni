# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


def test_reverse_tensor_ipc_preserves_decoder_captured_by_eager_reader(monkeypatch):
    from vllm.v1.engine.core_client import AsyncMPClient
    from vllm.v1.serial_utils import MsgpackDecoder

    import vllm_omni.engine.stage_engine_core_client as module

    captured = []
    provider = object()

    def parent_init(self, *_args, **_kwargs):
        self.decoder = MsgpackDecoder()
        # Model the upstream reader's closure, started during parent init.
        captured.append(self.decoder)

    monkeypatch.setattr(AsyncMPClient, "__init__", parent_init)
    monkeypatch.setattr(module, "TensorIpcReceiver", lambda _queue: provider)
    monkeypatch.setattr(StageEngineCoreClient, "_resolve_contact_host", lambda _self: "127.0.0.1")
    monkeypatch.setattr(StageEngineCoreClient, "_initialize_kv_sender_endpoint", lambda _self: None)
    client = _make_client()
    client.__init__(
        vllm_config=SimpleNamespace(model_config=None),
        executor_class=object,
        log_stats=False,
        output_tensor_queue=object(),
    )
    assert client.decoder is captured[0]
    assert captured[0].oob_tensor_provider is provider
