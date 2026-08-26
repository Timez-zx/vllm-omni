# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_process_engine_outputs_records_pre_queue_timestamp(monkeypatch):
    client = _make_client(engine_dead=False)
    client._output_queue_received_mono = {}
    outputs = SimpleNamespace(outputs=[])
    monkeypatch.setattr(
        "vllm_omni.engine.stage_engine_core_client.time.monotonic",
        lambda: 123.5,
    )

    await client.process_engine_outputs(outputs)

    assert client._output_queue_received_mono[id(outputs)] == 123.5


def test_get_output_nowait_drains_decoded_queue(monkeypatch):
    client = _make_client(engine_dead=False)
    client._output_queue_received_mono = {}
    client.outputs_queue = asyncio.Queue()
    outputs = SimpleNamespace(outputs=[])
    client.outputs_queue.put_nowait(outputs)
    monkeypatch.setattr(client, "_ensure_output_queue_task", lambda: None)

    assert client.get_output_nowait() is outputs
    assert client.get_output_nowait() is None
