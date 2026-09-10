# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from vllm_omni.experimental.fullduplex.minicpmo45 import speech_probe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_disabled_probe_does_not_read_tensors_or_create_files(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_probe, "DIRECTORY", str(tmp_path / "unused"))
    monkeypatch.setattr(speech_probe, "ENABLED", False)
    monkeypatch.setattr(speech_probe, "SAVE_TENSORS", True)
    assert speech_probe.emit("unused", tensor=object()) is None
    assert speech_probe.new_batch_id() is None
    assert speech_probe.dump_condition(condition=object()) is None
    assert not (tmp_path / "unused").exists()


def test_enabled_probe_saves_pre_mutation_condition_and_json(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_probe, "DIRECTORY", str(tmp_path))
    monkeypatch.setattr(speech_probe, "ENABLED", True)
    monkeypatch.setattr(speech_probe, "SAVE_TENSORS", True)
    original = torch.arange(4, dtype=torch.bfloat16)
    path = speech_probe.dump_condition(condition=original)
    original.add_(100)
    seq = speech_probe.emit("test", positions=[0, 1], condition_file=path, turn_id=torch.tensor(7))
    record = json.loads(next(tmp_path.glob("speech-*.jsonl")).read_text())
    assert record["seq"] == seq
    assert record["positions"] == [0, 1]
    assert record["turn_id"] == 7
    assert torch.equal(torch.load(tmp_path / path, weights_only=True)["condition"], torch.arange(4))
