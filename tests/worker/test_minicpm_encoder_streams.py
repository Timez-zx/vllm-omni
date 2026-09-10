from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.worker.gpu_ar_worker import GPUARWorker


@pytest.mark.parametrize("placement", ["", "cuda:0", "cuda:1"])
def test_encoder_stream_is_private_reused_and_fenced(monkeypatch, placement):
    worker = GPUARWorker.__new__(GPUARWorker)
    worker.device = torch.device("cuda:0")
    stream = MagicMock()
    event = MagicMock()
    factory = MagicMock(return_value=stream)
    monkeypatch.setenv("MINICPMO45_AUDIO_ENCODER_DEVICE", placement)
    monkeypatch.setattr(torch.cuda, "Stream", factory)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device: "loaded-weights")
    monkeypatch.setattr(torch.cuda, "stream", lambda value: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda value: nullcontext())

    def forbidden(*args):
        raise AssertionError("must not synchronize the whole GPU")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    for _ in range(2):
        assert worker._run_minicpmo_encoder("audio", lambda jobs: jobs, [1]) == [1]
    factory.assert_called_once_with(device=torch.device(placement or "cuda:0"))
    stream.wait_stream.assert_called_once_with("loaded-weights")
    assert event.record.call_count == event.synchronize.call_count == 2

    def failing(jobs):
        raise RuntimeError("encoding failed")

    with pytest.raises(RuntimeError, match="encoding failed"):
        worker._run_minicpmo_encoder("audio", failing, [])
    assert event.synchronize.call_count == 3
