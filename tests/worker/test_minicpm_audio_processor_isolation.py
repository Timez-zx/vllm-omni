"""Real-processor regressions for per-session streaming Mel state isolation."""

import copy
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch

from tests.worker.test_minicpm_reference_audio import official_processor as official_processor
from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45Stage0SessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runtime_and_processors(official_processor):
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.processor = copy.deepcopy(official_processor)
    runtime._stage_param = lambda _name, default: default
    states = [_MiniCPMO45Stage0SessionState(session_id=f"audio-{index}") for index in range(2)]
    processors = [runtime._configure_streaming_processor(state) for state in states]
    return runtime, states, processors


def _waveform(seconds):
    t = np.arange(16000 * seconds, dtype=np.float32) / 16000
    return (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_native_streaming_sessions_own_distinct_feature_extractors(official_processor):
    runtime, states, processors = _runtime_and_processors(official_processor)
    extractors = [runtime.processor.audio_processor, *(processor.audio_processor for processor in processors)]
    assert len({id(extractor) for extractor in extractors}) == 3
    assert processors[0]._streaming_mel_processor is not processors[1]._streaming_mel_processor
    for state, processor in zip(states, processors, strict=True):
        assert processor._streaming_mel_processor.feature_extractor is processor.audio_processor
        assert runtime._configure_streaming_processor(state) is processor


def test_streaming_normalization_and_snapshot_restore_do_not_touch_another_session(official_processor):
    runtime, _, processors = _runtime_and_processors(official_processor)
    first, second = [processor._streaming_mel_processor for processor in processors]
    first.feature_extractor.set_spac_log_norm(dynamic_range_db=8)
    snapshot = first.get_snapshot()
    second.feature_extractor.set_spac_log_norm(log_floor_db=-10)
    first.restore_snapshot(snapshot)
    assert first.feature_extractor.dynamic_log_norm is True
    assert second.feature_extractor.dynamic_log_norm is False
    assert runtime.processor.audio_processor.dynamic_log_norm is True


def test_concurrent_young_and_old_sessions_match_isolated_real_mel(official_processor, monkeypatch):
    _, _, processors = _runtime_and_processors(official_processor)
    short, long = [processor._streaming_mel_processor for processor in processors]
    short.buffer = _waveform(1)
    long.buffer = _waveform(6)
    expected = short._extract_full().clone()
    configured_short = Event()
    release_short = Event()
    set_norm = short.feature_extractor.set_spac_log_norm

    def pause_after_configuring_short(*args, **kwargs):
        result = set_norm(*args, **kwargs)
        if kwargs.get("log_floor_db") is not None:
            configured_short.set()
            assert release_short.wait(5)
        return result

    monkeypatch.setattr(short.feature_extractor, "set_spac_log_norm", pause_after_configuring_short)
    with ThreadPoolExecutor(max_workers=2) as executor:
        short_result = executor.submit(short._extract_full)
        try:
            assert configured_short.wait(5)
            # The older session switches to dynamic normalization while the
            # first session is paused before its actual mel calculation.
            executor.submit(long._extract_full).result(timeout=5)
        finally:
            release_short.set()
        actual = short_result.result(timeout=5)
    assert torch.equal(actual, expected), f"Cross-session mel max error: {(actual - expected).abs().max().item()}"


def test_live_audio_does_not_mutate_reference_processor(official_processor):
    runtime, _, processors = _runtime_and_processors(official_processor)
    reference = _waveform(6)
    before = runtime.processor.process_audio([reference])["audio_features"].clone()
    mel = processors[0]._streaming_mel_processor
    mel.buffer = _waveform(1)
    mel._extract_full()
    after = runtime.processor.process_audio([reference])["audio_features"]
    assert torch.equal(before, after)
