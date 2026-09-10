"""Reference PCM, processor lengths, and native scheduler-prefix agreement."""

import base64
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.experimental.fullduplex.minicpmo45.adapter import (
    MiniCPMO45ClientRuntimeConfigError,
    MiniCPMO45NativeDuplexServingAdapter,
)
from vllm_omni.experimental.fullduplex.minicpmo45.policy import MiniCPMO45DuplexPolicy
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import duplex_first_append_context_reserve
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSessionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"


@pytest.fixture(scope="module")
def official_processor():
    """Use released processor code and mel extraction, not a copied formula.

    This optional integration fixture reads an existing local HF snapshot only;
    it neither downloads model weights nor initializes a GPU. Tokenizer strings
    suffice for the audio-only public placeholder method.
    """
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache("openbmb/MiniCPM-o-4_5", "processing_minicpmo.py", revision=_REVISION)
    if not isinstance(cached, str):
        pytest.skip("Official MiniCPM-o 4.5 processor snapshot is not cached")
    source = Path(cached)
    spec = importlib.util.spec_from_file_location("_minicpm_reference_audio_official", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor_config = json.loads((source.parent / "preprocessor_config.json").read_text())
    processor = module.MiniCPMOProcessor.__new__(module.MiniCPMOProcessor)
    processor.audio_processor = module.MiniCPMAAudioProcessor.from_pretrained(source.parent, local_files_only=True)
    processor.pool_step = processor_config["audio_pool_step"]
    processor.tokenizer = SimpleNamespace(audio_start="<|audio_start|>", audio_end="<|audio_end|>")
    assert processor.audio_processor.hop_length == 160
    assert processor.pool_step == 5
    return processor


def _official_rows(processor, waveform):
    features = processor.process_audio([waveform])
    return sum(int((((lens - 1) // 2 + 1) // processor.pool_step).sum()) for lens in features["audio_feature_lens"])


@pytest.mark.parametrize(
    ("samples", "rows"),
    [
        (0, 0),
        (1, 0),
        (1280, 0),
        (1281, 1),
        (1599, 1),
        (1600, 1),
        (1601, 1),
        (2881, 2),
        (16000, 10),
        (96000, 60),
        (96256, 60),
        (97281, 61),
        (481281, 301),
    ],
)
def test_reference_audio_count_includes_partial_mel_frames(samples, rows):
    assert MiniCPMO45DuplexPolicy.audio_token_count(samples) == rows


@pytest.mark.parametrize("samples", [0, 1, 1280, 1281, 1599, 1601, 2881, 96256, 97281, 481281])
def test_reference_count_matches_real_official_processor(official_processor, samples):
    waveform = np.zeros(samples, dtype=np.float32)
    expected = MiniCPMO45DuplexPolicy.audio_token_count(samples)
    assert _official_rows(official_processor, waveform) == expected
    assert official_processor.get_audio_placeholder(samples, chunk_input=False).count("<unk>") == expected


async def _prepare_reference(monkeypatch, waveform):
    async def resolve_ref_audio(_uri, *, model_config):
        return waveform, 16000

    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: list(range(8)) if text.startswith("<|im_start|>") else [8, 9]
    )
    monkeypatch.setattr(MiniCPMO45NativeDuplexServingAdapter, "resolve_ref_audio", staticmethod(resolve_ref_audio))
    monkeypatch.setattr(
        MiniCPMO45NativeDuplexServingAdapter, "_load_native_tokenizer", staticmethod(lambda _: tokenizer)
    )
    config = DuplexSessionConfig(
        ref_audio="data:audio/wav;base64,fixture", extra_body={"minicpmo45_native_duplex": True}
    )
    prepared = await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(config, model_config=SimpleNamespace())
    return config, prepared, tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [1281, 1599, 2881, 96256, 97281])
async def test_reference_pcm_is_not_trimmed_or_padded_to_fit_budget(monkeypatch, samples):
    waveform = np.linspace(-0.2, 0.3, samples, dtype=np.float32)
    config, prepared, _ = await _prepare_reference(monkeypatch, waveform)
    assert base64.b64decode(prepared["ref_audio_data"]) == waveform.tobytes()
    assert config.ref_audio is None
    assert prepared["ref_audio_sample_rate_hz"] == 16000
    assert prepared["duplex_first_append_context_tokens"] == 10 + MiniCPMO45DuplexPolicy.audio_token_count(samples)
    assert duplex_first_append_context_reserve(prepared) == prepared["duplex_first_append_context_tokens"]
    fallback = dict(prepared)
    fallback.pop("duplex_first_append_context_tokens")
    assert duplex_first_append_context_reserve(fallback) == 56 + MiniCPMO45DuplexPolicy.audio_token_count(samples)


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [0, 1, 1280])
async def test_empty_or_unencodable_reference_is_rejected_without_padding(monkeypatch, samples):
    with pytest.raises(MiniCPMO45ClientRuntimeConfigError) as error:
        await _prepare_reference(monkeypatch, np.zeros(samples, dtype=np.float32))
    assert error.value.code == "ref_audio_too_short"


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [1281, 96256, 97281])
async def test_reference_budget_matches_worker_prefix_and_rebuild(monkeypatch, official_processor, samples):
    from vllm_omni.experimental.fullduplex.minicpmo45.stage0 import (
        MiniCPMO45Stage0DuplexRuntime,
        _MiniCPMO45Stage0SessionState,
    )

    waveform = np.linspace(-0.1, 0.1, samples, dtype=np.float32)
    config, prepared, tokenizer = await _prepare_reference(monkeypatch, waveform)
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.unit_token_id = 151683
    runtime.processor = official_processor
    runtime.stage_model = SimpleNamespace(get_audio_embedding=lambda *_: None)
    runtime.thinker = SimpleNamespace()
    runtime._session_context_cache = OrderedDict()
    runtime._stage_runtime_ready = lambda: True
    runtime._require_special_token_ids = lambda: None
    runtime._encode_text = tokenizer.encode
    runtime._embed_token = lambda token: torch.full((1, 2), float(token))
    encoded_waveforms = []

    def encode_ref_audio(ref_audio, state=None):
        encoded_waveforms.append(ref_audio.copy())
        # Only the heavyweight encoder weights are replaced. The actual PCM
        # decode, official mel processor, prefix construction and cache are real.
        return torch.zeros((_official_rows(official_processor, ref_audio), 2))

    runtime._stage_ref_audio_embeddings = encode_ref_audio
    first = _MiniCPMO45Stage0SessionState(session_id="reference-prefix")
    rebuilt = _MiniCPMO45Stage0SessionState(session_id="reference-prefix-rebuilt")
    for state in (first, rebuilt):
        runtime._prepare_session_context(state, config.as_dict(), runtime_config=prepared)
        assert len(state.context_token_ids) == duplex_first_append_context_reserve(prepared)
        assert sum(embedding.shape[0] for embedding in state.context_embeds) == len(state.context_token_ids)
    assert first.context_token_ids == rebuilt.context_token_ids
    assert len(encoded_waveforms) == 1
    np.testing.assert_array_equal(encoded_waveforms[0], waveform)
    changed_tail = waveform.copy()
    changed_tail[-1] += 0.25
    assert runtime._session_context_cache_key(config.instructions, waveform) != runtime._session_context_cache_key(
        config.instructions, changed_tail
    )
