# SPDX-License-Identifier: Apache-2.0
"""Keep native text semantics without re-decoding long AV prompt histories."""
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    IncrementalDetokenizer,
    SlowIncrementalDetokenizer,
)

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor


@pytest.fixture(scope="module")
def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<media>", "<listen>"])
    return get_cached_tokenizer(TokenizersBackend(
        tokenizer_object=backend, additional_special_tokens=["<media>", "<listen>"],
    ))


def request(tokenizer, *, duplex=True, detokenize=True, skip=True, stop=None):
    prompt = [tokenizer.convert_tokens_to_ids("<media>")] * 18000
    prompt += tokenizer.encode("之前的问题：", add_special_tokens=False)
    return OmniEngineCoreRequest(
        request_id="r", external_req_id="r", prompt_token_ids=prompt,
        mm_features=None, sampling_params=SamplingParams(
            detokenize=detokenize, skip_special_tokens=skip, stop=stop,
        ), pooling_params=None, arrival_time=0, lora_request=None,
        cache_salt=None, data_parallel_rank=None,
        model_intermediate_buffer={"duplex": {}} if duplex else None,
    )


@pytest.mark.parametrize("duplex,detokenize,expected", [
    (True, True, SlowIncrementalDetokenizer),
    (False, True, FastIncrementalDetokenizer),
    (True, False, IncrementalDetokenizer),
])
def test_selects_native_incremental_path_only_for_duplex(tokenizer, duplex, detokenize, expected):
    processor = MultimodalOutputProcessor(tokenizer, log_stats=False)
    req = request(tokenizer, duplex=duplex, detokenize=detokenize)
    original = req.prompt_token_ids.copy()
    processor.add_request(req, None)
    assert type(processor.request_states["r"].detokenizer) is expected
    assert req.prompt_token_ids == original
    assert req.sampling_params.detokenize is detokenize


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("stop", [None, ["结束"]])
def test_long_prompt_text_unicode_specials_and_stop_match_native_fast(tokenizer, skip, stop):
    req = request(tokenizer, skip=skip, stop=stop)
    processor = MultimodalOutputProcessor(tokenizer, log_stats=False)
    processor.add_request(req, None)
    actual = processor.request_states["r"].detokenizer
    reference = FastIncrementalDetokenizer(tokenizer, req)
    ids = [tokenizer.convert_tokens_to_ids("<listen>")] * 12
    ids += tokenizer.encode("你好👋🏽！ Hello  world，结束。", add_special_tokens=False)
    for token in ids:
        assert actual.update([token], False) == reference.update([token], False)
        assert actual.get_next_output_text(False, True) == reference.get_next_output_text(False, True)
        assert actual.output_token_ids == reference.output_token_ids
        assert actual.num_output_tokens() == reference.num_output_tokens()


def test_native_fallback_only_converts_prompt_boundary(tokenizer, monkeypatch):
    calls = []
    convert = tokenizer.convert_ids_to_tokens
    def traced(ids, *args, **kwargs):
        calls.append(len(ids) if isinstance(ids, list) else 1)
        return convert(ids, *args, **kwargs)
    monkeypatch.setattr(tokenizer, "convert_ids_to_tokens", traced)
    req = request(tokenizer)
    SlowIncrementalDetokenizer(tokenizer, req)
    assert max(calls) <= 7
