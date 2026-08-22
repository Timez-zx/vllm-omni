# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for OmniRequest substitutability with the base Request.

vllm-omni rebinds ``vllm.v1.request.Request`` to ``OmniRequest`` at import
time (see ``vllm_omni/patch.py``). vLLM core constructs ``Request`` positionally
in some paths (notably ``vllm/v1/worker/gpu/warmup.py::warmup_kernels`` for V2
model-runner architectures like Qwen3ForCausalLM). The omni-specific params
must therefore be keyword-only so positional construction stays Liskov-
substitutable with the base class — including base-style calls that pass
``prompt_embeds`` (itself a positional-capable base param) positionally.
"""

import inspect

import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_omni.engine import PromptEmbedsPayload
from vllm_omni.request import OmniRequest, OmniStreamingUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_params_are_keyword_only():
    """The three omni params must be keyword-only after ``*args``.

    Guards against re-introducing the import-time rebind bug where omni params
    came first positionally and broke positional ``Request(...)`` construction.
    """
    params = inspect.signature(OmniRequest.__init__).parameters
    for name in (
        "prompt_embeds",
        "external_req_id",
        "additional_information",
        "model_intermediate_buffer",
        "pd_prefill_payload",
        "cache_token_ids",
        "prefill_only",
    ):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_positional_construction_matches_base_request():
    """Reproduces the warmup_kernels call: Request(id, tokens, sp, pp)."""
    req = OmniRequest("req-0", [1, 2, 3], SamplingParams(), None)

    assert isinstance(req, Request)
    assert req.request_id == "req-0"
    assert req.prompt_token_ids == [1, 2, 3]
    # Omni params default cleanly when constructed positionally.
    assert req.external_req_id is None
    assert req.additional_information is None
    assert req.prompt_embeds_payload is None


def test_positional_prompt_embeds_does_not_collide():
    """Base-style positional call that includes ``prompt_embeds``.

    ``prompt_embeds`` is a positional-capable base param (after
    ``arrival_time``). A base-style call must not trip
    ``got multiple values for argument 'prompt_embeds'`` — the subclass must
    not also inject its keyword override when the value arrived positionally.
    Positional order: request_id, prompt_token_ids, sampling_params,
    pooling_params, client_index, arrival_time, prompt_embeds.
    """
    embeds = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    req = OmniRequest("req-pe", [1, 2], SamplingParams(), None, 0, None, embeds)

    assert isinstance(req, Request)
    assert torch.equal(req.prompt_embeds, embeds)
    # Positional tensor is not a serialized payload.
    assert req.prompt_embeds_payload is None
    assert req.external_req_id is None


def test_keyword_omni_params_round_trip():
    """Keyword omni params are preserved; serialized embeds are decoded."""
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = PromptEmbedsPayload(data=arr.tobytes(), shape=[2, 3], dtype="float32")

    req = OmniRequest(
        request_id="req-1",
        prompt_token_ids=[7, 8],
        sampling_params=SamplingParams(),
        pooling_params=None,
        prompt_embeds=payload,
        external_req_id="ext-1",
    )

    assert req.external_req_id == "ext-1"
    # The serialized payload is retained and decoded into a tensor on the base.
    assert req.prompt_embeds_payload is payload
    assert torch.equal(req.prompt_embeds, torch.from_numpy(arr))


def test_prefill_only_round_trips_from_engine_request():
    from vllm_omni.engine import OmniEngineCoreRequest

    core_request = OmniEngineCoreRequest(
        request_id="cache-warm",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        prefill_only=True,
    )

    request = OmniRequest.from_engine_core_request(core_request, block_hasher=None)

    assert request.prefill_only is True


def test_pd_prefill_payload_round_trips_from_engine_request():
    from vllm_omni.engine import OmniEngineCoreRequest, OmniPDPrefillPayload

    payload = OmniPDPrefillPayload(
        prompt_layer_0=torch.ones(2, 3),
        prompt_layer_24=torch.full((2, 3), 24.0),
        prompt_token_ids=[1, 2],
    )
    core_request = OmniEngineCoreRequest(
        request_id="pd-decode",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        pd_prefill_payload=payload,
    )

    request = OmniRequest.from_engine_core_request(core_request, block_hasher=None)

    assert request.pd_prefill_payload is payload


def test_model_intermediate_buffer_round_trips_to_streaming_update():
    buffer = {"duplex": {"turn_id": 3}}
    request = OmniRequest(
        request_id="req-model-buffer",
        prompt_token_ids=[1, 2],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        resumable=True,
        model_intermediate_buffer=buffer,
    )

    update = OmniStreamingUpdate.from_request(request)

    assert update is not None
    assert request.model_intermediate_buffer is buffer
    assert update.model_intermediate_buffer is buffer


def test_cache_token_ids_hash_conditioning_without_changing_execution_ids():
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    block_hasher = get_request_block_hasher(4, hash_fn)

    first = OmniRequest(
        request_id="first",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(),
        pooling_params=None,
        cache_salt="session-a",
        cache_token_ids=[100, 101, 102, 103, 200, 201, 202, 203],
        block_hasher=block_hasher,
    )
    same = OmniRequest(
        request_id="same",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(),
        pooling_params=None,
        cache_salt="session-a",
        cache_token_ids=[100, 101, 102, 103, 200, 201, 202, 203],
        block_hasher=block_hasher,
    )
    changed_tail = OmniRequest(
        request_id="changed-tail",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(),
        pooling_params=None,
        cache_salt="session-a",
        cache_token_ids=[100, 101, 102, 103, 300, 301, 302, 303],
        block_hasher=block_hasher,
    )
    other_session = OmniRequest(
        request_id="other-session",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(),
        pooling_params=None,
        cache_salt="session-b",
        cache_token_ids=[100, 101, 102, 103, 200, 201, 202, 203],
        block_hasher=block_hasher,
    )

    assert first.prompt_token_ids == [0] * 8
    assert list(first.all_token_ids) == [0] * 8
    assert first.block_hashes == same.block_hashes
    assert first.block_hashes[0] == changed_tail.block_hashes[0]
    assert first.block_hashes[1] != changed_tail.block_hashes[1]
    assert first.block_hashes[0] != other_session.block_hashes[0]


def test_disposable_kv_lineage_hashes_only_the_changed_tail():
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    block_hasher = get_request_block_hasher(4, hash_fn)
    first = OmniRequest(
        request_id="lineage-first",
        prompt_token_ids=list(range(12)),
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_hasher=block_hasher,
        kv_lineage_id="session-a",
        kv_lineage_revision=1,
    )
    changed = OmniRequest(
        request_id="lineage-changed",
        prompt_token_ids=[*range(8), 90, 91, 92, 93],
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_hasher=block_hasher,
        kv_lineage_id="session-a",
        kv_lineage_parent_revision=1,
        kv_lineage_revision=2,
        kv_lineage_prefix_tokens=8,
        kv_lineage_snapshot_block_hashes=first.block_hashes,
        kv_lineage_snapshot_num_computed_tokens=12,
        kv_lineage_snapshot_hash_block_size=4,
    )

    assert changed.kv_lineage_seeded_tokens == 8
    assert changed.block_hashes[:2] == first.block_hashes[:2]
    assert changed.block_hashes[2] != first.block_hashes[2]


def test_missing_kv_lineage_snapshot_falls_back_to_full_hashing():
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    block_hasher = get_request_block_hasher(4, hash_fn)
    request = OmniRequest(
        request_id="lineage-miss",
        prompt_token_ids=list(range(12)),
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_hasher=block_hasher,
        kv_lineage_id="evicted",
        kv_lineage_parent_revision=99,
        kv_lineage_revision=100,
        kv_lineage_prefix_tokens=8,
    )

    assert request.kv_lineage_seeded_tokens == 0
    assert len(request.block_hashes) == 3


def test_kv_lineage_falls_back_when_multiple_media_items_append_together():
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    block_hasher = get_request_block_hasher(4, hash_fn)
    first = OmniRequest(
        request_id="lineage-media-first",
        prompt_token_ids=list(range(8)),
        sampling_params=SamplingParams(),
        pooling_params=None,
        block_hasher=block_hasher,
    )
    features = [
        MultiModalFeatureSpec(None, "image", "image-a", PlaceholderRange(8, 2)),
        MultiModalFeatureSpec(None, "image", "image-b", PlaceholderRange(10, 2)),
    ]
    kwargs = {
        "prompt_token_ids": list(range(12)),
        "sampling_params": SamplingParams(),
        "pooling_params": None,
        "block_hasher": block_hasher,
        "mm_features": features,
    }
    expected = OmniRequest(request_id="lineage-media-expected", **kwargs)
    seeded = OmniRequest(
        request_id="lineage-media-seeded",
        kv_lineage_id="session-media",
        kv_lineage_parent_revision=1,
        kv_lineage_revision=2,
        kv_lineage_prefix_tokens=8,
        kv_lineage_snapshot_block_hashes=first.block_hashes,
        kv_lineage_snapshot_num_computed_tokens=8,
        kv_lineage_snapshot_hash_block_size=4,
        **kwargs,
    )

    assert seeded.kv_lineage_seeded_tokens == 0
    assert seeded.block_hashes == expected.block_hashes
