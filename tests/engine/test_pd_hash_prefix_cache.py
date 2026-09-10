"""CPU hash reuse must exactly match native hashing, not relax KV matching."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

from vllm_omni.request import OmniRequest, TokenBlockHashPrefixCache


def make(ids, hasher, *, salt="session", remote="p", **kwargs):
    return OmniRequest(
        request_id="d",
        prompt_token_ids=list(ids),
        block_hasher=hasher,
        sampling_params=SamplingParams(
            max_tokens=20,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_engine_id": "engine",
                    "remote_request_id": remote,
                }
            }
            if remote is not None
            else None,
        ),
        pooling_params=None,
        cache_salt=salt,
        **kwargs,
    )


@pytest.fixture
def hashers():
    init_none_hash(sha256)
    calls = []

    def counted(value):
        calls.append(value)
        return sha256(value)

    native = get_request_block_hasher(16, sha256)
    cached = TokenBlockHashPrefixCache(get_request_block_hasher(16, counted), 16, 2)
    return native, cached, calls


def test_growing_logical_history_only_hashes_delta_and_matches_native(hashers):
    native, cached, calls = hashers
    ids = list(range(64003))
    first = make(ids, cached)
    assert first.block_hashes == make(ids, native).block_hashes
    assert len(calls) == 4000
    calls.clear()
    ids += list(range(229))
    second = make(ids, cached)
    assert second.block_hashes == make(ids, native).block_hashes
    assert second._omni_reused_hash_tokens == 64000
    assert len(calls) == len(ids) // 16 - 4000
    calls.clear()
    # Cache-only import and formal D admission share hashes, not Request/KV state.
    third = make(ids, cached, remote=None)
    assert third.block_hashes == second.block_hashes
    assert third.block_hashes is not second.block_hashes
    assert not calls
    third.append_output_token_ids([7] * 16)
    expected = make(ids, native)
    expected.append_output_token_ids([7] * 16)
    assert third.block_hashes == expected.block_hashes


@pytest.mark.parametrize("change", ["tokens", "salt", "unsalted", "lora", "conditioning", "mm", "embeds"])
def test_different_hash_inputs_never_reuse_incorrect_prefix(hashers, change):
    native, cached, calls = hashers
    ids = list(range(128))
    make(ids, cached)
    calls.clear()
    kwargs = {}
    if change == "tokens":
        ids[0] += 1000
    elif change == "salt":
        kwargs["salt"] = "other"
    elif change == "unsalted":
        kwargs["salt"] = None
    elif change == "lora":
        kwargs["lora_request"] = LoRARequest("adapter", 1, "/unused")
    elif change == "conditioning":
        kwargs["cache_token_ids"] = [x + 1000 for x in ids]
    elif change == "mm":
        kwargs["mm_features"] = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="image-a",
                mm_position=PlaceholderRange(offset=16, length=32),
            )
        ]
    else:
        kwargs["prompt_embeds"] = torch.zeros(128, 4)
    assert make(ids, cached, **kwargs).block_hashes == make(ids, native, **kwargs).block_hashes
    assert len(calls) >= 8


def test_shorter_prefix_partial_block_eviction_and_failure_restore(hashers):
    native, cached, calls = hashers
    make(range(130), cached)
    calls.clear()
    assert make(range(65), cached).block_hashes == make(range(65), native).block_hashes
    assert not calls
    make(range(80), cached, salt="two")
    make(range(80), cached, salt="three")
    assert len(cached._entries) == 2
    calls.clear()
    make(range(65), cached)
    assert len(calls) == 4
    request = make(range(80), native)
    original = request.block_hashes = []

    def fail(_request):
        raise ValueError("native failure")

    cached.delegate = fail
    with pytest.raises(ValueError, match="native failure"):
        cached(request)
    assert request.block_hashes is original


def test_concurrent_cache_sync_and_add_keep_exact_native_hashes(hashers):
    native, cached, _ = hashers
    jobs = [(n, salt) for n in (160, 384, 208, 512) for salt in ("one", "two", "three")]

    def run(job):
        n, salt = job
        assert make(range(n), cached, salt=salt).block_hashes == make(range(n), native, salt=salt).block_hashes

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, jobs * 4))
    assert len(cached._entries) <= 2
