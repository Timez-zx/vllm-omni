"""Real vLLM cache manager: sliding preserves lineage, not substring matches."""

import hashlib

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.request import Request


def test_window_cache_hit_survives_eviction_but_not_changed_history():
    def hash_fn(value):
        return hashlib.sha256(repr(value).encode()).digest()

    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(16, hash_fn)
    spec = SlidingWindowSpec(block_size=16, num_kv_heads=2, head_size=16, dtype=torch.float16, sliding_window=64)
    config = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    manager = KVCacheManager(
        config, max_model_len=4096, scheduler_block_size=16, hash_block_size=16, max_in_flight_tokens=32
    )

    def request(name, tokens):
        return Request(name, tokens, SamplingParams(max_tokens=1), None, block_hasher=hasher, cache_salt="session-one")

    original = request("p", list(range(256)))
    assert manager.allocate_slots(original, 256) is not None
    original.num_computed_tokens = 256
    manager.remove_skipped_blocks("p", 256)
    blocks = manager.get_blocks("p").blocks[0]
    assert sum(not block.is_null for block in blocks) == 4
    manager.free(original)

    # Remove old prefix from the global cache map as well: the suffix alone
    # must suffice, but it still carries its original chained identity.
    for block in list(manager.block_pool.blocks):
        if not block.is_null and block.block_hash is not None and block not in blocks[-4:]:
            manager.block_pool._maybe_evict_cached_block(block)
    continuation = request("d", list(range(272)))
    reused, count, _ = manager.get_computed_blocks(continuation)
    assert count == 256
    assert sum(not block.is_null for block in reused.blocks[0]) == 4
    unrelated = request("other", [999] + list(range(1, 272)))
    assert manager.get_computed_blocks(unrelated)[1] == 0
    cropped = request("cropped", list(range(192, 272)))
    assert manager.get_computed_blocks(cropped)[1] == 0
    other_session = Request(
        "other-session",
        list(range(272)),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=hasher,
        cache_salt="session-two",
    )
    assert manager.get_computed_blocks(other_session)[1] == 0


def test_native_av_placeholders_are_salted_per_session_and_restart():
    from vllm_omni.experimental.fullduplex.engine.duplex_runtime import DuplexInputMode
    from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
    from vllm_omni.experimental.fullduplex.minicpmo45.runtime import MiniCPMO45DuplexRuntimeExtension

    def plan(extension, sid, seq=1):
        return extension.plan_append(
            request_id=sid,
            fence=DuplexFence(sid),
            seq=seq,
            turn_seq=seq,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload={},
            final=False,
            session_config={},
            sampling_params=[],
            runtime_config={"duplex_scheduler_token_id": 0, "duplex_kv_window_tokens": 1024},
        ).prompt

    extension = MiniCPMO45DuplexRuntimeExtension()
    a, b = plan(extension, "a"), plan(extension, "b")
    assert a["prompt_token_ids"] == b["prompt_token_ids"]
    assert a["cache_salt"] != b["cache_salt"]
    assert plan(extension, "a")["cache_salt"] == a["cache_salt"]
    assert plan(extension, "a", 2)["cache_salt"] == a["cache_salt"]
    assert plan(MiniCPMO45DuplexRuntimeExtension(), "a")["cache_salt"] != a["cache_salt"]

    def hash_fn(value):
        return hashlib.sha256(repr(value).encode()).digest()

    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(16, hash_fn)
    requests = [
        Request(
            str(i),
            p["prompt_token_ids"],
            SamplingParams(max_tokens=1),
            None,
            block_hasher=hasher,
            cache_salt=p["cache_salt"],
        )
        for i, p in enumerate((a, b))
    ]
    assert requests[0].block_hashes and requests[0].block_hashes != requests[1].block_hashes
