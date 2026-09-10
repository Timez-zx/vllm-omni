import hashlib

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_omni.engine.pinned_prefix_window import PinnedPrefixWindowSpec, compact_window_view


def fixture():
    def h(value):
        return hashlib.sha256(repr(value).encode()).digest()

    init_none_hash(h)
    hasher = get_request_block_hasher(16, h)
    spec = PinnedPrefixWindowSpec(
        block_size=16, num_kv_heads=2, head_size=32, dtype=torch.float16, sliding_window=64, pinned_prefix_tokens=32
    )
    cfg = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)])
    manager = KVCacheManager(
        cfg, max_model_len=4096, scheduler_block_size=16, hash_block_size=16, max_in_flight_tokens=32
    )

    def request(name, n, salt="a"):
        return Request(name, list(range(n)), SamplingParams(max_tokens=1), None, block_hasher=hasher, cache_salt=salt)

    return manager, request


def test_pin_retention_hit_and_reference_counts():
    m, request = fixture()
    p = request("p", 256)
    assert m.allocate_slots(p, 256)
    p.num_computed_tokens = 256
    m.remove_skipped_blocks("p", 256)
    blocks = m.get_blocks("p").blocks[0]
    head_ids = [b.block_id for b in blocks[:2]]
    assert all(not b.is_null for b in blocks[:2])
    assert all(b.is_null for b in blocks[2:-4])
    assert sum(not b.is_null for b in blocks) == 6
    m.free(p)
    d = request("d", 272)
    hit, count, _ = m.get_computed_blocks(d)
    assert count == 256
    assert [b.block_id for b in hit.blocks[0][:2]] == head_ids
    assert m.allocate_slots(d, 16, num_new_computed_tokens=count, new_computed_blocks=hit)
    held = m.get_blocks("d").blocks[0]
    assert [b.block_id for b in held[:2]] == head_ids
    assert all(b.ref_cnt == 1 for b in held if not b.is_null)
    assert m.get_computed_blocks(request("unrelated", 272, "other"))[1] == 0


def test_missing_pin_rejects_tail_hit_and_remote_alloc_restores_head():
    m, request = fixture()
    p = request("p", 256)
    m.allocate_slots(p, 256)
    p.num_computed_tokens = 256
    m.remove_skipped_blocks("p", 256)
    heads = list(m.get_blocks("p").blocks[0][:2])
    m.free(p)
    for b in heads:
        m.block_pool._maybe_evict_cached_block(b)
    d = request("d", 272)
    assert m.get_computed_blocks(d)[1] == 0
    assert m.allocate_slots(d, 16, num_external_computed_tokens=256)
    blocks = m.get_blocks("d").blocks[0]
    assert all(not b.is_null for b in blocks[:2])
    assert all(b.is_null for b in blocks[2:12])
    assert sum(not b.is_null for b in blocks) == 7


def test_compact_attention_view_preserves_head_and_absolute_mapping():
    table = torch.arange(32, dtype=torch.int32).reshape(1, -1)
    compact, lens = compact_window_view(
        table, torch.tensor([263]), torch.tensor([0, 7]), prefix_tokens=32, window_tokens=64, block_size=16
    )
    assert compact.tolist() == [[0, 1, 12, 13, 14, 15, 16]]
    assert lens.tolist() == [103]
    assert table[0, 12] == 12
    assert (
        PinnedPrefixWindowSpec(
            block_size=16, num_kv_heads=2, head_size=32, dtype=torch.float16, sliding_window=64, pinned_prefix_tokens=32
        ).max_admission_blocks_per_request(32, 4096)
        == 9
    )


def test_compact_view_supports_mixed_session_lengths_without_cross_row_reuse():
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)
    compact, lens = compact_window_view(
        table,
        torch.tensor([263, 71, 280]),
        torch.tensor([0, 7, 14, 30]),
        prefix_tokens=32,
        window_tokens=64,
        block_size=16,
    )
    assert lens.tolist() == [103, 71, 120]
    assert compact[0, :7].tolist() == [0, 1, 12, 13, 14, 15, 16]
    assert compact[1, :5].tolist() == [32, 33, 34, 35, 36]
    assert compact[2].tolist() == [64, 65, 76, 77, 78, 79, 80, 81]
