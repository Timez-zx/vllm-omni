"""Opt-in pinned prefix + rolling causal KV, with native incremental requests.

The logical block table and RoPE positions never change. Only the attention
*view* omits the evicted middle gap. vLLM's R-SWA mask then implements
``causal & (key < pinned_prefix_tokens | query - key < window)``.
This is a token-prefix policy, not learned scalar attention sinks.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class PinnedPrefixWindowSpec(SlidingWindowSpec):
    pinned_prefix_tokens: int

    def __post_init__(self):
        super().__post_init__()
        if self.pinned_prefix_tokens <= 0 or self.pinned_prefix_tokens % self.block_size:
            raise ValueError("Pinned prefix must be positive and block aligned")

    def max_admission_blocks_per_request(self, max_in_flight_tokens, max_model_len):
        return min(
            cdiv(max_model_len, self.block_size),
            super().max_admission_blocks_per_request(max_in_flight_tokens, max_model_len)
            + self.pinned_prefix_tokens // self.block_size,
        )

    def is_uniform_with_collection(self, specs):
        return all(type(s) is type(self) and s == self for s in specs.values())


class PinnedPrefixWindowManager(SlidingWindowManager):
    def __init__(self, kv_cache_spec, **kwargs):
        super().__init__(kv_cache_spec, **kwargs)
        self.prefix_blocks = kv_cache_spec.pinned_prefix_tokens // self.block_size

    def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
        end = self.get_num_skipped_tokens(processed_computed_tokens) // self.block_size
        self._remove_blocks_in_range(request_id, self.prefix_blocks, end)

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size=1,
        pcp_world_size=1,
    ):
        result, length = super().find_longest_cache_hit(
            block_hashes,
            max_length,
            kv_cache_group_ids,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        if not length:
            return result, length
        if block_pool.hash_block_size != kv_cache_spec.block_size:
            raise NotImplementedError("Pinned prefix requires equal hash and physical block sizes")
        count = min(kv_cache_spec.pinned_prefix_tokens, length) // kv_cache_spec.block_size
        for i in range(count):
            cached = block_pool.get_cached_block(block_hashes[i], kv_cache_group_ids)
            if cached is None:
                # A cached tail without its required prefix is NOT a valid hit.
                # P can restore the resident prefix + tail via the connector.
                return tuple([] for _ in kv_cache_group_ids), 0
            for group, block in zip(result, cached):
                group[i] = block
        return result, length

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        count = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        if request_id in self.num_cached_block:
            return count
        skipped = self.get_num_skipped_tokens(total_computed_tokens) // self.block_size
        head = min(skipped, self.prefix_blocks)
        cached_head = new_computed_blocks[:head]
        return count + head - len(cached_head) + self._get_num_evictable_blocks(cached_head)

    def add_local_computed_blocks(
        self, request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens
    ):
        total = num_local_computed_tokens + num_external_computed_tokens
        skipped = self.get_num_skipped_tokens(total) // self.block_size
        if not skipped:
            return super().add_local_computed_blocks(
                request_id, new_computed_blocks, num_local_computed_tokens, num_external_computed_tokens
            )
        head = min(skipped, self.prefix_blocks)
        cached_head = list(new_computed_blocks[:head])
        if any(b.is_null for b in cached_head):
            raise RuntimeError("Prefix cache hit is missing pinned blocks")
        tail = list(new_computed_blocks[skipped:])
        if self.enable_caching:
            self.block_pool.touch(cached_head + tail)
        else:
            assert not cached_head and not tail
        blocks = self.req_to_blocks[request_id]
        assert not blocks
        blocks.extend(cached_head)
        blocks.extend([self._null_block] * (skipped - len(cached_head)))
        blocks.extend(tail)
        # When importing the pinned prefix, it still needs hashing on completion.
        self.num_cached_block[request_id] = len(blocks) if len(cached_head) == head else 0

    def allocate_external_computed_blocks(self, request_id, num_local_computed_tokens, num_external_computed_tokens):
        # Called only after ALL groups' cache hits have been touched.
        total = num_local_computed_tokens + num_external_computed_tokens
        skipped = self.get_num_skipped_tokens(total) // self.block_size
        blocks = self.req_to_blocks[request_id]
        for i in range(min(skipped, self.prefix_blocks)):
            if blocks[i].is_null:
                blocks[i] = self.block_pool.get_new_blocks(1)[0]
        super().allocate_external_computed_blocks(request_id, num_local_computed_tokens, num_external_computed_tokens)

    @classmethod
    def reachable_block_mask(
        cls,
        start_block,
        end_block,
        alignment_tokens,
        kv_cache_spec,
        use_eagle,
        retention_interval=None,
        reachable_boundaries=(),
    ):
        mask = super().reachable_block_mask(
            start_block,
            end_block,
            alignment_tokens,
            kv_cache_spec,
            use_eagle,
            retention_interval,
            reachable_boundaries,
        )
        if mask is not None:
            for i in range(start_block, min(end_block, kv_cache_spec.pinned_prefix_tokens // kv_cache_spec.block_size)):
                mask[i - start_block] = True
        return mask


def compact_window_view(block_table, seq_lens_cpu, query_start_cpu, *, prefix_tokens, window_tokens, block_size):
    """Build a bounded attention view; no KV tensor moves or RoPE changes."""
    qlens = query_start_cpu[1:] - query_start_cpu[:-1]
    history = seq_lens_cpu - qlens
    prefix_blocks = prefix_tokens // block_size
    gap_blocks = ((history - window_tokens + 1).clamp(min=0) // block_size - prefix_blocks).clamp(min=0)
    compact_lengths = seq_lens_cpu - gap_blocks * block_size
    width = cdiv(int(compact_lengths.max()), block_size)
    cols = torch.arange(width, device=block_table.device).unsqueeze(0)
    gaps = gap_blocks.to(block_table.device).unsqueeze(1)
    source_cols = cols + torch.where(cols >= prefix_blocks, gaps, 0)
    # Row padding is never attended; clamp it to a valid table index.
    source_cols = source_cols.clamp(max=block_table.shape[1] - 1).long()
    compact = torch.gather(block_table, 1, source_cols)
    return compact, compact_lengths


def install_pinned_prefix_window():
    """Register a spec/manager and opt-in Triton metadata hooks, once per process."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionMetadataBuilder

    if getattr(Attention.get_kv_cache_spec, "_pinned_prefix_window", False):
        return
    # vLLM's lazy registration uses a nonempty registry as its sentinel.
    # Register builtins before adding a custom type, otherwise they are skipped.
    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(
        PinnedPrefixWindowSpec, PinnedPrefixWindowManager, uniform_type_base_spec=PinnedPrefixWindowSpec
    )
    original_spec = Attention.get_kv_cache_spec

    def get_spec(self, config):
        spec = original_spec(self, config)
        pin = int(getattr(config.model_config.hf_config, "vllm_omni_pinned_prefix_tokens", 0))
        if not pin or spec is None:
            return spec
        if type(spec) is not SlidingWindowSpec or self.attn_backend.get_name() != "TRITON_ATTN":
            raise ValueError("Pinned prefix requires uniform sliding-window Triton attention")
        return PinnedPrefixWindowSpec(**{f.name: getattr(spec, f.name) for f in fields(spec)}, pinned_prefix_tokens=pin)

    get_spec._pinned_prefix_window = True
    Attention.get_kv_cache_spec = get_spec
    original_init = Attention.__init__

    def attention_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = get_current_vllm_config()
        if int(getattr(config.model_config.hf_config, "vllm_omni_pinned_prefix_tokens", 0)):
            if self.attn_backend.get_name() != "TRITON_ATTN" or self.sliding_window is None:
                raise ValueError("Pinned prefix requires Triton sliding-window attention")
            # Do not apply ordinary SW tile pruning: it would hide the sinks.
            # The bounded attention view + R-SWA mask below handles both sets.
            self.impl.sliding_window = (-1, -1)

    Attention.__init__ = attention_init

    class PinnedPrefixMetadataBuilder(TritonAttentionMetadataBuilder):
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self.pinned_prefix_tokens = getattr(kv_cache_spec, "pinned_prefix_tokens", 0)
            if not self.pinned_prefix_tokens:
                return
            parallel = vllm_config.parallel_config
            if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
                raise NotImplementedError("Pinned prefix does not yet support context parallelism")
            n = vllm_config.scheduler_config.max_num_seqs
            width = cdiv(vllm_config.model_config.max_model_len, self.block_size)
            self.pinned_view = torch.empty((n, width), dtype=torch.int32, device=device)
            self.pinned_lengths = torch.empty(n, dtype=torch.int32, device=device)
            self.pinned_heads = torch.full((n,), self.pinned_prefix_tokens, dtype=torch.int32, device=device)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            if not self.pinned_prefix_tokens:
                return super().build(common_prefix_len, common_attn_metadata, fast_build)
            if common_prefix_len or not common_attn_metadata.causal or common_attn_metadata.mm_req_doc_ranges:
                raise NotImplementedError("Pinned prefix supports causal non-cascade attention")
            n = common_attn_metadata.num_reqs
            view, lengths = compact_window_view(
                common_attn_metadata.block_table_tensor[:n],
                common_attn_metadata.seq_lens_cpu[:n],
                common_attn_metadata.query_start_loc_cpu[: n + 1],
                prefix_tokens=self.pinned_prefix_tokens,
                window_tokens=self.kv_cache_spec.sliding_window,
                block_size=self.block_size,
            )
            width = view.shape[1]
            self.pinned_view[:n, :width].copy_(view)
            self.pinned_lengths[:n].copy_(lengths, non_blocking=True)
            compact = copy.copy(common_attn_metadata)
            compact.block_table_tensor = self.pinned_view[:n, :width]
            compact.seq_lens = self.pinned_lengths[:n]
            compact.max_seq_len = int(lengths.max())
            result = super().build(0, compact, fast_build)
            result.rswa_prefix_lens = self.pinned_heads[:n]
            result.rswa_window = self.kv_cache_spec.sliding_window
            return result

    TritonAttentionBackend.get_builder_cls = staticmethod(lambda: PinnedPrefixMetadataBuilder)
