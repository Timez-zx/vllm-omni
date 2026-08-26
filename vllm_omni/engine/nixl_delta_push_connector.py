# SPDX-License-Identifier: Apache-2.0
"""Prefix-safe, suffix-only NIXL push connector for finite P/D requests.

The upstream :class:`NixlPushConnector` lets the decode scheduler allocate only
the blocks that missed its local prefix cache.  Its worker, however, aligns a
shorter destination list with the *front* of the prefiller's full block list.
That is correct only when D has no local prefix hit.  This connector carries the
matched prefix offset in D's registration and selects the corresponding source
suffix on P before delegating the actual NIXL WRITE to upstream.

This is deliberately narrower than vLLM's bidirectional NIXL protocol: the
prefiller uses its own ordinary prefix cache (and recomputes after eviction),
while the decoder independently retains disposable prefix blocks.  It preserves
finite request lifetimes and never treats a session id as proof of KV identity.
"""

from __future__ import annotations

import os
from time import monotonic
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

_LOG_CONNECTOR_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _delta_registration_fields(
    *,
    remote_prompt_tokens: int,
    external_tokens: int,
    block_size: int,
) -> dict[str, int]:
    """Build the positional contract sent from D to P.

    vLLM's local prefix lookup returns only complete cache blocks, so the
    matched token count must be block aligned.  Failing loudly here is safer
    than writing KV from the wrong token positions into D's suffix blocks.
    """
    if not 0 < external_tokens <= remote_prompt_tokens:
        raise ValueError(
            f"external_tokens must be in (0, remote_prompt_tokens], got {external_tokens=} and {remote_prompt_tokens=}"
        )
    matched_prefix_tokens = remote_prompt_tokens - external_tokens
    if matched_prefix_tokens % block_size:
        raise ValueError(
            f"NIXL delta push requires a block-aligned local prefix hit: {matched_prefix_tokens=} {block_size=}"
        )
    return {
        "matched_prefix_tokens": matched_prefix_tokens,
        "source_block_offset": matched_prefix_tokens // block_size,
        "decode_block_size": block_size,
        "remote_prompt_tokens": remote_prompt_tokens,
    }


def _select_delta_source_blocks(
    source_block_ids: BlockIds,
    destination_block_ids: BlockIds,
    *,
    source_block_offset: int,
    source_block_size: int,
    decode_block_size: int,
) -> BlockIds:
    """Select P blocks corresponding exactly to D's cache-miss suffix.

    The current Thinker P/D deployment uses identical logical block sizes and
    full-attention cache groups on both sides.  Explicitly reject a different
    layout instead of falling back to upstream's front-truncation, which could
    silently install KV for the wrong token positions.
    """
    if source_block_size != decode_block_size:
        raise ValueError(
            "NIXL delta push currently requires identical P/D block sizes: "
            f"P={source_block_size}, D={decode_block_size}"
        )
    if source_block_offset < 0:
        raise ValueError(f"source_block_offset must be non-negative, got {source_block_offset}")
    if len(source_block_ids) != len(destination_block_ids):
        raise ValueError(f"P/D KV cache group count differs: P={len(source_block_ids)}, D={len(destination_block_ids)}")

    selected: list[list[int]] = []
    for group_idx, (source_group, destination_group) in enumerate(zip(source_block_ids, destination_block_ids)):
        count = len(destination_group)
        end = source_block_offset + count
        if end > len(source_group):
            raise ValueError(
                "D requested KV blocks outside P's completed prompt: "
                f"group={group_idx}, offset={source_block_offset}, "
                f"count={count}, available={len(source_group)}"
            )
        selected.append(list(source_group[source_block_offset:end]))
    return tuple(selected)


class NixlDeltaPushConnectorScheduler(NixlPushConnectorScheduler):
    """Attach D's exact local-prefix offset to each push registration."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if self._is_hma_required:
            raise NotImplementedError("NixlDeltaPushConnector currently supports full-attention cache groups only.")

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        is_decode_registration = bool(params is not None and params.get("do_remote_prefill"))
        fields: dict[str, int] | None = None
        if is_decode_registration and num_external_tokens > 0:
            prompt_tokens = len(request.prompt_token_ids or ())
            remote_prompt_tokens = self._get_remote_prefill_token_count(prompt_tokens)
            fields = _delta_registration_fields(
                remote_prompt_tokens=remote_prompt_tokens,
                external_tokens=num_external_tokens,
                block_size=self.block_size,
            )

        super().update_state_after_alloc(request, blocks, num_external_tokens)

        if fields is not None:
            registration = self._push_pending_registrations.get(request.request_id)
            if registration is None:
                raise RuntimeError(f"NIXL delta push registration was not staged for {request.request_id}")
            registration.update(fields)
        elif is_decode_registration:
            # A full D prefix hit needs no WRITE, but P may already be running
            # because push mode dispatches both legs concurrently. Send an
            # empty registration so P can release its leased blocks promptly.
            assert params is not None
            prompt_tokens = len(request.prompt_token_ids or ())
            remote_prompt_tokens = self._get_remote_prefill_token_count(prompt_tokens)
            self._push_pending_registrations[request.request_id] = {
                "request_id": request.request_id,
                "decode_engine_id": self.engine_id,
                "decode_host": self.side_channel_host,
                "decode_port": self.side_channel_port,
                "decode_tp_size": (self.vllm_config.parallel_config.tensor_parallel_size),
                "local_block_ids": (),
                "remote_engine_id": params["remote_engine_id"],
                "remote_host": params["remote_host"],
                "remote_port": params["remote_port"],
                "remote_tp_size": params["tp_size"],
                "remote_pp_size": params.get("pp_size", 1),
                "matched_prefix_tokens": remote_prompt_tokens,
                "source_block_offset": remote_prompt_tokens // self.block_size,
                "decode_block_size": self.block_size,
                "remote_prompt_tokens": remote_prompt_tokens,
            }
            params["do_remote_prefill"] = False


class NixlDeltaPushConnectorWorker(NixlPushConnectorWorker):
    """Select the positional KV suffix before upstream submits the WRITE."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delta_load_started: dict[str, float] = {}
        # Cache-only D imports are driven by an EngineCore control operation,
        # not by ModelRunner.execute_model().  Keep their completions out of
        # the ordinary connector output or Scheduler would mistake them for
        # inference requests.
        self._direct_cache_sync_req_ids: set[str] = set()
        self._direct_cache_sync_finished: set[str] = set()
        self._deferred_regular_finished_sending: set[str] = set()
        self._deferred_regular_finished_recving: set[str] = set()

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Timestamp D registration through completed KV installation."""
        now = monotonic()
        for req_id in metadata.reqs_to_recv:
            self._delta_load_started.setdefault(req_id, now)
        super().start_load_kv(metadata)

    def _partition_finished(self) -> None:
        # Unit tests and rolling upgrades may construct this worker without
        # running the newest __init__; initialize the routing sets lazily.
        if not hasattr(self, "_direct_cache_sync_req_ids"):
            self._direct_cache_sync_req_ids = set()
            self._direct_cache_sync_finished = set()
            self._deferred_regular_finished_sending = set()
            self._deferred_regular_finished_recving = set()
        done_sending, done_recving = super().get_finished()
        now = monotonic()
        for req_id in done_recving:
            started = self._delta_load_started.pop(req_id, None)
            if started is not None:
                logger.info(
                    "[nixl-delta-load] request=%s transfer_load_ms=%.3f",
                    req_id,
                    (now - started) * 1000.0,
                )
        direct_done = done_recving & self._direct_cache_sync_req_ids
        if direct_done:
            self._direct_cache_sync_req_ids.difference_update(direct_done)
            self._direct_cache_sync_finished.update(direct_done)
        self._deferred_regular_finished_sending.update(done_sending)
        self._deferred_regular_finished_recving.update(done_recving - direct_done)

    def start_direct_cache_sync(self, metadata: NixlConnectorMetadata) -> None:
        """Start cache-only D imports without a model-runner invocation."""
        self._direct_cache_sync_req_ids.update(metadata.reqs_to_recv)
        self.start_load_kv(metadata)

    def poll_direct_cache_sync(self) -> set[str]:
        """Return only cache-only completions, preserving normal completions."""
        self._partition_finished()
        finished = set(self._direct_cache_sync_finished)
        self._direct_cache_sync_finished.clear()
        return finished

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Hide cache-only completions from the inference scheduler."""
        self._partition_finished()
        done_sending = set(self._deferred_regular_finished_sending)
        done_recving = set(self._deferred_regular_finished_recving)
        self._deferred_regular_finished_sending.clear()
        self._deferred_regular_finished_recving.clear()
        return done_sending, done_recving

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        diag_start = monotonic() if _LOG_CONNECTOR_DIAG else 0.0
        required = (
            "source_block_offset",
            "matched_prefix_tokens",
            "decode_block_size",
            "remote_prompt_tokens",
        )
        missing = [key for key in required if key not in registration_data]
        if missing:
            raise ValueError("NIXL delta push registration is missing positional metadata: " + ", ".join(missing))

        source_groups = self._as_grouped_block_ids(local_block_ids)
        destination_groups = self._as_grouped_block_ids(registration_data["local_block_ids"])
        if not any(destination_groups):
            # No data is needed on a full D prefix hit. Materialize an empty
            # P-side transfer entry; the normal completion poll will report it
            # as done and release P's request-scoped block lease.
            with self._sending_transfers_lock:
                self._sending_transfers[request_id] = []
            logger.info(
                "[nixl-delta-push] request=%s prefix_tokens=%d full_hit=true",
                request_id,
                int(registration_data["matched_prefix_tokens"]),
            )
            return
        selected_source = _select_delta_source_blocks(
            source_groups,
            destination_groups,
            source_block_offset=int(registration_data["source_block_offset"]),
            source_block_size=self.block_size,
            decode_block_size=int(registration_data["decode_block_size"]),
        )
        diag_selected = monotonic() if _LOG_CONNECTOR_DIAG else 0.0
        total_source_blocks = sum(len(group) for group in source_groups)
        delta_source_blocks = sum(len(group) for group in selected_source)
        logger.info(
            "[nixl-delta-push] request=%s prefix_tokens=%d source_blocks=%d delta_blocks=%d",
            request_id,
            int(registration_data["matched_prefix_tokens"]),
            total_source_blocks,
            delta_source_blocks,
        )
        super()._do_start_push_kv(
            request_id,
            selected_source,
            registration_data,
        )
        if _LOG_CONNECTOR_DIAG:
            diag_end = monotonic()
            logger.info(
                "[NIXL-PUSH-DIAG] request=%s source_blocks=%d delta_blocks=%d "
                "select_ms=%.3f submit_ms=%.3f total_ms=%.3f",
                request_id,
                total_source_blocks,
                delta_source_blocks,
                (diag_selected - diag_start) * 1000.0,
                (diag_end - diag_selected) * 1000.0,
                (diag_end - diag_start) * 1000.0,
            )


class NixlDeltaPushConnector(NixlPushConnector):
    """External vLLM connector combining delta scheduler and worker."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # Skip NixlPushConnector.__init__, which hard-codes the upstream
        # scheduler/worker classes, while retaining its worker entry point.
        NixlBaseConnector.__init__(self, vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlDeltaPushConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = NixlDeltaPushConnectorWorker(vllm_config, self.engine_id, kv_cache_config)
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def start_load_kv(self, forward_context, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)


__all__ = [
    "NixlDeltaPushConnector",
    "NixlDeltaPushConnectorScheduler",
    "NixlDeltaPushConnectorWorker",
]
