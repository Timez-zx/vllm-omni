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
import queue
import threading
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

_DIRECT_COMPLETION_WAKE_TIMEOUT_S = 0.001

_KV_TRANSFER_SELECTED_BLOCKS = "kv_transfer_selected_blocks"
_KV_TRANSFER_SELECTED_TOKENS = "kv_transfer_selected_tokens"
_KV_TRANSFER_SELECTED_BYTES = "kv_transfer_selected_bytes"
_KV_TRANSFER_WRITE_SUBMIT_TO_D_READY_MS = (
    "kv_transfer_write_submit_to_d_ready_ms"
)
_KV_TRANSFER_EVIDENCE_FIELDS = (
    _KV_TRANSFER_SELECTED_BLOCKS,
    _KV_TRANSFER_SELECTED_TOKENS,
    _KV_TRANSFER_SELECTED_BYTES,
    _KV_TRANSFER_WRITE_SUBMIT_TO_D_READY_MS,
)

_LOG_CONNECTOR_DIAG = os.environ.get("VLLM_OMNI_LOG_HANDOFF_DIAG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _is_formal_handoff_diagnostic(request_id: str) -> bool:
    """Keep added hot-path diagnostics off high-frequency arrival requests."""
    return "-warm-" not in request_id


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


def _delta_transfer_evidence(
    destination_block_ids: BlockIds | None,
    *,
    external_tokens: int,
    bytes_per_block_by_group: tuple[int, ...] | None,
) -> dict[str, int | float]:
    """Describe one D registration without inspecting tensors.

    ``external_tokens`` is the semantic KV suffix requested by D. The block
    count and byte count describe the whole block-granular WRITE, so they may
    include padding after the last external token. Byte counts are exact only
    when the scheduler can map every registered cache group to its static KV
    page size; otherwise they fail closed to ``-1``.
    """
    if destination_block_ids is None:
        group_counts: tuple[int, ...] | None = None
    elif destination_block_ids and isinstance(
        destination_block_ids[0], (int, bool)
    ):
        # Some upstream single-group paths collapse BlockIds to a flat list.
        group_counts = (len(destination_block_ids),)
    else:
        try:
            group_counts = tuple(len(group) for group in destination_block_ids)
        except TypeError:
            group_counts = None

    selected_blocks = sum(group_counts) if group_counts is not None else -1
    selected_bytes = -1
    if selected_blocks == 0 and external_tokens == 0:
        selected_bytes = 0
    elif (
        group_counts is not None
        and bytes_per_block_by_group is not None
        and len(group_counts) == len(bytes_per_block_by_group)
        and all(value >= 0 for value in bytes_per_block_by_group)
    ):
        selected_bytes = sum(
            count * bytes_per_block
            for count, bytes_per_block in zip(
                group_counts,
                bytes_per_block_by_group,
            )
        )

    return {
        _KV_TRANSFER_SELECTED_BLOCKS: selected_blocks,
        _KV_TRANSFER_SELECTED_TOKENS: (
            int(external_tokens) if int(external_tokens) >= 0 else -1
        ),
        _KV_TRANSFER_SELECTED_BYTES: selected_bytes,
        # P's WRITE submission timestamp lives in another process and is not
        # carried by the existing completion notification. Do not substitute
        # registration or D-service time for this interval.
        _KV_TRANSFER_WRITE_SUBMIT_TO_D_READY_MS: -1.0,
    }


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

    def _bytes_per_block_by_group(self) -> tuple[int, ...] | None:
        """Return request-level bytes per logical block for each cache group.

        KV specs are per tensor-parallel rank. Multiplying by D TP size gives
        the aggregate bytes written for one physical request. This connector
        rejects hybrid cache groups, so the static full-attention page sizes
        are the exact NIXL WRITE sizes and require no device inspection.
        """
        kv_cache_config = getattr(self, "kv_cache_config", None)
        vllm_config = getattr(self, "vllm_config", None)
        groups = getattr(kv_cache_config, "kv_cache_groups", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        tp_size = getattr(parallel_config, "tensor_parallel_size", None)
        if not isinstance(groups, list | tuple) or not isinstance(tp_size, int):
            return None
        if tp_size <= 0:
            return None

        result: list[int] = []
        for group in groups:
            layer_names = getattr(group, "layer_names", None)
            spec = getattr(group, "kv_cache_spec", None)
            page_size_bytes = getattr(spec, "page_size_bytes", None)
            if (
                not isinstance(layer_names, list | tuple)
                or not layer_names
                or not isinstance(page_size_bytes, int)
                or page_size_bytes < 0
            ):
                return None
            result.append(page_size_bytes * len(layer_names) * tp_size)
        return tuple(result)

    def take_immediate_push_metadata(
        self,
    ) -> NixlConnectorMetadata | None:
        """Detach newly finished P blocks from the next model batch.

        Upstream NIXL normally carries this state in the following
        ``SchedulerOutput``.  For a streaming P/D stage that means a completed
        segment cannot start its WRITE until an unrelated next P batch has
        finished input preparation.  Return a self-contained worker message so
        EngineCore can wake the push writer immediately after processing the
        segment output.

        ``_finished_request_blocks`` deliberately remains scheduler-owned: it
        is the block lease and is released only after the worker reports
        ``finished_sending`` (or the lease expires).
        """
        if not self._newly_finished_push_blocks:
            return None

        blocks = dict(self._newly_finished_push_blocks)
        missing_leases = set(blocks).difference(self._reqs_need_send)
        if missing_leases:
            raise RuntimeError(
                "Finished P/D blocks are missing worker lease metadata: "
                f"{sorted(missing_leases)}"
            )
        self._newly_finished_push_blocks.clear()

        metadata = NixlConnectorMetadata()
        metadata.push_finished_blocks = blocks
        # A normal model step has already told the worker these requests are in
        # process. Repeating the id is idempotent and also makes this control
        # path safe if a connector implementation changes that ordering.
        metadata.reqs_in_batch = set(blocks)
        for request_id in blocks:
            metadata.reqs_to_send[request_id] = self._reqs_need_send.pop(
                request_id
            )
        return metadata

    def _attach_transfer_evidence(
        self,
        request: Request,
        registration: dict[str, Any],
        *,
        external_tokens: int,
    ) -> None:
        """Attach immutable request-scoped evidence to D and its PUSH_REG."""
        params = request.kv_transfer_params
        if params is None:
            return
        evidence = _delta_transfer_evidence(
            registration.get("local_block_ids"),
            external_tokens=external_tokens,
            bytes_per_block_by_group=self._bytes_per_block_by_group(),
        )
        registration.update(evidence)
        params.update(evidence)

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
            remote_prompt_tokens = int(
                params.get(
                    "remote_prompt_tokens",
                    self._get_remote_prefill_token_count(prompt_tokens),
                )
            )
            fields = _delta_registration_fields(
                remote_prompt_tokens=remote_prompt_tokens,
                external_tokens=num_external_tokens,
                block_size=self.block_size,
            )

        if is_decode_registration:
            # Ordinary vLLM admission records this split before calling the
            # connector.  Direct P/D cache-sync admission calls the connector
            # itself, so preserve the same PrefillStats contract here as well.
            # The matched prefix is local on D; the remaining remote prefix is
            # supplied by the P -> D transfer.
            prompt_tokens = len(request.prompt_token_ids or ())
            remote_prompt_tokens = int(
                params.get(
                    "remote_prompt_tokens",
                    self._get_remote_prefill_token_count(prompt_tokens),
                )
            )
            prefill_stats = getattr(request, "prefill_stats", None)
            if (
                prefill_stats is not None
                and int(getattr(prefill_stats, "num_prompt_tokens", 0) or 0)
                == 0
            ):
                prefill_stats.set(
                    num_prompt_tokens=int(request.num_prompt_tokens),
                    num_local_cached_tokens=(
                        remote_prompt_tokens - num_external_tokens
                    ),
                    num_external_cached_tokens=num_external_tokens,
                )

        if _LOG_CONNECTOR_DIAG and is_decode_registration:
            logger.info(
                "[NIXL-D-TRACE] event=after-alloc request=%s "
                "computed=%d external=%d prompt=%d remote_prompt=%s",
                request.request_id,
                int(getattr(request, "num_computed_tokens", 0)),
                num_external_tokens,
                len(request.prompt_token_ids or ()),
                params.get("remote_prompt_tokens") if params else None,
            )

        super().update_state_after_alloc(request, blocks, num_external_tokens)

        if fields is not None:
            registration = self._push_pending_registrations.get(request.request_id)
            if registration is None:
                raise RuntimeError(f"NIXL delta push registration was not staged for {request.request_id}")
            registration.update(fields)
            self._attach_transfer_evidence(
                request,
                registration,
                external_tokens=num_external_tokens,
            )
        elif is_decode_registration:
            # A full D prefix hit needs no WRITE, but P may already be running
            # because push mode dispatches both legs concurrently. Send an
            # empty registration so P can release its leased blocks promptly.
            assert params is not None
            prompt_tokens = len(request.prompt_token_ids or ())
            remote_prompt_tokens = int(
                params.get(
                    "remote_prompt_tokens",
                    self._get_remote_prefill_token_count(prompt_tokens),
                )
            )
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
            self._attach_transfer_evidence(
                request,
                self._push_pending_registrations[request.request_id],
                external_tokens=0,
            )
            params["do_remote_prefill"] = False

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Import P's exact prefix while leaving its sampled token to D."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_prefill"):
            explicit = params.get("remote_prompt_tokens")
            if explicit is not None:
                count = max(0, int(explicit) - num_computed_tokens)
                if _LOG_CONNECTOR_DIAG:
                    logger.info(
                        "[NIXL-D-TRACE] event=match request=%s computed=%d "
                        "remote_prompt=%d external=%d",
                        request.request_id,
                        num_computed_tokens,
                        int(explicit),
                        count,
                    )
                # The explicit P boundary is authoritative even when D's
                # local prefix cache already covers all of it.  Falling back
                # to upstream here makes it infer the boundary from the full
                # D prompt (which also contains P's sampled token), inventing
                # one external token and a non-block-aligned suffix.
                return count, count > 0
        return super().get_num_new_matched_tokens(
            request,
            num_computed_tokens,
        )

    def request_finished(
        self,
        request: Request,
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free_blocks, output_params = super().request_finished(
            request,
            block_ids,
        )
        if (
            _LOG_CONNECTOR_DIAG
            and _is_formal_handoff_diagnostic(request.request_id)
            and request.request_id in self._newly_finished_push_blocks
        ):
            logger.info(
                "[NIXL-P-TRACE] event=finished-blocks-staged request=%s mono=%.6f",
                request.request_id,
                monotonic(),
            )
        params = request.kv_transfer_params
        if isinstance(params, dict) and any(
            name in params for name in _KV_TRANSFER_EVIDENCE_FIELDS
        ):
            output_params = dict(output_params or {})
            for name in _KV_TRANSFER_EVIDENCE_FIELDS:
                output_params[name] = params.get(name, -1)
        return delay_free_blocks, output_params

class _TimestampedNotifQueue(queue.Queue[bytes]):
    """Record when the NIXL writer first exposes a completion notification."""

    def __init__(self, callback: Any) -> None:
        super().__init__()
        self._callback = callback

    def put(
        self,
        item: bytes,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().put(item, block=block, timeout=timeout)
        # Publish the wake event only after the notification is drainable;
        # otherwise the waiting Core thread can win another empty-queue race.
        self._callback(item)


class NixlDeltaPushConnectorWorker(NixlPushConnectorWorker):
    """Select the positional KV suffix before upstream submits the WRITE."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delta_load_started: dict[str, float] = {}
        self._delta_registration_enqueued: dict[str, float] = {}
        # Cache-only D imports are driven by an EngineCore control operation,
        # not by ModelRunner.execute_model().  Keep their completions out of
        # the ordinary connector output or Scheduler would mistake them for
        # inference requests.
        self._direct_cache_sync_req_ids: set[str] = set()
        self._direct_cache_sync_finished: set[str] = set()
        self._deferred_regular_finished_sending: set[str] = set()
        self._deferred_regular_finished_recving: set[str] = set()
        self._completion_notif_seen: dict[str, float] = {}
        self._completion_notif_lock = threading.Lock()
        # The inherited writer parks when it has no unmatched P blocks.  A
        # get_finished() call wakes it and may race its notification drain.
        # The callback/event below lets the same Core poll wait at most 1 ms
        # and retry, avoiding both an extra model step and continuous polling.
        self._completion_notif_available = threading.Event()
        self._pending_completion_notifs = _TimestampedNotifQueue(
            self._record_completion_notification
        )

    def _record_completion_notification(self, notification: bytes) -> None:
        if notification.startswith(b"HB:"):
            return
        # The event is part of the transfer protocol: publish it even when
        # diagnostics are disabled.  Decoding request identity and retaining
        # timestamps, however, are observability-only work and must not tax
        # every cache handoff in a formal capacity run.
        self._completion_notif_available.set()
        if not _LOG_CONNECTOR_DIAG:
            return
        try:
            message = notification.decode("utf-8")
            request_id, _ = message.rsplit(":", 1)
        except Exception:
            return
        now = monotonic()
        if _is_formal_handoff_diagnostic(request_id):
            with self._completion_notif_lock:
                self._completion_notif_seen.setdefault(request_id, now)
            logger.info(
                "[NIXL-D-TRACE] event=completion-forwarded request=%s mono=%.6f",
                request_id,
                now,
            )

    def _ensure_direct_progress_state(self) -> None:
        """Initialize fields lazily for tests and rolling worker upgrades."""
        if not hasattr(self, "_completion_notif_seen"):
            self._completion_notif_seen = {}
            self._completion_notif_lock = threading.Lock()
        if not hasattr(self, "_completion_notif_available"):
            self._completion_notif_available = threading.Event()

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Timestamp D registration through completed KV installation."""
        if not hasattr(self, "_delta_registration_enqueued"):
            self._delta_registration_enqueued = {}
        if _LOG_CONNECTOR_DIAG:
            now = monotonic()
            for req_id in metadata.reqs_to_recv:
                self._delta_load_started.setdefault(req_id, now)
                self._delta_registration_enqueued.setdefault(req_id, now)
                logger.info(
                    "[NIXL-D-TRACE] event=registration-enqueued request=%s mono=%.6f",
                    req_id,
                    now,
                )
            for req_id in metadata.push_finished_blocks:
                if not _is_formal_handoff_diagnostic(req_id):
                    continue
                logger.info(
                    "[NIXL-P-TRACE] event=finished-metadata-received "
                    "request=%s mono=%.6f",
                    req_id,
                    now,
                )
        # A resumable P request may be cancelled after publishing a segment
        # but before its paired D request registers.  Upstream NIXL assumes a
        # request reported in reqs_not_processed cannot still own a send
        # lease and asserts otherwise.  That assumption is true for finite
        # requests, but not for this segment-boundary publication path.  Drop
        # the orphaned lease and writer-side unmatched snapshot before the
        # normal bookkeeping consumes the cancellation metadata.
        cancelled_leases = metadata.reqs_not_processed.intersection(self._reqs_to_send)
        for req_id in cancelled_leases:
            self._reqs_to_send.pop(req_id, None)
            self._evict_finished_inbox.put(req_id)
        if cancelled_leases:
            self._push_writer_wake.set()
        super().start_load_kv(metadata)

    def _do_send_reg_notif(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        super()._do_send_reg_notif(req_id, reg_data)
        if not _LOG_CONNECTOR_DIAG:
            return
        now = monotonic()
        started = self._delta_registration_enqueued.get(req_id, now)
        logger.info(
            "[NIXL-D-TRACE] event=registration-sent request=%s mono=%.6f queue_ms=%.3f",
            req_id,
            now,
            (now - started) * 1000.0,
        )

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        # This method executes on P's writer thread.  Decode only enough of
        # the inherited framing to identify the request in diagnostics; the
        # parent remains authoritative for validation and matching.
        if _LOG_CONNECTOR_DIAG:
            now = monotonic()
            try:
                import msgspec
                from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
                    PUSH_REG_NOTIF_PREFIX,
                )

                registration = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
                req_id = registration.get("request_id", "?") if isinstance(registration, dict) else "?"
            except Exception:
                req_id = "?"
            logger.info(
                "[NIXL-P-TRACE] event=registration-received request=%s mono=%.6f",
                req_id,
                now,
            )
        super()._handle_push_reg_notif(notif)

    def _partition_finished(self) -> None:
        # Unit tests and rolling upgrades may construct this worker without
        # running the newest __init__; initialize the routing sets lazily.
        if not hasattr(self, "_direct_cache_sync_req_ids"):
            self._direct_cache_sync_req_ids = set()
            self._direct_cache_sync_finished = set()
            self._deferred_regular_finished_sending = set()
            self._deferred_regular_finished_recving = set()
        self._ensure_direct_progress_state()
        done_sending, done_recving = super().get_finished()
        if _LOG_CONNECTOR_DIAG:
            now = monotonic()
            for req_id in done_recving:
                with self._completion_notif_lock:
                    forwarded = self._completion_notif_seen.pop(req_id, None)
                started = self._delta_load_started.pop(req_id, None)
                if started is not None:
                    self._delta_registration_enqueued.pop(req_id, None)
                    logger.info(
                        "[nixl-delta-load] request=%s transfer_load_ms=%.3f",
                        req_id,
                        (now - started) * 1000.0,
                    )
                    logger.info(
                        "[NIXL-D-TRACE] event=completion-observed request=%s mono=%.6f total_ms=%.3f",
                        req_id,
                        now,
                        (now - started) * 1000.0,
                    )
                if forwarded is not None:
                    logger.info(
                        "[NIXL-D-TRACE] event=completion-core-observed "
                        "request=%s mono=%.6f notify_to_core_ms=%.3f",
                        req_id,
                        now,
                        (now - forwarded) * 1000.0,
                    )
        direct_done = done_recving & self._direct_cache_sync_req_ids
        if direct_done:
            self._direct_cache_sync_req_ids.difference_update(direct_done)
            self._direct_cache_sync_finished.update(direct_done)
        self._deferred_regular_finished_sending.update(done_sending)
        self._deferred_regular_finished_recving.update(done_recving - direct_done)

    def start_direct_cache_sync(self, metadata: NixlConnectorMetadata) -> None:
        """Start cache-only D imports without a model-runner invocation."""
        self._ensure_direct_progress_state()
        self._direct_cache_sync_req_ids.update(metadata.reqs_to_recv)
        self.start_load_kv(metadata)

    def poll_direct_cache_sync(self) -> set[str]:
        """Return only cache-only completions, preserving normal completions."""
        self._ensure_direct_progress_state()
        self._completion_notif_available.clear()
        self._partition_finished()
        # Upstream get_finished() wakes the NIXL writer and immediately drains
        # its forwarded-notification queue.  If the writer was parked, that
        # first drain can win the race.  Wait for the writer's queue callback
        # and retry inside this same Core step instead of deferring activation
        # until another potentially long model batch completes.
        if (
            not self._direct_cache_sync_finished
            and self._direct_cache_sync_req_ids
            and self._completion_notif_available.wait(
                _DIRECT_COMPLETION_WAKE_TIMEOUT_S
            )
        ):
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
            evidence_blocks = registration_data.get(
                _KV_TRANSFER_SELECTED_BLOCKS,
                -1,
            )
            if int(evidence_blocks) not in (-1, 0):
                raise ValueError(
                    "NIXL delta evidence disagrees with an empty WRITE: "
                    f"request={request_id} selected_blocks={evidence_blocks}"
                )
            # No data is needed on a full D prefix hit. Materialize an empty
            # P-side transfer entry; the normal completion poll will report it
            # as done and release P's request-scoped block lease.
            with self._sending_transfers_lock:
                self._sending_transfers[request_id] = []
            if _LOG_CONNECTOR_DIAG:
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
        selected_block_count = sum(len(group) for group in selected_source)
        evidence_blocks = int(
            registration_data.get(_KV_TRANSFER_SELECTED_BLOCKS, -1)
        )
        if evidence_blocks >= 0 and evidence_blocks != selected_block_count:
            raise ValueError(
                "NIXL delta evidence disagrees with P's selected source "
                f"blocks: request={request_id} evidence={evidence_blocks} "
                f"selected={selected_block_count}"
            )
        diag_selected = monotonic() if _LOG_CONNECTOR_DIAG else 0.0
        if _LOG_CONNECTOR_DIAG:
            total_source_blocks = sum(len(group) for group in source_groups)
            delta_source_blocks = selected_block_count
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
                "[NIXL-P-TRACE] event=write-submitted request=%s mono=%.6f "
                "delta_blocks=%d submit_ms=%.3f",
                request_id,
                diag_end,
                delta_source_blocks,
                (diag_end - diag_selected) * 1000.0,
            )
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

    def take_immediate_push_metadata(
        self,
    ) -> NixlConnectorMetadata | None:
        """Return P completions that should bypass the next model batch."""
        scheduler = self.connector_scheduler
        assert isinstance(scheduler, NixlDeltaPushConnectorScheduler)
        return scheduler.take_immediate_push_metadata()


__all__ = [
    "NixlDeltaPushConnector",
    "NixlDeltaPushConnectorScheduler",
    "NixlDeltaPushConnectorWorker",
]
