from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.minicpmo45.input import (
    MiniCPMO45PcmAppendBuffer,
)

_DEFAULT_MAX_PENDING_VISION_PREENCODES_PER_SESSION = 0


def _max_pending_vision_preencodes_per_session() -> int | None:
    """Return the arrival-vision lookahead, or ``None`` when unbounded.

    Native-duplex capacity experiments must submit every arriving camera frame
    to the independent encoder.  A positive environment override retains the
    bounded speculative policy for deployments that prefer memory protection.
    """
    raw = os.environ.get(
        "MINICPMO45_MAX_PENDING_VISION_PREENCODES_PER_SESSION",
        str(_DEFAULT_MAX_PENDING_VISION_PREENCODES_PER_SESSION),
    )
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return _DEFAULT_MAX_PENDING_VISION_PREENCODES_PER_SESSION


@dataclass(slots=True)
class MiniCPMO45ServingSessionState:
    """Mutable serving state owned by one MiniCPM duplex session."""

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None
    vision_preencode_tasks: dict[str, tuple[int, asyncio.Task[bool]]] = field(default_factory=dict)
    audio_preencode_seq: int = 0
    audio_preencode_tasks: dict[str, tuple[int, int, asyncio.Task[bool]]] = field(
        default_factory=dict
    )

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None

    def can_start_vision_preencode(
        self,
        *,
        frame_count: int,
        epoch: int,
    ) -> bool:
        """Bound speculative work that is ahead of the formal append chain."""
        stale_ids = [
            preencode_id
            for preencode_id, (task_epoch, _task) in self.vision_preencode_tasks.items()
            if task_epoch != epoch
        ]
        stale_tasks = {self.vision_preencode_tasks[preencode_id][1] for preencode_id in stale_ids}
        for preencode_id in stale_ids:
            self.vision_preencode_tasks.pop(preencode_id, None)
        for task in stale_tasks:
            if not task.done():
                task.cancel()
        limit = _max_pending_vision_preencodes_per_session()
        return frame_count > 0 and (
            limit is None
            or len(self.vision_preencode_tasks) + frame_count <= limit
        )

    def track_vision_preencode(
        self,
        preencode_ids: list[str],
        *,
        epoch: int,
        task: asyncio.Task[bool],
    ) -> bool:
        # Completed tasks still own cached GPU embeddings until the matching
        # formal append reaches the head of the per-session wire-order chain.
        # Do not remove them merely because the RPC has completed: doing so
        # allowed speculative work to run arbitrarily far ahead, evict its own
        # cache entries, and force a second request-local vision encode.
        if not self.can_start_vision_preencode(
            frame_count=len(preencode_ids),
            epoch=epoch,
        ):
            return False
        for preencode_id in preencode_ids:
            self.vision_preencode_tasks[preencode_id] = (epoch, task)
        return True

    def pop_vision_preencode_tasks(
        self,
        preencode_ids: list[str],
        *,
        epoch: int,
    ) -> list[asyncio.Task[bool]]:
        tasks: list[asyncio.Task[bool]] = []
        seen: set[int] = set()
        for preencode_id in preencode_ids:
            item = self.vision_preencode_tasks.pop(preencode_id, None)
            if item is None or item[0] != epoch:
                continue
            task = item[1]
            if id(task) not in seen:
                seen.add(id(task))
                tasks.append(task)
        return tasks

    def cancel_vision_preencode_tasks(self) -> None:
        tasks = {task for _, task in self.vision_preencode_tasks.values()}
        self.vision_preencode_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()

    def allocate_audio_preencode(self, *, epoch: int) -> tuple[int, str]:
        """Allocate a stable identity for one complete PCM model unit.

        The sequence is monotonic for the lifetime of the serving session,
        including barge-in epochs and resumable WebSocket attachments.  The
        epoch remains part of the cache fence; the sequence is an ordering
        aid, not a replacement for that fence.
        """
        stale_ids = [
            preencode_id
            for preencode_id, (task_epoch, _seq, _task) in self.audio_preencode_tasks.items()
            if task_epoch != epoch
        ]
        stale_tasks = {
            self.audio_preencode_tasks[preencode_id][2]
            for preencode_id in stale_ids
        }
        for preencode_id in stale_ids:
            self.audio_preencode_tasks.pop(preencode_id, None)
        for task in stale_tasks:
            if not task.done():
                task.cancel()
        self.audio_preencode_seq += 1
        return self.audio_preencode_seq, uuid.uuid4().hex

    def track_audio_preencode(
        self,
        preencode_id: str,
        *,
        seq: int,
        epoch: int,
        task: asyncio.Task[bool],
    ) -> None:
        self.audio_preencode_tasks[preencode_id] = (epoch, seq, task)

    def pop_audio_preencode_task(
        self,
        preencode_id: str,
        *,
        seq: int,
        epoch: int,
    ) -> asyncio.Task[bool] | None:
        item = self.audio_preencode_tasks.pop(preencode_id, None)
        if item is None:
            return None
        task_epoch, task_seq, task = item
        if task_epoch == epoch and task_seq == seq:
            return task
        if not task.done():
            task.cancel()
        return None

    def cancel_audio_preencode_tasks(self) -> None:
        tasks = {task for _, _, task in self.audio_preencode_tasks.values()}
        self.audio_preencode_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()

    def cancel_preencode_tasks(self) -> None:
        self.cancel_vision_preencode_tasks()
        self.cancel_audio_preencode_tasks()
