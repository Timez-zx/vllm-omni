# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Application-owned identity and lifecycle state for realtime video sessions.

The engine streaming API preserves submission order, but it does not currently
echo application turn identifiers on every stage output.  This module keeps the
ordered part explicit and typed: a request owns a ledger of submitted segments,
and each segment carries the session incarnation, engine-request epoch, turn,
and segment identifiers that the websocket protocol exposes to clients.

It deliberately contains no scheduler, KV-cache, or model-execution policy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class SegmentKind(str, Enum):
    """Why an application segment was submitted to a persistent request."""

    TURN = "turn"
    SHADOW_SEED = "shadow_seed"


@dataclass(frozen=True, slots=True)
class SegmentIdentity:
    """Stable application identity for one ordered engine segment."""

    session_id: str
    incarnation: str
    epoch: int
    segment_id: int
    kind: SegmentKind
    turn_id: int | None = None

    def event_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "incarnation": self.incarnation,
            "epoch": self.epoch,
            "segment_id": self.segment_id,
            "segment_kind": self.kind.value,
        }
        if self.turn_id is not None:
            fields["turn_id"] = self.turn_id
        return fields


@dataclass(slots=True)
class SessionIdentity:
    """Monotonic identity allocator owned by one websocket session actor."""

    session_id: str = field(default_factory=lambda: f"vs-{uuid4().hex[:16]}")
    incarnation: str = field(default_factory=lambda: uuid4().hex)
    turn_id: int = 0
    _next_epoch: int = 0
    _next_segment_id: int = 0

    def allocate_epoch(self) -> int:
        epoch = self._next_epoch
        self._next_epoch += 1
        return epoch

    def new_segment(
        self,
        *,
        epoch: int,
        kind: SegmentKind,
        turn_id: int | None = None,
    ) -> SegmentIdentity:
        segment = SegmentIdentity(
            session_id=self.session_id,
            incarnation=self.incarnation,
            epoch=epoch,
            segment_id=self._next_segment_id,
            kind=kind,
            turn_id=turn_id,
        )
        self._next_segment_id += 1
        return segment

    def advance_turn(self, expected_turn_id: int) -> None:
        if expected_turn_id != self.turn_id:
            raise RuntimeError(f"turn fence mismatch: expected {self.turn_id}, got {expected_turn_id}")
        self.turn_id += 1

    def session_fields(self, *, epoch: int | None = None) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "session_id": self.session_id,
            "incarnation": self.incarnation,
        }
        if epoch is not None:
            fields["epoch"] = epoch
        return fields


@dataclass(slots=True)
class SegmentLedger:
    """Ordered segment attribution for a single engine request epoch."""

    epoch: int
    _pending: deque[SegmentIdentity] = field(default_factory=deque)

    def submit(self, segment: SegmentIdentity) -> None:
        if segment.epoch != self.epoch:
            raise RuntimeError(f"segment epoch {segment.epoch} submitted to request epoch {self.epoch}")
        self._pending.append(segment)

    def current(self) -> SegmentIdentity | None:
        return self._pending[0] if self._pending else None

    def finish_current(self) -> SegmentIdentity | None:
        return self._pending.popleft() if self._pending else None

    def clear(self) -> list[SegmentIdentity]:
        abandoned = list(self._pending)
        self._pending.clear()
        return abandoned

    def __bool__(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)
