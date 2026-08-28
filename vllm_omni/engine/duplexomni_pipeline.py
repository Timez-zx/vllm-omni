# SPDX-License-Identifier: Apache-2.0
"""Application-level slot ordering for the DuplexOmni stage pipeline.

The vLLM engines still receive finite requests.  This coordinator only keeps
the small amount of application state needed to let Thinker(t+1) overlap with
Talker(t), while preserving Talker's codec-history order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PIPELINE_SESSION_ID = "duplexomni_pipeline_session_id"
PIPELINE_EPOCH = "duplexomni_pipeline_epoch"
PIPELINE_SLOT = "duplexomni_pipeline_slot"
PIPELINE_FINAL = "duplexomni_pipeline_final"
PIPELINE_BASE_TURNS = "duplexomni_pipeline_base_turns"


@dataclass(frozen=True)
class DuplexOmniPipelineIdentity:
    session_id: str
    epoch: int
    slot: int
    final: bool = False
    base_turns: int = 0


@dataclass
class _SessionState:
    epoch: int
    base_turns: int
    history_indices: list[int] = field(default_factory=list)
    codec_history: list[list[list[int]]] = field(default_factory=list)
    registered_slots: dict[int, str] = field(default_factory=dict)
    completed_through: int = -1


def _prompt_dict(prompt: Any) -> dict[str, Any] | None:
    if isinstance(prompt, dict):
        return prompt
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], dict):
        return prompt[0]
    return None


def _additional_information(prompt: Any) -> dict[str, Any] | None:
    item = _prompt_dict(prompt)
    info = item.get("additional_information") if item is not None else None
    return info if isinstance(info, dict) else None


def pipeline_identity_from_prompt(prompt: Any) -> DuplexOmniPipelineIdentity | None:
    info = _additional_information(prompt)
    meta = info.get("meta") if info is not None else None
    if not isinstance(meta, dict):
        return None
    session_id = meta.get(PIPELINE_SESSION_ID)
    epoch = meta.get(PIPELINE_EPOCH)
    slot = meta.get(PIPELINE_SLOT)
    if not isinstance(session_id, str) or not session_id:
        return None
    if not isinstance(epoch, int) or epoch < 0 or not isinstance(slot, int) or slot < 0:
        raise ValueError("DuplexOmni pipeline epoch and slot must be non-negative integers")
    base_turns = meta.get(PIPELINE_BASE_TURNS, 0)
    if not isinstance(base_turns, int) or base_turns < 0:
        raise ValueError("DuplexOmni pipeline base-turn count must be a non-negative integer")
    return DuplexOmniPipelineIdentity(
        session_id=session_id,
        epoch=epoch,
        slot=slot,
        final=bool(meta.get(PIPELINE_FINAL, False)),
        base_turns=base_turns,
    )


def _normalize_codec_turns(value: Any) -> list[list[list[int]]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("DuplexOmni codec history must be a list or tensor")
    result: list[list[list[int]]] = []
    for turn in value:
        if hasattr(turn, "detach"):
            turn = turn.detach().cpu().tolist()
        if not isinstance(turn, (list, tuple)) or len(turn) != 16:
            raise ValueError("DuplexOmni codec turn must contain 16 codebooks")
        normalized_turn: list[list[int]] = []
        for codebook in turn:
            if hasattr(codebook, "detach"):
                codebook = codebook.detach().cpu().tolist()
            if not isinstance(codebook, (list, tuple)) or len(codebook) != 6:
                raise ValueError("DuplexOmni codec codebook must contain six frames")
            normalized = [int(token) for token in codebook]
            if any(token < 0 or token >= 2048 for token in normalized):
                raise ValueError("DuplexOmni codec id must be in [0, 2048)")
            normalized_turn.append(normalized)
        result.append(normalized_turn)
    return result


def _seed_history(prompt: Any, base_turns: int) -> tuple[list[int], list[list[list[int]]]]:
    info = _additional_information(prompt) or {}
    raw_codes = info.get("codes", {}).get("ref") if isinstance(info.get("codes"), dict) else None
    codes = _normalize_codec_turns(raw_codes)
    raw_indices = info.get("ids", {}).get("duplex_history_indices") if isinstance(info.get("ids"), dict) else None
    indices = list(range(len(codes))) if raw_indices is None else [int(index) for index in raw_indices]
    if len(indices) != len(codes):
        raise ValueError("DuplexOmni base history index/codec count mismatch")
    if any(index < 0 or index >= base_turns for index in indices):
        raise ValueError("DuplexOmni base history index is outside the retained prompt")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise ValueError("DuplexOmni base history indices must be strictly increasing")
    return indices, codes


def inject_codec_history(
    prompt: Any,
    *,
    history_indices: list[int],
    codec_history: list[list[list[int]]],
) -> Any:
    """Return a shallow prompt copy carrying the authoritative Talker history."""
    if len(history_indices) != len(codec_history):
        raise ValueError("DuplexOmni pipeline history index/codec count mismatch")
    item = _prompt_dict(prompt)
    if item is None:
        raise TypeError("DuplexOmni pipeline requires one dictionary prompt")
    updated = dict(item)
    info = dict(updated.get("additional_information") or {})
    codes = dict(info.get("codes") or {})
    ids = dict(info.get("ids") or {})
    codes["ref"] = codec_history
    ids["duplex_history_indices"] = history_indices
    info["codes"] = codes
    info["ids"] = ids
    updated["additional_information"] = info
    return [updated] if isinstance(prompt, list) else updated


class DuplexOmniPipelineCoordinator:
    """Order Talker histories without serializing the preceding Thinker."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionState] = {}
        self._requests: dict[str, DuplexOmniPipelineIdentity] = {}

    def register(
        self,
        request_id: str,
        identity: DuplexOmniPipelineIdentity,
        prompt: Any,
    ) -> None:
        state = self._sessions.get(identity.session_id)
        if state is None or identity.epoch > state.epoch:
            base_indices, base_codes = _seed_history(prompt, identity.base_turns)
            state = _SessionState(
                epoch=identity.epoch,
                base_turns=identity.base_turns,
                history_indices=base_indices,
                codec_history=base_codes,
            )
            self._sessions[identity.session_id] = state
        elif identity.epoch < state.epoch:
            raise ValueError("stale DuplexOmni pipeline epoch")
        elif identity.base_turns != state.base_turns:
            raise ValueError("DuplexOmni pipeline base-turn count changed within an epoch")

        existing = state.registered_slots.get(identity.slot)
        if existing is not None and existing != request_id:
            raise ValueError(f"DuplexOmni pipeline slot {identity.slot} was submitted twice")
        if identity.slot > 0 and identity.slot - 1 not in state.registered_slots:
            raise ValueError(f"DuplexOmni pipeline slot {identity.slot} has no registered predecessor")
        state.registered_slots[identity.slot] = request_id
        self._requests[request_id] = identity

    def identity(self, request_id: str) -> DuplexOmniPipelineIdentity | None:
        return self._requests.get(request_id)

    def talker_ready(self, identity: DuplexOmniPipelineIdentity) -> bool:
        state = self._state(identity)
        return state.completed_through >= identity.slot - 1

    def history_for(
        self,
        identity: DuplexOmniPipelineIdentity,
    ) -> tuple[list[int], list[list[list[int]]]]:
        state = self._state(identity)
        if not self.talker_ready(identity):
            raise RuntimeError("DuplexOmni Talker predecessor is not complete")
        return list(state.history_indices), list(state.codec_history)

    def complete(
        self,
        identity: DuplexOmniPipelineIdentity,
        *,
        codec_codes: Any,
        valid_turn: bool,
    ) -> None:
        state = self._state(identity)
        expected = state.completed_through + 1
        if identity.slot != expected:
            raise RuntimeError(
                f"DuplexOmni Talker completion is out of order: expected {expected}, got {identity.slot}"
            )
        if valid_turn:
            turns = _normalize_codec_turns([codec_codes])
            state.history_indices.append(state.base_turns + identity.slot)
            state.codec_history.append(turns[0])
        state.completed_through = identity.slot

    def successor_request_id(self, identity: DuplexOmniPipelineIdentity) -> str | None:
        state = self._state(identity)
        return state.registered_slots.get(identity.slot + 1)

    def release_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_if_final(self, identity: DuplexOmniPipelineIdentity) -> None:
        if identity.final:
            self._sessions.pop(identity.session_id, None)

    def _state(self, identity: DuplexOmniPipelineIdentity) -> _SessionState:
        state = self._sessions.get(identity.session_id)
        if state is None or state.epoch != identity.epoch:
            raise RuntimeError("DuplexOmni pipeline session is not active")
        return state


__all__ = [
    "DuplexOmniPipelineCoordinator",
    "DuplexOmniPipelineIdentity",
    "PIPELINE_BASE_TURNS",
    "PIPELINE_EPOCH",
    "PIPELINE_FINAL",
    "PIPELINE_SESSION_ID",
    "PIPELINE_SLOT",
    "inject_codec_history",
    "pipeline_identity_from_prompt",
]
