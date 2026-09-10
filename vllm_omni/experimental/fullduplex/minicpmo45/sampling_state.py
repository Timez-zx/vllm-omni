# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded model-policy state carried alongside KV, never media or embeddings."""

from dataclasses import dataclass, field
from typing import Any

from .policy import MiniCPMO45DuplexPolicy

SAMPLING_STATE_KEY = "duplex_sampling_state"
SAMPLING_STATE_WIRE_KEY = f"meta.{SAMPLING_STATE_KEY}"
_FLAGS = ("current_turn_ended", "pending_speech_context", "pending_speech_response_open")
_TOKENS = ("pending_terminator_token", "last_terminator_token")
_U64_MASK = (1 << 64) - 1


def _wire_u64(value: int) -> int:
    if not 0 <= value <= _U64_MASK:
        raise ValueError("CUDA RNG state must contain unsigned 64-bit values")
    return value if value < (1 << 63) else value - (1 << 64)


@dataclass
class DecodeSamplingState:
    current_turn_ended: bool = True
    pending_speech_context: bool = False
    pending_speech_response_open: bool = False
    pending_terminator_token: int | None = None
    last_terminator_token: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    current_segment_output_tokens: list[int] = field(default_factory=list)
    rng_seed: int | None = None
    rng_offset: int | None = None


def pack_sampling_state(state: Any, *, incarnation: int, epoch: int | None, seq: int | None) -> list[int]:
    history = list(state.generated_tokens[-MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE :])
    segment = list(state.current_segment_output_tokens)
    seed, offset = getattr(state, "rng_seed", None), getattr(state, "rng_offset", None)
    if (seed is None) != (offset is None):
        raise ValueError("Incomplete CUDA RNG state")
    return [
        2,
        incarnation,
        -1 if epoch is None else epoch,
        -1 if seq is None else seq,
        *(int(getattr(state, key, key == "current_turn_ended")) for key in _FLAGS),
        *(-1 if getattr(state, key, None) is None else int(getattr(state, key)) for key in _TOKENS),
        len(history),
        len(segment),
        int(seed is not None),
        0 if seed is None else _wire_u64(seed),
        0 if offset is None else _wire_u64(offset),
        *history,
        *segment,
    ]


def unpack_sampling_state(value: Any) -> tuple[DecodeSamplingState, tuple[int, int | None, int | None]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 14:
        raise ValueError("Missing or malformed MiniCPM P/D sampling state")
    data = [int(item) for item in value]
    version, incarnation, epoch, seq = data[:4]
    nh, ns = data[9:11]
    if (
        version != 2
        or not 0 <= nh <= MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE
        or not 0 <= ns <= 4096
        or len(data) != 14 + nh + ns
        or any(flag not in (0, 1) for flag in data[4:7])
        or data[11] not in (0, 1)
        or any(not -(1 << 63) <= item < (1 << 63) for item in data[12:14])
        or (not data[11] and any(data[12:14]))
        or any(token < 0 for token in data[14:])
    ):
        raise ValueError("Invalid MiniCPM P/D sampling state")
    state = DecodeSamplingState()
    for key, item in zip(_FLAGS, data[4:7]):
        setattr(state, key, bool(item))
    for key, item in zip(_TOKENS, data[7:9]):
        setattr(state, key, None if item == -1 else item)
    state.generated_tokens = data[14 : 14 + nh]
    state.current_segment_output_tokens = data[14 + nh :]
    if data[11]:
        state.rng_seed, state.rng_offset = (item & _U64_MASK for item in data[12:14])
    return state, (incarnation, None if epoch == -1 else epoch, None if seq == -1 else seq)


def restore_sampling_state(target: Any, source: DecodeSamplingState) -> None:
    for key in (*_FLAGS, *_TOKENS, "rng_seed", "rng_offset"):
        setattr(target, key, getattr(source, key))
    target.generated_tokens = list(source.generated_tokens)
    target.current_segment_output_tokens = list(source.current_segment_output_tokens)
    target._rng_restored_identity = None


def resume_sampling_rng(state: Any, generator: Any, identity: object) -> None:
    """Restore the model session's Philox stream once per physical unit.

    CUDA seed/offset are host-side generator bookkeeping. No device-state
    tensor copy or synchronization is needed; CPU and unseeded RNG are untouched.
    """
    if state is None or getattr(getattr(generator, "device", None), "type", None) != "cuda":
        return
    restore_key = (identity, id(generator))
    if getattr(state, "_rng_restored_identity", None) == restore_key:
        return
    seed, offset = getattr(state, "rng_seed", None), getattr(state, "rng_offset", None)
    if seed is not None:
        generator.manual_seed(seed)
        generator.set_offset(offset)
    state._rng_restored_identity = restore_key


def snapshot_sampling_rng(state: Any, generator: Any) -> None:
    if state is None or getattr(getattr(generator, "device", None), "type", None) != "cuda":
        return
    state.rng_seed = generator.initial_seed()
    state.rng_offset = generator.get_offset()
