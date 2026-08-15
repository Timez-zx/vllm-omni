# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime flags for the streaming thinker/talker/code2wav pipeline.

Every flag here defaults ON: they are the transport and kernel optimizations the
baseline needs to serve many concurrent realtime voice sessions, and each one is
measured in workflow.md. Setting a flag to 0 opts OUT, which is how the
corresponding ablation arm is built.

    VLLM_OMNI_INLINE_RECV        take chunk delivery on the scheduler thread
                                 instead of parking the consumer for a
                                 round-trip through the recv thread
    VLLM_OMNI_INLINE_RECV_ASYNC  allow that on a stage running async scheduling
    VLLM_OMNI_INLINE_SEND        put the outgoing chunk on the shared-memory
                                 edge from the scheduler thread at T+0
    VLLM_OMNI_MAILBOX            persistent two-slot shared-memory mailbox per
                                 (session, edge) instead of one POSIX segment
                                 per chunk
    VLLM_OMNI_STREAM_VOCODER     windowed convolution in code2wav: emitted
                                 samples are bit-identical, ~48% less work
    VLLM_OMNI_FUSED_SNAKE        run code2wav's Snake activations through the
                                 fused Triton kernel instead of transformers'
                                 five-kernel elementwise chain
    VLLM_OMNI_T2T_LEAN_DECODE    ship only the fields the talker reads on a
                                 decode payload
"""
from __future__ import annotations

import os

_DEFAULTS = {
    "VLLM_OMNI_INLINE_RECV": "1",
    "VLLM_OMNI_INLINE_RECV_ASYNC": "1",
    "VLLM_OMNI_INLINE_SEND": "1",
    "VLLM_OMNI_MAILBOX": "1",
    "VLLM_OMNI_STREAM_VOCODER": "1",
    "VLLM_OMNI_FUSED_SNAKE": "1",
    "VLLM_OMNI_T2T_LEAN_DECODE": "1",
}


def flag(name: str) -> str:
    """Effective value: an empty string counts as unset, so `NAME=` inherits
    the default rather than silently disabling a feature."""
    v = os.environ.get(name)
    if v is None or v == "":
        return _DEFAULTS.get(name, "0")
    return v


def flag_on(name: str) -> bool:
    return flag(name) not in ("0", "", "false", "False")
