# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime flags for the streaming thinker/talker/code2wav pipeline.

Every flag here defaults ON. They are transport and kernel optimizations still
used by the finite-request realtime baseline. Setting a flag to 0 opts out for
an ablation.

    VLLM_OMNI_MAILBOX            persistent two-slot shared-memory mailbox per
                                 (request, edge) instead of one POSIX segment
                                 per chunk
    VLLM_OMNI_STREAM_VOCODER     windowed convolution in code2wav: emitted
                                 samples are bit-identical, ~48% less work
    VLLM_OMNI_FUSED_SNAKE        run code2wav's Snake activations through the
                                 fused Triton kernel instead of transformers'
                                 five-kernel elementwise chain
"""
from __future__ import annotations

import os

_DEFAULTS = {
    "VLLM_OMNI_MAILBOX": "1",
    "VLLM_OMNI_STREAM_VOCODER": "1",
    "VLLM_OMNI_FUSED_SNAKE": "1",
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
