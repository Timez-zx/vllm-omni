# SPDX-License-Identifier: Apache-2.0

import hashlib

from benchmarks.live_agent.web_client.probe import synth_speechlike_pcm


def test_synthetic_audio_variants_are_deterministic_and_distinct() -> None:
    first = synth_speechlike_pcm(0.16, variant=101)
    repeated = synth_speechlike_pcm(0.16, variant=101)
    second = synth_speechlike_pcm(0.16, variant=102)

    assert first == repeated
    assert hashlib.sha256(first).digest() != hashlib.sha256(second).digest()
    assert len(first) == len(second) == int(16000 * 0.16) * 2
