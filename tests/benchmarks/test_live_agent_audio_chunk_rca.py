# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from benchmarks.live_agent.analysis.audio_chunk_rca import analyze_cell


def test_audio_chunk_rca_separates_talker_wait_and_code2wav(tmp_path: Path) -> None:
    summary = {
        "users": 1,
        "warmup_turns": 0,
        "expected_measured_turns": 1,
        "ttfa_p50_ms": 210.0,
        "ttfa_p99_ms": 210.0,
        "playback_start_p50_ms": 510.0,
        "playback_start_p99_ms": 510.0,
        "stall_max_ms_p99": 0.0,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    turn = {
        "user": "u0",
        "turn": 1,
        "status": "ok",
        "t_q": 1.0,
        "t_fa": 1.21,
        "t_done": 2.0,
    }
    (tmp_path / "turns.jsonl").write_text(json.dumps(turn) + "\n")
    log = "\n".join(
        [
            "[turnprobe] recv rid=video-sess-abc turn=0 mono=1.001000",
            "[CHUNK-EMIT] rid=video-sess-abc-engine chunk_id=0 frames=4 mono=1.200000",
            "[AUDIO-CHUNK] stage=2 req=video-sess-abc-engine ts=10.0 mono=1.210000 frames=7000 sr=24000",
            "[SCHED-STEP] stage=0 mono=1.300000 nreq=1 ntok=1 run=1 wait=0 irecv=0/0/0/0",
            "[REQ-STEP] stage=0 mono=1.300000 reqs=video-sess-abc-engine:1",
            "[SCHED-STEP] stage=1 mono=1.250000 nreq=1 ntok=1 run=1 wait=0 irecv=2/1/0/0",
            "[REQ-STEP] stage=1 mono=1.250000 reqs=video-sess-abc-engine:1",
            "[SCHED-STEP] stage=1 mono=1.500000 nreq=1 ntok=1 run=1 wait=0 irecv=3/1/0/0",
            "[REQ-STEP] stage=1 mono=1.500000 reqs=video-sess-abc-engine:1",
            "[CHUNK-EMIT] rid=video-sess-abc-engine chunk_id=1 frames=25 mono=1.500000",
            "[AUDIO-CHUNK] stage=2 req=video-sess-abc-engine ts=10.3 mono=1.512000 frames=48000 sr=24000",
        ]
    )
    (tmp_path / "engine.log").write_text(log + "\n")

    report = analyze_cell(tmp_path, wait_threshold_ms=25.0)

    assert report["mapped_second_chunks"] == 1
    assert report["second_chunk_gap_ms"]["p50"] == pytest.approx(300.0)
    assert report["talker_request_steps"]["p50"] == 2
    assert report["talker_wait_excess_ms"]["p50"] == pytest.approx(225.0)
    assert report["long_gap_same_thinker_request_share"] == 1.0
    assert report["thinker_to_talker_inline_receive"]["hit_rate"] == 0.75
    assert report["code2wav_emit_to_audio_ms"]["p99"] == pytest.approx(12.0)
