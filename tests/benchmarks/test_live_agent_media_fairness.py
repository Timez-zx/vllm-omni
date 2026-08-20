# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.live_agent.analysis.media_fairness import (
    compare,
    parse_log,
    summarize,
)


def _line(*, sid: str = "u0", turn: int = 0, frame_sha: str = "a" * 64, audio_sha: str = "b" * 64) -> str:
    return (
        f"[media-ledger] sid={sid} turn={turn} frames_selected=2 frames_submitted=2 "
        f"frame_selected_sha={frame_sha} frame_submitted_sha={frame_sha} "
        f"audio_selected_bytes=64000 audio_submitted_bytes=64000 "
        f"audio_selected_sha={audio_sha} audio_submitted_sha={audio_sha} "
        f"prefill_chunks=222,333 prefill_tokens=555 prefill_token_sha={'c' * 64} dropped=0"
    )


def test_media_fairness_accepts_exact_cell(tmp_path: Path) -> None:
    log = tmp_path / "engine.log"
    log.write_text(_line() + "\n" + _line(sid="u1", turn=1) + "\n")

    records = parse_log(log)
    summary = summarize(records)

    assert summary["turns"] == 2
    assert summary["frames_selected"] == 4
    assert summary["frames_submitted"] == 4
    assert summary["audio_selected_bytes"] == 128000
    assert summary["prefill_chunks"] == 4
    assert summary["prefill_tokens"] == 1110
    assert summary["cell_media_exact"] is True


def test_media_fairness_rejects_cross_arm_input_drift(tmp_path: Path) -> None:
    baseline_log = tmp_path / "baseline.log"
    candidate_log = tmp_path / "candidate.log"
    baseline_log.write_text(_line() + "\n")
    candidate_log.write_text(_line(frame_sha="d" * 64) + "\n")

    report = compare(parse_log(baseline_log), parse_log(candidate_log))

    assert report["paired_turns"] == 1
    assert report["frame_input_exact_turns"] == 0
    assert report["audio_input_exact_turns"] == 1
    assert report["paired_media_exact"] is False
