# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.live_agent.playback_metrics import simulate_playback


def test_playback_without_stall() -> None:
    report = simulate_playback(
        [(0.20, 4800), (0.35, 4800), (0.50, 4800)],
        sample_rate=24000,
        prebuffer_s=0.06,
    )
    assert report.start_s == 0.20
    assert report.stalls_s == ()


def test_playback_reports_actual_dry_period() -> None:
    report = simulate_playback(
        [(0.20, 2400), (0.50, 4800)],
        sample_rate=24000,
        prebuffer_s=0.06,
    )
    assert report.start_s == 0.20
    assert report.stall_max_s == pytest.approx(0.20)
    assert report.stall_total_s == pytest.approx(0.20)


def test_prebuffer_can_cover_first_gap() -> None:
    report = simulate_playback(
        [(0.20, 2400), (0.25, 7200), (0.50, 2400)],
        sample_rate=24000,
        prebuffer_s=0.30,
    )
    assert report.start_s == 0.25
    assert report.stalls_s == ()


def test_playback_rejects_non_monotonic_input() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        simulate_playback([(0.2, 100), (0.1, 100)])
