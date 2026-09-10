from benchmarks.minicpmo.analyze_rtf import _pd_long_horizon_summary


def test_rounded_display_does_not_hide_rtf_below_one():
    result = _pd_long_horizon_summary(
        [{"request_id": "bench-a", "seq": 1, "ready_epoch": 0.0, "done_epoch": 1.0004, "e2e_ms": 1000.4}],
        1,
    )
    assert result["per_session_stream_rtf"]["min"] == 1.0
    assert result["min_stream_rtf_unrounded"] < 1.0
