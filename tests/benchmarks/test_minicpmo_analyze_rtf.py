from benchmarks.minicpmo.analyze_rtf import (
    _expected_input_units,
    _input_units_per_session,
    _measurement_is_complete,
    _pd_chain_summary,
    _pd_context_cycle_summary,
    _pd_long_horizon_summary,
    _parse_vision_timing,
    _periodic_pd_records,
    _rtf_miss_count,
)


def test_input_units_per_session_requires_one_common_positive_value() -> None:
    assert _input_units_per_session({"users": [{"units_sent": 30}, {"units_sent": 30}]}) == 30
    assert _input_units_per_session({"users": [{"units_sent": 29}, {"units_sent": 30}]}) is None
    assert _input_units_per_session({"users": []}) is None


def test_periodic_pd_records_exclude_setup_and_continuations() -> None:
    completions = [
        {"stage": 0, "generation": 0},
        {"stage": 0, "generation": 1},
        {"stage": 0, "generation": 30},
        {"stage": 0, "generation": 31},
        {"stage": 1, "generation": 30},
        {"stage": 1, "generation": 32},
        {"stage": 2, "generation": 50},
    ]
    slots = [{"seq": str(sequence)} for sequence in (1, 30, 31, 32)]

    periodic_completions, periodic_slots = _periodic_pd_records(completions, slots, 30)

    assert periodic_completions == [
        {"stage": 0, "generation": 1},
        {"stage": 0, "generation": 30},
        {"stage": 1, "generation": 30},
        {"stage": 2, "generation": 50},
    ]
    assert periodic_slots == [{"seq": "1"}, {"seq": "30"}]


def test_rtf_only_values_below_one_are_capacity_misses() -> None:
    assert _rtf_miss_count([0.999, 1.0, 1.001]) == 1


def test_expected_input_units_requires_a_known_nonempty_user_set() -> None:
    assert _expected_input_units({"users": [{}, {}]}, 180) == 360
    assert _expected_input_units({"users": []}, 180) is None
    assert _expected_input_units({"users": [{}]}, None) is None


def test_pd_measurement_uses_terminal_decode_as_completion_witness() -> None:
    run = {"failed_users": 0}
    assert _measurement_is_complete(
        run,
        600,
        {"0": 582, "1": 600},
        is_pd=True,
    )
    assert not _measurement_is_complete(
        run,
        600,
        {"0": 600, "1": 599},
        is_pd=True,
    )
    assert not _measurement_is_complete(
        {"failed_users": 1},
        600,
        {"0": 600, "1": 600},
        is_pd=True,
    )


def test_pd_long_horizon_reports_recoverable_tail_and_backlog() -> None:
    slots = [
        {
            "request_id": "session-a",
            "seq": str(sequence),
            "ready_epoch": float(sequence),
            "done_epoch": done,
            "e2e_ms": latency,
            "wait_previous_d_ms": 0.0,
        }
        for sequence, done, latency in (
            (1, 1.8, 800.0),
            (2, 2.9, 900.0),
            (3, 3.7, 700.0),
        )
    ]

    summary = _pd_long_horizon_summary(slots, 3)

    assert summary["sessions"] == 1
    assert summary["complete_sessions"] == 1
    assert summary["per_session_stream_rtf"]["min"] == 1.111
    assert summary["mean_latency_budget_rtf"] == 1.25
    assert summary["per_session_mean_latency_budget_rtf"]["min"] == 1.25
    assert summary["completion_cadence_rtf"]["min"] == 1.053
    assert summary["terminal_stream_backlog_ms"]["p50"] == -300.0
    assert summary["e2e_growth_ms"]["p50"] == -100.0


def test_pd_chain_decomposes_matched_p_and_d_records() -> None:
    completions = [
        {
            "stage": 0,
            "request_id": "session-a",
            "generation": 1,
            "submit_epoch": 1.1,
            "done_epoch": 1.4,
            "service_ms": 300.0,
        },
        {
            "stage": 1,
            "request_id": "session-a",
            "generation": 1,
            "submit_epoch": 1.39,
            "done_epoch": 1.8,
            "service_ms": 410.0,
        },
    ]
    slots = [
        {
            "request_id": "session-a",
            "seq": "1",
            "ready_epoch": 1.0,
            "done_epoch": 1.8,
            "e2e_ms": 800.0,
        }
    ]

    summary = _pd_chain_summary(completions, slots)

    assert summary["matched_units"] == 1
    assert summary["ready_to_p_submit_ms"]["p50"] == 100.0
    assert summary["p_service_ms"]["p50"] == 300.0
    assert summary["p_done_to_d_done_ms"]["p50"] == 400.0
    assert summary["d_service_ms"]["p50"] == 410.0


def test_pd_context_cycle_uses_matching_decode_prompt_resets() -> None:
    completions = [
        {
            "stage": 1,
            "request_id": "session-a",
            "generation": generation,
            "done_epoch": done_epoch,
            "tokens_in": tokens_in,
        }
        for generation, done_epoch, tokens_in in (
            (1, 1.0, 500),
            (100, 100.0, 20_000),
            (101, 101.0, 500),
            (200, 200.0, 20_000),
            (201, 202.0, 500),
        )
    ]

    summary = _pd_context_cycle_summary(completions)

    assert summary["sessions_with_rollover"] == 1
    assert summary["sessions_with_complete_cycle"] == 1
    assert summary["rollovers_per_session"]["p50"] == 2.0
    assert summary["cycle_span_units"]["p50"] == 100.0
    assert summary["context_cycle_rtf"]["p50"] == 0.99


def test_parse_vision_timing_keeps_only_measurement_window(tmp_path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "[MINICPM-PREP-ARRIVAL] jobs=2 encoded_frames=2 "
                "cpu_prepare_ms=20.000 vision_encoder_ms=50.000 "
                "cache_ms=1.000 total_ms=71.000 done_epoch=10.000000",
                "[MINICPM-PREP-ARRIVAL-READY] success=True "
                "arrival_to_ready_ms=121.000 done_epoch=10.100000",
                "[MINICPM-PREP-FORMAL-WAIT] tasks=1 wait_ms=0.200 "
                "done_epoch=10.200000",
                "[MINICPM-PREP-ARRIVAL] jobs=9 encoded_frames=9 "
                "cpu_prepare_ms=99.000 vision_encoder_ms=99.000 "
                "cache_ms=1.000 total_ms=199.000 done_epoch=20.000000",
            ]
        )
    )

    summary = _parse_vision_timing(log, 9.0, 11.0)

    assert summary["batches"] == 1
    assert summary["jobs"] == 2
    assert summary["vision_encoder_ms"]["p50"] == 50.0
    assert summary["arrival_to_ready_ms"]["p50"] == 121.0
    assert summary["formal_wait_ms"]["p50"] == 0.2
