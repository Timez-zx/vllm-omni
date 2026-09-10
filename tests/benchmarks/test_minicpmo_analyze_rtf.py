import base64

from benchmarks.minicpmo.analyze_rtf import (
    _client_pd_completion_slots,
    _client_pd_measurement_is_complete,
    _expected_input_units,
    _input_units_per_session,
    _measurement_is_complete,
    _measurement_sequence_ranges,
    _parse_frame_audit,
    _parse_vision_timing,
    _pd_chain_summary,
    _pd_context_cycle_summary,
    _pd_long_horizon_summary,
    _pd_session_recurrence_summary,
    _periodic_pd_records,
    _physical_d_kv_transfer_summary,
    _rtf_miss_count,
)


def _request_id(session_id: str) -> str:
    encoded = base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")
    return f"duplex-s.{encoded}.i.0.e.0.r.stage0"


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


def test_periodic_pd_records_use_per_session_formal_ranges() -> None:
    request_a = _request_id("bench-a")
    request_b = _request_id("bench-b")
    ranges = {"bench-a": (4, 6), "bench-b": (8, 10)}
    completions = [{"stage": 0, "request_id": request_a, "generation": generation} for generation in (3, 4, 6, 7)] + [
        {"stage": 1, "request_id": request_b, "generation": generation} for generation in (7, 8, 10, 11)
    ]
    slots = [{"request_id": request_a, "seq": str(sequence)} for sequence in (3, 4, 6, 7)]

    periodic_completions, periodic_slots = _periodic_pd_records(
        completions,
        slots,
        3,
        ranges,
    )

    assert [(item["stage"], item["generation"]) for item in periodic_completions] == [
        (0, 4),
        (0, 6),
        (1, 8),
        (1, 10),
    ]
    assert [item["seq"] for item in periodic_slots] == ["4", "6"]


def test_measurement_ranges_come_from_formal_user_sequences() -> None:
    assert _measurement_sequence_ranges(
        {
            "users": [
                {
                    "session_id": "bench-a",
                    "formal_seq_start": 4,
                    "formal_seq_end": 33,
                }
            ]
        }
    ) == {"bench-a": (4, 33)}
    assert _measurement_sequence_ranges(
        {
            "users": [
                {
                    "session_id": "bench-a",
                    "formal_seq_start": 40,
                    "formal_seq_end": 69,
                    "formal_input_unit_start": 4,
                    "formal_input_unit_end": 33,
                }
            ]
        }
    ) == {"bench-a": (4, 33)}


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


def test_clean_pd_measurement_uses_client_visible_exact_sequences() -> None:
    run = {
        "failed_users": 0,
        "pd_completion_witness": {
            "source": "client-visible engine stage_metrics",
            "expected": 2,
            "completed": 2,
            "complete": True,
        },
        "users": [
            {
                "session_id": "bench-a",
                "pd_completion_witness": {
                    "records": [
                        {
                            "sequence": 42,
                            "input_unit_index": 7,
                            "ready_at_s": 10.0,
                            "done_at_s": 10.4,
                            "e2e_ms": 400.0,
                        },
                        {
                            "sequence": 8,
                            "ready_at_s": 11.0,
                            "done_at_s": 11.5,
                            "e2e_ms": 500.0,
                        },
                    ]
                },
            }
        ],
    }

    assert _client_pd_measurement_is_complete(run, 2)
    slots = _client_pd_completion_slots(run)
    assert [slot["seq"] for slot in slots] == ["7", "8"]
    assert slots[0]["request_id"] == "bench-a"
    assert slots[0]["source"] == "client_physical_d_completion_witness"
    assert slots[0]["timing_schema"] == "legacy_single_origin"
    assert "model_unit_ready_epoch" not in slots[0]
    assert "ready_to_d_ms" not in slots[0]
    assert "input_aggregation_ms" not in slots[0]
    assert not _measurement_is_complete(
        {"failed_users": 1},
        600,
        {"0": 600, "1": 600},
        is_pd=True,
    )


def test_client_pd_completion_slots_preserve_dual_input_origins() -> None:
    run = {
        "users": [
            {
                "session_id": "bench-a",
                "pd_completion_witness": {
                    "records": [
                        {
                            "sequence": 42,
                            "input_unit_index": 7,
                            # The compatibility field stays input-start E2E.
                            "ready_at_s": 10.0,
                            "first_media_arrival_at_s": 10.0,
                            "model_unit_ready_at_s": 10.8,
                            "done_at_s": 11.1,
                            "e2e_ms": 1100.0,
                            "input_start_e2e_ms": 1100.0,
                            "ready_to_d_ms": 300.0,
                            "input_aggregation_ms": 800.0,
                            "external_cached_tokens": 20,
                            "kv_transfer_selected_blocks": 2,
                            "kv_transfer_selected_tokens": 20,
                            "kv_transfer_selected_bytes": 16_384,
                            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
                        }
                    ]
                },
            }
        ]
    }

    [slot] = _client_pd_completion_slots(run)

    assert slot["seq"] == "7"
    assert slot["input_unit_index"] == 7
    assert slot["physical_seq"] == 42
    # Stream/cadence RTF keeps the historical first-media origin.
    assert slot["ready_epoch"] == 10.0
    assert slot["first_media_arrival_epoch"] == 10.0
    assert slot["model_unit_ready_epoch"] == 10.8
    assert slot["e2e_ms"] == 1100.0
    assert slot["input_start_e2e_ms"] == 1100.0
    assert slot["ready_to_d_ms"] == 300.0
    assert slot["input_aggregation_ms"] == 800.0
    assert slot["kv_transfer_selected_blocks"] == 2
    assert slot["kv_transfer_selected_tokens"] == 20
    assert slot["kv_transfer_selected_bytes"] == 16_384
    assert slot["kv_transfer_write_submit_to_d_ready_ms"] == -1.0
    assert slot["timing_schema"] == "dual_input_origin"
    # Capacity RTF intentionally keeps its previous input-start window; it is
    # not recomputed from the later model-ready timestamp.
    assert _pd_long_horizon_summary([slot], 1)["per_session_stream_rtf"]["min"] == 0.909


def test_physical_d_kv_transfer_summary_validates_delta_without_inventing_timing() -> None:
    records = [
        {
            "request_id": "bench-a",
            "sequence": 1,
            "external_cached_tokens": 18,
            "kv_transfer_selected_blocks": 2,
            "kv_transfer_selected_tokens": 18,
            "kv_transfer_selected_bytes": 16_384,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        },
        {
            "request_id": "bench-a",
            "sequence": 2,
            "external_cached_tokens": 0,
            "kv_transfer_selected_blocks": 0,
            "kv_transfer_selected_tokens": 0,
            "kv_transfer_selected_bytes": 0,
            "kv_transfer_write_submit_to_d_ready_ms": 3.5,
        },
    ]

    summary = _physical_d_kv_transfer_summary(records)

    assert summary["valid"] is True
    assert summary["full_hit_records"] == 1
    assert summary["non_full_hit_records"] == 1
    assert summary["selected_tokens"]["max"] == 18.0
    assert summary["remote_replay_tokens"]["max"] == 0.0
    assert summary["selected_blocks"]["max"] == 2.0
    assert summary["selected_bytes"]["max"] == 16_384.0
    assert summary["write_submit_to_d_ready_ms"]["count"] == 1
    assert summary["write_submit_to_d_ready_ms"]["max"] == 3.5
    assert summary["write_timing_unavailable_records"] == 1


def test_physical_d_kv_transfer_summary_rejects_unproven_non_full_hit() -> None:
    summary = _physical_d_kv_transfer_summary(
        [
            {
                "request_id": "bench-a",
                "sequence": 1,
                "external_cached_tokens": 17,
                "kv_transfer_selected_blocks": 0,
                "kv_transfer_selected_tokens": 16,
                "kv_transfer_selected_bytes": -1,
                "kv_transfer_write_submit_to_d_ready_ms": -2.0,
            }
        ]
    )

    assert summary["valid"] is False
    assert summary["mismatches"] == 1
    reasons = summary["mismatch_examples"][0]["reasons"]
    assert "selected_tokens_not_external_cached_tokens" in reasons
    assert "non_full_hit_missing_selected_blocks" in reasons
    assert "selected_bytes_unavailable_or_negative" in reasons
    assert "invalid_write_submit_to_d_ready_ms" in reasons


def test_physical_d_kv_transfer_rejects_obsolete_remote_token_replay() -> None:
    summary = _physical_d_kv_transfer_summary([{
        "external_cached_tokens": 17,
        "kv_transfer_selected_blocks": 2,
        "kv_transfer_selected_tokens": 18,
        "kv_transfer_selected_bytes": 16_384,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
    }])
    assert summary["valid"] is False
    assert summary["mismatch_examples"][0]["reasons"] == ["selected_tokens_not_external_cached_tokens"]


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


def test_pd_long_horizon_accepts_nonzero_formal_sequence_start() -> None:
    request_id = _request_id("bench-a")
    slots = [
        {
            "request_id": request_id,
            "seq": str(sequence),
            "ready_epoch": float(sequence),
            "done_epoch": float(sequence) + 0.5,
            "e2e_ms": 500.0,
            "wait_previous_d_ms": 0.0,
        }
        for sequence in (7, 8, 9)
    ]

    summary = _pd_long_horizon_summary(slots, 3, {"bench-a": (7, 9)})

    assert summary["complete_sessions"] == 1
    assert summary["completed_units_per_session"]["p50"] == 3.0


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


def test_pd_chain_uses_model_ready_but_preserves_input_start_e2e() -> None:
    completions = [
        {
            "stage": 0,
            "request_id": "session-a",
            "generation": 1,
            "submit_epoch": 1.85,
            "done_epoch": 2.0,
            "service_ms": 150.0,
        },
        {
            "stage": 1,
            "request_id": "session-a",
            "generation": 1,
            "submit_epoch": 2.0,
            "done_epoch": 2.2,
            "service_ms": 200.0,
        },
    ]
    slots = [
        {
            "request_id": "session-a",
            "seq": "1",
            "ready_epoch": 1.0,
            "model_unit_ready_epoch": 1.8,
            "done_epoch": 2.2,
            "e2e_ms": 1200.0,
            "ready_to_d_ms": 400.0,
        }
    ]

    summary = _pd_chain_summary(completions, slots)

    assert summary["model_ready_to_p_submit_ms"]["p50"] == 50.0
    assert summary["model_ready_to_d_done_ms"]["p50"] == 400.0
    assert summary["input_start_to_d_done_ms"]["p50"] == 1200.0


def test_pd_session_recurrence_separates_inherited_backlog() -> None:
    slots = [
        {
            "request_id": "session-a",
            "seq": str(sequence),
            "model_unit_ready_epoch": ready,
            "done_epoch": done,
            "d_service_ms": service,
        }
        for sequence, ready, done, service in (
            (1, 1.0, 1.9, 200.0),
            # D submit is 2.4. Previous D finishes at 1.9, so this unit has
            # 0.5 s fresh pre-D work and no inherited backlog.
            (2, 1.9, 2.6, 200.0),
            # D submit is 3.1. This unit becomes ready at 2.0, inherits
            # 0.6 s, then spends another 0.5 s before a 0.2 s D service.
            (3, 2.0, 3.3, 200.0),
        )
    ]

    summary = _pd_session_recurrence_summary(slots)

    assert summary["matched_consecutive_transitions"] == 2
    assert summary["previous_d_incomplete_at_ready"] == 1
    assert summary["d_submit_order_violations"] == 0
    assert summary["identity_max_abs_error_ms"] == 0.0
    all_transitions = summary["all_transitions"]
    assert all_transitions["inherited_previous_d_wait_ms"]["max"] == 600.0
    assert all_transitions["current_pre_d_after_barrier_ms"]["p50"] == 500.0
    assert all_transitions["current_d_service_ms"]["p50"] == 200.0
    assert all_transitions["fresh_serial_cycle_ms"]["p50"] == 700.0


def test_pd_session_recurrence_deduplicates_and_rejects_sequence_gaps() -> None:
    slots = [
        {
            "request_id": "session-a",
            "seq": "1",
            "model_unit_ready_epoch": 1.0,
            "done_epoch": 1.5,
            "d_service_ms": 100.0,
        },
        {
            "request_id": "session-a",
            "seq": "1",
            "model_unit_ready_epoch": 1.0,
            "done_epoch": 1.4,
            "d_service_ms": 100.0,
        },
        {
            "request_id": "session-a",
            "seq": "3",
            "model_unit_ready_epoch": 3.0,
            "done_epoch": 3.5,
            "d_service_ms": 100.0,
        },
    ]

    summary = _pd_session_recurrence_summary(slots)

    assert summary["matched_consecutive_transitions"] == 0
    assert summary["sequence_gap_count"] == 1
    assert summary["excluded_first_formal_units"] == 1


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


def test_pd_context_cycle_uses_clean_physical_witness_without_cadence() -> None:
    slots = [
        {
            "request_id": "session-a",
            "input_unit_index": generation,
            "done_epoch": done_epoch,
            "prompt_tokens": tokens_in,
        }
        for generation, done_epoch, tokens_in in (
            (1, 1.0, 500),
            (100, 100.0, 20_000),
            (101, 101.0, 500),
            (200, 200.0, 20_000),
            (201, 202.0, 500),
        )
    ]

    summary = _pd_context_cycle_summary([], slots)

    assert summary["sessions_with_rollover"] == 1
    assert summary["sessions_with_complete_cycle"] == 1
    assert summary["context_cycle_rtf"]["p50"] == 0.99


def test_parse_vision_timing_keeps_only_measurement_window(tmp_path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                "[MINICPM-PREP-ARRIVAL] jobs=2 encoded_frames=2 "
                "cpu_prepare_ms=20.000 vision_encoder_ms=50.000 "
                "cache_ms=1.000 total_ms=71.000 done_epoch=10.000000",
                "[MINICPM-PREP-ARRIVAL-READY] success=True arrival_to_ready_ms=121.000 done_epoch=10.100000",
                "[MINICPM-PREP-ARRIVAL-ACCEPTED] success=True arrival_to_accepted_ms=50.000 done_epoch=10.150000",
                "[MINICPM-PREP-FORMAL-WAIT] tasks=1 wait_ms=0.200 done_epoch=10.200000",
                "[MINICPM-PREP-FORMAL-WAIT] tasks=2 wait_ms=0.000 detached_ack=true done_epoch=10.300000",
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
    assert summary["arrival_dispatch_successes"] == 2
    assert summary["arrival_to_dispatch_ms"]["count"] == 2
    assert summary["arrival_to_ready_ms"] == summary["arrival_to_dispatch_ms"]
    assert summary["formal_wait_calls"] == 2
    assert summary["formal_wait_tasks"] == 3
    assert summary["formal_wait_ms"]["max"] == 0.2


def test_parse_frame_audit_deduplicates_and_filters_formal_sequences(tmp_path) -> None:
    request_id = _request_id("bench-a")
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                f"[MINICPM-FRAME-CONSUMED] req={request_id} seq=3 frames=1 source=arrival done_epoch=10.000000",
                f"[MINICPM-FRAME-CONSUMED] req={request_id} seq=4 frames=1 source=arrival done_epoch=10.100000",
                f"[MINICPM-FRAME-CONSUMED] req={request_id} seq=4 frames=1 source=arrival done_epoch=10.200000",
                f"[MINICPM-FRAME-CONSUMED] req={request_id} seq=5 frames=1 source=formal_fallback done_epoch=10.300000",
            ]
        )
    )

    summary = _parse_frame_audit(log, 9.0, 11.0, {"bench-a": (4, 5)})

    assert summary == {
        "units": 2,
        "frames_consumed": 2,
        "by_source": {"arrival": 1, "formal_fallback": 1},
    }
