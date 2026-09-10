from benchmarks.minicpmo.analyze_nonpd_placement import analyze
from benchmarks.minicpmo.analyze_placement_nsys import Intervals

REQUEST = "duplex-s.cw.i.0.e.0.r.stage0"


def test_nsys_interval_union_does_not_double_count_overlapping_kernels():
    spans = Intervals([(5, 9), (1, 4), (3, 6), (12, 15)])
    assert spans.prefix[-1] == 11
    assert spans.overlap(2, 13) == 8
    assert spans.overlap(9, 12) == 0
    assert spans.overlap(0, 20) == 11
    assert spans.overlap(3, 5) == 2
    assert Intervals([]).overlap(0, 20) == 0


def run():
    return {
        "failed_users": 0,
        "audio_chunks": 1,
        "users": [
            {
                "session_id": "s",
                "input_stream_complete": True,
                "input_unit_timings": [
                    {"input_unit_index": 1, "first_media_arrival_at_s": 100.0, "model_unit_ready_at_s": 100.8}
                ],
            }
        ],
    }


def admit(at=100.8):
    return (
        f"[duplex_cadence] stage=0 ADMIT req={REQUEST} generation=1 "
        f"admit_epoch={at} prompt_tokens=211 input_seq=1 input_origin=client input_unit_index=1\n"
    )


def test_rtf_includes_wait_before_engine_admission():
    text = admit(101.2) + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=101.4"
    result = analyze(run(), text)
    assert result["all_complete"]
    assert not result["stage0_rtf_pass"]
    # Finite-window RTF includes the initial audio aggregation interval;
    # it alone does not imply overlapping unfinished complete input units.
    assert result["capacity_pass"]
    assert 0.71 < result["rtf_min"] < 0.72


def test_forward_is_not_a_completed_model_unit():
    text = admit() + f"[duplex_cadence] stage=0 RUNNER_DONE req={REQUEST} generation=1 step=1 runner_done_epoch=100.9"
    result = analyze(run(), text)
    assert not result["all_complete"]
    assert result["rtf_min"] is None


def test_completed_unit_can_sustain_realtime():
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=100.9"
    assert analyze(run(), text)["capacity_pass"]


def test_optional_unit_records_do_not_change_capacity_audit():
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=100.9"
    default = analyze(run(), text)
    detailed = analyze(run(), text, include_unit_records=True)
    records = detailed["sessions"][0].pop("unit_records")
    assert records[0]["input_unit_index"] == 1
    assert records[0]["done_epoch"] == 100.9
    assert detailed == default


def test_downstream_finish_is_reported_separately_from_input_rtf():
    workload = run()
    workload["users"][0]["audio"] = {"last_received_epoch_s": 101.5}
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=100.9"
    result = analyze(workload, text)
    assert result["stage0_rtf_pass"]
    assert "Diagnostic only" in result["pipeline_caveat"]
    assert result["pipeline_rtf_min"] == 1 / 1.5


def test_silent_broken_pipeline_is_not_a_capacity_pass():
    workload = run()
    workload["audio_chunks"] = 0
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=100.9"
    assert not analyze(workload, text)["capacity_pass"]


def test_queued_updates_keep_distinct_fifo_completions():
    workload = run()
    workload["users"][0]["input_unit_timings"].append(
        {"input_unit_index": 2, "first_media_arrival_at_s": 101.0, "model_unit_ready_at_s": 101.8}
    )
    text = admit() + admit(101.8).replace("generation=1", "generation=2").replace(
        "input_unit_index=1", "input_unit_index=2"
    )
    text += f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=2 done_epoch=101.85 context_tokens=211\n"
    text += f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=2 done_epoch=101.95 context_tokens=422\n"
    result = analyze(workload, text)
    assert result["all_complete"]
    assert result["sessions"][0]["completed_units"] == 2
    assert result["no_skipped_units"]
    assert not result["capacity_pass"]
    assert result["max_earlier_unfinished_units"] == 1
    assert result["stage0_rtf_pass"]  # Catch-up cannot erase earlier backlog.


def test_final_unit_cannot_hide_in_drain():
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=102.0"
    result = analyze(run(), text)
    assert not result["capacity_pass"]
    assert result["sessions"][0]["backlog"]["final_unit_exceeds_period"]


def test_normal_inflight_unit_is_not_backlog():
    workload = run()
    workload["users"][0]["input_unit_timings"].append(
        {"input_unit_index": 2, "first_media_arrival_at_s": 101.0, "model_unit_ready_at_s": 101.75}
    )
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=101.75\n"
    text += admit(101.75).replace("generation=1", "generation=2").replace("input_unit_index=1", "input_unit_index=2")
    text += f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=2 done_epoch=102.3\n"
    assert analyze(workload, text)["capacity_pass"]


def test_talker_backlog_cannot_be_hidden_by_fast_thinker():
    workload = run()
    workload["users"][0]["input_unit_timings"].append(
        {"input_unit_index": 2, "first_media_arrival_at_s": 101.0, "model_unit_ready_at_s": 101.8}
    )
    text = admit() + f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=1 done_epoch=100.9\n"
    text += f"[duplex_cadence] stage=1 ADMIT req={REQUEST} generation=1 admit_epoch=100.95\n"
    text += f"[duplex_cadence] stage=1 ADMIT req={REQUEST} generation=2 admit_epoch=101.2\n"
    text += f"[duplex_cadence] stage=1 UNIT_DONE req={REQUEST} generation=2 done_epoch=101.3\n"
    text += f"[duplex_cadence] stage=1 UNIT_DONE req={REQUEST} generation=2 done_epoch=101.4\n"
    text += admit(101.8).replace("generation=1", "generation=2").replace("input_unit_index=1", "input_unit_index=2")
    text += f"[duplex_cadence] stage=0 UNIT_DONE req={REQUEST} generation=2 done_epoch=101.95\n"
    result = analyze(workload, text)
    assert result["no_input_backlog"]
    assert result["talker_backlog_events"] == 1
    assert not result["capacity_pass"]
