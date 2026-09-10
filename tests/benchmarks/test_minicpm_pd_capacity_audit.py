from copy import deepcopy

import pytest

from benchmarks.minicpmo.capacity_audit import (
    pd_inherited_backlog,
    pd_input_backlog,
    pd_input_budget_rtf,
    pd_sliding_window_backlog,
    sender_timing_audit,
    sliding_window_engine_audit,
)


def sample_run():
    return {
        "config": {"max_send_drift_ms": 10},
        "users": [
            {
                "session_id": "s",
                "input_stream_complete": True,
                "units_sent": 3,
                "send_drift_ms": {"count": 15, "max": 2.0},
                "input_unit_timings": [{"input_unit_index": i, "model_unit_ready_at_s": float(i)} for i in range(1, 4)],
                "pd_completion_witness": {
                    "records": [{"input_unit_index": i, "done_at_s": i + 0.5} for i in range(1, 4)]
                },
            }
        ],
    }


def test_no_backlog_and_valid_sender():
    run = sample_run()
    assert pd_input_backlog(run)["no_backlog"]
    assert sender_timing_audit(run)["valid"]


def test_transient_backlog_fails_even_when_last_unit_catches_up():
    run = sample_run()
    run["users"][0]["pd_completion_witness"]["records"][0]["done_at_s"] = 2.1
    audit = pd_input_backlog(run)
    assert audit["valid"] and not audit["no_backlog"]
    assert audit["late_units"] == 1


def test_final_unit_cannot_hide_lateness_in_drain():
    run = sample_run()
    run["users"][0]["pd_completion_witness"]["records"][-1]["done_at_s"] = 4.01
    assert not pd_input_backlog(run)["no_backlog"]


def test_missing_old_clocks_or_duplicate_completion_cannot_pass():
    run = sample_run()
    bad = deepcopy(run)
    del bad["users"][0]["input_unit_timings"][0]["model_unit_ready_at_s"]
    assert not pd_input_backlog(bad)["valid"]
    run["users"][0]["pd_completion_witness"]["records"] *= 2
    assert not pd_input_backlog(run)["valid"]
    assert not pd_input_backlog({})["no_backlog"]


def test_sender_delay_is_invalid_evidence_not_a_model_capacity_failure():
    run = sample_run()
    run["users"][0]["send_drift_ms"]["max"] = 11
    assert not sender_timing_audit(run)["valid"]
    assert pd_input_backlog(run)["no_backlog"]
    del run["config"]["max_send_drift_ms"]
    assert not sender_timing_audit(run)["valid"]


def test_pd_placement_keeps_fixed_budgets_and_all_nonplacement_settings():
    from pathlib import Path

    from vllm_omni.config.stage_config import load_deploy_config

    folder = Path(__file__).resolve().parents[2] / "benchmarks/minicpmo"
    a = load_deploy_config(folder / "deploy_capacity_pd_4gpu_fp8.yaml")
    b = load_deploy_config(folder / "deploy_capacity_pd_4gpu_encoders_on_p_fp8.yaml")
    assert [s.devices for s in a.stages] == ["0", "1", "3", "3"]
    assert [s.devices for s in b.stages] == ["0", "1", "2", "3"]
    for original, colocated in zip(a.stages, b.stages, strict=True):
        fields = vars(original).copy()
        fields["devices"] = colocated.devices
        if original.stage_id == 0:
            fields["env"] = dict(
                fields["env"],
                VLLM_OMNI_AUX_VISIBLE_DEVICES="",
                MINICPMO45_VISION_ENCODER_DEVICE="cuda:0",
                MINICPMO45_AUDIO_ENCODER_DEVICE="cuda:0",
            )
        assert vars(colocated) == fields
    assert a.stages[0].engine_extras["kv_cache_memory_bytes"] == 64 * 1024**3
    assert a.stages[1].engine_extras["kv_cache_memory_bytes"] == 64 * 1024**3
    assert a.stages[2].engine_extras["kv_cache_memory_bytes"] == 8 * 1024**3
    assert a.stages[2].max_model_len == 65536
    assert a.stages[2].engine_extras["enable_prefix_caching"] is False
    assert a.stages[0].engine_extras["enable_prefix_caching"] is True
    assert a.stages[1].engine_extras["enable_prefix_caching"] is True
    assert a.stages[2].engine_extras["hf_overrides"]["vllm_omni_minicpmo_talker_sliding_window_tokens"] == 4096
    # Overlaying the memory budget must not erase the connector/dtype config.
    assert a.stages[0].engine_extras["kv_cache_dtype"] == "fp8"
    assert a.stages[0].engine_extras["kv_transfer_config"]["kv_connector"] == "NixlDeltaPushConnector"
    assert a.stages[0].engine_extras["hf_overrides"]["vllm_omni_minicpmo_pd_prefill"] is True
    assert a.stages[1].engine_extras["hf_overrides"]["vllm_omni_minicpmo_pd_decode"] is True
    for stage in a.stages[:2]:
        assert stage.engine_extras["hf_overrides"]["sliding_window"] == 36000
        assert stage.engine_extras["hf_overrides"]["use_sliding_window"] is True


def test_window_capacity_allows_recovered_backlog_below_limit():
    run = sample_run()
    records = run["users"][0]["pd_completion_witness"]["records"]
    for record, length in zip(records, [500, 1200, 2200]):
        record["prompt_tokens"] = length
    records[0]["done_at_s"] = 2.1
    result = pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)
    assert result["capacity_pass"] and result["whole_run"]["late_units"] == 1
    records[1]["done_at_s"] = 3.1
    result = pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)
    assert result["capacity_pass"]
    assert result["after_window_full"]["late_units"] == 1
    assert result["long_horizon_rtf"]["min_rtf_unrounded"] == 2 / 1.5
    # Ending beyond the entire post-window input budget is a long-RTF failure.
    records[-1]["done_at_s"] = 4.01
    result = pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)
    assert not result["capacity_pass"]
    assert result["long_horizon_rtf"]["min_rtf_unrounded"] < 1


@pytest.mark.parametrize("delay_ms,passed", [(499.999, True), (500.0, True), (500.0001, False), (750.0, False)])
def test_backlog_bound_uses_unrounded_maximum_even_when_recovered(delay_ms, passed):
    run = sample_run()
    records = run["users"][0]["pd_completion_witness"]["records"]
    records[0]["done_at_s"] = 2.0 + delay_ms / 1000
    records[1]["done_at_s"] = 2.9
    assert pd_input_budget_rtf(run)["capacity_pass"]
    result = pd_inherited_backlog(run)
    assert result["valid"]
    assert result["limit_ms"] == 500
    assert result["max_backlog_ms"] == pytest.approx(delay_ms)
    assert result["capacity_pass"] is passed


def test_backlog_excludes_current_execution_and_input_assembly():
    run = sample_run()
    for t in run["users"][0]["input_unit_timings"]:
        t["first_media_arrival_at_s"] = t["model_unit_ready_at_s"] - 0.8
    for r in run["users"][0]["pd_completion_witness"]["records"]:
        r["done_at_s"] = r["input_unit_index"] + 0.9
    result = pd_inherited_backlog(run)
    assert result["capacity_pass"] and result["max_backlog_ms"] == 0


def test_one_users_recovered_spike_fails_the_whole_point():
    run = sample_run()
    slow = deepcopy(run["users"][0])
    slow["session_id"] = "spike"
    slow["pd_completion_witness"]["records"][0]["done_at_s"] = 2.75
    slow["pd_completion_witness"]["records"][1]["done_at_s"] = 2.9
    run["users"].append(slow)
    result = pd_inherited_backlog(run)
    assert pd_input_budget_rtf(run)["capacity_pass"]
    assert result["sessions"][0]["capacity_pass"]
    assert not result["sessions"][1]["capacity_pass"]
    assert not result["capacity_pass"] and result["exceeded_units"] == 1


def test_first_sliding_unit_keeps_its_prewindow_predecessor():
    run = sample_run()
    records = run["users"][0]["pd_completion_witness"]["records"]
    for record, length in zip(records, [500, 1200, 2200]):
        record["prompt_tokens"] = length
    records[0]["done_at_s"] = 2.75
    records[1]["done_at_s"] = 2.9
    result = pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)
    assert result["long_horizon_rtf"]["capacity_pass"]
    assert result["bounded_backlog"]["max_backlog_ms"] == 750
    assert not result["capacity_pass"]
    # A spike entirely before the chosen evaluation interval stays diagnostic.
    assert pd_inherited_backlog(run, first_units={"s": 3})["capacity_pass"]


@pytest.mark.parametrize("mutation", ["missing", "gap", "reverse", "nan"])
def test_backlog_invalid_evidence_cannot_pass(mutation):
    run = sample_run()
    records = run["users"][0]["pd_completion_witness"]["records"]
    if mutation == "missing":
        records.pop(0)
    elif mutation == "gap":
        records.pop(1)
        run["users"][0]["input_unit_timings"].pop(1)
    elif mutation == "reverse":
        records[0]["done_at_s"] = 2.75
    else:
        records[0]["done_at_s"] = float("nan")
    assert not pd_inherited_backlog(run)["capacity_pass"]
    assert not pd_inherited_backlog({})["capacity_pass"]


@pytest.mark.parametrize("limit", [-1, float("nan"), float("inf")])
def test_invalid_backlog_bound_is_rejected(limit):
    with pytest.raises(ValueError):
        pd_inherited_backlog(sample_run(), max_backlog_ms=limit)


def test_long_rtf_equal_to_one_is_not_a_pass():
    run = sample_run()
    run["users"][0]["pd_completion_witness"]["records"][-1]["done_at_s"] = 4.0
    result = pd_input_budget_rtf(run)
    assert result["min_rtf_unrounded"] == 1.0
    assert not result["capacity_pass"]


def test_long_rtf_uses_complete_input_and_a_continuous_span():
    run = sample_run()
    for timing in run["users"][0]["input_unit_timings"]:
        timing["first_media_arrival_at_s"] = timing["model_unit_ready_at_s"] - 0.8
    records = run["users"][0]["pd_completion_witness"]["records"]
    records[0]["done_at_s"] = 2.1
    audit = pd_input_budget_rtf(run)
    assert audit["capacity_pass"]
    assert audit["sessions"][0]["wall_s"] == 2.5
    assert audit["min_rtf_unrounded"] == 3 / 2.5
    assert not pd_input_backlog(run)["no_backlog"]


def test_long_rtf_does_not_round_or_average_away_a_slow_user():
    run = sample_run()
    slow = deepcopy(run["users"][0])
    slow["session_id"] = "slow"
    slow["pd_completion_witness"]["records"][-1]["done_at_s"] = 4.0004
    run["users"].append(slow)
    audit = pd_input_budget_rtf(run)
    assert round(audit["min_rtf_unrounded"], 3) == 1.0
    assert not audit["capacity_pass"]


def test_long_rtf_requires_complete_contiguous_valid_clocks():
    assert not pd_input_budget_rtf({})["capacity_pass"]
    run = sample_run()
    run["users"][0]["pd_completion_witness"]["records"].pop()
    assert not pd_input_budget_rtf(run)["capacity_pass"]
    run = sample_run()
    run["users"][0]["input_unit_timings"].pop(1)
    run["users"][0]["pd_completion_witness"]["records"].pop(1)
    assert not pd_input_budget_rtf(run)["capacity_pass"]
    run = sample_run()
    run["users"][0]["pd_completion_witness"]["records"][-1]["done_at_s"] = float("nan")
    assert not pd_input_budget_rtf(run)["capacity_pass"]


def test_window_capacity_requires_every_user_to_fill_and_replace_window():
    run = sample_run()
    records = run["users"][0]["pd_completion_witness"]["records"]
    for record, length in zip(records, [500, 1200, 1800]):
        record["prompt_tokens"] = length
    assert not pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)["capacity_pass"]
    records[-1]["prompt_tokens"] = 2200
    assert not pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=3)["capacity_pass"]
    assert pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)["capacity_pass"]
    run["users"].append(deepcopy(run["users"][0]))
    run["users"][-1]["session_id"] = "short"
    run["users"][-1]["pd_completion_witness"]["records"][-1]["prompt_tokens"] = 400
    assert not pd_sliding_window_backlog(run, window_tokens=1024, min_post_window_units=2)["capacity_pass"]


def test_window_engine_audit_requires_both_stages_and_bounded_real_kv():
    deploy = {
        "stages": [
            {
                "stage_id": i,
                "hf_overrides": {
                    "sliding_window": 1024,
                    "use_sliding_window": True,
                    "vllm_omni_minicpmo_sliding_window_tokens": 1024,
                    "vllm_omni_minicpmo_pd_prefill" if i == 0 else "vllm_omni_minicpmo_pd_decode": True,
                },
            }
            for i in (0, 1)
        ]
    }
    log = "\n".join(
        f"(StageEngineCoreProc_stage{i}_replica0 pid=123) INFO [kv-window] "
        f"request=s{'-0010' if i else ''} logical_tokens=2200 window_tokens=1024 resident_blocks=[65]"
        for i in (0, 1)
    )
    assert sliding_window_engine_audit(log, deploy, window_tokens=1024, users=1)["valid"]
    assert not sliding_window_engine_audit(log, deploy, window_tokens=1024, users=2)["valid"]
    assert not sliding_window_engine_audit(log.replace("[65]", "[80]"), deploy, window_tokens=1024, users=1)["valid"]
    assert not sliding_window_engine_audit(log.splitlines()[0], deploy, window_tokens=1024, users=1)["valid"]
    pinned = deepcopy(deploy)
    for stage in pinned["stages"]:
        stage["hf_overrides"]["vllm_omni_pinned_prefix_tokens"] = 128
    pinned_log = "\n".join(
        line.replace("[65]", "[73]") + " pinned_prefix_tokens=128 pinned_resident_blocks=[8]"
        for line in log.splitlines()
    )
    assert sliding_window_engine_audit(pinned_log, pinned, window_tokens=1024, users=1)["valid"]
    assert not sliding_window_engine_audit(log, pinned, window_tokens=1024, users=1)["valid"]
    assert not sliding_window_engine_audit(
        pinned_log.replace("pinned_resident_blocks=[8]", "pinned_resident_blocks=[7]"),
        pinned, window_tokens=1024, users=1,
    )["valid"]
    deploy["stages"][1]["hf_overrides"]["use_sliding_window"] = False
    assert not sliding_window_engine_audit(log, deploy, window_tokens=1024, users=1)["valid"]
    deploy["stages"][1]["hf_overrides"]["use_sliding_window"] = True
    del deploy["stages"][0]["hf_overrides"]["vllm_omni_minicpmo_pd_prefill"]
    assert not sliding_window_engine_audit(log, deploy, window_tokens=1024, users=1)["valid"]
