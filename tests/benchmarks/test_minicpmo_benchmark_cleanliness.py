import base64
import hashlib
import json
import shutil
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks.minicpmo.analyze_rtf import (
    _benchmark_cleanliness,
    _client_audio_sidecar_audit,
    _client_frame_audit,
    _formal_capacity_workload_validity,
)
from benchmarks.minicpmo.analyze_rtf import main as analyze_main
from benchmarks.minicpmo.clean_server import (
    DIAGNOSTIC_CLI_FLAGS,
    DIAGNOSTIC_ENV_VARS,
    _deploy_config_provenance,
    formal_environment,
    launch_environment,
)
from benchmarks.minicpmo.clean_server import main as clean_server_main
from benchmarks.minicpmo.continuous_av import _pd_completion_records


def test_looped_av_tracks_share_one_boundary_without_resampling():
    from benchmarks.minicpmo.continuous_av import _align_loop_media

    pcm = bytes(range(256)) * 4415  # 35.32 s at 16 kHz PCM16
    frames = [str(i) for i in range(35)]
    trimmed, kept, duration = _align_loop_media(pcm, frames)
    assert duration == 35.0
    assert trimmed == pcm[:35 * 32000]
    assert kept == frames
    assert _align_loop_media(pcm[:64000], frames)[1:] == (frames[:2], 2.0)
    with pytest.raises(ValueError, match="complete second"):
        _align_loop_media(pcm[:30000], frames)


def _request_id(session_id: str, sequence: int) -> str:
    encoded = base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")
    return f"duplex-s.{encoded}.i.0.e.0.r.stage0-{sequence:08x}"


def _provenance(repo: str) -> dict[str, object]:
    return {
        "run_id": "a" * 32,
        "formal": True,
        "git": {"repo": repo, "head": "1" * 40, "dirty": False},
        "imports": {"vllm_omni": f"{repo}/vllm_omni/__init__.py"},
        "diagnostics": {"enabled": [], "cli_enabled": [], "all_disabled": True},
        "deploy_config": {
            "argument": "deploy.yaml",
            "resolved_path": f"{repo}/deploy.yaml",
            "sha256": "2" * 64,
            "size_bytes": 123,
            "exists": True,
        },
    }


def _clean_run(repo: str, **values: object) -> dict[str, object]:
    return {
        "client_provenance": _provenance(repo),
        "input_stream_complete": True,
        **values,
    }


def test_formal_environment_removes_every_diagnostic_switch() -> None:
    source = {name: "1" for name in DIAGNOSTIC_ENV_VARS}
    source["CUDA_VISIBLE_DEVICES"] = "0,1"

    cleaned, removed = formal_environment(source)

    assert set(removed) == set(DIAGNOSTIC_ENV_VARS)
    assert not set(DIAGNOSTIC_ENV_VARS).intersection(cleaned)
    assert cleaned["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert source["MINICPMO45_LOG_PREP_DIAG"] == "1"


def test_launch_environment_finds_adjacent_ninja_and_pip_cuda(tmp_path) -> None:
    environment = tmp_path / "env"
    python = environment / "bin" / "python"
    ninja = environment / "bin" / "ninja"
    cuda = environment / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    nvcc = cuda / "bin" / "nvcc"
    cudart = cuda / "lib64" / "libcudart.so"
    for executable in (python, ninja, nvcc):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
    cudart.parent.mkdir(parents=True)
    cudart.write_bytes(b"")

    prepared = launch_environment(
        {"PATH": "/usr/bin"},
        python_executable=python,
        purelib=environment / "lib" / "python3.12" / "site-packages",
    )

    assert prepared["CUDA_HOME"] == str(cuda.resolve())
    assert prepared["PATH"].split(":")[:2] == [
        str(environment.joinpath("bin").resolve()),
        str(cuda.joinpath("bin").resolve()),
    ]
    assert shutil.which("ninja", path=prepared["PATH"]) == str(ninja.resolve())
    assert shutil.which("nvcc", path=prepared["PATH"]) == str(nvcc.resolve())


def test_deploy_config_provenance_resolves_and_hashes_exact_yaml(tmp_path) -> None:
    deploy = tmp_path / "capacity.yaml"
    payload = b"stages:\n  - stage_id: 0\n"
    deploy.write_bytes(payload)

    result = _deploy_config_provenance(
        tmp_path,
        ["python", "-m", "server", "--deploy-config", deploy.name],
        cwd=tmp_path,
    )

    assert result == {
        "argument": deploy.name,
        "resolved_path": str(deploy.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "exists": True,
    }


@pytest.mark.parametrize("functional_only", [False, True])
def test_formal_capacity_workload_requires_long_randomized_production_run(functional_only) -> None:
    run = {
        "config": {
            "duration_s": 180,
            "max_send_drift_ms": 10,
            "workload_profile": "production",
            "phase_window_s": 1.0,
            "force_listen_count": None,
            "server_trace_frame_audit_requested": False,
        },
        "users": [{"phase_s": phase, "units_sent": 180, "send_drift_ms": {"count": 900, "max": 2}}
                  for phase in (0.1, 0.7)],
    }

    if functional_only:
        run["measurement_purpose"] = "serving_contract_validation"
    result = _formal_capacity_workload_validity(run)

    if functional_only:
        assert result["valid"] is False
        assert result["classification"] == "serving_contract_validation"
        assert result["violations"] == ["capacity_measurement_requested"]
        return

    assert result["valid"] is True
    assert result["classification"] == "formal_capacity"
    assert not result["violations"]


@pytest.mark.parametrize(
    ("update", "violation", "classification"),
    [
        ({"duration_s": 30}, "duration_at_least_180s", "development_screening"),
        (
            {"workload_profile": "synchronized"},
            "production_workload_profile",
            "diagnostic_control",
        ),
        ({"force_listen_count": 1}, "force_listen_disabled", "diagnostic_control"),
    ],
)
def test_nonformal_workloads_remain_classified_but_invalid(
    update: dict[str, object], violation: str, classification: str
) -> None:
    config = {
        "duration_s": 180,
        "workload_profile": "production",
        "phase_window_s": 1.0,
        "force_listen_count": None,
        "server_trace_frame_audit_requested": False,
        **update,
    }
    result = _formal_capacity_workload_validity(
        {"config": config, "users": [{"phase_s": 0.1}, {"phase_s": 0.7}]}
    )

    assert result["valid"] is False
    assert result["classification"] == classification
    assert violation in result["violations"]


@pytest.mark.parametrize(
    "argument",
    [
        spelling
        for flag in DIAGNOSTIC_CLI_FLAGS
        for spelling in (flag, f"{flag}=true")
    ],
)
def test_formal_server_rejects_diagnostic_cli_flags(
    tmp_path, monkeypatch, argument: str
) -> None:
    provenance_path = tmp_path / "provenance.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_server.py",
            "--provenance-out",
            str(provenance_path),
            "--",
            "/bin/echo",
            argument,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        clean_server_main()

    assert exc_info.value.code == 2
    assert not provenance_path.exists()


def test_diagnostic_server_preserves_cli_flags_in_provenance(
    tmp_path, monkeypatch
) -> None:
    provenance_path = tmp_path / "provenance.json"
    command = [
        "/bin/echo",
        "--enable-ar-profiler",
        "--enable-orch-monitor=true",
        "--enable-diffusion-pipeline-profiler=1",
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_server.py",
            "--provenance-out",
            str(provenance_path),
            "--allow-diagnostics",
            "--",
            *command,
        ],
    )

    launched: dict[str, object] = {}

    def fake_execvpe(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        launched.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(
        "benchmarks.minicpmo.clean_server.os.execvpe",
        fake_execvpe,
    )

    with pytest.raises(RuntimeError, match="exec intercepted"):
        clean_server_main()

    provenance = json.loads(provenance_path.read_text())
    assert launched["arguments"] == command
    assert provenance["launched_command"] == command
    assert provenance["formal"] is False
    assert provenance["diagnostics"]["cli_enabled"] == command[1:]
    assert provenance["diagnostics"]["all_disabled"] is False


@pytest.mark.parametrize("extra_remote_replay", [False, True])
def test_cleanliness_accepts_complete_clean_prefix_lineage(tmp_path, extra_remote_replay) -> None:
    repo = str(tmp_path / "repo")
    log = tmp_path / "server.log"
    log.write_text(
        "MiniCPM-o duplex append produced 491 embeddings but the scheduler "
        "reserved only 490 prompt slots; the tail will be truncated.\n"
        f"[benchmark-provenance] run_id={'a' * 32} captured_epoch=1.0\n"
        f"[benchmark-provenance] run_id={'b' * 32} captured_epoch=2.0\n"
        "[duplex_cadence] belongs_to_the_next_run\n"
    )

    result = _benchmark_cleanliness(
        log,
        _clean_run(
            repo,
            users=[
                {
                    "session_id": "bench-a",
                    "pd_completion_witness": {
                        "records": [
                            {
                                "sequence": 4,
                                "prompt_tokens": 1000,
                                "cached_tokens": 999 - int(extra_remote_replay),
                                "local_cached_tokens": 500,
                                "external_cached_tokens": 499 - int(extra_remote_replay),
                                "computed_tokens": 1 + int(extra_remote_replay),
                                "kv_transfer_selected_blocks": 32,
                                "kv_transfer_selected_tokens": 499,
                                "kv_transfer_selected_bytes": 262_144,
                                "kv_transfer_write_submit_to_d_ready_ms": -1.0,
                            },
                            {
                                "sequence": 5,
                                "prompt_tokens": 1210,
                                "cached_tokens": 1209,
                                "local_cached_tokens": 1200,
                                "external_cached_tokens": 9,
                                "computed_tokens": 1,
                                "kv_transfer_selected_blocks": 1,
                                "kv_transfer_selected_tokens": 9,
                                "kv_transfer_selected_bytes": 8192,
                                "kv_transfer_write_submit_to_d_ready_ms": -1.0,
                            },
                        ]
                    },
                }
            ],
        ),
        {"bench-a": (4, 5)},
        2,
        True,
        _provenance(repo),
    )

    if extra_remote_replay:
        assert result["valid"] is False
        assert result["decode_prefix_cache"]["mismatches"] == 1
        assert result["physical_d_kv_transfer"]["valid"] is False
        return
    assert result["valid"] is True
    assert result["decode_prefix_cache"]["records"] == 2
    assert (
        result["decode_prefix_cache"]["source"]
        == "client_physical_completion_witness"
    )
    assert result["decode_prefix_cache"]["mismatches"] == 0
    assert result["decode_prefix_cache"]["expected_uncached_suffix_tokens"] == [1]
    assert result["decode_prefix_cache"]["expected_locally_computed_tokens"] == [1]
    assert result["checks"]["physical_d_kv_transfer_evidence_complete"] is True
    assert result["physical_d_kv_transfer"]["valid"] is True
    assert (
        result["physical_d_kv_transfer"]["write_timing_unavailable_records"]
        == 2
    )


def test_cleanliness_fails_closed_on_truncation_diagnostics_and_cache_miss(tmp_path) -> None:
    repo = str(tmp_path / "repo")
    log = tmp_path / "server.log"
    log.write_text(
        "\n".join(
            [
                f"[benchmark-provenance] run_id={'a' * 32} captured_epoch=1.0",
                "[duplex_cadence] stage=0 ADMIT req=req generation=1 admit_epoch=1.0",
                "MiniCPM-o duplex append produced 491 embeddings but the scheduler "
                "reserved only 490 prompt slots; the tail will be truncated.",
                f"[prefix-cache] request={_request_id('bench-a', 4)} "
                "hit_tokens=900 prompt_tokens=1000",
                "Prepared P/D prefix changed before activation; using ordinary remote-prefill path",
                # This is the actual aggregate line emitted by vLLM's metrics
                # logger; it must invalidate a formal delta-only run.
                "Preemptions: 1",
            ]
        )
    )

    result = _benchmark_cleanliness(
        log,
        _clean_run(repo),
        {"bench-a": (4, 4)},
        1,
        True,
        _provenance(repo),
    )

    assert result["valid"] is False
    assert result["truncation_count"] == 1
    assert result["decode_prefix_cache"]["mismatches"] == 1
    assert result["diagnostic_marker_counts"]["duplex_cadence"] == 1
    assert result["runtime_failure_counts"]["pd_prefix_changed"] == 1
    assert result["runtime_failure_counts"]["remote_prefill_fallback"] == 1
    assert result["preemption_or_eviction_count"] == 1


def test_dirty_source_tree_invalidates_formal_result_without_hiding_analysis(
    tmp_path,
) -> None:
    repo = str(tmp_path / "repo")
    log = tmp_path / "server.log"
    log.write_text(f"[benchmark-provenance] run_id={'a' * 32} captured_epoch=1.0\n")
    provenance = deepcopy(_provenance(repo))
    provenance["git"]["dirty"] = True

    result = _benchmark_cleanliness(
        log,
        _clean_run(repo),
        {},
        None,
        True,
        provenance,
    )

    assert result["valid"] is False
    assert result["checks"]["server_source_tree_clean"] is False
    assert "server_source_tree_clean" in result["violations"]
    # Runtime diagnostics are still returned for a development measurement.
    assert result["diagnostic_marker_counts"]


def test_client_stage_metrics_are_an_exact_pd_completion_witness() -> None:
    stage1 = {
        "request_id": "duplex-request",
        "engine_request_id": "duplex-request-0000002a",
        "batch_id": 7,
        # Logical response accumulation may make this much larger; the
        # physical witness must ignore it.
        "num_tokens_in": 99_999,
        "engine_prompt_tokens": 10_002,
        "num_cached_tokens": 10_000,
        "stage_gen_time_ms": 123.0,
        "completed_epoch_s": 20.2,
    }
    client = SimpleNamespace(
        events=SimpleNamespace(
            events=[
                {"type": "response.listen", "vllm_omni": {"stage_metrics": {"1": stage1}}},
                # Repeated protocol projection must not double-count one D request.
                {"type": "response.audio.delta", "metadata": {"vllm_omni": {"stage_metrics": {"1": stage1}}}},
            ],
            event_received_at_s=[20.25, 20.5],
        )
    )

    records = _pd_completion_records(
        client,
        sequence_start=42,
        sequence_end=42,
        unit_ready_at=[20.0],
        unit_ready_epoch_s=[20.0],
        model_unit_ready_at=[20.15],
        model_unit_ready_epoch_s=[20.15],
    )

    assert records == [
        {
            "sequence": 42,
            "input_unit_index": 42,
            "source": "real_input",
            "request_id": "duplex-request-0000002a",
            "ready_at_s": 20.0,
            "first_media_arrival_at_s": 20.0,
            "model_unit_ready_at_s": 20.15,
            "done_at_s": 20.2,
            "e2e_ms": 200.0,
            "input_start_e2e_ms": 200.0,
            "ready_to_d_ms": 50.0,
            "input_aggregation_ms": 150.0,
            "completion_clock": "engine_stage_epoch",
            "prompt_tokens": 10_002,
            "cached_tokens": 10_000,
            "local_cached_tokens": -1,
            "external_cached_tokens": -1,
            "computed_tokens": -1,
            "uncached_suffix_tokens": 2,
            "d_service_ms": 123.0,
            "batch_id": 7,
            "input_video_frames": 0,
            "arrival_video_frames": 0,
            "vision_fallback_frames": 0,
            "arrival_audio_units": 0,
            "audio_fallback_units": 0,
            "kv_transfer_selected_blocks": -1,
            "kv_transfer_selected_tokens": -1,
            "kv_transfer_selected_bytes": -1,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        }
    ]
    assert (
        _pd_completion_records(
            client,
            sequence_start=42,
            sequence_end=42,
            unit_ready_at=[20.0],
            unit_ready_epoch_s=[20.0],
            not_after_s=20.2,
        )
        == []
    )


def test_compact_metadata_is_an_exact_pd_completion_witness() -> None:
    witness = {
        "stage_id": 1,
        "engine_request_id": "duplex-request-0000002a",
        "prompt_tokens": 10_002,
        "cached_tokens": 10_000,
        "batch_id": 7,
        "submit_epoch_s": 20.05,
        "completed_epoch_s": 20.2,
        "service_ms": 150.0,
        "arrival_audio_units": 1,
        "audio_fallback_units": 0,
    }
    client = SimpleNamespace(
        events=SimpleNamespace(
            events=[
                {
                    "type": "response.audio.delta",
                    "metadata": {"vllm_omni": {"completion_witness": witness}},
                }
            ],
            event_received_at_s=[20.25],
        )
    )

    records = _pd_completion_records(
        client,
        sequence_start=42,
        sequence_end=42,
        unit_ready_at=[20.0],
        unit_ready_epoch_s=[20.0],
    )

    assert len(records) == 1
    assert records[0]["request_id"] == "duplex-request-0000002a"
    assert records[0]["input_unit_index"] == 42
    assert records[0]["e2e_ms"] == 200.0
    # Legacy callers without a second timestamp remain valid.  The analyzer
    # must not invent an input-aggregation interval for old artifacts.
    assert records[0]["first_media_arrival_at_s"] == 20.0
    assert records[0]["input_start_e2e_ms"] == 200.0
    assert "model_unit_ready_at_s" not in records[0]
    assert "ready_to_d_ms" not in records[0]
    assert "input_aggregation_ms" not in records[0]
    assert records[0]["uncached_suffix_tokens"] == 2
    assert records[0]["d_service_ms"] == 150.0
    assert records[0]["arrival_audio_units"] == 1
    assert records[0]["audio_fallback_units"] == 0
    assert records[0]["kv_transfer_selected_blocks"] == -1
    assert records[0]["kv_transfer_selected_tokens"] == -1
    assert records[0]["kv_transfer_selected_bytes"] == -1
    assert records[0]["kv_transfer_write_submit_to_d_ready_ms"] == -1.0


def test_completion_witness_matches_real_input_identity_not_physical_sequence() -> None:
    def event(
        physical_sequence: int,
        *,
        input_unit_index: int,
        source: str,
    ) -> dict[str, object]:
        request_id = f"duplex-request-{physical_sequence:08x}"
        return {
            "type": "response.model_unit.done",
            "vllm_omni": {
                "completion_witness": {
                    "stage_id": 1,
                    "engine_request_id": request_id,
                    "physical_sequence": physical_sequence,
                    "input_unit_index": input_unit_index,
                    "source": source,
                    "prompt_tokens": 102,
                    "cached_tokens": 100,
                    "local_cached_tokens": 80,
                    "external_cached_tokens": 20,
                    "computed_tokens": 2,
                    "kv_transfer_selected_blocks": 2,
                    "kv_transfer_selected_tokens": 20,
                    "kv_transfer_selected_bytes": 16_384,
                    "kv_transfer_write_submit_to_d_ready_ms": -1.0,
                    "completed_epoch_s": 20.2,
                    "service_ms": 50.0,
                }
            },
        }

    client = SimpleNamespace(
        events=SimpleNamespace(
            events=[
                {
                    "type": "duplex.response.model_unit.done",
                    "event": event(
                        42,
                        input_unit_index=7,
                        source="real_input",
                    ),
                },
                event(43, input_unit_index=43, source="auto_continuation"),
            ],
            event_received_at_s=[20.25, 20.3],
        )
    )

    records = _pd_completion_records(
        client,
        sequence_start=7,
        sequence_end=7,
        unit_ready_at=[20.0],
        unit_ready_epoch_s=[20.0],
    )

    assert len(records) == 1
    assert records[0]["sequence"] == 42
    assert records[0]["input_unit_index"] == 7
    assert records[0]["source"] == "real_input"
    assert records[0]["computed_tokens"] == 2
    assert records[0]["kv_transfer_selected_blocks"] == 2
    assert records[0]["kv_transfer_selected_tokens"] == 20
    assert records[0]["kv_transfer_selected_bytes"] == 16_384
    assert records[0]["kv_transfer_write_submit_to_d_ready_ms"] == -1.0


def test_client_frame_audit_counts_arrival_hits_and_fallbacks() -> None:
    run = {
        "pd_completion_witness": {
            "source": "client-visible physical D completion witness"
        },
        "users": [
            {
                "pd_completion_witness": {
                    "records": [
                        {
                            "input_video_frames": 1,
                            "arrival_video_frames": 1,
                            "vision_fallback_frames": 0,
                            "arrival_audio_units": 1,
                            "audio_fallback_units": 0,
                        },
                        {
                            "input_video_frames": 1,
                            "arrival_video_frames": 0,
                            "vision_fallback_frames": 1,
                            "arrival_audio_units": 0,
                            "audio_fallback_units": 1,
                        },
                    ]
                }
            }
        ],
    }

    assert _client_frame_audit(run) == {
        "source": "client_physical_completion_witness",
        "units": 2,
        "frames_consumed": 2,
        "by_source": {"arrival": 1, "formal_fallback": 1},
    }
    assert _client_audio_sidecar_audit(run) == {
        "source": "client_physical_completion_witness",
        "records": 2,
        "audited_records": 2,
        "malformed_records": 0,
        "arrival_audio_units": 1,
        "audio_fallback_units": 1,
    }

    del run["users"][0]["pd_completion_witness"]["records"][1][
        "audio_fallback_units"
    ]
    assert _client_audio_sidecar_audit(run)["malformed_records"] == 1


def test_audio_sidecar_fallback_invalidates_cleanliness(tmp_path) -> None:
    repo = str(tmp_path / "repo")
    log = tmp_path / "server.log"
    log.write_text(f"[benchmark-provenance] run_id={'a' * 32} captured_epoch=1.0\n")
    record = {
        "sequence": 1,
        "prompt_tokens": 1000,
        "cached_tokens": 999,
        "local_cached_tokens": 500,
        "external_cached_tokens": 499,
        "computed_tokens": 1,
        "kv_transfer_selected_blocks": 32,
        "kv_transfer_selected_tokens": 499,
        "kv_transfer_selected_bytes": 262_144,
        "kv_transfer_write_submit_to_d_ready_ms": -1.0,
        "input_video_frames": 1,
        "arrival_video_frames": 1,
        "vision_fallback_frames": 0,
        "arrival_audio_units": 0,
        "audio_fallback_units": 1,
    }

    result = _benchmark_cleanliness(
        log,
        _clean_run(
            repo,
            frames_sent=1,
            pd_completion_witness={
                "source": "client-visible physical D completion witness"
            },
            users=[
                {
                    "session_id": "bench-a",
                    "pd_completion_witness": {"records": [record]},
                }
            ],
        ),
        {"bench-a": (1, 1)},
        1,
        True,
        _provenance(repo),
    )

    assert result["checks"]["client_audio_sidecar_complete"] is False
    assert result["checks"]["zero_runtime_fallback"] is False
    assert result["client_audio_sidecar_audit"]["complete"] is False
    assert result["client_audio_sidecar_audit"]["audio_fallback_units"] == 1


@pytest.mark.parametrize("functional_only", [False, True])
def test_clean_formal_analysis_needs_no_per_request_server_log(tmp_path, monkeypatch, functional_only) -> None:
    log = tmp_path / "server.log"
    run_path = tmp_path / "run.json"
    provenance_path = tmp_path / "provenance.json"
    output = tmp_path / "analysis.json"
    log.write_text(f"[benchmark-provenance] run_id={'a' * 32} captured_epoch=1.0\n")
    records = [
        {
            "sequence": 1,
            "input_unit_index": 1,
            "ready_at_s": 10.0,
            "done_at_s": 10.4,
            "e2e_ms": 400.0,
            "prompt_tokens": 1000,
            "cached_tokens": 999,
            "local_cached_tokens": 500,
            "external_cached_tokens": 499,
            "computed_tokens": 1,
            "kv_transfer_selected_blocks": 32,
            "kv_transfer_selected_tokens": 499,
            "kv_transfer_selected_bytes": 262_144,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
            "input_video_frames": 1,
            "arrival_video_frames": 1,
            "vision_fallback_frames": 0,
            "arrival_audio_units": 1,
            "audio_fallback_units": 0,
        },
        {
            "sequence": 2,
            "input_unit_index": 2,
            "ready_at_s": 11.0,
            "done_at_s": 11.5,
            "e2e_ms": 500.0,
            "prompt_tokens": 1210,
            "cached_tokens": 1209,
            "local_cached_tokens": 1200,
            "external_cached_tokens": 9,
            "computed_tokens": 1,
            "kv_transfer_selected_blocks": 1,
            "kv_transfer_selected_tokens": 9,
            "kv_transfer_selected_bytes": 8192,
            "kv_transfer_write_submit_to_d_ready_ms": -1.0,
            "input_video_frames": 1,
            "arrival_video_frames": 1,
            "vision_fallback_frames": 0,
            "arrival_audio_units": 1,
            "audio_fallback_units": 0,
        },
    ]
    run_path.write_text(
        json.dumps(
            {
                "measurement_purpose": "serving_contract_validation" if functional_only else "capacity_exploration",
                "started_epoch_s": 1.0,
                "ended_epoch_s": 2.0,
                "failed_users": 0,
                "frames_sent": 2,
                "input_stream_complete": True,
                "config": {
                    "duration_s": 2,
                    "workload_profile": "production",
                    "phase_window_s": 1.0,
                    "force_listen_count": None,
                    "server_trace_frame_audit_requested": False,
                    "frame_audit_required": False,
                    "audio_sidecar_audit_required": True,
                },
                "client_provenance": _provenance(str(tmp_path / "repo")),
                "pd_completion_witness": {
                    "source": "client-visible physical D completion witness",
                    "expected": 2,
                    "completed": 2,
                    "complete": True,
                },
                "users": [
                    {
                        "session_id": "bench-a",
                        "phase_s": 0.25,
                        "formal_seq_start": 1,
                        "formal_seq_end": 2,
                        "units_sent": 2,
                        "pd_completion_witness": {
                            "expected": 2,
                            "completed": 2,
                            "complete": True,
                            "records": records,
                        },
                    }
                ],
            }
        )
    )
    provenance_path.write_text(json.dumps(_provenance(str(tmp_path / "repo"))))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_rtf.py",
            "--server-log",
            str(log),
            "--server-provenance-json",
            str(provenance_path),
            "--run-json",
            str(run_path),
            "--out",
            str(output),
        ],
    )

    analyze_main()

    result = json.loads(output.read_text())
    assert result["measurement_completion_witness"] == (
        "client_physical_d_completion_witness"
    )
    assert result["completed_input_units"] == {"0": 2, "1": 2}
    assert result["completed_input_units_source"] == (
        "client_physical_d_completion_witness"
    )
    assert result["server_cadence_completed_input_units"] == {"0": 0, "1": 0}
    assert result["measurement_complete"] is True
    assert result["benchmark_cleanliness"]["valid"] is True
    assert result["benchmark_cleanliness"]["checks"][
        "client_audio_sidecar_complete"
    ] is True
    assert result["pd_slot"]["physical_kv_transfer"]["valid"] is True
    assert (
        result["pd_slot"]["physical_kv_transfer"]
        ["write_timing_unavailable_records"]
        == 2
    )
    assert result["benchmark_valid"] is False
    assert result["capacity_pass"] is False
    assert result["formal_capacity_validity"]["classification"] == (
        "serving_contract_validation" if functional_only else "development_screening"
    )
    assert result["benchmark_invalid_reasons"] == (["workload:capacity_measurement_requested"] if functional_only else []) + [
        "workload:duration_at_least_180s", "workload:sender_timing_valid"
    ]
    if functional_only:
        assert result["input_capacity_pass"] is False
        assert result["end_to_end_capacity_pass"] is False
        assert result["capacity_exclusion_reason"]
    assert result["frame_audit"]["source"] == "client_physical_completion_witness"
    assert result["frame_audit"]["complete"] is True
    assert result["audio_sidecar_audit"]["arrival_audio_units"] == 2
    assert result["audio_sidecar_audit"]["audio_fallback_units"] == 0
    assert result["audio_sidecar_audit"]["expected_units"] == 2
    assert result["audio_sidecar_audit"]["complete"] is True
    assert not any(result["benchmark_cleanliness"]["diagnostic_marker_counts"].values())
