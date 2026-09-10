"""Keep the non-P/D capacity comparison limited to GPU placement."""

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config


def test_nonpd_colocation_changes_only_placement():
    root = Path(__file__).resolve().parents[5]
    folder = root / "benchmarks/minicpmo"
    colocated = load_deploy_config(folder / "deploy_capacity_nonpd_colocated_fp8.yaml")
    separated = load_deploy_config(folder / "deploy_capacity_nonpd_separated_fp8.yaml")
    assert colocated.pipeline == separated.pipeline == "minicpmo_4_5"
    assert [stage.devices for stage in colocated.stages] == ["0", "0", "0"]
    assert [stage.devices for stage in separated.stages] == ["0", "0", "2"]
    for a, b in zip(colocated.stages, separated.stages, strict=True):
        a_fields = vars(a).copy()
        b_fields = vars(b).copy()
        a_fields.pop("devices")
        b_fields.pop("devices")
        if a.stage_id == 0:
            a_env = dict(a_fields.pop("env"))
            b_env = dict(b_fields.pop("env"))
            assert b_env.pop("VLLM_OMNI_AUX_VISIBLE_DEVICES") == "1"
            for key in ("MINICPMO45_VISION_ENCODER_DEVICE", "MINICPMO45_AUDIO_ENCODER_DEVICE"):
                assert a_env.pop(key) == "cuda:0"
                assert b_env.pop(key) == "cuda:1"
            assert a_env == b_env
        assert a_fields == b_fields


def test_three_gpu_mps_changes_only_auxiliary_device_notation(tmp_path):
    from benchmarks.minicpmo.run_placement import mps_deploy_config

    folder = Path(__file__).resolve().parents[5] / "benchmarks/minicpmo"
    config = folder / "deploy_capacity_nonpd_separated_fp8.yaml"
    before = load_deploy_config(config)
    uuids = ["GPU-first", "GPU-second", "GPU-third"]
    after = load_deploy_config(mps_deploy_config(config, "split3", uuids, tmp_path))
    for a, b in zip(before.stages, after.stages, strict=True):
        expected = vars(a).copy()
        if a.stage_id == 0:
            expected["env"] = dict(expected["env"], VLLM_OMNI_AUX_VISIBLE_DEVICES=uuids[1])
        assert vars(b) == expected
    assert mps_deploy_config(config, "all1", uuids[:1], tmp_path) == config


@pytest.mark.parametrize("visible", ["2", "GPU-third", "GPU-third,GPU-second"])
def test_init_lock_uses_physical_device_for_uuid_visibility(tmp_path, monkeypatch, visible):
    import os

    import pynvml

    from vllm_omni.engine import stage_init_utils as init

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetHandleByUUID", lambda uuid: uuid)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetIndex", lambda uuid: {"GPU-third": 2, "GPU-second": 1}[uuid])
    opened = []
    real_open = os.open

    def open_local_lock(path, flags, mode):
        opened.append(path)
        return real_open(tmp_path / Path(path).name, flags, mode)

    monkeypatch.setattr(init.os, "open", open_local_lock)
    fds = init.acquire_device_locks(2, {"tensor_parallel_size": 1}, 1)
    try:
        assert len(fds) == 1
        assert opened == ["/tmp/vllm_omni_device_2_init.lock"]
    finally:
        init.release_device_locks(fds)


def test_nonpd_thinker_emits_one_complete_unit_without_changing_downstream_streaming():
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    from vllm_omni.experimental.fullduplex.minicpmo45.runtime import MiniCPMO45DuplexRuntimeExtension

    defaults = tuple(SamplingParams(output_kind=RequestOutputKind.DELTA) for _ in range(3))
    configured = MiniCPMO45DuplexRuntimeExtension().configure_sampling_params(
        runtime_config={"duplex_stage_max_tokens": {"0": 20, "1": 8192}},
        defaults=defaults,
    )
    assert configured[0].output_kind == RequestOutputKind.FINAL_ONLY
    assert configured[0].max_tokens == 20
    assert configured[1].output_kind == RequestOutputKind.DELTA
    assert configured[2].output_kind == RequestOutputKind.DELTA


@pytest.mark.parametrize("placement", ["encoder_out", "code2wav_out", "talker_out"])
def test_leave_one_out_changes_only_one_module_placement(placement):
    folder = Path(__file__).resolve().parents[5] / "benchmarks/minicpmo"
    baseline = load_deploy_config(folder / "deploy_capacity_nonpd_colocated_fp8.yaml")
    moved = load_deploy_config(folder / f"deploy_capacity_nonpd_{placement}_fp8.yaml")
    assert baseline.pipeline == moved.pipeline == "minicpmo_4_5"
    for before, after in zip(baseline.stages, moved.stages, strict=True):
        expected = vars(before).copy()
        if placement == "encoder_out" and before.stage_id == 0:
            expected["env"] = dict(expected["env"])
            expected["env"].update(
                {
                    "VLLM_OMNI_AUX_VISIBLE_DEVICES": "1",
                    "MINICPMO45_VISION_ENCODER_DEVICE": "cuda:1",
                    "MINICPMO45_AUDIO_ENCODER_DEVICE": "cuda:1",
                }
            )
        if placement == "code2wav_out" and before.stage_id == 2:
            expected["devices"] = "2"
        if placement == "talker_out" and before.stage_id == 1:
            expected["devices"] = "1"
        assert expected == vars(after)
