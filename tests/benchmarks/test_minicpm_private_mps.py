"""MPS benchmarks must use private state and prove actual client attachment."""

import subprocess

import pytest

from benchmarks.minicpmo.private_mps import PrivateMPS


def test_mps_rejects_ambient_policy(tmp_path):
    with pytest.raises(RuntimeError, match="inherited MPS"):
        PrivateMPS(tmp_path, {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "20"}, "GPU-test")


def test_mps_private_environment_attachment_and_cleanup(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        query = kwargs.get("input", "").strip()
        output = {"get_server_list": "999\n", "get_client_list 999": "11\n12\n13\n"}.get(query, "")
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("benchmarks.minicpmo.private_mps.subprocess.run", run)
    monkeypatch.setattr("benchmarks.minicpmo.private_mps.tempfile.mkdtemp", lambda **_: str(tmp_path / "pipes"))
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"(StageEngineCoreProc_stage{s}_replica0 pid={11+s})" for s in range(3)))
    original = {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    with PrivateMPS(tmp_path, original, "GPU-test") as mps:
        assert mps.client_env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert mps.client_env["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "pipes")
        assert mps.daemon_env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
        assert mps.snapshot("ready", log)["all_stages_connected"]
    assert original == {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    assert calls[-1][1]["input"] == "quit\n"
    assert all(item[1]["env"]["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "pipes") for item in calls)


def test_mps_refuses_to_measure_a_stage_not_connected(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.minicpmo.private_mps.tempfile.mkdtemp", lambda **_: str(tmp_path / "pipes"))
    mps = PrivateMPS(tmp_path, {}, "GPU-test")
    monkeypatch.setattr(mps, "command", lambda text: "999" if text == "get_server_list" else "11\n12")
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"(StageEngineCoreProc_stage{s}_replica0 pid={11+s})" for s in range(3)))
    with pytest.raises(RuntimeError, match="Not all 3 stages"):
        mps.snapshot("ready", log)


def test_mps_three_gpu_visibility_preserves_device_order(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.minicpmo.private_mps.tempfile.mkdtemp", lambda **_: str(tmp_path / "pipes"))
    uuids = ["GPU-first", "GPU-second", "GPU-third"]
    mps = PrivateMPS(tmp_path, {}, uuids)
    assert mps.client_env["CUDA_VISIBLE_DEVICES"] == ",".join(uuids)
    assert mps.daemon_env["CUDA_VISIBLE_DEVICES"] == ",".join(uuids)
    assert mps.record["gpu_uuids"] == uuids
    assert mps.record["gpu_uuid"] is None


def test_pd_mps_requires_fourth_stage_attachment(tmp_path, monkeypatch):
    mps = PrivateMPS(tmp_path, {}, [f"GPU-{i}" for i in range(4)], stage_count=4)
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"(StageEngineCoreProc_stage{s}_replica0 pid={11+s})" for s in range(4)))
    monkeypatch.setattr(mps, "command", lambda text: "999" if text == "get_server_list" else "11\n12\n13")
    with pytest.raises(RuntimeError, match="Not all 4 stages"):
        mps.snapshot("missing_decoder", log)
    monkeypatch.setattr(mps, "command", lambda text: "999" if text == "get_server_list" else "11\n12\n13\n14")
    assert mps.snapshot("all_connected", log)["all_stages_connected"]


@pytest.mark.parametrize("uuids", [[], ["0", "1"], ["GPU-a", "GPU-a"], ["GPU-a,GPU-b"]])
def test_mps_rejects_ambiguous_device_visibility(tmp_path, uuids):
    with pytest.raises(ValueError):
        PrivateMPS(tmp_path, {}, uuids)
