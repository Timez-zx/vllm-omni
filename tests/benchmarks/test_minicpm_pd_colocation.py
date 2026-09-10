"""Device-only co-location: do not change the serving pipeline or workload."""

import ast
import copy

import pytest
from test_minicpm_pd_placement_diagnostics import SOURCE, _execute, _parse_cli


@pytest.mark.parametrize("topology,encoder,decoder", [
    ("3gpu-encoder-p", 0, 2),
    ("3gpu-encoder-d", 1, 2),
    ("3gpu-decoder-p", 2, 0),
    ("3gpu-decoder-d", 2, 1),
])
def test_colocation_changes_only_device_placement(monkeypatch, tmp_path, topology, encoder, decoder):
    args, main = _parse_cli(monkeypatch, tmp_path, ["--topology", topology])
    state = {}
    helper = next(n for n in ast.parse(SOURCE.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == "_apply_three_gpu_placement")
    _execute([helper], state)
    apply = state["_apply_three_gpu_placement"]
    resolved = {"async_chunk": True, "stages": [
        {"stage_id": sid, "devices": str(sid), "async_scheduling": sid < 2,
         "env": {"KEEP": "unchanged"}, "kv_cache_memory_bytes": (32 if sid < 2 else 8) * 1024**3}
        for sid in range(4)
    ]}
    original = copy.deepcopy(resolved)
    uuids = [f"GPU-{i}" for i in range(3)]
    apply(resolved, topology, uuids)
    assert [s["devices"] for s in resolved["stages"]] == ["0", "1", "1", str(decoder)]
    env = resolved["stages"][0]["env"]
    assert env.pop("VLLM_OMNI_AUX_VISIBLE_DEVICES") == ("" if encoder == 0 else uuids[encoder])
    for modality in ("VISION", "AUDIO"):
        assert env.pop(f"MINICPMO45_{modality}_ENCODER_DEVICE") == ("cuda:0" if encoder == 0 else "cuda:1")
    for stage, old in zip(resolved["stages"], original["stages"]):
        stage["devices"] = old["devices"]
    assert resolved == original
    with pytest.raises(ValueError, match="exactly three"):
        apply(resolved, topology, uuids + ["GPU-3"])
    assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "client_cmd")
    import sys
    state = {"args": args, "sys": sys, "out": tmp_path}
    _execute([assignment], state)
    cmd = state["client_cmd"]
    assert cmd[cmd.index("--gpus") + 1:cmd.index("--out")] == ["0", "1", "2"]
    assert cmd[cmd.index("--max-slice-nums") + 1] == "4"
    assert cmd[cmd.index("--connect-stagger-s") + 1] == "0"
