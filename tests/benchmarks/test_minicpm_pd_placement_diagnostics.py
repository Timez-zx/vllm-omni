"""Execute wrapper configuration branches without importing/starting GPU code."""

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "benchmarks/minicpmo/run_pd_placement.py"


def _execute(nodes, state):
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), state)


@pytest.mark.parametrize("speech,numerical", [(True, False), (False, True), (False, False)])
def test_diagnostic_probe_configuration_and_capacity_invalidation(tmp_path, speech, numerical):
    module = ast.parse(SOURCE.read_text())
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert any(isinstance(n, ast.Constant) and n.value == "--speech-probe-dir" for n in ast.walk(main))
    stage_loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "stage")
    parent_env = next(n for n in main.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "env")
    parent_probe = next(
        n for n in main.body if isinstance(n, ast.If) and ast.unparse(n.test) == "args.speech_probe_dir"
    )
    invalidation = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "args.numerical_probe_dir or args.speech_probe_dir"
    )
    out = tmp_path / "run"
    out.mkdir()
    args = SimpleNamespace(
        speech_probe_dir=tmp_path / "speech" if speech else None,
        numerical_probe_dir=tmp_path / "numeric" if numerical else None,
        numerical_probe_seqs=None,
        numerical_probe_sessions=None,
        numerical_probe_layer_io=False,
        numerical_probe_layers=None,
    )
    stages = [
        {"stage_id": 0, "hf_overrides": {"vllm_omni_minicpmo_pd_prefill": True}},
        {"stage_id": 1, "hf_overrides": {"vllm_omni_minicpmo_pd_decode": True}},
        {"stage_id": 2},  # Talker
        {"stage_id": 3},  # Code2Wav
    ]
    import json

    state = {
        "args": args,
        "resolved": {"stages": stages},
        "ROOT": SOURCE.parents[2],
        "out": out,
        "os": SimpleNamespace(environ={}),
        "json": json,
        "uuids": [f"GPU-{i}" for i in range(4)],
        "report": {"capacity_pass": True, "input_capacity_pass": True},
    }
    _execute([stage_loop, parent_env, parent_probe, invalidation], state)
    for env in [stage.get("env", {}) for stage in stages] + [state["env"]]:
        if speech:
            assert env["VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR"] == str(args.speech_probe_dir.resolve())
            assert env["VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS"] == "1"
        else:
            assert "VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR" not in env
            assert "VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS" not in env
    assert state["report"]["capacity_pass"] is (not speech and not numerical)
    assert state["report"]["input_capacity_pass"] is (not speech and not numerical)
    if speech or numerical:
        saved = json.loads((out / "analysis.json").read_text())
        assert not saved["capacity_pass"] and not saved["input_capacity_pass"]
        assert saved["diagnostic_only"]
    else:
        assert not (out / "analysis.json").exists()
    # AST execution used a private environment, never changed the caller.
    assert state["os"].environ is not os.environ


def _parse_cli(monkeypatch, tmp_path, extra=()):
    module = ast.parse(SOURCE.read_text())
    main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    ready_index = next(
        i for i, n in enumerate(main.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "healthy(args.port)"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SOURCE),
            "--topology",
            "d-talker",
            "--users",
            "2",
            "--duration-s",
            "420",
            "--out-dir",
            str(tmp_path),
            *extra,
        ],
    )
    state = {"argparse": argparse, "Path": Path, "json": json, "math": math, "__doc__": "CPU parser test"}
    helpers = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_probe_")]
    _execute(helpers + main.body[:ready_index], state)
    return state["args"], main


def test_functional_only_records_purpose_without_enabling_probes(monkeypatch, tmp_path):
    args, main = _parse_cli(monkeypatch, tmp_path, ["--functional-only"])
    assert args.functional_only is True
    assert args.numerical_probe_dir is args.speech_probe_dir is None
    assert args.kv_window_tokens == 36000 and args.kv_cache_dtype == "fp8"
    purpose = next(n for n in ast.walk(main) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "args.functional_only")
    state = {"args": args, "run_result": {"measurement_complete": False}}
    _execute([purpose], state)
    assert state["run_result"] == {"measurement_complete": False,
                                   "measurement_purpose": "serving_contract_validation"}


@pytest.mark.parametrize("seconds", [None, 90])
def test_post_stream_observation_is_explicit_without_changing_input(monkeypatch, tmp_path, seconds):
    extra = [] if seconds is None else ["--post-stream-s", str(seconds)]
    args, main = _parse_cli(monkeypatch, tmp_path, extra)
    assert args.post_stream_s == (30 if seconds is None else seconds)
    assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "client_cmd")
    state = {"args": args, "sys": sys, "out": tmp_path}
    _execute([assignment], state)
    command = state["client_cmd"]
    assert command[command.index("--post-stream-s") + 1] == str(args.post_stream_s)
    assert command[command.index("--duration-s") + 1] == "420"
    assert command[command.index("--expected-kv-window-tokens") + 1] == "36000"
    timeout = next(k.value for n in ast.walk(main) if isinstance(n, ast.Call)
                   and ast.unparse(n.func) == "subprocess.run"
                   and n.args and ast.unparse(n.args[0]) == "client_cmd"
                   for k in n.keywords if k.arg == "timeout")
    assert eval(compile(ast.Expression(timeout), str(SOURCE), "eval"), {"args": args}) >= 420 + args.post_stream_s + 120


def test_negative_post_stream_observation_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exc:
        _parse_cli(monkeypatch, tmp_path, ["--post-stream-s", "-1"])
    assert exc.value.code == 2


@pytest.mark.parametrize("limit", [None, 250.0])
def test_backlog_bound_is_recorded_without_changing_input_cadence(monkeypatch, tmp_path, limit):
    extra = [] if limit is None else ["--max-backlog-ms", str(limit)]
    args, main = _parse_cli(monkeypatch, tmp_path, extra)
    assert args.max_backlog_ms == (500 if limit is None else limit)
    assignment = next(n for n in main.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "client_cmd")
    state = {"args": args, "sys": sys, "out": tmp_path}
    _execute([assignment], state)
    command = state["client_cmd"]
    assert command[command.index("--max-backlog-ms") + 1] == str(args.max_backlog_ms)
    assert command[command.index("--duration-s") + 1] == "420"


@pytest.mark.parametrize("limit", ["-1", "nan", "inf"])
def test_invalid_backlog_cli_limit_is_rejected(monkeypatch, tmp_path, limit):
    with pytest.raises(SystemExit) as exc:
        _parse_cli(monkeypatch, tmp_path, ["--max-backlog-ms", limit])
    assert exc.value.code == 2


@pytest.mark.parametrize("selection", ["all", "none", "0", "0,17,35"])
def test_probe_kv_layer_selection_is_explicit_pd_only(monkeypatch, tmp_path, selection):
    args, main = _parse_cli(monkeypatch, tmp_path, [
        "--numerical-probe-dir", str(tmp_path / "probe"),
        "--numerical-probe-layers", selection,
    ])
    stages = [
        {"stage_id": 0, "hf_overrides": {"vllm_omni_minicpmo_pd_prefill": True}},
        {"stage_id": 1, "hf_overrides": {"vllm_omni_minicpmo_pd_decode": True}},
        {"stage_id": 2}, {"stage_id": 3},
    ]
    stage_loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "stage")
    _execute([stage_loop], {"args": args, "resolved": {"stages": stages}})
    for stage in stages[:2]:
        assert stage["enforce_eager"] is True
        assert stage["env"]["MINICPMO45_NUMERICAL_PROBE_LAYERS"] == selection
    assert all("env" not in stage for stage in stages[2:])


@pytest.mark.parametrize("extra", [
    ["--numerical-probe-layers", "none"],
    ["--numerical-probe-dir", "probe", "--numerical-probe-layers", "0-999999"],
    ["--numerical-probe-dir", "probe", "--numerical-probe-layers", "bogus"],
])
def test_invalid_or_orphan_probe_layers_fail_before_gpu(monkeypatch, tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        _parse_cli(monkeypatch, tmp_path, extra)
    assert exc.value.code == 2


@pytest.mark.parametrize("gib", [None, 1, 32])
@pytest.mark.parametrize("existing_bytes", [None, 64 * 1024**3])
def test_thinker_kv_memory_override_is_pd_only_and_archives_resolved_bytes(
    monkeypatch, tmp_path, gib, existing_bytes
):
    extra = [] if gib is None else ["--thinker-kv-cache-memory-gib", str(gib)]
    args, main = _parse_cli(monkeypatch, tmp_path, extra)
    assert args.thinker_kv_cache_memory_gib == gib
    stages = [
        {"stage_id": 0, "env": {}, "hf_overrides": {"vllm_omni_minicpmo_pd_prefill": True}},
        {"stage_id": 1, "env": {}, "hf_overrides": {"vllm_omni_minicpmo_pd_decode": True}},
        {"stage_id": 2, "kv_cache_memory_bytes": 8 * 1024**3, "quantization": "fp8"},
        {"stage_id": 3},
    ]
    if existing_bytes is not None:
        for stage in stages[:2]:
            stage["kv_cache_memory_bytes"] = existing_bytes
    loops = [n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "stage"]
    precision = next(
        n for n in main.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "deployment_precision"
    )
    state = {"args": args, "resolved": {"stages": stages}, "uuids": [f"GPU-{i}" for i in range(4)]}
    _execute(loops + [precision], state)
    expected = existing_bytes if gib is None else gib * 1024**3
    for stage in stages[:2]:
        assert stage.get("kv_cache_memory_bytes") == expected
        assert state["deployment_precision"][str(stage["stage_id"])]["kv_cache_memory_bytes"] == expected
        if expected is None:
            assert "kv_cache_memory_bytes" not in stage
        else:
            assert type(stage["kv_cache_memory_bytes"]) is int
    assert stages[2:] == [
        {"stage_id": 2, "kv_cache_memory_bytes": 8 * 1024**3, "quantization": "fp8"},
        {"stage_id": 3},
    ]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan", "abc", "True"])
def test_invalid_thinker_kv_memory_fails_before_gpu_checks(monkeypatch, tmp_path, value):
    with pytest.raises(SystemExit) as exc:
        _parse_cli(monkeypatch, tmp_path, ["--thinker-kv-cache-memory-gib", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize("force_2d", [False, True])
def test_q_precision_is_explicit_pd_only_and_saved(monkeypatch, tmp_path, disable, force_2d):
    extra = ["--triton-disable-q-quantization"] if disable else []
    if force_2d:
        extra.append("--triton-force-2d-attention")
    args, main = _parse_cli(monkeypatch, tmp_path, extra)
    stages = [
        {
            "stage_id": 0,
            "env": {},
            "hf_overrides": {"vllm_omni_minicpmo_pd_prefill": True},
            "additional_config": {"existing": "p"},
        },
        {"stage_id": 1, "env": {}, "hf_overrides": {"vllm_omni_minicpmo_pd_decode": True}},
        {"stage_id": 2, "additional_config": {"existing": "talker"}},
        {"stage_id": 3},
    ]
    loops = [n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "stage"]
    precision = next(
        n for n in main.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "deployment_precision"
    )
    state = {"args": args, "resolved": {"stages": stages}, "uuids": [f"GPU-{i}" for i in range(4)]}
    _execute(loops + [precision], state)
    for stage in stages[:2]:
        assert stage["additional_config"]["triton_disable_q_quantization"] is disable
        assert stage["additional_config"]["triton_force_2d_attention"] is force_2d
        assert stage["kv_cache_dtype"] == "fp8"
        assert stage["quantization"] == "fp8"
        assert stage["hf_overrides"]["sliding_window"] == 36000
        assert "enforce_eager" not in stage
        if disable or force_2d:
            assert stage["attention_config"]["backend"] == "TRITON_ATTN"
        else:
            assert "attention_config" not in stage
    assert stages[0]["additional_config"]["existing"] == "p"
    assert stages[2] == {"stage_id": 2, "additional_config": {"existing": "talker"}}
    assert stages[3] == {"stage_id": 3}
    assert set(state["deployment_precision"]) == {"0", "1"}
    # Execute the real final run.json annotation statements without a server.
    (tmp_path / "run.json").write_text(json.dumps({"users": [{"session_id": "s"}]}))
    run_nodes = []
    for n in ast.walk(main):
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in (
            "run_path",
            "run_result",
            "run_result['deployment_precision']",
        ):
            run_nodes.append(n)
        elif isinstance(n, ast.Expr) and ast.unparse(n.value).startswith("run_path.write_text("):
            run_nodes.append(n)
    state.update(out=tmp_path, json=json)
    _execute(sorted(run_nodes, key=lambda n: n.lineno), state)
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["users"] == [{"session_id": "s"}]
    assert saved["deployment_precision"] == state["deployment_precision"]
    commands_write = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
        and "commands.json" in ast.unparse(n.func.value)
    )
    assert "'triton_force_2d_attention': args.triton_force_2d_attention" in ast.unparse(commands_write)


@pytest.mark.parametrize(
    "extra",
    [
        ["--triton-disable-q-quantization", "--attention-backend", "FLASHINFER"],
        ["--triton-force-2d-attention", "--attention-backend", "FLASHINFER"],
        ["--numerical-probe-seqs", "37-39"],
        ["--numerical-probe-sessions", '["user-0"]'],
        ["--numerical-probe-layer-io"],
        ["--numerical-probe-dir", "/tmp/probe", "--numerical-probe-seqs", "-1"],
        ["--numerical-probe-dir", "/tmp/probe", "--numerical-probe-seqs", "4-2"],
        ["--numerical-probe-dir", "/tmp/probe", "--numerical-probe-seqs", "0-1001"],
        ["--numerical-probe-dir", "/tmp/probe", "--numerical-probe-sessions", "[1]"],
    ],
)
def test_invalid_controls_fail_before_gpu_checks(monkeypatch, tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        _parse_cli(monkeypatch, tmp_path, extra)
    assert exc.value.code == 2


def test_force_2d_can_be_explicitly_disabled_without_selecting_triton(monkeypatch, tmp_path):
    args, _ = _parse_cli(
        monkeypatch,
        tmp_path,
        ["--triton-force-2d-attention", "--no-triton-force-2d-attention", "--attention-backend", "FLASHINFER"],
    )
    assert args.triton_force_2d_attention is False
    assert args.attention_backend == "FLASHINFER"


@pytest.mark.parametrize("layer_io", [False, True])
def test_selected_numerical_probe_scope_is_pd_only(monkeypatch, tmp_path, layer_io):
    args, main = _parse_cli(
        monkeypatch,
        tmp_path,
        [
            "--numerical-probe-dir",
            str(tmp_path / "probe"),
            "--numerical-probe-seqs",
            "37-39,239-241",
            "--numerical-probe-sessions",
            '["user-0"]',
            "--triton-force-2d-attention",
            "--triton-disable-q-quantization",
            "--attention-backend",
            "TRITON_ATTN",
            *(["--numerical-probe-layer-io"] if layer_io else []),
        ],
    )
    stages = [
        {"stage_id": 0, "hf_overrides": {"vllm_omni_minicpmo_pd_prefill": True}},
        {"stage_id": 1, "hf_overrides": {"vllm_omni_minicpmo_pd_decode": True}},
        {"stage_id": 2},
        {"stage_id": 3},
    ]
    loop = next(n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == "stage")
    _execute([loop], {"args": args, "resolved": {"stages": stages}})
    for stage in stages[:2]:
        assert stage["env"]["MINICPMO45_NUMERICAL_PROBE_SEQS"] == "37-39,239-241"
        assert json.loads(stage["env"]["MINICPMO45_NUMERICAL_PROBE_SESSIONS"]) == ["user-0"]
        assert stage["enforce_eager"] is True
        assert ("MINICPMO45_NUMERICAL_PROBE_LAYER_IO" in stage["env"]) is layer_io
        if layer_io:
            assert stage["env"]["MINICPMO45_NUMERICAL_PROBE_LAYER_IO"] == "1"
    assert stages[2:] == [{"stage_id": 2}, {"stage_id": 3}]
    commands_write = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "write_text"
        and "commands.json" in ast.unparse(n.func.value)
    )
    assert "'numerical_probe_layer_io': args.numerical_probe_layer_io" in ast.unparse(commands_write)


@pytest.mark.parametrize(
    "completion", [{}, {"measurement_complete": None}, {"measurement_complete": False}, {"measurement_complete": True}]
)
@pytest.mark.parametrize("capacity_pass", [False, True])
def test_measurement_completion_failure_is_not_a_capacity_failure(tmp_path, completion, capacity_pass):
    module = ast.parse(SOURCE.read_text())
    run_try = next(
        n
        for n in ast.walk(module)
        if isinstance(n, ast.Try)
        and any(isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "unchanged" for child in n.body)
    )
    start = next(
        i for i, n in enumerate(run_try.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "unchanged"
    )
    events = []
    report = {"capacity_pass": capacity_pass, "input_capacity_pass": capacity_pass, **completion}
    state = {
        "report": report,
        "out": tmp_path,
        "ROOT": tmp_path / "repo",
        "vllm_root": tmp_path / "vllm",
        "source_manifest": lambda _path: "stable-hash",
        "source_before": "stable-hash",
        "vllm_before": "stable-hash",
        "mps": SimpleNamespace(snapshot=lambda *args: events.append(("mps", args))),
        "print": lambda *args, **kwargs: events.append(("print", args)),
        "json": json,
    }
    if completion.get("measurement_complete") is True:
        # Complete short/probe runs and real capacity failures must still exit
        # normally: capacity_pass=False does not mean the measurement failed.
        _execute(run_try.body[start:], state)
    else:
        with pytest.raises(RuntimeError, match="Measurement incomplete"):
            _execute(run_try.body[start:], state)
    assert json.loads((tmp_path / "source-stability.json").read_text()) == {"unchanged": True}
    assert [event[0] for event in events] == ["mps", "print"]
    assert json.loads(events[1][1][0])["capacity_pass"] is capacity_pass
