"""One matched P/D placement run; explicit private MPS, never an ambient daemon.

Input capacity uses exact D witnesses, per-user long RTF and bounded backlog. This runner does
not relabel that result as end-to-end audio capacity.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.minicpmo.private_mps import PrivateMPS  # noqa: E402
from benchmarks.minicpmo.run_placement import healthy  # noqa: E402


def _probe_seq_selection(value):
    """Validate the probe's bounded comma/range syntax without importing torch."""
    for item in value.split(","):
        bounds = item.strip().split("-")
        if len(bounds) not in (1, 2) or not all(part.isdigit() for part in bounds):
            raise argparse.ArgumentTypeError("probe seqs must be nonnegative integers/ranges, e.g. 37-39,239-241")
        start, end = int(bounds[0]), int(bounds[-1])
        if end < start or end - start > 1000:
            raise argparse.ArgumentTypeError("probe sequence ranges must be ordered and span at most 1000")
    return value


def _probe_session_selection(value):
    try:
        selection = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("probe sessions must be a JSON array of strings or null") from exc
    if selection is not None and not (isinstance(selection, list) and all(isinstance(item, str) for item in selection)):
        raise argparse.ArgumentTypeError("probe sessions must be a JSON array of strings or null")
    return json.dumps(selection)


def _probe_layer_selection(value):
    """A subset bounds KV capture; 'none' still captures actual inputs/hidden."""
    return value if value in {"all", "none"} else _probe_seq_selection(value)


def _apply_three_gpu_placement(resolved, topology, uuids):
    """Move devices only; retain sidecar pipelines and the four engine stages.

    Encoder means both AV encoders; decoder means Code2Wav. Talker always
    stays with D. Auxiliary UUIDs are translated to P's local cuda:1, even
    when that device is also used by the independent D/Talker processes.
    """
    placements = {
        "3gpu-encoder-p": (0, 2),
        "3gpu-encoder-d": (1, 2),
        "3gpu-decoder-p": (2, 0),
        "3gpu-decoder-d": (2, 1),
    }
    if topology not in placements:
        return
    if len(uuids) != 3:
        raise ValueError("Three-GPU placement requires exactly three visible devices")
    encoder, decoder = placements[topology]
    stages = {stage["stage_id"]: stage for stage in resolved["stages"]}
    if set(stages) != {0, 1, 2, 3}:
        raise ValueError("P/D placement requires P, D, Talker and Code2Wav stages")
    for sid, device in enumerate((0, 1, 1, decoder)):
        stages[sid]["devices"] = str(device)
    env = stages[0].setdefault("env", {})
    env["VLLM_OMNI_AUX_VISIBLE_DEVICES"] = "" if encoder == 0 else uuids[encoder]
    for modality in ("VISION", "AUDIO"):
        env[f"MINICPMO45_{modality}_ENCODER_DEVICE"] = "cuda:0" if encoder == 0 else "cuda:1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=(
        "isolated", "encoders-on-p", "d-talker", "3gpu-encoder-p", "3gpu-encoder-d",
        "3gpu-decoder-p", "3gpu-decoder-d",
    ), required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--duration-s", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--context-age-max-units", type=int, default=0)
    parser.add_argument("--kv-window-tokens", type=int, default=36000)
    parser.add_argument("--min-post-window-units", type=int, default=120)
    parser.add_argument("--max-backlog-ms", type=float, default=500.0,
                        help="Offline capacity gate only; does not throttle or drop inputs")
    parser.add_argument("--warmup-s", type=int, default=12)
    parser.add_argument(
        "--post-stream-s", type=int, default=30,
        help="Observe outstanding input/output after sending ends; does not change input cadence or RTF accounting",
    )
    parser.add_argument("--mps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--functional-only", action="store_true",
        help="Validate serving contracts without claiming representative-dialogue capacity; execution is unchanged",
    )
    parser.add_argument(
        "--quality-capture", action="store_true", help="Export full measured-session text and PCM after input completes"
    )
    parser.add_argument("--pinned-prefix-tokens", type=int, default=0)
    parser.add_argument(
        "--numerical-probe-dir",
        type=Path,
        help="Synchronizing P/D tensor capture; diagnostic only, invalidates capacity",
    )
    parser.add_argument(
        "--numerical-probe-seqs",
        type=_probe_seq_selection,
        help="Capture only these native unit sequences, e.g. 37-39,239-241; requires --numerical-probe-dir",
    )
    parser.add_argument(
        "--numerical-probe-sessions",
        type=_probe_session_selection,
        help="JSON array of selected session IDs; requires --numerical-probe-dir",
    )
    parser.add_argument(
        "--numerical-probe-layer-io",
        action="store_true",
        help="Capture first-layer P/D intermediate tensors for selected units; requires --numerical-probe-dir",
    )
    parser.add_argument(
        "--numerical-probe-layers",
        type=_probe_layer_selection,
        help="KV layer subset, 'all', or 'none' for inputs/hidden only; requires --numerical-probe-dir",
    )
    parser.add_argument(
        "--speech-probe-dir",
        type=Path,
        help="Capture Talker/Code2Wav boundaries and tensors; diagnostic only, not capacity",
    )
    parser.add_argument("--attention-backend", choices=("FLASHINFER", "TRITON_ATTN"))
    parser.add_argument(
        "--triton-disable-q-quantization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="P/D only: keep Q in model dtype with FP8 KV storage; selects Triton, default unchanged",
    )
    parser.add_argument(
        "--triton-force-2d-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="P/D only: explicitly use Triton 2D attention for decode as well as prefill; default unchanged",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("fp8", "bfloat16"),
        default="fp8",
        help="P/D KV precision; BF16 is an explicit correctness control, not a new default",
    )
    parser.add_argument("--triton-decode-split-k-threshold", type=int,
                        help="D-only native split-K request threshold; omitted preserves native dispatch")
    parser.add_argument(
        "--thinker-quantization",
        choices=("fp8", "none"),
        default="fp8",
        help="Explicit P/D weight precision control; Talker precision is unchanged",
    )
    parser.add_argument(
        "--thinker-kv-cache-memory-gib",
        type=int,
        default=None,
        help="Positive integer GiB per P/D stage; omitted preserves deployment budgets and other stages",
    )
    args = parser.parse_args()
    if args.users < 1 or args.duration_s < 1 or args.context_age_max_units < 0 or args.warmup_s < 0:
        parser.error("users/duration must be positive and context age nonnegative")
    if args.post_stream_s < 0:
        parser.error("--post-stream-s must be nonnegative")
    if not math.isfinite(args.max_backlog_ms) or args.max_backlog_ms < 0:
        parser.error("--max-backlog-ms must be finite and nonnegative")
    if args.kv_window_tokens < 1024 or args.kv_window_tokens % 16:
        parser.error("KV window must be >=1024 and aligned to 16 tokens")
    if args.thinker_kv_cache_memory_gib is not None and args.thinker_kv_cache_memory_gib <= 0:
        parser.error("--thinker-kv-cache-memory-gib must be a positive integer")
    if args.pinned_prefix_tokens < 0 or args.pinned_prefix_tokens % 16:
        parser.error("Pinned prefix must be nonnegative and aligned to 16 tokens")
    if args.pinned_prefix_tokens and args.attention_backend == "FLASHINFER":
        parser.error("Pinned prefix currently requires TRITON_ATTN")
    if args.triton_disable_q_quantization and args.attention_backend == "FLASHINFER":
        parser.error("--triton-disable-q-quantization requires TRITON_ATTN")
    if args.triton_force_2d_attention and args.attention_backend == "FLASHINFER":
        parser.error("--triton-force-2d-attention requires TRITON_ATTN")
    if args.triton_decode_split_k_threshold is not None:
        if args.triton_decode_split_k_threshold < 1 or args.triton_force_2d_attention or args.attention_backend == "FLASHINFER":
            parser.error("split-K threshold must be positive, requires Triton and conflicts with forced 2D")
    if (
        args.numerical_probe_seqs is not None
        or args.numerical_probe_sessions is not None
        or args.numerical_probe_layer_io
        or args.numerical_probe_layers is not None
    ) and not args.numerical_probe_dir:
        parser.error("Numerical probe selections require --numerical-probe-dir")
    if healthy(args.port):
        raise RuntimeError("Existing server on this port; leave it untouched")
    import pynvml
    import yaml

    from benchmarks.minicpmo.pd_capacity_search import archive_source, source_manifest
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    pynvml.nvmlInit()
    try:
        uuids = []
        for gpu in range(3 if args.topology.startswith("3gpu-") else 4):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                raise RuntimeError(f"GPU {gpu} already used by {[p.pid for p in processes]}")
            uuids.append(pynvml.nvmlDeviceGetUUID(handle))
    finally:
        pynvml.nvmlShutdown()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir.resolve()
    source_before = archive_source(ROOT, out / "source")
    import importlib.util

    vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
    vllm_before = archive_source(vllm_root, out / "vllm-source")
    name = "deploy_capacity_pd_4gpu" + ("_encoders_on_p" if args.topology == "encoders-on-p" else "") + "_fp8.yaml"
    if args.topology == "d-talker" or args.topology.startswith("3gpu-"):
        name = "deploy_capacity_pd_d_talker_fp8.yaml"
    resolved = resolve_deploy_yaml(ROOT / "benchmarks/minicpmo" / name)
    for stage in resolved["stages"]:
        if args.speech_probe_dir:
            stage.setdefault("env", {})["VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR"] = str(args.speech_probe_dir.resolve())
            stage["env"]["VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS"] = "1"
        if stage["stage_id"] in (0, 1):
            if args.numerical_probe_dir:
                stage["enforce_eager"] = True
                stage.setdefault("env", {})["MINICPMO45_NUMERICAL_PROBE_DIR"] = str(args.numerical_probe_dir.resolve())
                if args.numerical_probe_seqs is not None:
                    stage["env"]["MINICPMO45_NUMERICAL_PROBE_SEQS"] = args.numerical_probe_seqs
                if args.numerical_probe_sessions is not None:
                    stage["env"]["MINICPMO45_NUMERICAL_PROBE_SESSIONS"] = args.numerical_probe_sessions
                if args.numerical_probe_layer_io:
                    stage["env"]["MINICPMO45_NUMERICAL_PROBE_LAYER_IO"] = "1"
                if args.numerical_probe_layers is not None:
                    stage["env"]["MINICPMO45_NUMERICAL_PROBE_LAYERS"] = args.numerical_probe_layers
            mode_key = "vllm_omni_minicpmo_pd_prefill" if stage["stage_id"] == 0 else "vllm_omni_minicpmo_pd_decode"
            if stage.get("hf_overrides", {}).get(mode_key) is not True:
                raise RuntimeError(f"Missing required stage mode {mode_key}; refuse an invalid P/D experiment")
    # Store the complete resolved config, not merely an overlay whose base
    # might change later. UUID auxiliary visibility also works under MPS.
    for stage in resolved["stages"]:
        if stage["stage_id"] in (0, 1):
            stage["quantization"] = None if args.thinker_quantization == "none" else "fp8"
            stage["kv_cache_dtype"] = args.kv_cache_dtype
            if args.thinker_kv_cache_memory_gib is not None:
                stage["kv_cache_memory_bytes"] = args.thinker_kv_cache_memory_gib * 1024**3
                # Explicit bytes determine KV allocation. Lower only the
                # startup free-memory check for co-located processes, equally
                # for the matched four-GPU control and three-GPU treatments.
                stage["gpu_memory_utilization"] = 0.1
            stage["hf_overrides"]["sliding_window"] = args.kv_window_tokens
            stage["hf_overrides"]["vllm_omni_minicpmo_sliding_window_tokens"] = args.kv_window_tokens
            stage["hf_overrides"]["vllm_omni_pinned_prefix_tokens"] = args.pinned_prefix_tokens
            stage.setdefault("additional_config", {})["triton_disable_q_quantization"] = (
                args.triton_disable_q_quantization
            )
            stage["additional_config"]["triton_force_2d_attention"] = args.triton_force_2d_attention
            if stage["stage_id"] == 1 and args.triton_decode_split_k_threshold is not None:
                stage["additional_config"]["triton_decode_split_k_threshold"] = args.triton_decode_split_k_threshold
            if (
                args.pinned_prefix_tokens
                or args.attention_backend
                or args.triton_disable_q_quantization
                or args.triton_force_2d_attention
            ):
                stage.setdefault("attention_config", {})["backend"] = args.attention_backend or "TRITON_ATTN"
        if stage["stage_id"] == 0 and args.topology != "encoders-on-p":
            stage["env"]["VLLM_OMNI_AUX_VISIBLE_DEVICES"] = uuids[2]
    _apply_three_gpu_placement(resolved, args.topology, uuids)
    config = out / "deploy.yaml"
    config.write_text(yaml.safe_dump(resolved))
    deployment_precision = {
        str(stage["stage_id"]): {
            "quantization": stage["quantization"],
            "kv_cache_dtype": stage["kv_cache_dtype"],
            "kv_cache_memory_bytes": stage.get("kv_cache_memory_bytes"),
            "attention_config": stage.get("attention_config", {}),
            "additional_config": stage["additional_config"],
        }
        for stage in resolved["stages"]
        if stage["stage_id"] in (0, 1)
    }
    env = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_USE_FLASHINFER_SAMPLER="0", CUDA_VISIBLE_DEVICES=",".join(uuids))
    if args.speech_probe_dir:
        # Also configure the parent before import: the API owns handoff
        # conversion and spawned workers may import the probe very early.
        env["VLLM_OMNI_MINICPMO_SPEECH_PROBE_DIR"] = str(args.speech_probe_dir.resolve())
        env["VLLM_OMNI_MINICPMO_SPEECH_PROBE_TENSORS"] = "1"
    if any(key.startswith("CUDA_MPS_") for key in env):
        raise RuntimeError("Refuse inherited MPS settings")
    # No-MPS controls use an empty private socket directory too: this prevents
    # accidental attachment to a machine-wide default MPS daemon.
    if not args.mps:
        import tempfile

        env["CUDA_MPS_PIPE_DIRECTORY"] = tempfile.mkdtemp(prefix="minicpm-no-mps-")
    server_cmd = [
        sys.executable,
        "benchmarks/minicpmo/clean_server.py",
        "--provenance-out",
        str(out / "provenance.json"),
        "--",
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        "openbmb/MiniCPM-o-4_5",
        "--omni",
        "--deploy-config",
        str(config),
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    client_cmd = [
        sys.executable,
        "benchmarks/minicpmo/continuous_av.py",
        "--url",
        f"ws://127.0.0.1:{args.port}/v1/realtime",
        "--users",
        str(args.users),
        "--duration-s",
        str(args.duration_s),
        "--seed",
        str(args.seed),
        "--context-age-max-units",
        str(args.context_age_max_units),
        "--workload-profile",
        "production",
        "--connect-stagger-s",
        "0",
        "--post-stream-s",
        str(args.post_stream_s),
        "--close-timeout-s",
        "60",
        "--timeout-s",
        "60",
        "--media",
        "/home/ubuntu/data/minicpmo-benchmark-assets/assets/omni_duplex1.mp4",
        "--ref-audio",
        "/home/ubuntu/data/minicpmo-benchmark-assets/assets/HT_ref_audio.wav",
        "--loop-media",
        "--frame-max-side",
        "0",
        "--max-slice-nums",
        "4",
        "--context-window-trigger-tokens",
        str(args.kv_window_tokens),
        "--expected-kv-window-tokens",
        str(args.kv_window_tokens),
        "--min-post-window-units",
        str(args.min_post_window_units),
        "--max-backlog-ms",
        str(args.max_backlog_ms),
        "--max-send-drift-ms",
        "10",
        "--gpus",
        *(str(gpu) for gpu in range(3 if args.topology.startswith("3gpu-") else 4)),
        "--out",
        str(out / "run.json"),
    ]
    # Initialize lazy encoders, connector handshakes and output caches before
    # timed capacity input. This is a separate session, never seeded history
    # for measured users: all measured sessions still start at context zero.
    warmup_cmd = client_cmd.copy()
    for flag, value in {
        "--users": "1",
        "--duration-s": str(args.warmup_s),
        "--post-stream-s": "5",
        "--context-age-max-units": "0",
        "--out": str(out / "warmup.json"),
    }.items():
        warmup_cmd[warmup_cmd.index(flag) + 1] = value
    if args.quality_capture:
        client_cmd.extend(["--quality-capture-dir", str(out / "quality")])
    (out / "commands.json").write_text(
        json.dumps(
            {
                "server": server_cmd,
                "warmup": warmup_cmd if args.warmup_s else None,
                "client": client_cmd,
                "mps": args.mps,
                "deployment_precision": deployment_precision,
                "triton_force_2d_attention": args.triton_force_2d_attention,
                "numerical_probe_seqs": args.numerical_probe_seqs,
                "numerical_probe_sessions": args.numerical_probe_sessions,
                "numerical_probe_layer_io": args.numerical_probe_layer_io,
                "numerical_probe_layers": args.numerical_probe_layers,
            },
            indent=2,
        )
    )
    with contextlib.ExitStack() as stack:
        mps = stack.enter_context(PrivateMPS(out, env, uuids, stage_count=4)) if args.mps else None
        if mps:
            env = mps.client_env
        log = stack.enter_context((out / "server.log").open("w"))
        server = subprocess.Popen(
            server_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"START {args.topology} mps={args.mps} pid={server.pid} out={out}", flush=True)
        try:
            deadline = time.monotonic() + 600
            while not healthy(args.port):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Server startup failed; inspect server.log")
                time.sleep(1)
            if mps:
                print(json.dumps(mps.snapshot("ready", out / "server.log")), flush=True)
            if args.warmup_s:
                with (out / "warmup.log").open("w") as warmup_log:
                    subprocess.run(
                        warmup_cmd,
                        cwd=ROOT,
                        env=env,
                        stdout=warmup_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.warmup_s + 120,
                    )
                warmup = json.loads((out / "warmup.json").read_text())
                if any(
                    not u.get("input_stream_complete") or u.get("errors") or u.get("server_errors")
                    for u in warmup.get("users", [])
                ) or not warmup.get("users"):
                    raise RuntimeError("Warmup failed; inspect warmup.json")
                print("WARMUP complete; measured sessions start with fresh histories", flush=True)
            with (out / "client.log").open("w") as client_log:
                subprocess.run(
                    client_cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.duration_s + args.context_age_max_units + args.post_stream_s + 210,
                )
            run_path = out / "run.json"
            run_result = json.loads(run_path.read_text())
            run_result["deployment_precision"] = deployment_precision
            if args.functional_only:
                run_result["measurement_purpose"] = "serving_contract_validation"
            run_path.write_text(json.dumps(run_result, indent=2))
            with (out / "analysis.log").open("w") as analysis_log:
                subprocess.run(
                    [
                        sys.executable,
                        "benchmarks/minicpmo/analyze_rtf.py",
                        "--run-json",
                        str(out / "run.json"),
                        "--server-log",
                        str(out / "server.log"),
                        "--server-provenance-json",
                        str(out / "provenance.json"),
                        "--out",
                        str(out / "analysis.json"),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=True,
                    stdout=analysis_log,
                    stderr=subprocess.STDOUT,
                )
            report = json.loads((out / "analysis.json").read_text())
            if args.numerical_probe_dir or args.speech_probe_dir:
                report["capacity_pass"] = False
                report["input_capacity_pass"] = False
                report["diagnostic_only"] = "synchronizing model tensor / speech boundary capture"
                (out / "analysis.json").write_text(json.dumps(report, indent=2))
            unchanged = source_manifest(ROOT) == source_before and source_manifest(vllm_root) == vllm_before
            (out / "source-stability.json").write_text(json.dumps({"unchanged": unchanged}, indent=2))
            if not unchanged:
                raise RuntimeError("Source changed during measurement; do not use this point for capacity")
            if mps:
                mps.snapshot("after_measurement", out / "server.log")
            print(
                json.dumps(
                    {
                        k: report.get(k)
                        for k in (
                            "measurement_complete",
                            "capacity_pass",
                            "input_capacity_pass",
                            "capacity_metric",
                            "capacity_scope",
                            "input_backlog_audit",
                            "sliding_window_audit",
                            "sender_timing_audit",
                        )
                    }
                ),
                flush=True,
            )
            if report.get("measurement_complete") is not True:
                raise RuntimeError("Measurement incomplete; inspect analysis.json and client/server logs")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)


if __name__ == "__main__":
    main()
