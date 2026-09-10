"""Run one isolated, traced non-P/D placement capacity point."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmarks.minicpmo.analyze_nonpd_placement import analyze
from benchmarks.minicpmo.private_mps import PrivateMPS

ROOT = Path(__file__).resolve().parents[2]
PLACEMENTS = {
    "all1": ("colocated", (0,)),
    "split3": ("separated", (0, 1, 2)),
    "encoder_out": ("encoder_out", (0, 1)),
    "code2wav_out": ("code2wav_out", (0, 2)),
    "talker_out": ("talker_out", (0, 1)),
}


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def mps_deploy_config(config: Path, topology: str, gpu_uuids: list[str], out: Path) -> Path:
    """Resolve auxiliary visibility too, without changing model placement.

    Stage devices are translated to UUIDs by Omni. Its auxiliary visibility
    is appended verbatim, so retaining ordinal "1" would produce a mixed
    UUID/ordinal list that PyTorch truncates before the encoder's device.
    """
    if topology != "split3":
        return config
    import yaml

    # Stage env is a replacement, not a nested merge in deploy inheritance.
    # Preserve the complete original auxiliary-device and sidecar settings.
    source = yaml.safe_load(config.read_text())
    stage_env = dict(next(stage for stage in source["stages"] if stage["stage_id"] == 0)["env"])
    stage_env["VLLM_OMNI_AUX_VISIBLE_DEVICES"] = gpu_uuids[1]
    override = out / "deploy-mps.yaml"
    override.write_text(
        yaml.safe_dump({"base_config": str(config), "stages": [{"stage_id": 0, "env": stage_env}]})
    )
    return override


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=tuple(PLACEMENTS), required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--duration-s", type=int, required=True)
    parser.add_argument("--drain-s", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--debug-handoff", action="store_true")
    parser.add_argument("--sweep-users", type=int, nargs="*", default=[])
    parser.add_argument("--sweep-duration-s", type=int, default=360)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--mps", action="store_true", help="Private MPS on the all1 or split3 topology's GPUs")
    args = parser.parse_args()
    if args.mps and args.topology not in ("all1", "split3"):
        parser.error("--mps supports the all1 and split3 topologies")
    if healthy(args.port):
        raise RuntimeError("An existing server occupies the requested port; leave it untouched")
    import pynvml

    pynvml.nvmlInit()
    name, gpus = PLACEMENTS[args.topology]
    for gpu in gpus:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(gpu))
        if processes:
            raise RuntimeError(f"GPU {gpu} is already in use by {[p.pid for p in processes]}")
    gpu_uuids = [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(gpu)) for gpu in gpus]
    pynvml.nvmlShutdown()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    out = args.out_dir.resolve()
    config = ROOT / f"benchmarks/minicpmo/deploy_capacity_nonpd_{name}_fp8.yaml"
    if args.mps:
        config = mps_deploy_config(config, args.topology, gpu_uuids, out)
    env = dict(os.environ, PYTHONPATH=str(ROOT), VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_OMNI_LOG_DUPLEX_CADENCE="units")
    if args.debug_handoff:
        env["MINICPMO45_LOG_TTS_HANDOFF"] = "1"
    server_command = [
        sys.executable,
        "benchmarks/minicpmo/clean_server.py",
        "--allow-diagnostics",
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
    with contextlib.ExitStack() as stack:
        mps = stack.enter_context(PrivateMPS(out, env, gpu_uuids)) if args.mps else None
        if mps is not None:
            env = mps.client_env
        log = stack.enter_context((out / "server.log").open("w"))
        server = subprocess.Popen(
            server_command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        print(f"START {args.topology} {args.users}x{args.duration_s} server_pid={server.pid} out={out}", flush=True)
        try:
            deadline = time.monotonic() + 600
            while not healthy(args.port):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {server.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup exceeded 600 s")
                time.sleep(2)
            print("SERVER_READY; starting workload", flush=True)
            if mps is not None:
                print("MPS_VERIFIED " + json.dumps(mps.snapshot("server_ready", out / "server.log")), flush=True)
            command = [
                sys.executable,
                "benchmarks/minicpmo/continuous_av.py",
                "--completion-mode",
                "nonpd",
                "--url",
                f"ws://127.0.0.1:{args.port}/v1/realtime",
                "--users",
                str(args.users),
                "--duration-s",
                str(args.duration_s),
                "--workload-profile",
                "production",
                "--context-age-max-units",
                "0",
                "--seed",
                str(args.seed),
                "--connect-stagger-s",
                "0",
                "--post-stream-s",
                str(args.drain_s),
                "--close-timeout-s",
                "60",
                "--timeout-s",
                "60",
                "--media",
                "/home/ubuntu/data/minicpmo-benchmark-assets/assets/omni_duplex1.mp4",
                "--loop-media",
                "--ref-audio",
                "/home/ubuntu/data/minicpmo-benchmark-assets/assets/HT_ref_audio.wav",
                "--frame-max-side",
                "0",
                "--max-slice-nums",
                "4",
                "--context-window-trigger-tokens",
                "36000",
                "--out",
                str(out / "run.json"),
                "--gpus",
                *map(str, gpus),
            ]
            (out / "commands.json").write_text(json.dumps({"server": server_command, "client": command}, indent=2))
            with (out / "client.log").open("w") as client_log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.duration_s + args.drain_s + 240,
                )
            result = analyze(
                json.loads((out / "run.json").read_text()), (out / "server.log").read_text(errors="replace")
            )
            (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "sessions"}), flush=True)
            if args.sweep_users and not result["downstream_audio_observed"]:
                raise RuntimeError("Functional probe emitted no audio; refuse to measure capacity")
            for users in args.sweep_users:
                point = out / f"{users}x{args.sweep_duration_s}"
                point.mkdir(exist_ok=False)
                next_command = list(command)
                for flag, value in (
                    ("--users", users),
                    ("--duration-s", args.sweep_duration_s),
                    ("--out", point / "run.json"),
                ):
                    next_command[next_command.index(flag) + 1] = str(value)
                (point / "commands.json").write_text(json.dumps({"client": next_command}, indent=2))
                print(f"POINT_START {users}x{args.sweep_duration_s} out={point}", flush=True)
                if mps is not None:
                    mps.snapshot(f"before_{users}x{args.sweep_duration_s}", out / "server.log")
                with (point / "client.log").open("w") as client_log:
                    subprocess.run(
                        next_command,
                        cwd=ROOT,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.sweep_duration_s + args.drain_s + 240,
                    )
                result = analyze(
                    json.loads((point / "run.json").read_text()),
                    (out / "server.log").read_text(errors="replace"),
                )
                (point / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"users": users, **{k: v for k, v in result.items() if k != "sessions"}}), flush=True)
                if mps is not None:
                    mps.snapshot(f"after_{users}x{args.sweep_duration_s}", out / "server.log")
                if args.stop_on_failure and not result["capacity_pass"]:
                    break
        finally:
            # Terminate only the new process group created by THIS invocation.
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)


if __name__ == "__main__":
    main()
