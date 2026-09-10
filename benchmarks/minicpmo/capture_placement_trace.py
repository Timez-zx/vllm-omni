"""Start an existing Nsight session at a specific real-input model unit.

Launch run_placement.py through ``nsys launch`` first. This helper changes no
model/runtime setting and captures only a diagnostic window, not capacity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from benchmarks.minicpmo.analyze_nonpd_placement import ADMIT
from benchmarks.minicpmo.analyze_rtf import _request_session_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--unit", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--capture-s", type=float, default=20)
    parser.add_argument("--timeout-s", type=float, default=1200)
    args = parser.parse_args()
    if not 0 < args.capture_s <= 60:
        parser.error("capture-s must be in (0, 60]")
    control = args.out.with_suffix(".control.json")
    if control.exists() or args.out.with_suffix(".nsys-rep").exists():
        raise FileExistsError(f"Refuse to overwrite {args.out}")
    deadline = time.monotonic() + args.timeout_s
    prefix = f"minicpm-cap-{args.seed}-{args.users}-"
    print(f"Waiting for {prefix}* input unit {args.unit}", flush=True)
    trigger = None
    with args.server_log.open() as stream:
        while time.monotonic() < deadline:
            line = stream.readline()
            if not line:
                time.sleep(0.2)
                continue
            match = ADMIT.search(line)
            if not match:
                continue
            req, generation, epoch, prompt, seq, origin, unit = match.groups()
            if origin != "client" or not unit.isdigit() or int(unit) < args.unit:
                continue
            if not _request_session_id(req).startswith(prefix):
                continue
            trigger = {"request_id": req, "input_unit_index": int(unit), "admit_epoch": float(epoch)}
            break
    if trigger is None:
        raise TimeoutError("Target unit did not arrive; no profiling state changed")
    command = [
        "nsys",
        "start",
        f"--session={args.session}",
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-devices=none",
        f"--output={args.out}",
    ]
    record = {"kind": "diagnostic CUDA trace; not a capacity certificate", "trigger": trigger, "start_command": command}
    record["start_requested_epoch"] = time.time()
    start = subprocess.run(command, capture_output=True, text=True, timeout=60)
    record["start_stdout"] = start.stdout
    record["start_stderr"] = start.stderr
    record["start_returncode"] = start.returncode
    control.write_text(json.dumps(record, indent=2) + "\n")
    start.check_returncode()
    print(f"CAPTURE_STARTED {trigger}", flush=True)
    try:
        time.sleep(args.capture_s)
    finally:
        record["stop_requested_epoch"] = time.time()
        stop_command = ["nsys", "stop", f"--session={args.session}"]
        record["stop_command"] = stop_command
        stop = subprocess.run(stop_command, capture_output=True, text=True, timeout=180)
        record["stop_stdout"] = stop.stdout
        record["stop_stderr"] = stop.stderr
        record["stop_returncode"] = stop.returncode
        control.write_text(json.dumps(record, indent=2) + "\n")
        stop.check_returncode()
    print(f"CAPTURE_FINISHED {args.out}", flush=True)


if __name__ == "__main__":
    main()
