#!/usr/bin/env python3
"""Per-stage GPU accounting sampler for a vllm-omni deployment.

vllm-omni runs each pipeline stage (thinker / talker / code2wav) as its own OS
process, which makes per-stage GPU attribution possible on a single card
without patching the engine: NVML reports utilization and memory per PID.

Two independent signals are recorded:

  device level   SM utilization, memory used, power draw, SM clock, temperature
                 -- and clock/power are recorded specifically so that thermal
                 or power throttling can be detected after the fact rather
                 than silently distorting a duty-cycle number.

  process level  nvmlDeviceGetProcessUtilization gives per-PID SM percentages
                 sampled from NVML's internal ring buffer. Integrating a PID's
                 SM% over wall time yields that stage's share of GPU time.

Caveats recorded in the output so they cannot be forgotten downstream:
  * Per-process SM% from NVML is a sampled estimate, not exact kernel time. It
    is adequate for apportioning cost between stages (the question here) and
    not adequate for absolute kernel timing.
  * SM utilization means "at least one warp resident", not occupancy. A stage
    at 100% SM util may still leave most of the GPU idle. Treated as an upper
    bound on busy time, never as achieved FLOPs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import time

import pynvml as nv


def stage_map() -> dict[int, str]:
    """Map PIDs of vllm-omni stage workers to readable stage labels."""
    out: dict[int, str] = {}
    try:
        ps = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return out
    for line in ps.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, args = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if "vllm" not in args and "VLLM" not in args:
            continue
        m = re.search(r"stage[_-]?(\d+)", args)
        if m:
            out[pid] = f"stage{m.group(1)}"
        elif "vllm serve" in args or "api_server" in args:
            out[pid] = "frontend"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/zx/results/gpu_samples.jsonl")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = until killed")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(args.gpu)
    name = nv.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    total_mem = nv.nvmlDeviceGetMemoryInfo(h).total

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    f = outp.open("w")
    t0 = time.monotonic()

    try:
        max_sm = nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_SM)
    except Exception:
        max_sm = None
    try:
        plimit = nv.nvmlDeviceGetEnforcedPowerManagementLimit(h) / 1000.0
    except Exception:
        plimit = None

    f.write(json.dumps({
        "k": "meta",
        "gpu": name,
        "total_mem_bytes": int(total_mem),
        "max_sm_clock_mhz": max_sm,
        "power_limit_w": plimit,
        "hz": args.hz,
        "wall_start": time.time(),
        "caveats": [
            "per-process SM% is an NVML sampled estimate, not exact kernel time",
            "SM utilization = at least one warp resident, not occupancy",
        ],
    }) + "\n")

    smap = stage_map()
    f.write(json.dumps({"k": "stage_map", "map": {str(k): v for k, v in smap.items()}}) + "\n")

    period = 1.0 / args.hz
    last_ts = 0
    next_t = time.monotonic()
    remap_at = time.monotonic() + 5.0

    while True:
        now = time.monotonic()
        if args.duration_s and (now - t0) > args.duration_s:
            break

        rec: dict = {"t": now - t0, "k": "s"}
        try:
            u = nv.nvmlDeviceGetUtilizationRates(h)
            rec["sm"] = u.gpu
            rec["memio"] = u.memory
        except Exception:
            pass
        try:
            mi = nv.nvmlDeviceGetMemoryInfo(h)
            rec["mem_used"] = int(mi.used)
        except Exception:
            pass
        try:
            rec["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pass
        try:
            rec["sm_clock"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
        except Exception:
            pass
        try:
            rec["temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        # throttle reasons -- so a duty-cycle dip can be explained rather than guessed
        try:
            rec["throttle"] = int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
        except Exception:
            pass

        # per-process resident memory
        try:
            procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
            rec["proc_mem"] = {
                str(p.pid): int(p.usedGpuMemory or 0) for p in procs
            }
        except Exception:
            pass

        # per-process SM utilization samples since last_ts
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(h, last_ts)
            pu: dict[str, int] = {}
            newest = last_ts
            for s in samples:
                pu[str(s.pid)] = max(pu.get(str(s.pid), 0), int(s.smUtil))
                newest = max(newest, int(s.timeStamp))
            if newest > last_ts:
                last_ts = newest
            if pu:
                rec["proc_sm"] = pu
        except nv.NVMLError:
            pass

        f.write(json.dumps(rec) + "\n")

        if now > remap_at:
            smap2 = stage_map()
            if smap2 != smap:
                smap = smap2
                f.write(json.dumps(
                    {"t": now - t0, "k": "stage_map",
                     "map": {str(k): v for k, v in smap.items()}}) + "\n")
            remap_at = now + 5.0
            f.flush()

        next_t += period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()  # fell behind; re-anchor

    f.flush()
    f.close()
    nv.nvmlShutdown()
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
