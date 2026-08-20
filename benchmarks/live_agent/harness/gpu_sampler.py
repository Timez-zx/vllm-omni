#!/usr/bin/env python3
"""Sample effective GPU activity for a multi-stage vllm-omni deployment.

``utilization.gpu`` is busy time, not achieved occupancy. This sampler also
records NVML GPM counters and uses the same monotonic clock as ``mu_bench.py``
so resource samples can be joined directly to turn timestamps.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import signal
import subprocess
import threading
import time
from typing import Any

import pynvml as nv

GPM_METRICS = {
    "sm_active_pct": nv.NVML_GPM_METRIC_SM_UTIL,
    "sm_occupancy_pct": nv.NVML_GPM_METRIC_SM_OCCUPANCY,
    "tensor_active_pct": nv.NVML_GPM_METRIC_ANY_TENSOR_UTIL,
    "dram_active_pct": nv.NVML_GPM_METRIC_DRAM_BW_UTIL,
    "fp16_active_pct": nv.NVML_GPM_METRIC_FP16_UTIL,
    "fp32_active_pct": nv.NVML_GPM_METRIC_FP32_UTIL,
    "pcie_rx_mib_s": nv.NVML_GPM_METRIC_PCIE_RX_PER_SEC,
    "pcie_tx_mib_s": nv.NVML_GPM_METRIC_PCIE_TX_PER_SEC,
}


def stage_map() -> dict[int, str]:
    """Map vllm-omni worker PIDs to readable stage labels."""
    out: dict[int, str] = {}
    try:
        process_list = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return out
    for line in process_list.splitlines():
        pid_text, _, process_args = line.strip().partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if "vllm" not in process_args.lower():
            continue
        match = re.search(r"stage[_-]?(\d+)", process_args)
        if match:
            out[pid] = f"stage{match.group(1)}"
        elif "serve" in process_args or "api_server" in process_args:
            out[pid] = "frontend"
    return out


def optional(call, default: Any = None) -> Any:
    try:
        return call()
    except Exception:
        return default


class DeviceSampler:
    def __init__(self, index: int) -> None:
        self.index = index
        self.handle = nv.nvmlDeviceGetHandleByIndex(index)
        self.name = nv.nvmlDeviceGetName(self.handle)
        if isinstance(self.name, bytes):
            self.name = self.name.decode()
        self.total_memory = nv.nvmlDeviceGetMemoryInfo(self.handle).total
        self.max_sm_clock = optional(lambda: nv.nvmlDeviceGetMaxClockInfo(self.handle, nv.NVML_CLOCK_SM))
        self.power_limit_w = optional(lambda: nv.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0)
        self.last_process_timestamp = 0
        self.gpm_was_enabled = False
        self.gpm_enabled = False
        self.gpm_old = None
        self.gpm_new = None
        self._start_gpm()

    def _start_gpm(self) -> None:
        try:
            support = nv.nvmlGpmQueryDeviceSupport(self.handle)
            if not support.isSupportedDevice:
                return
            self.gpm_was_enabled = bool(nv.nvmlGpmQueryIfStreamingEnabled(self.handle))
            if not self.gpm_was_enabled:
                nv.nvmlGpmSetStreamingEnabled(self.handle, 1)
            self.gpm_old = nv.nvmlGpmSampleAlloc()
            self.gpm_new = nv.nvmlGpmSampleAlloc()
            nv.nvmlGpmSampleGet(self.handle, self.gpm_old)
            self.gpm_enabled = True
        except Exception:
            self.gpm_enabled = False

    def close(self) -> None:
        if self.gpm_old is not None:
            optional(lambda: nv.nvmlGpmSampleFree(self.gpm_old))
        if self.gpm_new is not None:
            optional(lambda: nv.nvmlGpmSampleFree(self.gpm_new))
        if self.gpm_enabled and not self.gpm_was_enabled:
            optional(lambda: nv.nvmlGpmSetStreamingEnabled(self.handle, 0))

    def _gpm_values(self) -> dict[str, float | None]:
        values: dict[str, float | None] = {name: None for name in GPM_METRICS}
        if not self.gpm_enabled:
            return values
        try:
            nv.nvmlGpmSampleGet(self.handle, self.gpm_new)
            request = nv.c_nvmlGpmMetricsGet_t()
            request.version = nv.NVML_GPM_METRICS_GET_VERSION
            request.numMetrics = len(GPM_METRICS)
            request.sample1 = self.gpm_old
            request.sample2 = self.gpm_new
            for slot, metric_id in zip(request.metrics, GPM_METRICS.values(), strict=False):
                slot.metricId = metric_id
            nv.nvmlGpmMetricsGet(request)
            for name, slot in zip(GPM_METRICS, request.metrics, strict=False):
                value = float(slot.value)
                if slot.nvmlReturn == nv.NVML_SUCCESS and math.isfinite(value):
                    values[name] = value
            self.gpm_old, self.gpm_new = self.gpm_new, self.gpm_old
        except Exception:
            pass
        return values

    def sample(self, monotonic_s: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "k": "sample",
            "monotonic_s": monotonic_s,
            "wall_time_s": time.time(),
            "gpu": self.index,
            **self._gpm_values(),
        }
        utilization = optional(lambda: nv.nvmlDeviceGetUtilizationRates(self.handle))
        if utilization is not None:
            record["gpu_busy_pct"] = int(utilization.gpu)
            record["memory_io_pct"] = int(utilization.memory)
        memory = optional(lambda: nv.nvmlDeviceGetMemoryInfo(self.handle))
        if memory is not None:
            record["memory_used_mib"] = memory.used / 2**20
        record["power_w"] = optional(lambda: nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
        record["sm_clock_mhz"] = optional(lambda: nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_SM))
        record["memory_clock_mhz"] = optional(lambda: nv.nvmlDeviceGetClockInfo(self.handle, nv.NVML_CLOCK_MEM))
        record["temperature_c"] = optional(lambda: nv.nvmlDeviceGetTemperature(self.handle, nv.NVML_TEMPERATURE_GPU))
        record["pstate"] = optional(lambda: int(nv.nvmlDeviceGetPerformanceState(self.handle)))
        record["throttle_reasons"] = optional(lambda: int(nv.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)))
        processes = optional(lambda: nv.nvmlDeviceGetComputeRunningProcesses(self.handle), [])
        record["process_memory_mib"] = {str(process.pid): (process.usedGpuMemory or 0) / 2**20 for process in processes}
        try:
            samples = nv.nvmlDeviceGetProcessUtilization(self.handle, self.last_process_timestamp)
            process_sm: dict[str, int] = {}
            newest = self.last_process_timestamp
            for sample in samples:
                pid = str(sample.pid)
                process_sm[pid] = max(process_sm.get(pid, 0), int(sample.smUtil))
                newest = max(newest, int(sample.timeStamp))
            self.last_process_timestamp = newest
            record["process_sm_pct"] = process_sm
        except nv.NVMLError:
            record["process_sm_pct"] = {}
        return record

    def metadata(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_mib": self.total_memory / 2**20,
            "max_sm_clock_mhz": self.max_sm_clock,
            "power_limit_w": self.power_limit_w,
            "gpm_enabled": self.gpm_enabled,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--devices", default="0,1,2")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 = until killed")
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("hz must be positive")
    try:
        indices = [int(value) for value in args.devices.split(",")]
    except ValueError:
        parser.error("devices must be comma-separated GPU indices")
    if not indices or len(set(indices)) != len(indices):
        parser.error("devices must contain unique GPU indices")

    stop = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop.set()

    # Bash starts background jobs with SIGINT ignored. Reinstall both handlers
    # explicitly so the capacity runner can always stop and flush the sampler.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    nv.nvmlInit()
    devices = [DeviceSampler(index) for index in indices]
    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period_s = 1.0 / args.hz
    start = time.monotonic()
    deadline = start
    mapping: dict[int, str] = {}
    next_remap = start
    try:
        with output_path.open("w") as output:
            output.write(
                json.dumps(
                    {
                        "k": "meta",
                        "monotonic_start_s": start,
                        "wall_start_s": time.time(),
                        "hz": args.hz,
                        "devices": [device.metadata() for device in devices],
                        "caveats": [
                            "gpu_busy_pct is busy time, not achieved occupancy",
                            "GPM activity counters are interval averages",
                            "per-process SM is an NVML sampled estimate",
                        ],
                    }
                )
                + "\n"
            )
            while not stop.is_set() and (not args.duration_s or time.monotonic() - start < args.duration_s):
                now = time.monotonic()
                if now >= next_remap:
                    updated = stage_map()
                    if updated != mapping:
                        mapping = updated
                        output.write(
                            json.dumps(
                                {
                                    "k": "stage_map",
                                    "monotonic_s": now,
                                    "map": {str(pid): stage for pid, stage in mapping.items()},
                                }
                            )
                            + "\n"
                        )
                    next_remap = now + 5.0
                for device in devices:
                    output.write(json.dumps(device.sample(now)) + "\n")
                output.flush()
                deadline += period_s
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0:
                    stop.wait(sleep_s)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        for device in devices:
            device.close()
        nv.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
