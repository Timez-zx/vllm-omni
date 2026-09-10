"""Isolated, same-user MPS lifecycle for free benchmark GPUs.

Never change the machine's default MPS socket or GPU compute mode. The caller
must check GPU ownership before entering, and stop its clients before exiting.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


class PrivateMPS:
    def __init__(self, output: Path, env: dict[str, str], gpu_uuid: str | list[str], *, stage_count: int = 3):
        if stage_count < 1:
            raise ValueError("stage_count must be positive")
        self.stage_count = stage_count
        unexpected = [key for key in env if key.startswith("CUDA_MPS_")]
        if unexpected:
            raise RuntimeError(f"Refuse inherited MPS policy: {unexpected}")
        gpu_uuids = [gpu_uuid] if isinstance(gpu_uuid, str) else list(gpu_uuid)
        if not gpu_uuids or any(not uuid.startswith("GPU-") or "," in uuid for uuid in gpu_uuids):
            raise ValueError("MPS requires an ordered, nonempty list of GPU UUIDs")
        if len(set(gpu_uuids)) != len(gpu_uuids):
            raise ValueError("MPS GPU UUIDs must be unique")
        visible_devices = ",".join(gpu_uuids)
        self.output = output
        self.pipe = Path(tempfile.mkdtemp(prefix="minicpm-mps-"))
        self.logs = output / "mps-logs"
        self.logs.mkdir()
        # PyTorch's NVML enumeration otherwise sees all four physical GPUs,
        # while the MPS CUDA client sees only the selected devices. Restrict by
        # UUID (not a possibly remapped ordinal) before Python imports torch.
        self.client_env = dict(
            env,
            CUDA_MPS_PIPE_DIRECTORY=str(self.pipe),
            CUDA_MPS_LOG_DIRECTORY=str(self.logs),
            CUDA_VISIBLE_DEVICES=visible_devices,
        )
        self.daemon_env = dict(self.client_env)
        self.record = {
            "gpu_uuid": gpu_uuids[0] if len(gpu_uuids) == 1 else None,
            "gpu_uuids": gpu_uuids,
            "pipe_directory": str(self.pipe),
            "log_directory": str(self.logs),
            "uid": os.getuid(),
            "compute_mode_changed": False,
            "active_thread_percentage": 100,
            "expected_stage_count": stage_count,
            "started_epoch": time.time(),
            "commands": [],
            "snapshots": [],
        }
        self.started = False

    def save(self):
        (self.output / "mps.json").write_text(json.dumps(self.record, indent=2) + "\n")

    def command(self, command):
        result = subprocess.run(
            ["nvidia-cuda-mps-control"],
            input=command + "\n",
            text=True,
            capture_output=True,
            env=self.daemon_env,
            timeout=30,
        )
        self.record["commands"].append(
            {
                "epoch": time.time(),
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        self.save()
        result.check_returncode()
        return result.stdout.strip()

    def __enter__(self):
        self.save()
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=self.daemon_env, check=True, timeout=30)
        self.started = True
        try:
            self.command("set_default_active_thread_percentage 100")
            self.command(f"start_server -uid {os.getuid()}")
            self.snapshot("daemon_started")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def snapshot(self, label, server_log=None):
        server_ids = [int(value) for value in self.command("get_server_list").split() if value.isdigit()]
        if not server_ids:
            raise RuntimeError("MPS did not start a server")
        clients = set()
        for pid in server_ids:
            clients.update(int(value) for value in self.command(f"get_client_list {pid}").split() if value.isdigit())
        snapshot = {
            "label": label,
            "epoch": time.time(),
            "servers": server_ids,
            "clients": sorted(clients),
            "client_device_listing": self.command("ps"),
            "device_client_listing": self.command("get_device_client_list"),
        }
        if server_log is not None:
            stages = dict(re.findall(r"\(StageEngineCoreProc_stage(\d+)_replica0 pid=(\d+)\)", server_log.read_text()))
            expected = {int(stages[str(stage)]) for stage in range(self.stage_count) if str(stage) in stages}
            snapshot["stage_pids"] = stages
            snapshot["all_stages_connected"] = len(expected) == self.stage_count and expected <= clients
            self.record["snapshots"].append(snapshot)
            self.save()
            if not snapshot["all_stages_connected"]:
                raise RuntimeError(f"Not all {self.stage_count} stages connected to MPS: {snapshot}")
        else:
            self.record["snapshots"].append(snapshot)
            self.save()
        return snapshot

    def __exit__(self, *_):
        if self.started:
            self.command("quit")
            self.record["stopped_epoch"] = time.time()
            self.save()
            self.started = False
