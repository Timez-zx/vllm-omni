"""Launch a benchmark server with a reproducible, non-diagnostic environment.

This wrapper deliberately uses only the Python standard library.  In
particular, it does not import vLLM or vLLM-Omni before replacing itself with
the server process.  Import resolution is recorded with ``find_spec`` so a
stale editable install cannot silently become part of a benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# These switches enable per-request/per-step logging, explicit CUDA timing
# synchronization, or both.  A formal run removes them rather than merely
# setting them to ``0`` so provenance describes the effective environment
# unambiguously.
DIAGNOSTIC_ENV_VARS = (
    "MINICPMO45_LOG_PREP_DIAG",
    "VLLM_OMNI_DIAG_STAGE",
    "VLLM_OMNI_LOG_AUDIO_CHUNKS",
    "VLLM_OMNI_LOG_CORE_STEP_DIAG",
    "VLLM_OMNI_LOG_DUPLEX_CADENCE",
    "VLLM_OMNI_LOG_ENC",
    "VLLM_OMNI_LOG_HANDOFF_DIAG",
    "VLLM_OMNI_LOG_MTP_GPU",
    "VLLM_OMNI_LOG_ORCH_LAG",
    "VLLM_OMNI_LOG_PD_ITER",
    "VLLM_OMNI_LOG_PD_SLOT",
    "VLLM_OMNI_LOG_PREFIX_CACHE",
    "VLLM_OMNI_LOG_REQ_STEPS",
    "VLLM_OMNI_LOG_RUNNER_DIAG",
    "VLLM_OMNI_LOG_SCHED_DIAG",
    "VLLM_OMNI_LOG_SCHED_STEPS",
    "VLLM_OMNI_LOG_SPAN",
    "VLLM_OMNI_LOG_STEP_GPU",
    "VLLM_OMNI_MINICPMO_PD_ONLY_DIAGNOSTIC",
)

# These command-line switches enable server-side profilers or monitoring that
# can perturb a capacity run.  Unlike environment switches, they cannot be
# made safe by cleaning the child environment, so a formal launch must reject
# the command instead of silently passing it through.
DIAGNOSTIC_CLI_FLAGS = (
    "--enable-ar-profiler",
    "--enable-orch-monitor",
    "--enable-diffusion-pipeline-profiler",
)

PROVENANCE_ENV_VARS = (
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "NIXL_PLUGIN_DIR",
    "PYTHONPATH",
    "MINICPMO45_VISION_ENCODER_BATCH_SIZE",
    "VLLM_OMNI_INPUT_THREADS",
    "VLLM_OMNI_PD_SNAPSHOT_CACHE_BYTES",
    "VLLM_OMNI_PD_SNAPSHOT_MAX_CHUNKS",
    "VLLM_OMNI_PREFILL_MICROBATCH_WINDOW_MS",
    "VLLM_OMNI_STAGE0_SHM_MM_CACHE",
    "VLLM_USE_FLASHINFER_SAMPLER",
    *DIAGNOSTIC_ENV_VARS,
)


def _valid_cuda_home(path: Path) -> bool:
    return (
        (path / "bin" / "nvcc").is_file()
        and (path / "lib64").is_dir()
        and any((path / "lib64").glob("libcudart.so*"))
    )


def launch_environment(
    env: Mapping[str, str],
    *,
    python_executable: str | Path | None = None,
    purelib: str | Path | None = None,
) -> dict[str, str]:
    """Expose the active environment's build tools to spawned stage workers.

    vLLM can JIT a missing FlashInfer kernel during startup.  The benchmark
    launcher may be invoked with an absolute Python path without activating
    its environment, so neither the adjacent ``ninja`` nor a pip-installed
    CUDA toolkit is necessarily on ``PATH``.
    """
    prepared = dict(env)
    executable = Path(python_executable or sys.executable).resolve()
    python_bin = executable.parent

    cuda_home = prepared.get("CUDA_HOME")
    if not cuda_home:
        site_packages = Path(purelib or sysconfig.get_path("purelib")).resolve()
        candidates = sorted(
            (site_packages / "nvidia").glob("cu*"),
            reverse=True,
        )
        path_nvcc = shutil.which("nvcc", path=prepared.get("PATH"))
        if path_nvcc:
            candidates.append(Path(path_nvcc).resolve().parent.parent)
        candidates.append(Path("/usr/local/cuda"))
        discovered = next((path for path in candidates if _valid_cuda_home(path)), None)
        if discovered is not None:
            cuda_home = str(discovered)
            prepared["CUDA_HOME"] = cuda_home

    prepend = [str(python_bin)]
    if cuda_home:
        prepend.append(str(Path(cuda_home).expanduser().resolve() / "bin"))
    current = [entry for entry in prepared.get("PATH", "").split(os.pathsep) if entry]
    prepared["PATH"] = os.pathsep.join(dict.fromkeys((*prepend, *current)))
    return prepared


def _git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin not in (None, "namespace"):
        return str(Path(spec.origin).resolve())
    locations = spec.submodule_search_locations
    return str(Path(next(iter(locations))).resolve()) if locations else None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


def diagnostic_cli_flags(command: Sequence[str]) -> list[str]:
    """Return diagnostic CLI arguments, preserving their launched spelling."""
    return [
        argument
        for argument in command
        if any(
            argument == flag or argument.startswith(f"{flag}=")
            for flag in DIAGNOSTIC_CLI_FLAGS
        )
    ]


def _command_option(command: Sequence[str], name: str) -> str | None:
    """Return the last value supplied for a conventional CLI option."""
    value: str | None = None
    for index, argument in enumerate(command):
        if argument == name and index + 1 < len(command):
            value = command[index + 1]
        elif argument.startswith(f"{name}="):
            value = argument.split("=", 1)[1]
    return value


def _deploy_config_provenance(
    repo: Path,
    command: Sequence[str],
    *,
    cwd: Path,
) -> dict[str, Any]:
    """Resolve and hash the exact deploy YAML named by the server command.

    This mirrors the two useful resolution forms accepted by vLLM-Omni: a
    path relative to the launch directory, or a bare name in
    ``vllm_omni/deploy``. Formal capacity runs deliberately require an
    explicit ``--deploy-config`` so the topology is reproducible.
    """
    argument = _command_option(command, "--deploy-config")
    if not argument:
        return {
            "argument": None,
            "resolved_path": None,
            "sha256": None,
            "size_bytes": None,
            "exists": False,
        }

    requested = Path(argument).expanduser()
    resolved = requested if requested.is_absolute() else cwd / requested
    if not resolved.exists() and requested.parent == Path("."):
        name = requested.name
        if not name.endswith(".yaml"):
            name = f"{name}.yaml"
        candidate = repo / "vllm_omni" / "deploy" / name
        if candidate.exists():
            resolved = candidate
    resolved = resolved.resolve()
    exists = resolved.is_file()
    payload = resolved.read_bytes() if exists else None
    return {
        "argument": argument,
        "resolved_path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest() if payload is not None else None,
        "size_bytes": len(payload) if payload is not None else None,
        "exists": exists,
    }


def collect_provenance(
    repo: Path,
    env: Mapping[str, str] | None = None,
    *,
    formal: bool,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    effective_env = os.environ if env is None else env
    status = _git(repo, "status", "--porcelain")
    selected_env = {name: effective_env[name] for name in PROVENANCE_ENV_VARS if name in effective_env}
    enabled_diagnostics = [
        name for name in DIAGNOSTIC_ENV_VARS if _enabled(effective_env.get(name))
    ]
    cli_enabled = diagnostic_cli_flags(command or ())
    cwd = Path.cwd().resolve()
    return {
        "captured_epoch_s": time.time(),
        "formal": formal,
        "command": list(sys.argv),
        "cwd": str(cwd),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
        },
        "git": {
            "repo": str(repo.resolve()),
            "head": _git(repo, "rev-parse", "HEAD"),
            "describe": _git(repo, "describe", "--always", "--dirty", "--tags"),
            "branch": _git(repo, "branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines() if status else [],
        },
        "imports": {
            "vllm": _module_origin("vllm"),
            "vllm_omni": _module_origin("vllm_omni"),
        },
        "distributions": {
            "vllm": _distribution_version("vllm"),
            "vllm-omni": _distribution_version("vllm-omni"),
            "torch": _distribution_version("torch"),
        },
        "environment": selected_env,
        "build_tools": {
            "cuda_home": effective_env.get("CUDA_HOME"),
            "nvcc": shutil.which("nvcc", path=effective_env.get("PATH")),
            "ninja": shutil.which("ninja", path=effective_env.get("PATH")),
        },
        "diagnostics": {
            "enabled": enabled_diagnostics,
            "cli_enabled": cli_enabled,
            "all_disabled": not enabled_diagnostics and not cli_enabled,
        },
        "deploy_config": _deploy_config_provenance(
            repo,
            command or (),
            cwd=cwd,
        ),
    }


def formal_environment(env: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Return a copy with all benchmark-perturbing diagnostics removed."""
    cleaned = dict(env)
    removed = {name: cleaned.pop(name) for name in DIAGNOSTIC_ENV_VARS if name in cleaned}
    return cleaned, removed


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="record import provenance, disable diagnostics, then exec the server"
    )
    parser.add_argument("--provenance-out", required=True)
    parser.add_argument(
        "--allow-diagnostics",
        action="store_true",
        help="diagnostic rerun only: preserve diagnostic environment switches",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a server command is required after --")
    if command[0] in {"python", "python3"}:
        # Provenance is collected in this interpreter.  Keep a bare Python
        # command on the exact same executable instead of re-resolving PATH
        # after the environment has been cleaned.
        command[0] = sys.executable

    cli_enabled = diagnostic_cli_flags(command)
    if cli_enabled and not args.allow_diagnostics:
        parser.error(
            "formal server command contains diagnostic CLI flags: "
            f"{', '.join(cli_enabled)}; use --allow-diagnostics only for a "
            "separate diagnostic run"
        )

    inherited = dict(os.environ)
    if args.allow_diagnostics:
        effective = inherited
        removed: dict[str, str] = {}
    else:
        effective, removed = formal_environment(inherited)
    effective = launch_environment(effective)
    repo = Path(__file__).resolve().parents[2]
    repo_text = str(repo.resolve())
    inherited_pythonpath = effective.get("PYTHONPATH")
    effective["PYTHONPATH"] = (
        repo_text
        if not inherited_pythonpath
        else os.pathsep.join((repo_text, inherited_pythonpath))
    )
    if sys.path[:1] != [repo_text]:
        sys.path.insert(0, repo_text)
    provenance = collect_provenance(
        repo,
        effective,
        formal=not args.allow_diagnostics,
        command=command,
    )
    provenance["run_id"] = uuid.uuid4().hex
    provenance["launched_command"] = command
    provenance["removed_diagnostic_environment"] = removed
    _write_json(Path(args.provenance_out), provenance)
    print(
        "[benchmark-provenance] "
        f"run_id={provenance['run_id']} "
        f"captured_epoch={provenance['captured_epoch_s']:.6f}",
        flush=True,
    )
    os.execvpe(command[0], command, effective)


if __name__ == "__main__":
    main()
