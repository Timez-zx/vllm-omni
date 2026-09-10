"""Source snapshots and coarse-to-fine capacity evidence helpers.

Each measured point uses a fresh server. Hashing and archiving occur outside
input timing; archived dirty-tree evidence is not formal certification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".cu", ".cuh", ".cpp", ".h", ".rs"}


def source_manifest(root):
    root = Path(root)
    if not (root / ".git").exists():
        return {
            "head": None,
            "files": {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix in SOURCE_SUFFIXES
            },
        }
    names = (
        subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root)
        .decode()
        .split("\0")
    )
    files = {}
    for name in sorted(set(names)):
        path = root / name
        if name and path.is_file() and path.suffix in SOURCE_SUFFIXES:
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(), "files": files}


def archive_source(root, folder):
    folder.mkdir()
    manifest = source_manifest(root)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["head"] is None:
        with tarfile.open(folder / "package-source.tar.gz", "w:gz") as archive:
            for name in manifest["files"]:
                archive.add(root / name, arcname=name, recursive=False)
        return manifest
    (folder / "tracked.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root))
    names = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    with tarfile.open(folder / "untracked.tar.gz", "w:gz") as archive:
        for name in names.decode().split("\0"):
            if name in manifest["files"]:
                archive.add(root / name, arcname=name, recursive=False)
    return manifest


def next_candidate(results, *, first, step, resolution, maximum):
    if not results:
        return first
    passed = [item["users"] for item in results if item["input_capacity_pass"]]
    failed = [item["users"] for item in results if not item["input_capacity_pass"]]
    low = max(passed, default=0)
    high = min(failed, default=maximum + 1)
    if failed:
        if high <= low:
            raise ValueError("non-monotonic capacity observations; repeat the conflicting points")
        if high - low <= resolution:
            return None
        return low + (high - low) // 2
    return min(low + step, maximum) if low < maximum else None


def context_coverage(run):
    rows = []
    for user in run["users"]:
        records = sorted(user["pd_completion_witness"]["records"], key=lambda row: row["input_unit_index"])
        prompts = [row["prompt_tokens"] for row in records]
        # A native lineage reset drops a long prompt back to the retained unit.
        resets = [records[i]["input_unit_index"] for i in range(1, len(prompts)) if prompts[i - 1] - prompts[i] > 1024]
        rows.append(
            {"session_id": user["session_id"], "max_prompt_tokens": max(prompts, default=0), "reset_units": resets}
        )
    return {"every_session_reset": bool(rows) and all(row["reset_units"] for row in rows), "sessions": rows}
