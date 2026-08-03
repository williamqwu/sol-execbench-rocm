#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Provenance stamping. Every artifact carries one of these.

An artifact without provenance is not usable for scoring: six months from now
nobody can tell whether a number was measured at the right clock, on the right
stack, from the right commit.

CPU-safe: every GPU/ROCm probe degrades to None rather than raising, so this
module imports and runs anywhere.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_sha(repo_root: Path | None = None) -> str | None:
    cwd = str(repo_root) if repo_root else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=cwd, timeout=30,
        )
        if out.returncode != 0:
            return None
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=cwd, timeout=30,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return None


def torch_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info = {
        "available": True,
        "version": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
    }
    try:
        info["device_count"] = torch.cuda.device_count()
        info["devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        info["device_count"] = 0
        info["devices"] = []
    return info


def rocm_info() -> dict:
    version = None
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"):
        try:
            version = Path(path).read_text().strip()
            break
        except Exception:
            continue
    return {
        "version": version,
        "driver": _run(["cat", "/sys/module/amdgpu/version"]),
        "amd_smi": _run(["amd-smi", "version"]),
    }


def f_lock_mhz() -> int | None:
    """F_LOCK as recorded by task 01, via env or the state file."""
    env = os.environ.get("SOLEXBENCH_F_LOCK_MHZ")
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    return None


def stamp(task: str, extra: dict | None = None) -> dict:
    """Build a provenance block. Attach to every artifact."""
    return {
        "_provenance": {
            "task": task,
            "utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha(Path(__file__).resolve().parent.parent),
            "host": platform.node(),
            "python": sys.version.split()[0],
            "torch": torch_info(),
            "rocm": rocm_info(),
            "f_lock_mhz": f_lock_mhz(),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            **(extra or {}),
        }
    }


def write_artifact(path: str | Path, task: str, payload: dict,
                   extra_provenance: dict | None = None) -> Path:
    """Write *payload* to *path* with a provenance block merged in."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {**stamp(task, extra_provenance), **payload}
    p.write_text(json.dumps(doc, indent=2, default=str))
    return p


if __name__ == "__main__":
    print(json.dumps(stamp(sys.argv[1] if len(sys.argv) > 1 else "adhoc"),
                     indent=2, default=str))
