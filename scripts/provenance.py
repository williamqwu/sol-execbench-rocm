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


def kernel_stack() -> dict:
    """Which kernel-authoring toolchains were available, and from where.

    Upstream needed none of this: a CUDA C++ solution is pinned by the CUDA
    version already in ``rocm``/``torch``. The AMD side is different in a way
    that matters for the agent scoreboard, because an agent may write a solution
    in any of these and the *build* of the toolchain then decides what the
    kernel is:

    - ``triton`` may resolve to a release wheel or to a development checkout.
      A Gluon kernel that compiles against a checkout may not exist at all in a
      release, so "triton 3.6.0" alone does not identify the compiler. The
      import path is recorded for that reason.
    - ``aiter`` is a source checkout with its own git SHA; the library *is* the
      kernel, so its SHA is part of the result.
    - ``ck`` / ``ck_tile`` / ``hipblaslt`` / ``miopen`` ship with ROCm, so the
      ROCm version pins them, but their presence is recorded so a failure to
      use one can be distinguished from an inability to.

    Never raises: a missing toolchain is recorded as absent, which is itself a
    fact worth having on the artifact.
    """
    import importlib.metadata as md
    import importlib.util

    def _pkg(name: str, module: str | None = None) -> dict:
        """Locate a package WITHOUT importing it.

        Deliberately uses ``find_spec`` rather than ``import_module``: importing
        ``aiter`` loads a compiled extension, and a provenance stamp taken in the
        middle of a timing run must not create a HIP context or perturb the
        device it is describing. Versions come from installed metadata, which
        needs no import either.
        """
        entry: dict = {"available": False}
        try:
            spec = importlib.util.find_spec(module or name)
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            return entry
        if spec is None:
            return entry
        entry["available"] = True
        entry["path"] = spec.origin
        try:
            entry["dist_version"] = md.version(name)
        except Exception:
            entry["dist_version"] = None
        return entry

    triton = _pkg("triton")
    if triton.get("available"):
        # A dist version carrying a git suffix, or an import path outside
        # site-packages, both mean "not a release wheel".
        path = triton.get("path") or ""
        dist = triton.get("dist_version") or ""
        triton["is_release_wheel"] = ("site-packages" in path) and ("git" not in dist)
        triton["gluon"] = _pkg("triton", "triton.experimental.gluon").get("available", False)

    stack: dict = {
        "triton": triton,
        "aiter": _pkg("aiter"),
        "hipcc": _run(["hipcc", "--version"]),
        "rocm_libraries": {
            name: Path(f"/opt/rocm/include/{name}").exists()
            for name in ("ck", "ck_tile", "hipblaslt", "miopen")
        },
    }

    aiter = stack["aiter"]
    if aiter.get("available") and aiter.get("path"):
        repo = Path(aiter["path"]).resolve().parent.parent
        aiter["git_sha"] = git_sha(repo)

    return stack


def part_name() -> str | None:
    """The Instinct part these measurements were taken on, e.g. ``MI355X``.

    Recorded explicitly because MI350X and MI355X are the same gfx950 die and
    are therefore indistinguishable from ``gcnArchName`` alone, while their
    measured quantities -- F_LOCK above all -- do not transfer between them.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from solexbench_rocm.parts import detect_part

        return detect_part().name
    except Exception:
        return None


def env_mode() -> dict:
    """Whether this ran in the pinned container or natively against it.

    ``env/solb`` runs inside ``solbench:rocm7.2-torch2.9.1``; ``env/solb-native``
    reproduces the same environment contract on a node with no docker, and
    asserts the stack matches rather than assuming it. The two are intended to
    be equivalent, so which one produced a number is exactly the kind of thing
    that should be on the record rather than inferred later.
    """
    return {
        "mode": os.environ.get("SOLEXBENCH_ENV_MODE", "unknown"),
        "in_docker": Path("/.dockerenv").exists(),
        "stack_drift_allowed": os.environ.get("SOLB_ALLOW_STACK_DRIFT") == "1",
    }


def f_lock_mhz() -> int | None:
    """F_LOCK for this artifact: the clock its measurements were taken at.

    Resolution order:
      1. ``SOLEXBENCH_F_LOCK_MHZ`` — an explicit override.
      2. The measured preset for the GPU this process can see.

    Step 2 exists because an env var is exactly the kind of thing a sweep
    forgets to export, and an artifact whose F_LOCK is null cannot be used for
    scoring. Reading it from the same table the lock is applied from means the
    recorded clock and the applied clock cannot disagree.

    Note this returns the ACHIEVED clock, not the requested one: on AMD they
    differ (see ``ClockPreset``), and the achieved value is the one every
    T_SOL and T_b is expressed at.
    """
    env = os.environ.get("SOLEXBENCH_F_LOCK_MHZ")
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    try:
        import torch

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from sol_execbench.core.bench.config import get_clock_preset

        preset = get_clock_preset(torch.cuda.get_device_name(0))
        return preset.f_lock_mhz if preset else None
    except Exception:
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
            "part": part_name(),
            "kernel_stack": kernel_stack(),
            "env": env_mode(),
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
