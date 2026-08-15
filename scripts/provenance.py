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


#: Parts an artifact can be attributed to, as the name appears inside a torch
#: device string ("AMD Instinct MI355X"). Deliberately the same literals as
#: `verify_artifacts.KNOWN_PARTS`: that resolver reads artifacts and this one
#: reads cards, but they answer one question, and a part known to only one of
#: them resolves differently depending on who asks. Literals rather than an
#: import of `solexbench_rocm.parts`, because this module must work with no
#: torch and no GPU.
KNOWN_PARTS = ("MI350X", "MI355X", "MI300X")


class PartConflict(RuntimeError):
    """A declared part and the cards this process can see disagree.

    Carries the provenance document that *would* have been written as `.block`,
    so a caller holding an expensive result can record the conflict and exit
    non-zero rather than losing the artifact to a traceback.
    """

    def __init__(self, message: str, block: dict):
        super().__init__(message)
        self.block = block


def detected_part(devices: list[str] | None = None) -> str | None:
    """The part of the cards THIS PROCESS can see, or None.

    Reads the device names `torch_info()` already collected -- no new hardware
    call, and None when `device_count` is 0. Deliberately NOT
    `solexbench_rocm.parts.detect_part()`: that needs a live GPU, raises on an
    unknown card, and would put a hardware probe inside a module whose contract
    is that every probe degrades to None.

    A device list naming two different parts returns None, not the first one.
    That is unresolvable, and unresolvable must not look like an answer --
    `leaderboard/ingest.py:774 run_part()` refuses a run that spans two parts
    for the same reason.
    """
    if devices is None:
        devices = torch_info().get("devices") or []
    seen = {part for dev in devices for part in KNOWN_PARTS if part in str(dev)}
    return seen.pop() if len(seen) == 1 else None


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


def stamp(task: str, extra: dict | None = None, *,
          part: str | None = None, allow_cross_part: bool = False) -> dict:
    """Build a provenance block. Attach to every artifact.

    Three keys answer the part question, because it is two questions:

    ==================  =========================================================
    ``part``            what this artifact is ABOUT: the declaration if the
                        caller made one, else what was detected, else None
    ``part_source``     ``"declared"`` | ``"detected"`` | None -- which of the
                        two the value above came from
    ``part_detected``   always the detection, even when a declaration overrode
                        it, so the evidence survives next to the statement
    ==================  =========================================================

    A **measured** artifact should detect: the part is a property of the machine
    the number came off. A **derived** artifact (``device="meta"`` -- every
    T_SOL tier, every manifest) should declare, from its ``--part`` or its arch
    config: the part is a parameter of the derivation and the host is
    irrelevant. Deriving MI350X bounds on the MI355X node is legitimate and
    stamping it MI355X would be a wrong statement that OUTRANKS its own
    evidence, because `verify_artifacts.artifact_part()` prefers an explicit
    `part` over the device list.

    So: a declaration that the visible cards contradict raises `PartConflict`,
    unless `allow_cross_part=True` (which records `part_cross_derived: true`).
    This is the only raise in a module whose contract is "never raise", and it
    is justified because it is not a probe failure -- it is a caller asserting
    something the machine contradicts, and both explanations (wrong flag, wrong
    node) invalidate the artifact. The exception carries the block it would have
    written, so a caller that has already paid for an expensive result can write
    it, marked, and exit non-zero instead of losing it.

    The declaration must come from a flag or a config the caller read -- never
    from the environment. `score_solutions.py` asks `stamp()` the *different*
    question "what part is this node" by reading a fresh stamp back; an
    env-sourced declaration would leak into that answer, make its
    `--part`-versus-detected comparison a tautology, and kill the guard a second
    time. `extra={"part": ...}` IS honoured as a declaration, because
    `artifacts/01/unlocked-clock.json` established that convention before the
    keyword existed and it is a caller statement like any other.
    """
    extra = dict(extra or {})
    declared = part if part is not None else extra.get("part")
    if not isinstance(declared, str) or not declared:
        declared = None
    info = torch_info()
    found = detected_part(info.get("devices") or [])
    conflict = declared is not None and found is not None and declared != found

    block = {
        "task": task,
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(Path(__file__).resolve().parent.parent),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": info,
        "rocm": rocm_info(),
        "f_lock_mhz": f_lock_mhz(),
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "part": declared or found,
        "part_source": "declared" if declared else ("detected" if found else None),
        "part_detected": found,
        **extra,
    }
    doc = {"_provenance": block}
    if conflict:
        if allow_cross_part:
            block["part_cross_derived"] = True
        else:
            block["part_conflict"] = {"declared": declared, "detected": found}
            raise PartConflict(
                f"{task}: declared part {declared!r} but this process sees "
                f"{found!r}. Either the flag or the node is wrong; pass "
                f"allow_cross_part=True only if this really is a {declared} "
                f"artifact derived on a {found} host.", doc)
    return doc


def write_artifact(path: str | Path, task: str, payload: dict,
                   extra_provenance: dict | None = None, *,
                   part: str | None = None,
                   allow_cross_part: bool = False) -> Path:
    """Write *payload* to *path* with a provenance block merged in.

    `part` declares what the artifact is about; see `stamp()`. A GPU-free
    deriver passes its own `--part` / arch name here and gets the cross-check
    for free.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {**stamp(task, extra_provenance, part=part,
                   allow_cross_part=allow_cross_part), **payload}
    p.write_text(json.dumps(doc, indent=2, default=str))
    return p


if __name__ == "__main__":
    # `--detect-part` prints the part of the visible cards and nothing else, so
    # a driver on a python WITHOUT torch can ask a python that has one. Empty
    # output means unresolvable, which the caller must not read as an answer.
    if len(sys.argv) > 1 and sys.argv[1] == "--detect-part":
        print(detected_part() or "")
    else:
        print(json.dumps(stamp(sys.argv[1] if len(sys.argv) > 1 else "adhoc"),
                         indent=2, default=str))
