#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refuse to run measurements on a stack that is not the pinned one.

Every baseline in this repo -- every tolerance, every T_b, every T_SOL in
milliseconds -- is relative to one torch/ROCm combination. A drifted stack does
not fail loudly; it produces numbers that look exactly as authoritative as the
pinned ones and cannot be told apart afterwards. Prime directive 6.

Called by ``env/solb-native``, and by ``env/Dockerfile`` at build time so the
image cannot be built around a stack it does not describe.

Exits 0 when the stack matches, 1 when it does not, printing what differs.

Environment:
  SOLB_WANT_TORCH       pinned torch version prefix, e.g. 2.9.1+rocm7.2.0
  SOLB_WANT_ROCM        pinned ROCm major.minor, e.g. 7.2
  SOLB_WANT_AITER_SHA   pinned aiter commit; unset means "aiter is not pinned
                        here", which is not the same as "any aiter will do"
  SOLB_COMPARE_RECORD   a record written by an earlier run; refuse if a pin
                        moved between then and now
  SOLB_STACK_RECORD     write the observed stack here, on pass and on failure
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(repo: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def aiter_info() -> dict[str, str | None]:
    """Identify the installed aiter WITHOUT importing it.

    ``find_spec`` rather than an import, for the reason scripts/provenance.py
    gives at its own ``_pkg``: importing aiter loads a compiled extension and
    creates a HIP context, and this check runs immediately before a measurement
    on the device it would perturb.

    Everything here comes from git because aiter publishes no version worth
    having -- it is installed from a shallow clone with no tags, so
    setuptools_scm has nothing to derive one from. The commit IS the version,
    which is also why ``scripts/provenance.py`` recovers it the same way.
    """
    info: dict[str, str | None] = {
        "aiter_path": None,
        "aiter_sha": None,
        "aiter_ck_sha": None,
        "aiter_dist_version": None,
        "aiter_worktree": None,
    }
    try:
        spec = importlib.util.find_spec("aiter")
    except Exception as exc:
        info["aiter_path"] = f"<find_spec failed: {type(exc).__name__}: {exc}>"
        return info
    if spec is None or not spec.origin:
        return info

    info["aiter_path"] = spec.origin
    root = Path(spec.origin).resolve().parent.parent
    # git searches upward, so a wheel install in site-packages can answer with
    # whatever repository happens to enclose the venv -- a SHA that is real,
    # unrelated, and indistinguishable from the right one on the artifact. Only
    # trust it when the import path IS the checkout root.
    top = _git(root, "rev-parse", "--show-toplevel")
    if top and Path(top).resolve() == root:
        info["aiter_sha"] = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain")
        if status is not None:
            info["aiter_worktree"] = "dirty" if status else "clean"
        info["aiter_ck_sha"] = _git(root / "3rdparty" / "composable_kernel", "rev-parse", "HEAD")

    try:
        import importlib.metadata as md

        info["aiter_dist_version"] = md.version("amd-aiter")
    except Exception:
        pass
    return info


def check_aiter(info: dict[str, str | None], want_sha: str) -> list[str]:
    """Assert the installed aiter is the commit the solutions were written for.

    29 of the harvested full-01 solutions ``import aiter``, so aiter is not a
    convenience here: for those problems it is the kernel. An unpinned aiter
    would move their timings the way an unpinned SOLAR would move every bound.

    The dirty-worktree refusal is not pedantry. The node those solutions were
    authored on stamped ``kernel_stack.aiter.git_sha`` as
    ``d9e5ef7ce08ee7045d583aed768cff41aa9210fe-dirty`` onto every artifact in
    artifacts/10/scores/full-01/, and nobody can now say what those uncommitted
    changes were -- aiter's own ``pretune`` rewrites tracked GEMM tables under
    hsa/, so a tuning run is enough to do it. A dirty checkout records a commit
    that does not describe the library that ran.
    """
    problems: list[str] = []
    if not want_sha:
        return problems

    if info.get("aiter_path") is None:
        problems.append(
            f"aiter is pinned to {want_sha} but is not installed; the 29 harvested "
            f"solutions that import it would be recorded as model failures"
        )
        return problems

    sha = info.get("aiter_sha")
    if sha is None:
        problems.append(
            f"aiter is installed at {info['aiter_path']} but its commit cannot be "
            f"read: it is a wheel install rather than a checkout, or git refused "
            f"the directory (dubious ownership needs a safe.directory entry). "
            f"Either way every artifact stamps aiter git_sha: null"
        )
    elif not sha.startswith(want_sha):
        problems.append(f"aiter is at {sha}, pinned to {want_sha}")

    if info.get("aiter_worktree") == "dirty":
        problems.append(
            f"the aiter checkout has uncommitted changes, so {sha} does not "
            f"identify the library that would run"
        )

    if info.get("aiter_ck_sha") is None:
        problems.append(
            "aiter's 3rdparty/composable_kernel submodule is not checked out; "
            "every CK-backed op (mha, the moe stages, the a8w8 gemms) fails at "
            "its first JIT build, which is mid-sweep rather than at startup"
        )
    return problems


# The pins that must survive anything added to the image after them. Not a
# check of *what* they are -- the base image decides that -- but that nothing
# moved them.
_INVARIANT_FIELDS = ("torch", "hip", "triton", "triton_path")


def check_against_record(observed: dict[str, str | None], path: str) -> list[str]:
    """Refuse when installing something moved a pin that was already in place.

    Concrete failure this exists for: aiter's setup.py runs
    ``.github/scripts/install_triton.sh`` whenever torch >= 2.9.1, and that
    script's first act is ``pip uninstall -y triton pytorch-triton-rocm`` before
    dropping in an aiter-built wheel. ``AITER_USE_SYSTEM_TRITON=1`` turns it
    off; this is the check that it stayed off. A Triton kernel's performance is
    a property of the compiler that built it (STATE.md D16), so the swap would
    invalidate every Triton measurement in the repo while breaking nothing.

    A pip constraint file cannot catch it: the script bypasses pip's resolver.
    """
    try:
        before = json.loads(Path(path).read_text())
    except Exception as exc:
        return [f"the earlier stack record {path} could not be read: {exc!r}"]

    problems: list[str] = []
    for key in _INVARIANT_FIELDS:
        if key in before and before[key] != observed.get(key):
            problems.append(
                f"{key} moved from {before[key]!r} to {observed.get(key)!r} since "
                f"{path} was written"
            )
    return problems


def write_record(observed: dict[str, str | None], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(observed, indent=2, sort_keys=True) + "\n")


def main() -> int:
    want_torch = os.environ.get("SOLB_WANT_TORCH", "")
    want_rocm = os.environ.get("SOLB_WANT_ROCM", "")
    want_aiter = os.environ.get("SOLB_WANT_AITER_SHA", "")

    problems: list[str] = []
    observed: dict[str, str | None] = {}

    try:
        import torch
    except Exception as exc:
        print(f"env/check_stack.py: torch is not importable: {exc!r}", file=sys.stderr)
        return 1

    observed["torch"] = torch.__version__
    observed["hip"] = str(torch.version.hip)
    if want_torch and not torch.__version__.startswith(want_torch):
        problems.append(f"torch {torch.__version__} does not start with pinned {want_torch}")
    if want_rocm and not str(torch.version.hip or "").startswith(want_rocm):
        problems.append(f"HIP {torch.version.hip} is not ROCm {want_rocm}.x")

    # Triton is not version-pinned here on purpose: this node resolves `triton`
    # to a development checkout rather than a release, and that is recorded as a
    # deviation in STATE.md rather than silently corrected. The import path is
    # reported so a reader can see which build produced a number, and
    # scripts/provenance.py stamps it onto every artifact.
    try:
        import triton

        observed["triton"] = triton.__version__
        observed["triton_path"] = triton.__file__
    except Exception as exc:
        problems.append(f"triton is not importable: {exc!r}")

    observed.update(aiter_info())
    problems += check_aiter(observed, want_aiter)

    compare = os.environ.get("SOLB_COMPARE_RECORD", "")
    if compare:
        problems += check_against_record(observed, compare)

    # Written whether or not the check passed: a stack that failed the pin is
    # exactly the one somebody will need the details of afterwards.
    record = os.environ.get("SOLB_STACK_RECORD", "")
    if record:
        write_record(
            {
                **observed,
                "utc": datetime.now(timezone.utc).isoformat(),
                "want_aiter_sha": want_aiter or None,
                "problems": "; ".join(problems) or None,
            },
            record,
        )

    if not problems:
        return 0

    # The prefix is the script, not env/solb-native, because env/Dockerfile runs
    # this too and a build failure that blames the native runner sends the reader
    # to the wrong file.
    print("env/check_stack.py: measurement stack does NOT match the pin.", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    for k, v in observed.items():
        print(f"  observed {k}: {v}", file=sys.stderr)
    print(
        "\n  Fix the environment, or (env/solb-native only) set\n"
        "  SOLB_ALLOW_STACK_DRIFT=1 and record the drift in STATE.md before\n"
        "  measuring anything.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
