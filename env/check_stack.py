#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refuse to run measurements on a stack that is not the pinned one.

Every baseline in this repo -- every tolerance, every T_b, every T_SOL in
milliseconds -- is relative to one torch/ROCm combination. A drifted stack does
not fail loudly; it produces numbers that look exactly as authoritative as the
pinned ones and cannot be told apart afterwards. Prime directive 6.

Called by ``env/solb-native``. Exits 0 when the stack matches, 1 when it does
not, printing what differs.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    want_torch = os.environ.get("SOLB_WANT_TORCH", "")
    want_rocm = os.environ.get("SOLB_WANT_ROCM", "")

    problems: list[str] = []
    observed: dict[str, str] = {}

    try:
        import torch
    except Exception as exc:
        print(f"env/solb-native: torch is not importable: {exc!r}", file=sys.stderr)
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

    if not problems:
        return 0

    print("env/solb-native: measurement stack does NOT match the pin.", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    for k, v in observed.items():
        print(f"  observed {k}: {v}", file=sys.stderr)
    print(
        "\n  Fix the environment, or set SOLB_ALLOW_STACK_DRIFT=1 and record the\n"
        "  drift in STATE.md before measuring anything.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
