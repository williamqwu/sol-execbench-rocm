#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve the in-container correctness-tolerance tree for a GPU part.

There is deliberately no fallback.  The two trees were calibrated separately
and differ materially, so an unknown part is an inability to score rather than
permission to borrow whichever tree happens to be oldest.
"""

from __future__ import annotations


TOLERANCE_ROOTS = {
    "MI350X": "/work/artifacts/05/workloads",
    "MI355X": "/work/artifacts/05-MI355X/workloads",
}


def container_tolerance_root(part: str) -> str:
    """Return the calibrated tolerance tree mounted inside ``env/solb``."""
    try:
        return TOLERANCE_ROOTS[part]
    except KeyError as exc:
        known = ", ".join(sorted(TOLERANCE_ROOTS))
        raise ValueError(
            f"no tolerance tree is registered for part {part!r}; known parts: "
            f"{known}"
        ) from exc
