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


def recorded_tolerance_root(payload: object,
                            measured_part: str | None) -> str | None:
    """Return the tree that produced a re-time, when the artifact proves it.

    New artifacts carry an explicit stamp.  Before that stamp existed, both
    authoritative re-time paths unconditionally selected the MI350X tree.  An
    old ``10-agent-eval`` artifact measured on MI350X is therefore still
    auditable and must not consume GPU time again merely to add metadata.

    The exception is deliberately one-way.  An unstamped MI355X artifact was
    evaluated through that same hard-coded MI350X tree, which is the defect
    this module closes, so it has no reusable correctness verdict.
    """
    if not isinstance(payload, dict):
        return None
    stamped = payload.get("tolerance_root")
    if isinstance(stamped, str) and stamped:
        return stamped
    provenance = payload.get("_provenance") or {}
    if (isinstance(provenance, dict)
            and provenance.get("task") == "10-agent-eval"
            and measured_part == "MI350X"):
        return TOLERANCE_ROOTS["MI350X"]
    return None
