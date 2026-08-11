#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's B200 figures are matched by axes, or not at all.

The overlay pairs each of our workloads with the one NVIDIA published for the
same axis values. Their records carry every axis including the constants; ours
carry only what varies — so the match is a subset probe, and a subset probe is
exactly the kind that can quietly answer for the wrong row.

Three problems in the real dataset have workloads that share axes (both ragged
prefill kernels, and L1__016 whose sixteen workloads declare no axes at all).
There, no honest pairing exists. The rule is that ambiguity yields nothing:
NULL in the column, a blank cell on the page, and a count on the page of how
many could not be matched. A nearest-match here would put another part's
milliseconds on a row they do not describe, and nothing downstream could ever
detect it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "leaderboard"))

import ingest  # noqa: E402

PUBLISHED = {
    "L1__001_alpha": {
        "workloads": [
            {"axes": {"n": 1, "h": 8}, "baseline_latency_ms": 1.0, "sol_ms": 0.1},
            {"axes": {"n": 2, "h": 8}, "baseline_latency_ms": 2.0, "sol_ms": 0.2},
            # Two workloads, identical axes: NVIDIA distinguishes them by
            # something their record does not carry.
            {"axes": {"n": 9, "h": 8}, "baseline_latency_ms": 9.0, "sol_ms": 0.9},
            {"axes": {"n": 9, "h": 8}, "baseline_latency_ms": 9.5, "sol_ms": 0.95},
        ]
    }
}


@pytest.fixture()
def index():
    return ingest.b200_by_axes(PUBLISHED, "L1__001_alpha")


def test_a_unique_axes_match_is_used(index):
    assert ingest.b200_for(index, {"n": 1}) == (1.0, 0.1)
    assert ingest.b200_for(index, {"n": 2, "h": 8}) == (2.0, 0.2)


def test_axes_compare_by_value_not_by_type(index):
    """Ours arrive from JSON, theirs from JSON, and one side has been an int
    where the other was a string before now."""
    assert ingest.b200_for(index, {"n": "1"}) == (1.0, 0.1)


def test_duplicate_axes_on_their_side_match_nothing(index):
    """Not the first of them, and not an average of the two."""
    assert ingest.b200_for(index, {"n": 9}) == (None, None)


def test_an_ambiguous_probe_matches_nothing(index):
    """`h=8` is true of every workload here. A subset that does not separate
    the rows is not a match."""
    assert ingest.b200_for(index, {"h": 8}) == (None, None)


def test_a_workload_with_no_axes_matches_nothing(index):
    """L1__016 is this case, sixteen times over."""
    assert ingest.b200_for(index, {}) == (None, None)


def test_an_unknown_value_matches_nothing(index):
    assert ingest.b200_for(index, {"n": 404}) == (None, None)


def test_a_problem_nvidia_does_not_publish_yields_an_empty_index():
    assert ingest.b200_by_axes(PUBLISHED, "L2__002_beta") == {}
    assert ingest.b200_for({}, {"n": 1}) == (None, None)


def test_the_overlay_never_reaches_a_scored_column(client):
    """The board's own bounds and the overlay are different columns, and the
    API — which is what anything downstream reads — carries only the former."""
    row = client.get("/api/v1/problems/L1__001_alpha").json()
    text = repr(row)
    assert "b200" not in text.lower()
