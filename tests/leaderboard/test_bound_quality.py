# SPDX-License-Identifier: Apache-2.0
"""The per-workload bound marking (STATE.md D39).

The board enforces exactly one invariant on a T_SOL -- nothing may beat it --
and that invariant is one-sided. It catches a bound too LARGE and is blind to
one too small, because a weak lower bound breaks no rule. 827 of 3,717
workloads (22.3%) sit above 100x headroom, where S collapses toward
T_b/(T_b+T_k) and carries no roofline content, and nothing reported it for the
whole life of the benchmark.

These tests are about the marking, not about any bound. Nothing here asserts
that a loose bound is wrong -- `FlashInfer-Bench__018` went from 185,274 cycles
to 8 in v1.1 and the new number is the correct one. Correct and vacuous is a
real state and the board should be able to say it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "leaderboard"))
from ingest import bound_quality  # noqa: E402


@pytest.mark.parametrize("t_sol,t_b,expected", [
    (1.0, 1.5, "narrow"),      # variance is a material share of S
    (1.0, 2.0, "ok"),          # boundary is inclusive upward
    (1.0, 99.0, "ok"),
    (1.0, 100.0, "loose"),
    (1.0, 999.0, "loose"),
    (1.0, 1000.0, "vacuous"),
    (1.0, 115005.0, "vacuous"),   # L2__006, the worst on the board
])
def test_the_bands_are_where_they_say_they_are(t_sol, t_b, expected):
    assert bound_quality(t_sol, t_b)[0] == expected


def test_a_workload_with_no_anchor_is_not_marked():
    """A deferred problem has a T_SOL -- it is architectural and needs no GPU
    -- and no T_b, because nothing ran. Marking that `narrow` would invent a
    judgement about a bound nobody has measured against."""
    assert bound_quality(1.0, None) == (None, None)
    assert bound_quality(None, 1.0) == (None, None)
    assert bound_quality(0.0, 1.0) == (None, None)


def test_the_headroom_comes_back_with_the_word(client):
    """Both, so a reader can disagree with the banding."""
    q, h = bound_quality(2.0, 50.0)
    assert q == "ok" and h == 25.0


def test_every_scoreable_workload_on_the_board_is_marked(client):
    """A NULL here would be a workload whose score nobody can weigh.

    Read off a rendered problem page rather than the table, because the point
    is that the marking REACHES a reader: a column populated in SQLite and
    dropped by the template is the same to the reader as one that was never
    computed.
    """
    listing = client.get("/problems").text
    keys = set(re.findall(r'/problems/([A-Za-z0-9_.-]+?)(?:\?|")', listing))
    assert keys, "no problems linked from /problems"
    checked = 0
    for key in sorted(keys)[:12]:
        page = client.get(f"/problems/{key}").text
        # `[^>]*` for the cell's other attributes: the headroom cell carries a
        # `data-sort` now that the workload table is sortable, and matching the
        # tag literally made this test pass by finding nothing.
        for cell in re.findall(r"<td class=\"r mono\"[^>]*>([\d.]+)×(.*?)</td>",
                               page, re.S):
            headroom, rest = float(cell[0]), cell[1]
            checked += 1
            if headroom >= 100 or headroom < 2:
                assert "bq bq-" in rest, (key, headroom, "unmarked")
    assert checked, "no headroom cells found on any problem page"


def test_the_marking_is_not_shown_for_the_ordinary_case(client):
    """`ok` renders nothing. A marker on every row is a marker on none."""
    page = client.get("/problems").text
    assert "bq-ok" not in page
