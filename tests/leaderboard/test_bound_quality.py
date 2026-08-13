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
        # Matched on `data-sort`, not on the rendered text. The headroom column
        # goes through the `ratio` helper now, which renders three significant
        # figures and a k/M suffix -- `115005.0` prints as `115k×`, and a test
        # that read the band off the printed string would compare 115 against a
        # 100x threshold and pass for the wrong reason. `data-sort` is the raw
        # float, which is what `bound_quality` banded on.
        for m in re.finditer(r'<td class="r mono" data-sort="([^"]*)">(.*?)</td>',
                             page, re.S):
            raw, rest = m.group(1), m.group(2)
            # ANY of the three ×-units, not just the bare one. `ratio` promotes
            # to `k×` at 1e3 and `M×` at 1e6, so matching only `×` excluded
            # every cell at or above 1000x headroom -- which is to say the
            # whole `vacuous` band, the band this file exists for (D39). The
            # skip was silent: `assert checked` stayed non-zero on the ×
            # cells, so the test kept passing with its reach cut off.
            if not re.search(r'<span class="qu">[kM]?×</span>', rest):
                continue        # a duration cell, not the headroom cell
            headroom = float(raw)
            checked += 1
            if headroom >= 100 or headroom < 2:
                assert "bq bq-" in rest, (key, headroom, "unmarked")
    assert checked, "no headroom cells found on any problem page"


def test_the_vacuous_band_is_marked_on_the_real_board(real_client, real_conn):
    """The fixture cannot reach this band; only the real board can.

    Every fixture headroom cell is 1.50-2.00x and renders with the bare `×`
    unit, so a cell-matcher that saw only `×` looked healthy while being blind
    to everything `ratio` promotes to `k×` or `M×` -- which is exactly the
    vacuous band (>= 1000x) this file was written for. This test picks the
    problem holding the largest `bound_headroom` on the board and requires the
    promoted cells to be found AND marked, so the blindness cannot come back.
    """
    key, worst = real_conn.execute(
        "SELECT problem_key, MAX(bound_headroom) FROM workload "
        "WHERE bound_headroom IS NOT NULL "
        "GROUP BY problem_key ORDER BY 2 DESC LIMIT 1").fetchone()
    assert worst >= 1000, (
        "no workload on this board is in the vacuous band; if D39 was fixed, "
        "retire this test deliberately rather than letting it go vacuous")
    page = real_client.get(f"/problems/{key}").text
    promoted = 0
    for m in re.finditer(r'<td class="r mono" data-sort="([^"]*)">(.*?)</td>',
                         page, re.S):
        raw, rest = m.group(1), m.group(2)
        if not re.search(r'<span class="qu">[kM]×</span>', rest):
            continue
        promoted += 1
        assert float(raw) >= 1000
        assert "bq bq-vacuous" in rest, (key, raw, "promoted but unmarked")
    assert promoted, (
        f"{key} holds a {worst:.0f}x bound and no k×/M× cell was seen; the "
        "matcher has lost the band again")


def test_the_marking_is_not_shown_for_the_ordinary_case(client):
    """`ok` renders nothing. A marker on every row is a marker on none."""
    page = client.get("/problems").text
    assert "bq-ok" not in page
