#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One score, two scopes, and neither may be computed twice.

The board prints a single score column whose denominator is chosen by a switch:

  attempted -- divided by the workloads the submission was actually run on
  full      -- divided by every scoreable workload in the benchmark

Both numbers, both ranks and both coverage figures are rendered by the server
into the same cell; the switch only changes which is displayed. That is the
whole reason this can be tested at all, and it is also the invariant most at
risk -- the obvious "simplification" is to render one number and have the
browser divide again for the other, at which point there are two
implementations of the score and they disagree in the fourth decimal.

The failure this guards against is not a wrong number. It is a *mislabelled*
one: a table whose header says one denominator while its cells hold the other,
or a rank column left over from the scope you switched away from. Nothing about
the page looks broken when that happens.

Not asserted: the switch itself. Clicking it is JavaScript and there is no
runtime on this node. What is asserted is that both scopes are present, correct
and correctly wired to the attributes the script reads -- so a script that runs
has something true to show, and a reader with no script gets the default scope
fully rendered rather than a blank column.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

ROW = re.compile(r'<tr class="kind-[^"]*"(.*?)</tr>', re.S)
SCORE_CELL = re.compile(
    r'<td class="r big" data-sort="([^"]*)"\s+'
    r'data-s-attempted="([^"]*)" data-s-full="([^"]*)"', re.S)


def _api(client) -> list[dict]:
    r = client.get("/api/v1/leaderboard")
    assert r.status_code == 200
    return r.json()


def test_the_two_scopes_are_different_questions(client):
    """If they were equal the switch would be decoration. They differ exactly
    when a submission did not attempt everything -- which is every row today."""
    for row in _api(client):
        if row["workloads_attempted"] < row["workloads_total"]:
            assert row["mean_score_attempted"] >= row["benchmark_score"], row["slug"]
        else:
            assert row["mean_score_attempted"] == pytest.approx(
                row["benchmark_score"]), row["slug"]


def test_each_scope_is_its_own_sum_over_its_own_denominator(client):
    """Recomputed from the parts, so a refactor that swaps a denominator is
    caught by arithmetic rather than by eye."""
    for row in _api(client):
        total = row["mean_score_attempted"] * row["workloads_attempted"]
        assert row["benchmark_score"] == pytest.approx(
            total / row["workloads_total"], abs=1e-9), row["slug"]


def test_both_ranks_are_dense_and_start_at_one(client):
    rows = _api(client)
    for key in ("rank", "rank_attempted"):
        got = sorted(r[key] for r in rows)
        assert got == list(range(1, len(rows) + 1)), key


def test_the_api_stays_in_full_benchmark_order(client):
    """The page's default changed; this list's did not. A consumer reading
    position N must not start getting a different row for that reason."""
    rows = _api(client)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_problems_column_counts_attempts_not_wins(client):
    """`problems_attempted` is what the column shows. It can only be >= the
    number swept clean, and a row with results has attempted at least one."""
    for row in _api(client):
        assert row["problems_attempted"] >= row["problems_complete"], row["slug"]
        assert row["problems_attempted"] <= row["problems_total"], row["slug"]
        if row["workloads_attempted"]:
            assert row["problems_attempted"] >= 1, row["slug"]


def test_the_page_carries_both_numbers_in_every_score_cell(client):
    """Rendered server-side, both of them. The browser divides nothing."""
    page = client.get("/").text
    api = {r["slug"]: r for r in _api(client)}
    cells = SCORE_CELL.findall(page)
    assert len(cells) == len(api), (len(cells), len(api))
    want_att = sorted(round(r["mean_score_attempted"], 9) for r in api.values())
    want_full = sorted(round(r["benchmark_score"], 9) for r in api.values())
    assert sorted(round(float(c[1]), 9) for c in cells) == want_att
    assert sorted(round(float(c[2]), 9) for c in cells) == want_full
    # `data-sort` must point at the scope the table is actually showing, or the
    # column sort silently orders by the hidden number.
    for sort_v, att, _full in cells:
        assert sort_v == att, (sort_v, att)


def test_the_default_scope_is_named_on_the_page_and_on_the_table(client):
    page = client.get("/")
    assert page.status_code == 200
    body = page.text
    assert '<table class="board sortable" data-scope="attempted">' in body
    assert '<div class="scopebar" data-scope="attempted">' in body
    # Both explanations ship; CSS shows one. With no CSS and no JS the reader
    # gets both, labelled, which is the right degradation for a denominator.
    assert 'data-scope-note="attempted"' in body
    assert 'data-scope-note="full"' in body


def test_the_partial_star_belongs_to_the_full_scope_only(client):
    """Under `attempted` nothing is missing from the denominator, so a star
    there would be marking a hazard that scope does not have."""
    body = client.get("/").text
    for chunk in ROW.findall(body):
        att = re.search(r'<span class="sc sc-attempted">(.*?)</span>', chunk, re.S)
        assert att, chunk[:200]
        assert "star" not in att.group(1)
