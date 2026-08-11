#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""`?category=` is a filter over the board, and the board must say so.

Two independent filters decide what a score on the front page means: the
category chip picks WHICH workloads are summed, and the scope switch picks
WHAT the sum is divided by. `leaderboard_rows()` has always honoured the first
one — the failure was never in the arithmetic. It was that every label around
the table went on quoting the manifest's whole-benchmark figures, so
`/?category=L1` printed "divide by all 3,717 scoreable workloads" and
"coverage — all 220 problems" above a column divided by 1,480 over 94. The
numbers were right and described as something they were not, which is the one
class of defect nobody can catch by looking at the page.

The other half is what an *unknown* category used to do: match nothing, and
render a full leaderboard of 0.0000 scores with empty coverage bars. Every row
looked measured and every one of those zeros was an artefact of a filter that
selected no workloads. That is now a 400, like an unknown `?part=`.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

# The fixture benchmark: L1 holds one problem of four workloads, out of two
# scoreable problems and six workloads in total. Written out rather than read
# back from the app, so a change to either has to be a deliberate one here.
WHOLE = {"problems": 2, "workloads": 6}
L1 = {"problems": 1, "workloads": 4}


def test_the_scope_notes_quote_the_filtered_denominator(client):
    body = client.get("/?category=L1").text
    assert f"divide by all <b>{L1['workloads']}</b>" in body
    assert f"{L1['workloads']}-workload" in body
    # The whole-benchmark figure must not survive anywhere in the prose that
    # names a denominator -- that is the mislabelling itself.
    assert f"divide by all <b>{WHOLE['workloads']}</b>" not in body
    assert f"{WHOLE['workloads']}-workload" not in body


def test_the_coverage_header_counts_the_problems_the_bars_are_drawn_over(client):
    """The header and `s.problems_total` are the same number or the bar is
    lying about what its grey tail is a fraction of."""
    body = client.get("/?category=L1").text
    assert f"coverage &mdash; all {L1['problems']} L1 problems" in body
    totals = {int(n) for n in re.findall(r'of (\d+)</span>\s*</div>', body)}
    assert totals == {L1["problems"]}, totals


def test_the_unfiltered_board_still_quotes_the_whole_benchmark(client):
    body = client.get("/").text
    assert f"divide by all <b>{WHOLE['workloads']}</b>" in body
    assert f"coverage &mdash; all {WHOLE['problems']} problems" in body


def test_both_filters_are_in_one_row_and_marked_apart(client):
    """They compose -- neither is a step before the other -- and they are two
    different questions, so the row gives each its own name and colour."""
    body = client.get("/").text
    row = body.split('<div class="filterbar">', 1)[1].split("<p class=", 1)[0]
    assert 'class="fgroup fg-cat"' in row
    assert 'class="fgroup fg-scope"' in row
    # The literal the score-scope test pins, still inside the row.
    assert '<div class="scopebar" data-scope="attempted">' in row


def test_the_active_chip_says_so_to_a_screen_reader(client):
    """`class="on"` is a colour. Exactly one chip carries the state as well."""
    for url, want in (("/", "all"), ("/?category=L1", "L1")):
        chips = re.findall(r'<a class="chip[^>]*>\s*([^<\s][^<]*?)\s*(?:<|$)',
                           client.get(url).text)
        assert chips, url
        current = re.findall(r'<a class="chip[^>]*aria-current="true"[^>]*>\s*'
                             r'([^<\s][^<]*?)\s*(?:<|$)', client.get(url).text)
        assert current == [want], (url, current)


@pytest.mark.parametrize("url", [
    "/?category=nope",
    "/problems?category=nope",
    "/api/v1/leaderboard?category=nope",
    "/api/v1/problems?category=nope",
    "/api/leaderboard?category=nope",
    "/api/problems?category=nope",
])
def test_an_unknown_category_is_refused_not_answered_with_zeros(client, url):
    r = client.get(url)
    assert r.status_code == 400, f"{url} -> {r.status_code}"
    assert "nope" in r.text


def test_a_known_category_is_still_served_everywhere(client):
    for url in ("/?category=Quant", "/problems?category=Quant",
                "/api/v1/leaderboard?category=Quant",
                "/api/v1/problems?category=Quant"):
        assert client.get(url).status_code == 200, url
