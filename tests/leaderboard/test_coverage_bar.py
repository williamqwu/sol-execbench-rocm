#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The coverage bar must account for every problem, exactly once.

It replaced two columns that asked overlapping questions in incompatible units
-- workloads passed out of 3,717, and problems swept clean out of 220 -- with
one bar in problems over the whole benchmark. That only works if the four
states are disjoint and complete: a problem is either never attempted, or
attempted with nothing passing, or partly passing, or clean. If they are not,
the bar still renders, still looks like a bar, and silently misstates how much
of the benchmark a submission has seen.

Two failure modes worth naming, because both shipped in the version before it:

  * `attempted` computed from result rows alone. Failures used to be discarded
    at ingest, so a problem where a variant passed NOTHING produced no rows and
    read as never attempted -- which is how `torch.compile` came to claim it
    had seen 213 of 220 problems when it had run all 220.
  * segment widths rounded before summing. Four widths rounded to one decimal
    do not add to 100%, and the remainder shows as a hairline of track colour
    at the right end of a bar that should be full.

Not asserted: the colours, or that the bar is visible. Both are CSS and there
is no browser here. The contract checked is the arithmetic behind it.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

BAR = re.compile(r'<div class="covbar".*?</div>', re.S)
WIDTH = re.compile(r'width:([\d.]+)%')
SEG = re.compile(r'class="cs cs-(\w+)"')


def _rows(client) -> list[dict]:
    r = client.get("/api/v1/leaderboard")
    assert r.status_code == 200
    return r.json()


def test_the_four_states_partition_the_benchmark(client):
    for r in _rows(client):
        total = (r["problems_clean"] + r["problems_partial"]
                 + r["problems_failed"] + r["problems_untouched"])
        assert total == r["problems_total"], r["slug"]


def test_attempted_is_everything_that_is_not_untouched(client):
    """`problems_failed` is the bucket that used to vanish. A submission that
    ran a problem and passed nothing on it has been run on that problem."""
    for r in _rows(client):
        assert r["problems_attempted"] == (
            r["problems_total"] - r["problems_untouched"]), r["slug"]
        assert r["problems_attempted"] == (
            r["problems_clean"] + r["problems_partial"]
            + r["problems_failed"]), r["slug"]


def test_the_reference_variants_attempted_every_problem(client):
    """They were run on the whole benchmark, and the board said otherwise:
    torch.compile read as 213/220 because it passed nothing on 7 problems."""
    variants = [r for r in _rows(client) if r["kind"] == "reference_variant"]
    assert variants, "no reference variants on the board"
    for r in variants:
        assert r["problems_untouched"] == 0, (r["slug"], r["problems_untouched"])


def test_problems_complete_is_the_clean_bucket(client):
    for r in _rows(client):
        assert r["problems_complete"] == r["problems_clean"], r["slug"]


def test_every_bar_sums_to_exactly_one_hundred_percent(client):
    """Unrounded, so they sum without a remainder. A zero-count state emits no
    segment at all rather than a 0%-wide one."""
    page = client.get("/").text
    bars = BAR.findall(page)
    assert bars, "no coverage bars rendered"
    for i, bar in enumerate(bars, 1):
        widths = [float(w) for w in WIDTH.findall(bar)]
        assert widths, f"bar {i} has no segments"
        assert sum(widths) == pytest.approx(100.0, abs=1e-9), (i, widths)
        assert 0.0 not in widths, f"bar {i} emitted a zero-width segment"


def test_the_bar_does_not_follow_the_score_switch(client):
    """Coverage is a fact about the submission; only the score's denominator
    changes. A second set of segments keyed to the scope would say otherwise."""
    page = client.get("/").text
    for bar in BAR.findall(page):
        assert "sc-attempted" not in bar and "sc-full" not in bar
        assert "data-s-attempted" not in bar


def test_the_colours_are_explained_on_the_page(client):
    """Four colours with no key is the defect that got `flagged` deleted."""
    page = client.get("/").text
    assert 'class="covkey"' in page
    for cls in ("k-clean", "k-partial", "k-failed", "k-none"):
        assert cls in page, cls


def test_the_flagged_column_is_gone_from_the_board(client):
    """It was always a dash, carried no explanation, and the reward-hack count
    it reported is still on `/api/v1` and the per-problem pages."""
    page = client.get("/")
    head = page.text.split("</thead>", 1)[0]
    assert "flagged" not in head.lower()
    assert "n_flagged" in _rows(client)[0]


def test_the_section_nav_is_not_a_row_of_links(client):
    """A bare `nav{display:flex}` for the header also matched the section nav
    and laid seven block links out in a row inside a 210px column. The rule is
    scoped now; this is the assertion that keeps it scoped."""
    css = client.get("/static/style.css").text
    assert not re.search(r'(?m)^nav\s*\{', css), \
        "unscoped `nav {` selector is back; it will match .sidenav nav too"
    assert re.search(r'(?m)^header nav\s*\{', css)
