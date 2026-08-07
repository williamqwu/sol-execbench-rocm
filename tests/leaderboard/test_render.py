#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What the rendered page says, where saying it wrongly is silent.

Not "does the page render" -- the URL sweep does that better and over all of
them. These are the places where the HTML makes a *claim*: a grid cell whose
class says which of five states a workload is in, a timestamp whose `datetime`
attribute is what the browser reparses, and a trajectory point whose position
asserts when an eval happened.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "leaderboard" / "static" / "style.css"

GRID_STATES = ["g-fail", "g-miss", "g-unmeasured", "g-nosc", "g-inv"]


def grid(html: str) -> list[str]:
    """The grid's own cells, by state class — not the legend's swatches."""
    block = re.search(r'<div class="wl-grid".*?</div>', html, re.S)
    assert block, "no workload grid on the page"
    return [m for m in re.findall(r'class="cell (g-[a-z0-9]+)"', block.group(0))]


# --------------------------------------------------------------- the grid

def test_unmeasured_is_not_rendered_as_never_attempted(client):
    """D23. A kernel was submitted, the authoritative re-time timed out, and so
    there are no result rows at all -- which the grid drew as "not attempted",
    thirty cells directly above a banner saying a kernel had been submitted and
    its re-time hit TimeoutExpired. The page contradicted itself."""
    html = client.get("/submissions/agent-timeout/problems/L2__002_beta").text
    cells = grid(html)
    assert cells == ["g-unmeasured", "g-unmeasured"]
    assert "g-miss" not in cells
    # On the cells, not merely in the legend, which lists all five states on
    # every page and would make this pass anywhere.
    assert html.count('data-state="submitted, never measured"') == 2
    assert 'data-state="not attempted"' not in html
    assert "TimeoutExpired" in html, "the reason the measurement is missing"


def test_a_null_retime_is_not_a_failed_retime(client):
    """The third state: the ingest recorded nothing about the re-time. `not
    retime_ok` would fold it into the failure and claim a kernel was measured
    and could not be."""
    html = client.get("/submissions/agent-alpha/problems/L2__002_beta").text
    assert grid(html) == ["g-miss", "g-miss"]
    # The legend names all five states on every page, so the claim has to be
    # read off a cell, not off the document.
    assert 'data-state="submitted, never measured"' not in html
    assert html.count('data-state="not attempted"') == 2


def test_the_five_states_are_visually_distinct(client):
    """Five states, five classes, each with its own rule -- a failed workload
    and an unattempted one must not look alike at a glance."""
    css = CSS.read_text()
    for state in GRID_STATES:
        assert f".cell.{state}{{" in css.replace(" ", ""), \
            f"{state} has no style of its own"
    html = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    legend = re.search(r'<div class="grid-legend">.*?</div>', html, re.S).group(0)
    for state in GRID_STATES:
        assert state in legend, f"{state} is not in the legend"


def test_the_grid_carries_the_numbers_server_side(client):
    """Rendered server-side, so "view source" shows the data and no request is
    needed to see it."""
    html = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    block = re.search(r'<div class="wl-grid".*?</div>', html, re.S).group(0)
    assert block.count("<button") == 4
    assert 'data-s="0.9800"' in block
    assert 'data-tsol=' in block and 'data-tb=' in block and 'data-tk=' in block


def test_a_bound_invalid_cell_is_its_own_state(client):
    """T_k below T_SOL is impossible, so no score is stored. Rendered as a
    failure it would read as a broken kernel; rendered as a pass with a blank
    score it would read as a missing measurement."""
    html = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    assert "g-inv" in grid(html)
    assert "bound invalid" in html


# ----------------------------------------------------------- the timestamps

def test_every_localtime_carries_a_parseable_utc_datetime(client):
    """The browser reparses the `datetime` attribute and rewrites the text into
    the reader's timezone. A value it cannot parse leaves a UTC string in place
    (survivable); a value that parses as *local* silently shifts every
    timestamp on the page by the reader's offset."""
    seen = 0
    for url in ("/", "/submissions/agent-trial-a",
                "/submissions/agent-trial-a/problems/L1__001_alpha"):
        html = client.get(url).text
        for tag in re.findall(r"<time class=\"localtime\".*?</time>", html, re.S):
            iso = re.search(r'datetime="([^"]*)"', tag).group(1)
            assert iso, f"empty datetime on {url}: {tag[:80]}"
            dt = datetime.fromisoformat(iso)
            assert dt.tzinfo is not None, f"{iso} on {url} has no offset"
            assert dt.utcoffset() == timezone.utc.utcoffset(None), \
                f"{iso} on {url} is not UTC"
            # With JS off the page still has to say which zone it is in.
            assert "UTC" in re.sub(r"<[^>]+>", "", tag)
            seen += 1
    assert seen >= 4, f"only {seen} timestamps checked; the markup moved"


def test_an_untimed_eval_is_listed_but_not_plotted(client, app_mod):
    """`minutes_in or 0` put an eval nothing timed at the origin and dragged
    the polyline back through it. It is dropped from the plot and counted --
    and it stays in the table, because it happened and hiding it would make a
    run that broke look monotonic."""
    d = client.get(
        "/api/v1/submissions/agent-trial-a/problems/L1__001_alpha").json()
    traj = d["trajectory"]
    untimed = [e for e in traj if e["minutes_in"] is None]
    assert len(untimed) == 1 and untimed[0]["n"] == 4

    chart = app_mod.trajectory_chart(traj)
    assert chart["n_untimed"] == 1
    assert [p["n"] for p in chart["points"]] == [1, 2]
    assert [m["n"] for m in chart["marks"]] == [3]   # ran, scored nothing

    html = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    assert 'data-eval="4"' in html
    assert "not recorded" in html


def test_the_nullable_clock_labels_are_none_not_placeholders(client):
    """A caller substituting a dash is making a display choice; a caller
    printing "+0m" is making a claim."""
    traj = client.get(
        "/api/v1/submissions/agent-trial-a/problems/L1__001_alpha"
    ).json()["trajectory"]
    by_n = {e["n"]: e for e in traj}
    assert by_n[1]["at_label"] == "+0m"       # really is minute zero
    assert by_n[1]["utc_label"] == "10:00 UTC"
    assert by_n[4]["at_label"] is None
    assert by_n[4]["utc_label"] is None


def test_a_harness_error_is_not_a_regression(client):
    """Nothing ran, so nothing regressed -- and nothing improved either."""
    traj = client.get(
        "/api/v1/submissions/agent-trial-a/problems/L1__001_alpha"
    ).json()["trajectory"]
    by_n = {e["n"]: e for e in traj}
    assert by_n[3]["harness_error"] is True
    assert by_n[3]["regression"] is False
    assert by_n[3]["delta_vs_best"] is None
    assert by_n[4]["regression"] is True      # 4 passing -> 3


# ------------------------------------------------------------- the evidence

def test_a_deferred_problem_page_says_why(client):
    """A bare "0 scoreable" is indistinguishable from a sweep that never ran."""
    html = client.get("/problems/Quant__003_gamma").text
    assert "nvfp4-no-rocm-path" in html
    assert "quantize_nvfp4 is not implemented" in html
