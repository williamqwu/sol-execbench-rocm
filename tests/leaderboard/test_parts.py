#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The part switch: which dataset a request is about (DESIGN-v2 §6).

The part is not a filter over one dataset, it *selects* the dataset, so every
mistake in here has the same shape -- a reader is shown one part's numbers
under another part's heading, and nothing about the page says so. That is why
the resolver is tested step by step rather than end to end: each rung of
`query > cookie > SOLBENCH_PART > sole > MI350X` has to beat the one below it,
and a rung that silently stops working leaves the page rendering fine.

`/api/v1/parts` gets its own tests because it regressed on exactly this: it
ignored `?part=`, so it answered 200 where every other route answers 400 and
marked the *resolved* part active rather than the requested one -- the switch's
own data was the one thing on the site that disagreed with its URL.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


def active(resp) -> str:
    parts = resp.json()
    return next(p["name"] for p in parts if p["active"])


def test_query_wins(client):
    assert active(client.get("/api/v1/parts?part=MI355X")) == "MI355X"
    assert active(client.get("/api/v1/parts?part=MI350X")) == "MI350X"


def test_the_query_sets_a_cookie_and_the_choice_sticks(make_client, board):
    c = make_client()
    assert active(c.get("/api/v1/parts?part=MI355X")) == "MI355X"
    assert c.cookies.get("part") == "MI355X"
    # No query this time: the cookie carries it.
    assert active(c.get("/api/v1/parts")) == "MI355X"
    # ...and an explicit query still beats the cookie it set.
    assert active(c.get("/api/v1/parts?part=MI350X")) == "MI350X"


def test_cookie_beats_the_environment(make_client, board, monkeypatch):
    monkeypatch.setenv("SOLBENCH_PART", "MI355X")
    c = make_client(cookies={"part": "MI350X"})
    assert active(c.get("/api/v1/parts")) == "MI350X"


def test_environment_beats_the_sole_measured_part(make_client, board, monkeypatch):
    """The pin is deliberate operator configuration, so it outranks "there is
    only one database" -- which is a fact about the disk, not a choice."""
    monkeypatch.setenv("SOLBENCH_PART", "MI355X")
    assert active(make_client().get("/api/v1/parts")) == "MI355X"
    assert make_client().get("/api/v1/leaderboard").status_code == 503


def test_the_sole_measured_part_wins_over_the_default(make_client, board):
    """Built as MI355X only. Resolving to MI350X here would be the default
    masquerading as a resolution -- and would 503 on a board that has data."""
    board.add("MI355X")
    board.drop("MI350X")
    c = make_client()
    assert active(c.get("/api/v1/parts")) == "MI355X"
    assert c.get("/api/v1/leaderboard").status_code == 200


def test_the_default_applies_only_when_nothing_is_measured(make_client, board):
    board.drop("MI350X")
    c = make_client()
    assert active(c.get("/api/v1/parts")) == "MI350X"
    assert c.get("/api/v1/leaderboard").status_code == 503


def test_an_unknown_part_is_400_everywhere(client):
    """Silently serving MI350X to someone who asked for something else is how a
    reader ends up comparing two parts without knowing it. MI300X is the
    interesting case: it is a real part in the registry, and it is CDNA3, so
    this port cannot have measured it."""
    for url in ("/", "/problems", "/problems/L1__001_alpha", "/methodology",
                "/api/v1/parts", "/api/v1/leaderboard", "/api/v1/problems",
                "/api/v1/stats", "/api/v1/submissions/agent-alpha",
                "/api/v1/submissions/agent-alpha/problems/L1__001_alpha",
                "/healthz"):
        for bad in ("BOGUS", "MI300X", "mi350x"):
            r = client.get(url, params={"part": bad})
            assert r.status_code == 400, f"{url}?part={bad} -> {r.status_code}"


def test_a_stale_cookie_is_ignored_not_fatal(make_client, board):
    """A cookie outlives a rename, and one stale value must not brick every
    page for that browser -- unlike the query, which the reader just typed."""
    c = make_client(cookies={"part": "MI999X"})
    r = c.get("/api/v1/parts")
    assert r.status_code == 200
    assert active(r) == "MI350X"


def test_a_bad_environment_pin_fails_loudly(make_client, board, monkeypatch):
    """Server misconfiguration, not a reader's typo: nobody would see a warning."""
    monkeypatch.setenv("SOLBENCH_PART", "MI999X")
    assert make_client().get("/api/v1/parts").status_code == 500


def test_a_part_with_no_database_renders_the_empty_state(client):
    """Not a 404 (the part exists), not a 500 (nothing broke), not an empty
    table (which reads as "measured, scored zero")."""
    r = client.get("/?part=MI355X")
    assert r.status_code == 200
    assert "Nothing has been measured on MI355X" in r.text
    # The path, not just the basename: a bare-basename assertion survives
    # the file moving and leaves the board pointing at nothing.
    assert "docs/TODO-MI355X.md" in r.text
    for url in ("/problems", "/problems/L1__001_alpha", "/methodology",
                "/submissions/agent-alpha"):
        assert client.get(url, params={"part": "MI355X"}).status_code == 200, url


def test_the_runbook_pointer_is_checked_against_the_disk():
    """The page names a runbook only when one was written.

    The path used to be f"TODO-{part}.md" -- synthesised, never checked. Add a
    third part to PARTS and the empty state would send a reader to a file that
    does not exist, which reads as "someone planned this" rather than "nobody
    has". The MI355X case must keep resolving, or the moved file is unlinked.
    """
    from leaderboard.app import todo_runbook
    assert todo_runbook("MI355X") == "docs/TODO-MI355X.md"
    assert todo_runbook("MI999X") is None


def test_the_api_answers_503_for_a_part_with_no_database(client):
    """503, not 200-with-an-empty-list: a client that cannot tell "no data" from
    "no submissions scored" will publish the second."""
    for url in ("/api/v1/leaderboard", "/api/v1/problems", "/api/v1/stats",
                "/healthz"):
        r = client.get(url, params={"part": "MI355X"})
        assert r.status_code == 503, f"{url} -> {r.status_code}"
    # The switch itself keeps working on an unmeasured part, or there is no way
    # back to a measured one.
    r = client.get("/api/v1/parts?part=MI355X")
    assert r.status_code == 200
    assert [p["n_results"] for p in r.json() if p["name"] == "MI355X"] == [None], \
        "n_results must be None, never 0: not measured is not measured-nothing"


def test_the_empty_state_shows_no_other_parts_numbers(client):
    """The footer's F_LOCK/device/ROCm line is per-part and comes from that
    part's own `meta`. Showing MI350X's 1300 MHz under an MI355X heading is the
    exact failure this whole split exists to prevent."""
    html = client.get("/?part=MI355X").text
    for leaked in ("1300", "6.4.1", "AMD Instinct MI350X", "fixture-v1"):
        assert leaked not in html, f"{leaked!r} leaked onto the MI355X page"


def test_links_carry_the_active_part_forward(client, board):
    """A reader who switches to MI355X and clicks a problem must not land back
    on MI350X data. `/api/docs` is exempt by design -- Swagger renders no
    measurement and does not read the parameter."""
    board.add("MI355X")
    html = client.get("/?part=MI355X").text
    import re
    links = [m for m in re.findall(r'href="(/[^"]*)"', html)
             if not m.startswith("/static") and m != "/api/docs"]
    assert links
    bare = [m for m in links if "part=" not in m]
    assert not bare, f"links dropped the part: {bare}"


def test_switch_urls_preserve_the_path_and_query(client):
    r = client.get("/api/v1/parts?part=MI350X&category=L1")
    urls = {p["name"]: p["url"] for p in r.json()}
    assert urls["MI355X"].startswith("/api/v1/parts?")
    assert "category=L1" in urls["MI355X"]
    assert "part=MI355X" in urls["MI355X"]


def test_known_parts_is_the_ports_own_registry(app_mod):
    """Not a second hardcoded list. MI300X is in `PARTS` and is gfx942, so
    offering it would advertise a dataset that cannot exist without a
    different port."""
    assert app_mod.known_parts() == ["MI350X", "MI355X"]
    assert "MI300X" in app_mod.PARTS
