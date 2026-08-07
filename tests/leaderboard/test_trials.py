#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trials — the same setup, run again under a different constraint.

A trial is a whole run, so the group is a set of `submission` rows. The two
things that can go wrong silently are the numbering (a rebuild that renumbers
trials invalidates every URL and every screenshot that named one) and the
switcher's own arithmetic, which sits six lines from the run card and must not
print a different number under the same label.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

PROBLEM = "L1__001_alpha"
UNTOUCHED = "L2__002_beta"
GROUP = ["agent-trial-a", "agent-trial-b"]


def test_a_group_of_one_gets_no_switcher(client):
    """An ungrouped run is not a trial, and a switcher with one entry invites
    the reader to look for the others."""
    d = client.get(f"/api/v1/submissions/agent-alpha/problems/{PROBLEM}").json()
    assert d["trials"] == []
    assert "you are here" not in client.get(
        f"/submissions/agent-alpha/problems/{PROBLEM}").text


def test_trials_group_and_are_numbered_by_creation(client):
    d = client.get(f"/api/v1/submissions/{GROUP[0]}/problems/{PROBLEM}").json()
    assert [t["slug"] for t in d["trials"]] == GROUP
    assert [t["trial_n"] for t in d["trials"]] == [1, 2]
    assert [t["trial_label"] for t in d["trials"]] == ["$8 / problem",
                                                       "$100 / problem"]
    assert [t["is_current"] for t in d["trials"]] == [True, False]
    # The switcher navigates. Two runs under two budgets are two measurements,
    # so the links go to the other run's page rather than merging them.
    assert d["trials"][1]["url"] == f"/submissions/{GROUP[1]}/problems/{PROBLEM}"


def test_trial_n_is_stable_across_a_rebuild_and_when_a_trial_is_added(board):
    """`assign_trial_numbers` is the ingest's, run against the fixture board.

    Stability matters because `trial_n` is printed ("trial 2") and a rebuild
    that renumbers makes every earlier reference wrong. Ordering is by
    `created_utc` with `id` breaking ties, so a *later* trial appended to the
    group must not move the ones before it.
    """
    import ingest
    conn = sqlite3.connect(board.path("MI350X"))
    try:
        def numbers():
            return {r[0]: r[1] for r in conn.execute(
                "SELECT slug, trial_n FROM submission WHERE group_slug IS NOT NULL")}

        ingest.assign_trial_numbers(conn)
        first = numbers()
        assert first == {GROUP[0]: 1, GROUP[1]: 2}
        ingest.assign_trial_numbers(conn)
        assert numbers() == first, "re-running the ingest renumbered the trials"

        conn.execute(
            """INSERT INTO submission (slug, name, kind, created_utc, board_visible,
                                       group_slug, group_name, trial_label, part)
               VALUES ('agent-trial-c','Agent Alpha (later)','agent',
                       '2026-09-01T00:00:00+00:00',1,'agent-trials',
                       'Fixture-Agent','$1000 / problem','MI350X')""")
        ingest.assign_trial_numbers(conn)
        assert numbers() == {**first, "agent-trial-c": 3}

        # And one inserted BEFORE both: it takes 1 and the others shift, which
        # is the ordering rule doing its job rather than an id accident.
        conn.execute(
            """INSERT INTO submission (slug, name, kind, created_utc, board_visible,
                                       group_slug, group_name, trial_label, part)
               VALUES ('agent-trial-0','Agent Alpha (first)','agent',
                       '2025-01-01T00:00:00+00:00',1,'agent-trials',
                       'Fixture-Agent','$1 / problem','MI350X')""")
        ingest.assign_trial_numbers(conn)
        assert numbers()["agent-trial-0"] == 1
        assert numbers()[GROUP[0]] == 2
    finally:
        conn.close()


def test_the_switcher_mean_and_the_run_card_cannot_disagree(client):
    """The bug this replaces: the switcher used `AVG(score)`, SQL skips NULLs,
    and a bound-invalid workload stores NULL -- so a trial read 0.9899 in the
    switcher against 0.3387 in the card six lines below it.

    Checked for every trial from every trial's page, not just the pair that
    broke, and the fixture guarantees the NULL is actually in play (asserted
    below, or this passes for the wrong reason).
    """
    for slug in GROUP:
        page = client.get(f"/api/v1/submissions/{slug}/problems/{PROBLEM}").json()
        for t in page["trials"]:
            own = client.get(
                f"/api/v1/submissions/{t['slug']}/problems/{PROBLEM}").json()
            assert t["mean_score"] == own["summary"]["mean_attempted"], (
                f"{t['slug']} reads {t['mean_score']} in {slug}'s switcher and "
                f"{own['summary']['mean_attempted']} on its own card")
            assert t["attempted"] == own["summary"]["attempted"]
            assert t["passed"] == own["summary"]["passed"]


def test_a_passed_workload_with_no_score_is_in_play(client):
    """Guards the test above: if the fixture ever loses its bound-invalid row,
    `AVG(score)` and `SUM(score)/attempted` agree and that test proves nothing."""
    d = client.get(f"/api/v1/submissions/{GROUP[0]}/problems/{PROBLEM}").json()
    assert any(w["passed"] and w["score"] is None for w in d["workloads"])
    assert any(w["bound_invalid"] for w in d["workloads"])


def test_a_trial_that_did_not_cover_the_problem_is_shown_disabled(client):
    """Not hidden. The reader needs to know the trial exists and did not cover
    this problem; a missing entry reads as a trial that does not exist."""
    d = client.get(f"/api/v1/submissions/{GROUP[0]}/problems/{UNTOUCHED}").json()
    other = [t for t in d["trials"] if t["slug"] == GROUP[1]]
    assert other, "the other trial vanished from the switcher"
    assert other[0]["touched"] is False
    assert other[0]["mean_score"] is None, "an untouched trial scored nothing, " \
                                           "which is not the same as scoring 0"
    html = client.get(f"/submissions/{GROUP[0]}/problems/{UNTOUCHED}").text
    assert "not in this trial" in html


def test_touched_is_true_for_a_kernel_with_no_results(board, client):
    """D23 inside the switcher: a run whose re-time timed out has no result
    rows, and "not in this trial" would turn a failed measurement into a
    problem the trial never opened."""
    board.write("UPDATE submission SET group_slug='timeouts', "
                "group_name='Timeouts', trial_n=1 WHERE slug='agent-timeout'")
    board.write("UPDATE submission SET group_slug='timeouts', "
                "group_name='Timeouts', trial_n=2 WHERE slug='agent-alpha'")
    d = client.get(f"/api/v1/submissions/agent-alpha/problems/{UNTOUCHED}").json()
    by_slug = {t["slug"]: t for t in d["trials"]}
    assert by_slug["agent-timeout"]["attempted"] == 0
    assert by_slug["agent-timeout"]["touched"] is True, \
        "a kernel was submitted and could not be measured; that is not absence"
