#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""`board_visible = 0`: off the ranking, still in the database.

Excluding a run from the ranking and deleting its evidence are different
decisions, and only the first one was made. Every test here holds one half of
that apart from the other. They run against the fixture board, where the hidden
run is deliberately the *best* run on its problem -- if the flag is read in the
wrong place, a headline moves and the assertion fires. On the real board today
the hidden run happens to be second on most problems, so the same bug would
show up on six rows out of 235 and only if you knew which six.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

HIDDEN = "agent-trial-a"
PROBLEM = "L1__001_alpha"


def test_the_board_is_unchanged_by_a_hidden_run(client, board):
    """DESIGN-v2 §1's verification requirement, as a comparison rather than a
    remembered number: the board with the hidden run ingested must equal the
    board with it deleted outright -- same rows, same ranks, same aggregates.
    """
    before = client.get("/api/v1/leaderboard").json()
    board.write("DELETE FROM result WHERE submission_id="
                "(SELECT id FROM submission WHERE slug=?)", (HIDDEN,))
    board.write("DELETE FROM submission WHERE slug=?", (HIDDEN,))
    after = client.get("/api/v1/leaderboard").json()
    assert before == after
    assert [r["slug"] for r in before] and HIDDEN not in [r["slug"] for r in before]


def test_a_hidden_run_is_reachable_by_url_and_says_why(client):
    d = client.get(f"/api/v1/submissions/{HIDDEN}").json()
    assert d["submission"]["board_visible"] == 0
    assert d["submission"]["exclusion_reason"]
    # Its evidence is all still there: results, kernel, trajectory, cost.
    assert d["problems"], "a hidden run with no per-problem rows is a deleted run"
    page = client.get(f"/submissions/{HIDDEN}")
    assert page.status_code == 200
    assert d["submission"]["exclusion_reason"][:40] in page.text


def test_a_hidden_run_sets_no_headline_but_is_listed_with_its_flag(client, board):
    """The hidden run holds the single best workload score on this problem
    (0.98) and must not be the number at the top of the page; it must still
    appear in the per-workload evidence, marked."""
    top = board.conn().execute(
        "SELECT MAX(score) FROM result r JOIN submission s ON s.id=r.submission_id "
        "WHERE r.problem_key=? AND s.slug=?", (PROBLEM, HIDDEN)).fetchone()[0]
    d = client.get(f"/api/v1/problems/{PROBLEM}").json()
    assert d["problem"]["best_score"] < top

    listed = {s["slug"]: s for s in d["submissions"]}
    assert HIDDEN in listed
    assert listed[HIDDEN]["board_visible"] == 0
    assert listed[HIDDEN]["exclusion_reason"]

    evidence = [r for w in d["workloads"] for r in w["results"]
                if r["slug"] == HIDDEN]
    assert any(r["score"] == top for r in evidence), \
        "the off-board score is not in the evidence: that is a deletion"


def test_the_headline_and_the_count_both_ignore_hidden_runs(client, board):
    """`best_score` and `n_submissions` on the problem list, which is what a
    reader sorts by. Compared against the same board with the run deleted --
    excluded and absent have to produce identical headlines."""
    before = {p["key"]: (p["best_score"], p["n_submissions"])
              for p in client.get("/api/v1/problems").json()}
    board.write("DELETE FROM result WHERE submission_id="
                "(SELECT id FROM submission WHERE slug=?)", (HIDDEN,))
    board.write("DELETE FROM submission WHERE slug=?", (HIDDEN,))
    after = {p["key"]: (p["best_score"], p["n_submissions"])
             for p in client.get("/api/v1/problems").json()}
    assert before == after


def test_the_problem_page_marks_the_off_board_row(client):
    """Rendered, not just in JSON: a row that is in the table unmarked reads as
    a ranked result, which is the comparison the exclusion exists to prevent."""
    html = client.get(f"/problems/{PROBLEM}").text
    assert HIDDEN in html
    i = html.index(HIDDEN)
    assert "off the ranking" in html[i:i + 4000]


def test_a_hidden_run_still_carries_its_peers_on_the_run_page(client):
    """`run_detail`'s peer table is evidence, not ranking, so it keeps them."""
    d = client.get(f"/api/v1/submissions/ref-v1-eager/problems/{PROBLEM}").json()
    peers = {p["slug"]: p for p in d["peers"]}
    assert HIDDEN in peers and peers[HIDDEN]["board_visible"] == 0
