#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A run measured on one part must never rank on another part's board.

This is the most damaging error the project can make and the only one with no
symptom: MI355X has a different power cap, so a different F_LOCK, so a
different T_SOL and T_b. An MI355X latency scored against MI350X bounds
produces a plausible number on every row and a rank nobody can tell is wrong. A
reviewer demonstrated it by relabelling glm-run1 -- it sat at #5, unmarked.

Two independent guards, tested separately because either alone is one bug away
from silence: `ingest.py` refuses to write the row, and `app.py` refuses to
rank it and reports the disagreement on `/healthz`.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


# --------------------------------------------------------------- the board

def test_a_foreign_part_row_is_dropped_from_the_ranking(client, board):
    before = [r["slug"] for r in client.get("/api/v1/leaderboard").json()]
    assert "agent-alpha" in before
    board.write("UPDATE submission SET part='MI355X' WHERE slug='agent-alpha'")
    after = client.get("/api/v1/leaderboard").json()
    assert "agent-alpha" not in [r["slug"] for r in after]
    # Ranks renumber rather than leaving a hole at 2.
    assert [r["rank"] for r in after] == list(range(1, len(after) + 1))


def test_the_drop_is_reported_not_silent(client, board):
    """A row that vanishes from a ranking with no explanation is its own
    failure mode: the reader cannot tell a dropped run from one never ingested."""
    assert client.get("/healthz").json()["part_mismatch"] == []
    board.write("UPDATE submission SET part='MI355X' WHERE slug='agent-alpha'")
    h = client.get("/healthz").json()
    assert h["ok"] is False
    assert h["part_mismatch"] == [{"slug": "agent-alpha", "name": "Agent Alpha",
                                   "submission_part": "MI355X"}]
    assert "agent-alpha" in client.get("/").text   # the banner names it


def test_a_null_part_still_ranks(client, board):
    """NULL is not a disagreement -- it is a run whose artifacts never named a
    part. Treating it as one would empty the board of every submission ingested
    before the column existed."""
    board.write("UPDATE submission SET part=NULL WHERE slug='agent-alpha'")
    assert "agent-alpha" in [r["slug"]
                             for r in client.get("/api/v1/leaderboard").json()]
    assert client.get("/healthz").json()["part_mismatch"] == []


def test_a_foreign_part_run_is_still_readable(client, board):
    """Refusing to rank it is not deleting it; its own page still says which
    part it was measured on, which is how a reader finds out why it is gone."""
    board.write("UPDATE submission SET part='MI355X' WHERE slug='agent-alpha'")
    d = client.get(
        "/api/v1/submissions/agent-alpha/problems/L1__001_alpha").json()
    assert d["part"] == "MI355X"
    assert client.get("/submissions/agent-alpha").status_code == 200


def test_the_run_part_is_the_runs_own_not_the_databases(client, board):
    """`RunDetail.part` is a fact about the run. Substituting the database's
    would give every run a part whether or not anything says so."""
    board.write("UPDATE submission SET part=NULL WHERE slug='agent-alpha'")
    d = client.get(
        "/api/v1/submissions/agent-alpha/problems/L1__001_alpha").json()
    assert d["part"] is None
    assert client.get("/healthz").json()["part"] == "MI350X"


# -------------------------------------------------------------- the ingest

def write_run(tmp_path, run_id, retimed_devices, scored_device=None):
    """A minimal agent-run directory: `retimed/*.json` plus `scored.json`."""
    d = tmp_path / run_id
    (d / "retimed").mkdir(parents=True)
    for i, dev in enumerate(retimed_devices):
        (d / "retimed" / f"p{i}.json").write_text(json.dumps(
            {"_provenance": {"torch": {"available": True, "devices": [dev]}}}))
    (d / "scored.json").write_text(json.dumps(
        {"_provenance": {"torch": ({"available": True, "devices": [scored_device]}
                                   if scored_device else {"available": False})}}))
    return d


def test_run_part_reads_the_retime_and_nothing_else(tmp_path):
    """`scored.json` is written by the driver process, which is usually
    torchless and, when it is not, names the host it was *scored* on. Taking it
    as a stand-in is how a run acquires a part it was never measured on."""
    import ingest
    d = write_run(tmp_path, "r1", ["AMD Instinct MI355X"],
                  scored_device="AMD Instinct MI350X")
    assert ingest.run_part(d) == "MI355X"

    # No re-time provenance at all: "this run does not say", which is not
    # MI350X and must not become it.
    bare = write_run(tmp_path, "r2", [], scored_device="AMD Instinct MI350X")
    assert ingest.run_part(bare) is None


def test_a_run_retimed_on_two_parts_is_refused(tmp_path):
    import ingest
    d = write_run(tmp_path, "r3",
                  ["AMD Instinct MI350X", "AMD Instinct MI355X"])
    with pytest.raises(SystemExit) as e:
        ingest.run_part(d)
    assert "more than one part" in str(e.value)


def test_check_run_part_refuses_a_mismatch_and_an_unknown(tmp_path):
    """It fails hard rather than warning or skipping: a warning is scrolled
    past, and a skip removes the run from the board without removing it from
    the reader's expectations, which is D24 wearing a different hat."""
    import ingest
    d = tmp_path / "r4"
    d.mkdir()
    assert ingest.check_run_part(d, "r4", "MI350X", "MI350X") == "MI350X"

    with pytest.raises(SystemExit) as mismatch:
        ingest.check_run_part(d, "r4", "MI355X", "MI350X")
    text = str(mismatch.value)
    assert "MI355X" in text and "MI350X" in text and "r4" in text

    with pytest.raises(SystemExit) as unknown:
        ingest.check_run_part(d, "r4", None, "MI350X")
    assert "does not say which part" in str(unknown.value)


def test_every_submission_in_the_fixture_board_names_its_part(board):
    """The state the guards are meant to maintain, asserted directly, so a
    fixture that drifts cannot make the tests above vacuous."""
    conn = sqlite3.connect(board.path())
    try:
        parts = {r[0] for r in conn.execute("SELECT DISTINCT part FROM submission")}
    finally:
        conn.close()
    assert parts == {"MI350X"}
