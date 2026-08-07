#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""`run_window`: when a run worked on a problem, and on whose clock.

No artifact says "this session started at T". What exists is the harness-eval
series and the authoritative re-time's provenance stamp, and those are
different quantities -- so `source` is NOT NULL, the UI prints it, and a run
with no timestamp evidence gets no row at all. A window derived from file
mtimes or from a neighbouring run would be a measurement nobody made.
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

SOURCES = {"first_last_eval", "session", "retime_only"}


def test_no_timestamp_evidence_means_no_row(client, board):
    """`agent-trial-b` has no eval series and no re-time stamp. The honest
    answer is absence; anything else is invented."""
    conn = board.conn()
    n = conn.execute(
        """SELECT COUNT(*) FROM run_window w JOIN submission s ON s.id=w.submission_id
            WHERE s.slug='agent-trial-b'""").fetchone()[0]
    conn.close()
    assert n == 0

    d = client.get(
        "/api/v1/submissions/agent-trial-b/problems/L1__001_alpha").json()
    assert d["window"] is None
    html = client.get("/submissions/agent-trial-b/problems/L1__001_alpha").text
    assert "what these timestamps are" not in html


def test_the_source_enum_is_enforced_by_the_schema(board):
    """A fourth source invented at ingest time -- "mtime", "guessed" -- must
    not reach the database, because the UI prints the label and a reader would
    believe it."""
    conn = sqlite3.connect(board.path())
    try:
        sub_id = conn.execute(
            "SELECT id FROM submission WHERE slug='agent-trial-b'").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO run_window (submission_id, problem_key, started_utc,
                                           finished_utc, source)
                   VALUES (?,?,?,?,'mtime')""",
                (sub_id, "L1__001_alpha", "2026-01-01T00:00:00+00:00",
                 "2026-01-01T01:00:00+00:00"))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO run_window (submission_id, problem_key, source)
                   VALUES (?,?,NULL)""", (sub_id, "L2__002_beta"))
    finally:
        conn.rollback()
        conn.close()


def test_every_stored_source_is_one_of_the_three(board):
    conn = board.conn()
    try:
        got = {r[0] for r in conn.execute("SELECT DISTINCT source FROM run_window")}
    finally:
        conn.close()
    assert got and got <= SOURCES


def test_a_one_ended_window_has_no_elapsed(client):
    """`retime_only` is when the kernel was SCORED on GPU 0, not when it was
    worked on, and it is written at the end -- a finish with no start.
    Computing a duration from "now", or from the created_utc, would invent one."""
    d = client.get(
        "/api/v1/submissions/agent-alpha/problems/L1__001_alpha").json()
    w = d["window"]
    assert w["source"] == "retime_only"
    assert w["started_utc"] is None
    assert w["elapsed_seconds"] is None

    html = client.get("/submissions/agent-alpha/problems/L1__001_alpha").text
    assert "retime_only" in html
    assert "not recorded" in html, "a missing start must say so, not render blank"


def test_elapsed_is_the_difference_and_the_caveat_travels_with_it(client):
    d = client.get(
        "/api/v1/submissions/agent-trial-a/problems/L1__001_alpha").json()
    w = d["window"]
    assert w["source"] == "first_last_eval"
    assert w["elapsed_seconds"] == 5400.0
    html = client.get("/submissions/agent-trial-a/problems/L1__001_alpha").text
    assert "first_last_eval" in html
    # The window is INSIDE the session, and the page has to say so or the
    # elapsed figure reads as the session length.
    assert "lower bound" in html
