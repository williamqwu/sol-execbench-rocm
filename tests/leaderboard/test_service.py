#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests that need the REAL board — the one `ingest.py` last built.

    leaderboard/.venv/bin/python -m pytest tests/leaderboard -q

Run in the LEADERBOARD venv, not the measurement container. The pinned ROCm
image deliberately has no fastapi -- adding it would change the environment
every baseline in this repo was measured under -- so these skip there rather
than fail, and `env/solb pytest tests/` stays green.

Everything here is a claim about the ingested artifacts that a hand-built
fixture cannot make: that no failed workload in 14k rows carries a score, that
every deferred problem in the manifest explains itself, that four consecutive
real rebuilds never expose an empty board. The invariants of the *service* --
board_visible, trials, the part resolver, the grid states -- live in the other
modules in this directory and run against a small fixture database, because an
invariant that only holds on today's artifacts has not been tested, it has been
observed.

Both fixtures here resolve the same file and then pin it with `SOLBENCH_DB`.
They used to disagree: `conn` opened `leaderboard/solbench.db` while `client`
resolved per-part through `part_databases()`, and they agreed only because both
happened to have been built from the same artifacts by hand. The skip guard
named the legacy path too, so on a fresh build -- which writes
`db/solbench-MI350X.db` and nothing else -- all seventeen tests skipped green.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "leaderboard"


# --------------------------------------------------------------- invariants

def test_the_two_fixtures_read_the_same_database(real_client, real_db, real_conn):
    """The bug this file had: two fixtures, two files, and comparisons between
    them that meant nothing. Asserted rather than remembered."""
    assert Path(real_client.get("/healthz").json()["db"]) == Path(real_db)
    assert real_conn.execute(
        "SELECT value FROM meta WHERE key='part'").fetchone()[0] == \
        real_client.get("/healthz").json()["part"]


def test_no_score_on_a_workload_that_did_not_pass(real_conn):
    """D22. A failed correctness check still has a latency, and `sol_score`
    will turn it into a plausible number -- the speed of the wrong answer."""
    n = real_conn.execute(
        "SELECT COUNT(*) FROM result WHERE status != 'PASSED' AND score IS NOT NULL"
    ).fetchone()[0]
    assert n == 0, f"{n} non-passing results carry a score"


def test_benchmark_score_denominator_is_every_scoreable_workload(real_client,
                                                                 real_conn):
    total = real_conn.execute(
        "SELECT COUNT(*) FROM workload WHERE scoreable=1").fetchone()[0]
    for row in real_client.get("/api/v1/leaderboard").json():
        assert row["workloads_total"] == total
        # The headline can never be raised by attempting less.
        assert row["benchmark_score"] <= row["mean_score_attempted"] + 1e-9


def test_partial_is_true_exactly_when_something_was_not_attempted(real_client):
    for row in real_client.get("/api/v1/leaderboard").json():
        assert row["partial"] == (row["workloads_attempted"] < row["workloads_total"])
        assert (row["workloads_untested"]
                == row["workloads_total"] - row["workloads_attempted"])


def test_mean_attempted_counts_failures_as_zero(real_client):
    """Averaging over passes only rewards giving up: it is what put
    torch.compile above eager PyTorch."""
    rows = {r["slug"]: r for r in real_client.get("/api/v1/leaderboard").json()}
    c = rows.get("baseline-v2-compile")
    if c is None:
        pytest.skip("this board has no torch.compile variant")
    assert c["workloads_failed"] > 0
    assert c["mean_score_attempted"] < c["mean_score_passed"]


def test_deferred_problems_carry_a_reason(real_conn):
    """A bare '0 scoreable' is indistinguishable from a sweep that never ran."""
    bad = [r["key"] for r in real_conn.execute(
        "SELECT key, deferred_reason FROM problem WHERE deferred=1")
        if not r["deferred_reason"]]
    assert not bad, f"deferred with no reason: {bad}"


def test_every_deferred_problem_page_renders(real_client, real_conn):
    """All 15 of these 500'd once, on a None/float in the headroom guard --
    and they are exactly the pages a reader opens to find out why it says 0."""
    for r in real_conn.execute("SELECT key FROM problem WHERE deferred=1"):
        assert real_client.get(f"/problems/{r['key']}").status_code == 200


def test_submitted_kernel_with_no_measurement_is_still_visible(real_conn):
    """D23. A kernel whose re-time timed out produces no result rows, so the
    board renders it identically to a problem nobody opened."""
    rows = list(real_conn.execute(
        "SELECT problem_key, retime_error FROM run_kernel WHERE retime_ok = 0"))
    for r in rows:
        assert r["retime_error"], f"{r['problem_key']} failed with no reason recorded"


def test_a_kernel_with_no_measurement_renders_as_unmeasured(client, board):
    """D23, on the fixture, so it is guarded whatever is on the real board.

    The real-artifact version below is the better test and stays. But it is
    conditional on some run in `artifacts/10/` having a re-time that failed
    outright, and on 2026-08-09 the only such run -- `glm-run1` -- was withdrawn
    from the board. The assertion did not fail; it skipped, silently, and the
    D23 rendering path went unguarded on a board that had just been rebuilt.

    A regression test whose coverage depends on which runs are currently
    published is a regression test that disappears exactly when the board
    changes, which is when it is most needed.
    """
    html = client.get("/submissions/agent-timeout/problems/L2__002_beta").text
    block = re.search(r'<div class="wl-grid".*?</div>', html, re.S)
    assert block, "no workload grid rendered"
    cells = re.findall(r'class="cell (g-[a-z0-9]+)"', block.group(0))
    assert cells and set(cells) == {"g-unmeasured"}, sorted(set(cells))


def test_the_real_unmeasured_run_renders_as_unmeasured(real_client, real_conn):
    """The concrete D23 case, on the artifacts rather than on a fixture:
    glm-run1's re-time of FlashInfer-Bench__014 hit TimeoutExpired after 1200 s,
    so every one of its workloads has a kernel and no measurement. Not one of
    them may draw as "not attempted"."""
    row = real_conn.execute(
        """SELECT s.slug, k.problem_key FROM run_kernel k
             JOIN submission s ON s.id = k.submission_id
            WHERE k.retime_ok = 0
              AND NOT EXISTS (SELECT 1 FROM result r
                               WHERE r.submission_id = k.submission_id
                                 AND r.problem_key = k.problem_key)
            LIMIT 1""").fetchone()
    if row is None:
        pytest.skip("no kernel in this board whose re-time failed outright")
    html = real_client.get(
        f"/submissions/{row['slug']}/problems/{row['problem_key']}").text
    block = re.search(r'<div class="wl-grid".*?</div>', html, re.S)
    assert block, f"no grid on {row['slug']} / {row['problem_key']}"
    cells = re.findall(r'class="cell (g-[a-z0-9]+)"', block.group(0))
    assert cells and set(cells) == {"g-unmeasured"}, sorted(set(cells))


# ------------------------------------------------------------ part hygiene

def test_the_real_board_holds_one_parts_measurements(real_client, real_conn):
    """`ingest.py` refuses to write a run measured on another part. This is the
    same check from the reader's side: if it is ever bypassed, `/healthz` stops
    being ok and the ranking silently loses a row."""
    part = real_conn.execute(
        "SELECT value FROM meta WHERE key='part'").fetchone()[0]
    foreign = [dict(r) for r in real_conn.execute(
        "SELECT slug, part FROM submission WHERE part IS NOT NULL AND part <> ?",
        (part,))]
    assert not foreign, f"measured elsewhere but stored here: {foreign}"
    h = real_client.get("/healthz").json()
    assert h["ok"] is True and h["part_mismatch"] == []


def test_every_run_window_names_one_of_the_three_sources(real_conn):
    """Never invented: three sources exist, they are three different
    quantities, and the UI prints which."""
    got = {r[0] for r in real_conn.execute("SELECT DISTINCT source FROM run_window")}
    assert got <= {"first_last_eval", "session", "retime_only"}


def test_no_hidden_run_is_ranked_and_each_says_why(real_client, real_conn):
    hidden = {r["slug"]: r["exclusion_reason"] for r in real_conn.execute(
        "SELECT slug, exclusion_reason FROM submission WHERE board_visible=0")}
    if not hidden:
        pytest.skip("nothing is excluded from this board")
    ranked = {r["slug"] for r in real_client.get("/api/v1/leaderboard").json()}
    assert not (ranked & set(hidden))
    for slug, reason in hidden.items():
        assert reason, f"{slug} is off the board with no reason recorded"
        assert real_client.get(f"/submissions/{slug}").status_code == 200


# ------------------------------------------------------------------- schema

def test_v1_responses_validate_on_the_real_data(real_client, real_conn):
    """A 500 here is the response model rejecting the handler's own output.
    Every submission and every problem it touched, not one sampled pair: the
    field that broke last time was NULL on 661 endpoints and fine on the rest."""
    key = real_conn.execute("SELECT key FROM problem LIMIT 1").fetchone()["key"]
    for url in ("/api/v1/stats", "/api/v1/leaderboard", "/api/v1/problems",
                f"/api/v1/problems/{key}", "/healthz"):
        assert real_client.get(url).status_code == 200, url
    for s in real_conn.execute("SELECT id, slug FROM submission"):
        assert real_client.get(
            f"/api/v1/submissions/{s['slug']}").status_code == 200
        for r in real_conn.execute(
                """SELECT DISTINCT problem_key FROM result WHERE submission_id=?
                   UNION SELECT problem_key FROM run_kernel WHERE submission_id=?""",
                (s["id"], s["id"])):
            url = f"/api/v1/submissions/{s['slug']}/problems/{r['problem_key']}"
            assert real_client.get(url).status_code == 200, url


# ------------------------------------------------------------- the rebuild

def test_rebuild_never_serves_an_empty_board(real_db, tmp_path):
    """The rebuild used to delete the live database and build in place, so a
    reader mid-rebuild got 200 with zero submissions -- not an error, and
    indistinguishable from 'nobody has submitted'.

    Runs the real `ingest.py` four times against a scratch `--db`, so it never
    touches the served board.
    """
    target = tmp_path / "board.db"
    roots = json.loads(
        sqlite3.connect(f"file:{real_db}?mode=ro", uri=True)
        .execute("SELECT value FROM meta WHERE key='input_extra_roots'")
        .fetchone()[0] or "[]")
    cmd = [str(LB / ".venv" / "bin" / "python"), str(LB / "ingest.py"),
           "--db", str(target)]
    if roots:
        cmd += ["--agent-runs", *roots]
    subprocess.run(cmd, check=True, capture_output=True, cwd=str(ROOT))
    expected = sqlite3.connect(target).execute(
        "SELECT COUNT(*) FROM submission").fetchone()[0]
    assert expected > 0

    seen, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            try:
                c = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
                seen.append(c.execute("SELECT COUNT(*) FROM submission").fetchone()[0])
                c.close()
            except sqlite3.Error:
                seen.append(-1)      # a reader that errors is also a failure

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    for _ in range(3):
        subprocess.run(cmd, check=True, capture_output=True, cwd=str(ROOT))
    time.sleep(0.05)
    stop.set()
    t.join(timeout=5)
    assert seen, "poller never ran"
    assert set(seen) == {expected}, (
        f"a reader saw {sorted(set(seen))} during the rebuild; expected only "
        f"{expected}")


def test_flagged_is_counted_over_every_result_not_just_the_scored_ones(tmp_path):
    """A flagged workload must reach the board row.

    It did not, from the day the board was built. `n_flagged` was computed
    inside the aggregate whose WHERE is `status='PASSED' AND score IS NOT NULL`
    -- and a flagged workload has status REWARD_HACK and a NULL score, so each
    of those clauses on its own excluded it. The column could not return
    anything but zero, /methodology asserted that zero in prose, and on
    2026-08-10 the harness caught 48 real ones while the board still read 0.

    A negative result guaranteed by construction is not evidence of anything,
    which is why this test builds a board with one flagged row rather than
    asserting against whatever the current artifacts happen to contain.
    """
    import sqlite3
    import app as app_mod

    db = tmp_path / "solbench-TEST.db"
    conn = sqlite3.connect(db)
    conn.executescript((ROOT / "leaderboard" / "schema.sql").read_text())
    conn.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     [("part", "TEST"), ("manifest_version", "vtest")])
    conn.execute("INSERT INTO problem(key,category,name,n_workloads,n_scoreable,deferred) "
                 "VALUES ('L1__x','L1','x',2,2,0)")
    conn.executemany(
        "INSERT INTO workload(problem_key,uuid,scoreable,t_sol_ms,t_b_ms) VALUES (?,?,?,?,?)",
        [("L1__x", "u1", 1, 0.1, 1.0), ("L1__x", "u2", 1, 0.1, 1.0)])
    conn.execute("INSERT INTO submission(id,slug,name,kind,board_visible,part) "
                 "VALUES (1,'probe','probe','agent',1,'TEST')")
    conn.executemany(
        "INSERT INTO result(submission_id,problem_key,workload_uuid,status,"
        "latency_ms,score,flagged,note) VALUES (?,?,?,?,?,?,?,?)",
        [(1, "L1__x", "u1", "PASSED", 0.5, 0.6, 0, ""),
         # The shape that was invisible: not PASSED, no score.
         (1, "L1__x", "u2", "REWARD_HACK", None, None, 1, "")])
    conn.commit()
    conn.close()

    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        rows = app_mod.leaderboard_rows(c)
    assert rows, "no board row built"
    assert rows[0]["n_flagged"] == 1, (
        f"a flagged workload did not reach the board row: {dict(rows[0])}")
