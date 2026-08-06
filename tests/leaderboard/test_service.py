#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the leaderboard service.

    leaderboard/.venv/bin/python -m pytest tests/leaderboard -q

Run in the LEADERBOARD venv, not the measurement container. The pinned ROCm
image deliberately has no fastapi -- adding it would change the environment
every baseline in this repo was measured under -- so these skip there rather
than fail, and `env/solb pytest tests/` stays green.

What is worth testing here is narrow and specific. Not "does the page render":
the URL sweep does that better and over all 3346 of them. These cover the
invariants that have actually broken, or that would break silently:

* the rebuild is atomic (it was not; a reader saw an empty board)
* a failed workload carries no score (it did; D22)
* untested is distinguishable from failed everywhere (the star, the summary)
* the write API fails CLOSED with no token file
* two workers cannot both take GPU 0
* the regression classifier does not invent a noise threshold
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="leaderboard venv only")

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "leaderboard"
sys.path.insert(0, str(LB))

DB = LB / "solbench.db"
pytestmark = pytest.mark.skipif(
    not DB.exists(), reason="board not built; run leaderboard/ingest.py")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app)


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


# --------------------------------------------------------------- invariants

def test_no_score_on_a_workload_that_did_not_pass(conn):
    """D22. A failed correctness check still has a latency, and `sol_score`
    will turn it into a plausible number -- the speed of the wrong answer."""
    n = conn.execute(
        "SELECT COUNT(*) FROM result WHERE status != 'PASSED' AND score IS NOT NULL"
    ).fetchone()[0]
    assert n == 0, f"{n} non-passing results carry a score"


def test_benchmark_score_denominator_is_every_scoreable_workload(client, conn):
    total = conn.execute(
        "SELECT COUNT(*) FROM workload WHERE scoreable=1").fetchone()[0]
    for row in client.get("/api/v1/leaderboard").json():
        assert row["workloads_total"] == total
        # The headline can never be raised by attempting less.
        assert row["benchmark_score"] <= row["mean_score_attempted"] + 1e-9


def test_partial_is_true_exactly_when_something_was_not_attempted(client):
    for row in client.get("/api/v1/leaderboard").json():
        assert row["partial"] == (row["workloads_attempted"] < row["workloads_total"])
        assert (row["workloads_untested"]
                == row["workloads_total"] - row["workloads_attempted"])


def test_mean_attempted_counts_failures_as_zero(client):
    """Averaging over passes only rewards giving up: it is what put
    torch.compile above eager PyTorch."""
    rows = {r["slug"]: r for r in client.get("/api/v1/leaderboard").json()}
    c = rows["baseline-v2-compile"]
    assert c["workloads_failed"] > 0
    assert c["mean_score_attempted"] < c["mean_score_passed"]


def test_deferred_problems_carry_a_reason(conn):
    """A bare '0 scoreable' is indistinguishable from a sweep that never ran."""
    bad = [r["key"] for r in conn.execute(
        "SELECT key, deferred_reason FROM problem WHERE deferred=1")
        if not r["deferred_reason"]]
    assert not bad, f"deferred with no reason: {bad}"


def test_every_deferred_problem_page_renders(client, conn):
    """All 15 of these 500'd once, on a None/float in the headroom guard --
    and they are exactly the pages a reader opens to find out why it says 0."""
    for r in conn.execute("SELECT key FROM problem WHERE deferred=1"):
        assert client.get(f"/problems/{r['key']}").status_code == 200


def test_submitted_kernel_with_no_measurement_is_still_visible(conn):
    """D23. A kernel whose re-time timed out produces no result rows, so the
    board renders it identically to a problem nobody opened."""
    rows = list(conn.execute(
        "SELECT problem_key, retime_error FROM run_kernel WHERE retime_ok = 0"))
    for r in rows:
        assert r["retime_error"], f"{r['problem_key']} failed with no reason recorded"


# ------------------------------------------------------------------- schema

def test_v1_responses_validate(client, conn):
    """A 500 here is the response model rejecting the handler's own output."""
    slug = conn.execute("SELECT slug FROM submission LIMIT 1").fetchone()["slug"]
    key = conn.execute("SELECT key FROM problem LIMIT 1").fetchone()["key"]
    for url in ("/api/v1/stats", "/api/v1/leaderboard", "/api/v1/problems",
                f"/api/v1/problems/{key}", f"/api/v1/submissions/{slug}",
                f"/api/v1/submissions/{slug}/problems/{key}", "/healthz"):
        assert client.get(url).status_code == 200, url


def test_every_v1_route_declares_a_response_schema(client):
    """The reason /api/v1 exists. Bare /api/* is exempt: it is the legacy alias."""
    spec = client.get("/openapi.json").json()
    missing = []
    for path, ops in spec["paths"].items():
        if not path.startswith("/api/v1") or path.endswith(("/transcript",)):
            continue
        for method, op in ops.items():
            # Not hardcoded 200: POST /submit answers 202, because nothing has
            # been measured when it returns. Look at whichever success code the
            # route actually declares.
            ok = [c for c in op["responses"] if c.isdigit() and c.startswith("2")]
            body = op["responses"].get(ok[0], {}).get("content", {}) if ok else {}
            schema = next(iter(body.values()), {}).get("schema", {})
            if not schema:
                missing.append(f"{method.upper()} {path}")
    assert not missing, f"no response schema: {missing}"


def test_unknown_slug_and_key_are_404_not_500(client):
    assert client.get("/api/v1/submissions/nope/problems/L1__001").status_code == 404
    assert client.get("/api/v1/submissions/baseline-v1-eager/problems/nope").status_code == 404


# ------------------------------------------------------------- write path

def test_write_api_fails_closed_without_a_token_file(client, monkeypatch, tmp_path):
    """A service that accepts anonymous submissions because its config is
    missing is worse than one that refuses everything."""
    import submit
    monkeypatch.setattr(submit, "TOKENS", tmp_path / "absent")
    r = client.post("/api/v1/submit", json={
        "slug": "x", "problem_key": "L1__085_geglu_activation", "kernel": "x"})
    assert r.status_code == 503


def test_submit_rejects_a_deferred_problem(client, monkeypatch, tmp_path, conn):
    """It cannot be scored, so it must not reach the queue and burn GPU time."""
    import submit
    tokens = tmp_path / "tokens"
    tokens.write_text("t0k:test\n")
    monkeypatch.setattr(submit, "TOKENS", tokens)
    monkeypatch.setattr(submit, "QUEUE_DB", tmp_path / "q.db")
    row = conn.execute("SELECT key FROM problem WHERE deferred=1 LIMIT 1").fetchone()
    if row is None:
        pytest.skip("no deferred problems in this manifest")
    r = client.post("/api/v1/submit",
                    headers={"Authorization": "Bearer t0k"},
                    json={"slug": "tt", "problem_key": row["key"], "kernel": "def run(): ..."})
    assert r.status_code == 409


def test_submit_rejects_bad_token_and_bad_slug(client, monkeypatch, tmp_path):
    import submit
    tokens = tmp_path / "tokens"
    tokens.write_text("good:test\n")
    monkeypatch.setattr(submit, "TOKENS", tokens)
    monkeypatch.setattr(submit, "QUEUE_DB", tmp_path / "q.db")
    body = {"slug": "ok", "problem_key": "L1__085_geglu_activation",
            "kernel": "def run(): ..."}
    assert client.post("/api/v1/submit", json=body).status_code == 401
    assert client.post("/api/v1/submit", json=body,
                       headers={"Authorization": "Bearer wrong"}).status_code == 403
    assert client.post("/api/v1/submit", json={**body, "slug": "Not A Slug"},
                       headers={"Authorization": "Bearer good"}).status_code == 422


def test_only_one_worker_can_hold_gpu0(tmp_path, monkeypatch):
    """Two jobs on GPU 0 invalidates both timings and every number they are
    compared against."""
    import worker
    monkeypatch.setattr(worker, "LOCK", tmp_path / "lock")
    fd = worker.acquire_lock()
    try:
        with pytest.raises(SystemExit):
            worker.acquire_lock()
    finally:
        os.close(fd)


def test_claim_is_atomic(tmp_path, monkeypatch):
    """Two workers racing the same row: exactly one wins."""
    import submit as submit_mod
    import worker
    monkeypatch.setattr(submit_mod, "QUEUE_DB", tmp_path / "q.db")
    with submit_mod.queue_db() as c:
        c.execute("""INSERT INTO job (token_name,slug,problem_key,kernel_sha256,
                                      kernel_bytes,submitted_utc)
                     VALUES ('t','s','p','sha',1,'now')""")
        c.commit()
    claims = []

    def go():
        with submit_mod.queue_db() as c:
            claims.append(worker.claim(c))

    ts = [threading.Thread(target=go) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(1 for c in claims if c is not None) == 1


# ------------------------------------------------------------- classifier

def test_regression_flag_is_correctness_only():
    """A dip in S is not a regression unless something measured says so, and
    nothing here measures the noise floor of an agent-side eval (D20)."""
    from app import _mark_regressions
    traj = [
        {"n": 1, "ok": 1, "workloads": 16, "passed": 16, "mean_score": 0.5665},
        {"n": 2, "ok": 1, "workloads": 16, "passed": 16, "mean_score": 0.5652},
        {"n": 3, "ok": 0, "workloads": 0, "passed": 0, "mean_score": None},
        {"n": 4, "ok": 1, "workloads": 16, "passed": 6, "mean_score": 0.4902},
    ]
    _mark_regressions(traj)
    assert traj[1]["regression"] is False           # 0.2% dip: not a claim
    assert traj[1]["delta_vs_best"] == pytest.approx(-0.0013)
    assert traj[2]["harness_error"] is True         # nothing ran
    assert traj[2]["regression"] is False           # so nothing regressed
    assert traj[3]["regression"] is True            # 16 -> 6 passing


# ------------------------------------------------------------- the rebuild

def test_rebuild_never_serves_an_empty_board(tmp_path):
    """The rebuild used to delete the live database and build in place, so a
    reader mid-rebuild got 200 with zero submissions -- not an error, and
    indistinguishable from 'nobody has submitted'."""
    import inputs
    target = tmp_path / "board.db"
    roots = json.loads(
        sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
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
