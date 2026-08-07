#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The write path: tokens, the queue, and the one GPU.

These were in `test_service.py`, behind a module-level skip on the existence of
`leaderboard/solbench.db`. None of them read the board -- the token check does
not, the GPU lock certainly does not -- so on a machine with no board built
they skipped green while testing nothing. They run against the fixture board
here, which also gives the deferred-problem check something real to refuse.
"""

from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")


def test_write_api_fails_closed_without_a_token_file(client, monkeypatch, tmp_path):
    """A service that accepts anonymous submissions because its config is
    missing is worse than one that refuses everything."""
    import submit
    monkeypatch.setattr(submit, "TOKENS", tmp_path / "absent")
    r = client.post("/api/v1/submit", json={
        "slug": "x", "problem_key": "L1__001_alpha", "kernel": "x"})
    assert r.status_code == 503


def test_submit_rejects_a_deferred_problem(client, monkeypatch, tmp_path):
    """It cannot be scored, so it must not reach the queue and burn GPU time."""
    import submit
    tokens = tmp_path / "tokens"
    tokens.write_text("t0k:test\n")
    monkeypatch.setattr(submit, "TOKENS", tokens)
    monkeypatch.setattr(submit, "QUEUE_DB", tmp_path / "q.db")
    r = client.post("/api/v1/submit",
                    headers={"Authorization": "Bearer t0k"},
                    json={"slug": "tt", "problem_key": "Quant__003_gamma",
                          "kernel": "def run(): ..."})
    assert r.status_code == 409


def test_submit_rejects_bad_token_and_bad_slug(client, monkeypatch, tmp_path):
    import submit
    tokens = tmp_path / "tokens"
    tokens.write_text("good:test\n")
    monkeypatch.setattr(submit, "TOKENS", tokens)
    monkeypatch.setattr(submit, "QUEUE_DB", tmp_path / "q.db")
    body = {"slug": "ok", "problem_key": "L1__001_alpha",
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


def test_regression_flag_is_correctness_only(app_mod):
    """A dip in S is not a regression unless something measured says so, and
    nothing here measures the noise floor of an agent-side eval (D20)."""
    traj = [
        {"n": 1, "ok": 1, "workloads": 16, "passed": 16, "mean_score": 0.5665},
        {"n": 2, "ok": 1, "workloads": 16, "passed": 16, "mean_score": 0.5652},
        {"n": 3, "ok": 0, "workloads": 0, "passed": 0, "mean_score": None},
        {"n": 4, "ok": 1, "workloads": 16, "passed": 6, "mean_score": 0.4902},
    ]
    app_mod._mark_regressions(traj)
    assert traj[1]["regression"] is False           # 0.2% dip: not a claim
    assert traj[1]["delta_vs_best"] == pytest.approx(-0.0013)
    assert traj[2]["harness_error"] is True         # nothing ran
    assert traj[2]["regression"] is False           # so nothing regressed
    assert traj[3]["regression"] is True            # 16 -> 6 passing
