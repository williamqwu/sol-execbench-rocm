#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""`/api/v1` is the contract: every route declares a schema and meets it.

Declaring is half of it. A `response_model` that the handler's own output does
not satisfy is a 500 at request time, not a startup error -- which is how a
`float | None` field written as `int = 0` got shipped once and only showed up
on 661 of 3343 endpoints. So every route is both read out of `/openapi.json`
*and* called, on a board that deliberately contains the awkward rows: a NULL
score on a passing workload, a NULL trial number, a run with no window, a
problem with no results at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="leaderboard venv only")

# Every (slug, problem) pair in the fixture board that exercises a different
# shape of run: complete, partial, hidden, kernel-with-no-results, nothing.
RUNS = [
    ("ref-v1-eager", "L1__001_alpha"),
    ("agent-alpha", "L1__001_alpha"),
    ("agent-alpha", "L2__002_beta"),          # a kernel, retime_ok NULL
    ("agent-trial-a", "L1__001_alpha"),       # hidden, bound-invalid row
    ("agent-trial-b", "L1__001_alpha"),       # no window row
    ("agent-timeout", "L2__002_beta"),        # re-time timed out
    ("agent-timeout", "Quant__003_gamma"),    # deferred, nothing at all
]


def v1_get_paths(client) -> list[str]:
    spec = client.get("/openapi.json").json()
    return [p for p, ops in spec["paths"].items()
            if p.startswith("/api/v1") and "get" in ops]


def test_every_v1_route_declares_a_response_schema(client):
    """The reason /api/v1 exists: a shape a client can be generated from, and
    a field rename that fails something. Bare /api/* is the legacy alias and is
    exempt. The transcript route streams a file and declares no JSON body."""
    spec = client.get("/openapi.json").json()
    missing = []
    for path, ops in spec["paths"].items():
        if not path.startswith("/api/v1") or path.endswith("/transcript"):
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


def test_every_v1_get_route_is_reached_by_this_test(client):
    """A guard on the two tests below: a route added to the app and not to
    `RUNS`/the URL list would otherwise be silently untested."""
    templated = {p for p in v1_get_paths(client) if "{" in p}
    assert templated == {
        "/api/v1/problems/{key}",
        "/api/v1/submissions/{slug}",
        "/api/v1/submissions/{slug}/problems/{key}",
        "/api/v1/submissions/{slug}/problems/{key}/kernel",
        "/api/v1/submissions/{slug}/problems/{key}/transcript",
        "/api/v1/jobs/{job_id}",
    }
    assert set(v1_get_paths(client)) - templated == {
        "/api/v1/parts", "/api/v1/stats", "/api/v1/leaderboard",
        "/api/v1/provisional", "/api/v1/problems", "/api/v1/jobs"}


def test_the_unparameterised_routes_validate(client):
    for url in ("/api/v1/parts", "/api/v1/stats", "/api/v1/leaderboard",
                "/api/v1/provisional",
                "/api/v1/problems", "/healthz",
                "/api/v1/leaderboard?category=L1", "/api/v1/problems?category=L2"):
        r = client.get(url)
        assert r.status_code == 200, f"{url} -> {r.status_code}: {r.text[:300]}"


def test_every_problem_and_run_validates(client):
    """A 500 here is the response model rejecting the handler's own output."""
    for p in client.get("/api/v1/problems").json():
        r = client.get(f"/api/v1/problems/{p['key']}")
        assert r.status_code == 200, f"{p['key']} -> {r.text[:300]}"
    for slug, key in RUNS:
        for url in (f"/api/v1/submissions/{slug}",
                    f"/api/v1/submissions/{slug}/problems/{key}"):
            r = client.get(url)
            assert r.status_code == 200, f"{url} -> {r.status_code}: {r.text[:300]}"


def test_a_missing_kernel_or_transcript_is_404_not_500(client):
    """Absence is an answer. `agent-trial-b` submitted no kernel that the
    ingest recorded, and nothing in this board has a transcript."""
    assert client.get("/api/v1/submissions/agent-trial-a/problems/"
                      "L1__001_alpha/kernel").status_code == 200
    assert client.get("/api/v1/submissions/agent-trial-b/problems/"
                      "L1__001_alpha/kernel").status_code == 404
    assert client.get("/api/v1/submissions/agent-trial-a/problems/"
                      "L1__001_alpha/transcript").status_code == 404


def test_unknown_slug_and_key_are_404_not_500(client):
    assert client.get("/api/v1/submissions/nope/problems/"
                      "L1__001_alpha").status_code == 404
    assert client.get("/api/v1/submissions/agent-alpha/problems/"
                      "nope").status_code == 404
    assert client.get("/api/v1/problems/nope").status_code == 404
    assert client.get("/api/v1/submissions/nope").status_code == 404


def test_absent_measurements_serialise_as_null_not_zero(client):
    """`int | None`, not `int = 0`. A missing measurement is not a small one,
    and a consumer cannot tell the difference after the coercion."""
    d = client.get(
        "/api/v1/submissions/agent-trial-a/problems/L1__001_alpha").json()
    invalid = [w for w in d["workloads"] if w["bound_invalid"]]
    assert invalid and invalid[0]["score"] is None
    failed = [w for w in d["workloads"] if w["status"] == "FAILED"]
    assert failed and failed[0]["score"] is None and failed[0]["speedup_vs_tb"] is None

    empty = client.get(
        "/api/v1/submissions/agent-timeout/problems/Quant__003_gamma").json()
    assert empty["summary"]["mean_attempted"] is None
    assert empty["summary"]["best"] is None
    assert empty["window"] is None

    p = {x["key"]: x for x in client.get("/api/v1/problems").json()}
    assert p["Quant__003_gamma"]["best_score"] is None


def test_the_queue_routes_validate_against_an_empty_queue(client, monkeypatch,
                                                          tmp_path):
    """`/api/v1/jobs` is served by `submit.py` and reads the *queue*, not the
    board, so it answers on a part with no database too. Pointed at a temp
    queue: the real one is a live spool and a test must not read or create it.
    """
    import submit
    monkeypatch.setattr(submit, "QUEUE_DB", tmp_path / "q.db")
    r = client.get("/api/v1/jobs")
    assert r.status_code == 200 and r.json() == []
    assert client.get("/api/v1/jobs/1").status_code == 404


def test_the_legacy_alias_is_a_superset_of_v1(client):
    """`/api/*` predates the versioned routes and is kept only because things
    call it. It has no response model, so it returns the handler's raw dict --
    every v1 field must still be present and equal, or the two have drifted
    into two contracts and only one of them is documented."""
    for bare, v1 in (("/api/leaderboard", "/api/v1/leaderboard"),
                     ("/api/provisional", "/api/v1/provisional"),
                     ("/api/problems", "/api/v1/problems")):
        raw = client.get(bare).json()
        typed = client.get(v1).json()
        assert len(raw) == len(typed), bare
        for a, b in zip(raw, typed):
            for k, v in b.items():
                assert a.get(k) == v, f"{bare}: {k} is {a.get(k)}, v1 says {v}"
    for bare, v1 in (("/api/problems/L1__001_alpha",
                      "/api/v1/problems/L1__001_alpha"),
                     ("/api/submissions/agent-alpha",
                      "/api/v1/submissions/agent-alpha")):
        raw, typed = client.get(bare).json(), client.get(v1).json()
        assert set(typed) <= set(raw), bare
