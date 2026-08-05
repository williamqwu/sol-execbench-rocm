#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SOL-ExecBench-ROCm leaderboard — local web app.

    leaderboard/run.sh              # http://127.0.0.1:8088

Pages render server-side from SQLite; the same data is available as JSON under
`/api/` for anything that wants to consume it.

Two ranking modes, both shown, because a leaderboard that offers only one is
easy to game:

* **Benchmark score** — the sum of per-workload scores over *every* scoreable
  workload in the benchmark, with anything not passed contributing zero. This
  is the headline. A submission that solves eight problems brilliantly cannot
  outrank one that solves two hundred.
* **Mean score (attempted)** — the average over the workloads a submission
  *attempted*, with an attempt that failed counting as zero. Always displayed
  next to its coverage. Useful for reading a partial run; meaningless without
  the coverage figure, so the UI never shows one without the other.

  This deliberately does **not** average over passes only. Averaging over
  passes rewards giving up: `torch.compile` reads 0.4907 that way, above eager
  PyTorch's 0.4548, purely because the 585 workloads it raised on vanish from
  the denominator instead of scoring zero. Both denominators are defensible in
  isolation, but only one of them cannot be improved by attempting less.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import inputs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = Path(os.environ.get("SOLBENCH_DB", HERE / "solbench.db"))

app = FastAPI(title="SOL-ExecBench-ROCm leaderboard", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))
# `meta` is a flat string->string table, so JSON-valued rows come back as text.
templates.env.filters["from_json"] = lambda s: json.loads(s) if s else []


def db() -> sqlite3.Connection:
    if not DB.exists():
        raise HTTPException(503, f"database not built: run leaderboard/ingest.py ({DB})")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def meta() -> dict:
    with db() as conn:
        m = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
    m["freshness"] = freshness(m)
    return m


def freshness(m: dict) -> dict:
    """Is the database still a faithful view of the artifacts it was built from?

    It is a *derived* store, so it goes stale the moment those artifacts move,
    and the failure is silent: every page still renders and every number still
    looks plausible.

    This compares the INPUTS, not the repo. An earlier version compared git
    HEAD, which fired on every commit -- including one that only touched a
    stylesheet -- and would still have called the board fresh when the
    untracked `glm-run1` agent run appeared. Both failures came from asking
    git a question about data git does not track. See `inputs.py`.
    """
    out: dict = {"stale": False, "reasons": []}
    out["db_built_utc"] = m.get("db_built_utc")
    out["built_from_git_sha"] = m.get("repo_git_sha")   # provenance, not a check
    try:
        recorded = json.loads(m.get("input_signature") or "{}")
        extra = [Path(p) for p in json.loads(m.get("input_extra_roots") or "[]")]
        current = inputs.signature(extra)
        out["inputs"] = {"recorded": recorded, "current": current}
        out["reasons"] = inputs.compare(recorded, current)
        out["stale"] = bool(out["reasons"])
        # The command must carry the roots this build actually used. A bare
        # `ingest.py` re-reads only artifacts/10, so following the banner
        # literally would silently drop every run kept outside the repo --
        # turning a freshness warning into a way to lose a submission.
        out["rebuild_command"] = "python leaderboard/ingest.py" + (
            " --agent-runs " + " ".join(str(p) for p in extra) if extra else "")
    except Exception as exc:                             # never 500 a page over this
        out["reasons"] = []
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def rows(conn, sql: str, args=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args)]


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------

def leaderboard_rows(conn, category: str | None = None) -> list[dict]:
    """One row per submission. Missing workloads score zero, by construction."""
    where_cat = "AND w.problem_key LIKE ?" if category else ""
    cat_arg = (f"{category}__%",) if category else ()

    total = conn.execute(
        f"SELECT COUNT(*) FROM workload w WHERE w.scoreable=1 {where_cat}",
        cat_arg).fetchone()[0]
    total_problems = conn.execute(
        f"""SELECT COUNT(*) FROM problem p WHERE p.n_scoreable > 0
            {'AND p.category = ?' if category else ''}""",
        (category,) if category else ()).fetchone()[0]

    out = []
    for s in rows(conn, "SELECT * FROM submission ORDER BY id"):
        agg = conn.execute(
            f"""SELECT COUNT(*) AS n_passed,
                       COALESCE(SUM(r.score),0) AS score_sum,
                       COALESCE(AVG(r.score),0) AS score_mean,
                       COALESCE(SUM(r.flagged),0) AS n_flagged
                  FROM result r
                  JOIN workload w ON w.problem_key=r.problem_key
                                 AND w.uuid=r.workload_uuid
                 WHERE r.submission_id=? AND r.status='PASSED'
                   AND r.score IS NOT NULL AND w.scoreable=1 {where_cat}""",
            (s["id"], *cat_arg)).fetchone()
        attempted = conn.execute(
            f"""SELECT COUNT(*) FROM result r
                  JOIN workload w ON w.problem_key=r.problem_key
                                 AND w.uuid=r.workload_uuid
                 WHERE r.submission_id=? AND w.scoreable=1 {where_cat}""",
            (s["id"], *cat_arg)).fetchone()[0]
        complete = conn.execute(
            f"""SELECT COUNT(*) FROM (
                  SELECT p.key, p.n_scoreable,
                         SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS ok
                    FROM problem p
                    JOIN workload w ON w.problem_key=p.key AND w.scoreable=1
                    LEFT JOIN result r ON r.problem_key=w.problem_key
                                      AND r.workload_uuid=w.uuid
                                      AND r.submission_id=?
                   WHERE p.n_scoreable > 0 {'AND p.category = ?' if category else ''}
                   GROUP BY p.key
                  HAVING ok = p.n_scoreable AND ok > 0)""",
            (s["id"], *((category,) if category else ()))).fetchone()[0]

        out.append({
            **s,
            "workloads_total": total,
            "workloads_passed": agg["n_passed"],
            "workloads_attempted": attempted,
            # Everything the benchmark score counts as zero, split by WHY it is
            # zero. Untested and failed are both zeroes in the headline, but
            # they mean different things about the submission, and a reader who
            # cannot tell them apart cannot tell a partial run from a bad one.
            "workloads_untested": total - attempted,
            "workloads_failed": attempted - agg["n_passed"],
            "partial": attempted < total,
            "problems_total": total_problems,
            "problems_complete": complete,
            "benchmark_score": (agg["score_sum"] / total) if total else 0.0,
            # Denominator is attempts, not passes: a workload that was tried
            # and failed scores 0 here, it does not leave the average. See the
            # module docstring for why the other denominator is unusable.
            "mean_score_attempted": (agg["score_sum"] / attempted) if attempted else 0.0,
            "mean_score_passed": agg["score_mean"],
            "coverage": (agg["n_passed"] / total) if total else 0.0,
            "n_flagged": agg["n_flagged"],
        })
    out.sort(key=lambda r: (-r["benchmark_score"], -r["mean_score_attempted"]))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def problem_rows(conn, category: str | None = None) -> list[dict]:
    sql = """SELECT p.*,
                    (SELECT MAX(r.score) FROM result r
                      WHERE r.problem_key=p.key AND r.status='PASSED') AS best_score,
                    (SELECT COUNT(DISTINCT r.submission_id) FROM result r
                      WHERE r.problem_key=p.key AND r.status='PASSED') AS n_submissions
               FROM problem p"""
    args: tuple = ()
    if category:
        sql += " WHERE p.category = ?"
        args = (category,)
    sql += " ORDER BY p.category, p.name"
    return rows(conn, sql, args)


def problem_detail(conn, key: str) -> dict:
    p = conn.execute("""
        SELECT p.*,
               (SELECT MAX(r.score) FROM result r
                 WHERE r.problem_key=p.key AND r.status='PASSED') AS best_score
          FROM problem p WHERE p.key=?""", (key,)).fetchone()
    if p is None:
        raise HTTPException(404, f"no such problem: {key}")
    p = dict(p)
    for f in ("axes_json", "inputs_json", "outputs_json"):
        p[f.replace("_json", "")] = json.loads(p.pop(f) or "{}")

    wls = rows(conn, "SELECT * FROM workload WHERE problem_key=? ORDER BY rowid", (key,))
    for w in wls:
        w["axes"] = json.loads(w.pop("axes_json") or "{}")
        # Both, not just t_sol_ms. A deferred problem has a T_SOL (it is
        # architectural and needs no GPU) but no T_b (nothing ran), so guarding
        # only the divisor left None/float and 500'd every one of the 15
        # deferred problem pages -- the exact pages a reader follows to find out
        # why the row shows 0.
        w["headroom"] = (w["t_b_ms"] / w["t_sol_ms"]) if (w["t_sol_ms"] and w["t_b_ms"]) else None
        w["results"] = rows(conn, """
            SELECT r.*, s.slug, s.name AS submission_name, s.kind
              FROM result r JOIN submission s ON s.id=r.submission_id
             WHERE r.problem_key=? AND r.workload_uuid=?
             ORDER BY (r.score IS NULL), r.score DESC""", (key, w["uuid"]))

    # No FILTER and no NULLS LAST: the system SQLite here is 3.26, which
    # predates both (3.30). They parse as syntax errors, not as ignored hints.
    per_sub = rows(conn, """
        SELECT s.slug, s.name, s.kind, s.model,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score,
               MIN(CASE WHEN r.status='PASSED' THEN r.latency_ms END) AS best_latency,
               SUM(r.flagged) AS flagged
          FROM result r JOIN submission s ON s.id=r.submission_id
         WHERE r.problem_key=?
         GROUP BY s.id
         ORDER BY (mean_score IS NULL), mean_score DESC""", (key,))

    return {"problem": p, "workloads": wls, "submissions": per_sub}


def submission_detail(conn, slug: str) -> dict:
    s = conn.execute("SELECT * FROM submission WHERE slug=?", (slug,)).fetchone()
    if s is None:
        raise HTTPException(404, f"no such submission: {slug}")
    s = dict(s)
    s["provenance"] = json.loads(s.pop("provenance_json") or "{}")
    per_problem = rows(conn, """
        SELECT p.key, p.category, p.name, p.n_scoreable,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               COUNT(*) AS attempted,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score,
               SUM(r.flagged) AS flagged
          FROM result r JOIN problem p ON p.key=r.problem_key
         WHERE r.submission_id=?
         GROUP BY p.key
         ORDER BY (mean_score IS NULL), mean_score DESC""", (s["id"],))
    return {"submission": s, "problems": per_problem}


def run_detail(conn, slug: str, key: str) -> dict:
    """One submission on one problem — the cell where the board's two axes meet.

    Rendered per request from the same tables as everything else. It is
    deliberately not pre-generated: there are 6 submissions x 235 problems =
    1410 of these today and the product grows with every run added, so a static
    build would be 1410 files that go stale the moment `ingest.py` runs again,
    which is exactly the failure mode the freshness check exists to catch.

    Rows come from `workload` LEFT JOINed to `result`, not from `result` alone.
    That is the whole point of the page: a workload this submission never
    attempted still has a T_SOL and a T_b, and it still contributes a zero to
    the benchmark score. Driving the table off `result` would silently omit
    precisely the rows that explain the score.
    """
    s = conn.execute("SELECT * FROM submission WHERE slug=?", (slug,)).fetchone()
    if s is None:
        raise HTTPException(404, f"no such submission: {slug}")
    p = conn.execute("SELECT * FROM problem WHERE key=?", (key,)).fetchone()
    if p is None:
        raise HTTPException(404, f"no such problem: {key}")
    s, p = dict(s), dict(p)
    s["provenance"] = json.loads(s.pop("provenance_json") or "{}")
    for f in ("axes_json", "inputs_json", "outputs_json"):
        p.pop(f, None)

    wls = rows(conn, """
        SELECT w.uuid, w.axes_json, w.scoreable, w.t_sol_ms, w.t_sol_source,
               w.sol_bottleneck, w.t_b_ms, w.t_b_variant,
               r.status, r.latency_ms, r.score, r.flagged, r.note
          FROM workload w
          LEFT JOIN result r ON r.problem_key = w.problem_key
                            AND r.workload_uuid = w.uuid
                            AND r.submission_id = ?
         WHERE w.problem_key = ?
         ORDER BY w.rowid""", (s["id"], key))

    n_att = n_pass = 0
    score_sum = 0.0
    for w in wls:
        w["axes"] = json.loads(w.pop("axes_json") or "{}")
        w["attempted"] = w["status"] is not None
        w["passed"] = w["status"] == "PASSED"
        # Speedup against the anchor, which is the number a kernel author
        # actually feels. S is the scored quantity but it is compressed; 1.8x
        # faster than optimized PyTorch reads as an outcome, S=0.64 does not.
        # Only on a pass, for the same reason the score is: T_k on a failed
        # workload is how fast the wrong answer was produced, and "0.99x vs
        # optimized PyTorch" printed next to FAILED reads as near-parity.
        w["speedup_vs_tb"] = ((w["t_b_ms"] / w["latency_ms"])
                              if w["passed"] and w["t_b_ms"] and w["latency_ms"]
                              else None)
        w["headroom"] = ((w["t_b_ms"] / w["t_sol_ms"])
                         if w["t_b_ms"] and w["t_sol_ms"] else None)
        # A score present on a row whose bound was beaten is impossible by
        # construction: ingest stores NULL there. Surface the reason instead of
        # an empty cell, or the page reads as a missing measurement.
        w["bound_invalid"] = (w["passed"] and w["score"] is None
                              and bool(w["latency_ms"]))
        if w["attempted"] and w["scoreable"]:
            n_att += 1
            if w["passed"]:
                n_pass += 1
                score_sum += w["score"] or 0.0

    n_scoreable = p["n_scoreable"] or 0
    summary = {
        "n_scoreable": n_scoreable,
        "attempted": n_att,
        "passed": n_pass,
        "failed": n_att - n_pass,
        "untested": max(0, n_scoreable - n_att),
        # Same three denominators as the board, for the same reason.
        "mean_attempted": (score_sum / n_att) if n_att else None,
        "mean_passed": (score_sum / n_pass) if n_pass else None,
        "problem_score": (score_sum / n_scoreable) if n_scoreable else None,
        # Passed rows only. Taking the max over every score let a submission
        # that passed nothing on this problem still advertise a "best workload
        # S", drawn from a row whose correctness check had failed.
        "best": max((w["score"] for w in wls
                     if w["passed"] and w["score"] is not None), default=None),
        "n_flagged": sum(1 for w in wls if w["flagged"]),
    }

    peers = rows(conn, """
        SELECT s.slug, s.name, s.kind,
               SUM(CASE WHEN r.status='PASSED' THEN 1 ELSE 0 END) AS passed,
               AVG(CASE WHEN r.status='PASSED' THEN r.score END) AS mean_score
          FROM result r JOIN submission s ON s.id = r.submission_id
         WHERE r.problem_key = ?
         GROUP BY s.id
         ORDER BY (mean_score IS NULL), mean_score DESC""", (key,))

    return {"submission": s, "problem": p, "workloads": wls,
            "summary": summary, "peers": peers}


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------

@app.get("/api/stats")
def api_stats():
    with db() as conn:
        m = {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}
        by_cat = rows(conn, """SELECT category, COUNT(*) AS n,
                                      SUM(CASE WHEN deferred=1 THEN 1 ELSE 0 END) AS deferred
                                 FROM problem GROUP BY category ORDER BY category""")
    return {"meta": m, "by_category": by_cat}


@app.get("/api/leaderboard")
def api_leaderboard(category: str | None = None):
    with db() as conn:
        return leaderboard_rows(conn, category)


@app.get("/api/problems")
def api_problems(category: str | None = None):
    with db() as conn:
        return problem_rows(conn, category)


@app.get("/api/problems/{key}")
def api_problem(key: str):
    with db() as conn:
        return problem_detail(conn, key)


@app.get("/api/submissions/{slug}")
def api_submission(slug: str):
    with db() as conn:
        return submission_detail(conn, slug)


@app.get("/api/submissions/{slug}/problems/{key}")
def api_run(slug: str, key: str):
    with db() as conn:
        return run_detail(conn, slug, key)


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

def page(request: Request, name: str, **ctx) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"meta": meta(), **ctx})


@app.get("/", response_class=HTMLResponse)
def index(request: Request, category: str | None = None):
    with db() as conn:
        board = leaderboard_rows(conn, category)
        cats = rows(conn, """SELECT category, COUNT(*) AS n,
                                    SUM(CASE WHEN deferred=1 THEN 1 ELSE 0 END) AS deferred
                               FROM problem GROUP BY category ORDER BY category""")
    return page(request, "index.html", board=board, categories=cats,
                category=category, nav="board")


@app.get("/problems", response_class=HTMLResponse)
def problems(request: Request, category: str | None = None, q: str | None = None):
    with db() as conn:
        items = problem_rows(conn, category)
        cats = rows(conn, "SELECT DISTINCT category FROM problem ORDER BY category")
    if q:
        needle = q.lower()
        items = [p for p in items
                 if needle in p["key"].lower()
                 or needle in (p["description"] or "").lower()]
    return page(request, "problems.html", problems=items,
                categories=[c["category"] for c in cats], category=category,
                q=q or "", nav="problems")


@app.get("/problems/{key}", response_class=HTMLResponse)
def problem(request: Request, key: str):
    with db() as conn:
        d = problem_detail(conn, key)
    return page(request, "problem.html", **d)


@app.get("/submissions/{slug}", response_class=HTMLResponse)
def submission(request: Request, slug: str):
    with db() as conn:
        d = submission_detail(conn, slug)
    return page(request, "submission.html", **d)


@app.get("/submissions/{slug}/problems/{key}", response_class=HTMLResponse)
def run(request: Request, slug: str, key: str):
    with db() as conn:
        d = run_detail(conn, slug, key)
    return page(request, "run.html", **d)


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request):
    with db() as conn:
        bounds = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("bound_sources") or "{}")
        excluded = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("excluded_submissions") or "{}")
        deferred = rows(conn, """SELECT key, n_workloads, deferred_reason,
                                        deferred_mechanism, deferred_error
                                   FROM problem WHERE deferred=1 ORDER BY key""")
        # Facts about each invalid-bound problem rather than a single asserted
        # cause: at least two distinct mechanisms are in play (D18, D21), so a
        # per-row explanation copied from the first one found would be wrong
        # for the others.
        bad_keys = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("problems_with_invalid_bound") or "[]")
        invalid_bound_info = {}
        for k in bad_keys:
            row = conn.execute(
                """SELECT n_workloads, median_headroom FROM problem WHERE key=?""",
                (k,)).fetchone()
            if row:
                invalid_bound_info[k] = {"n_workloads": row["n_workloads"],
                                         "headroom": row["median_headroom"]}
    return page(request, "methodology.html", bound_sources=bounds,
                deferred=deferred, excluded=excluded, nav="methodology",
                invalid_bound_info=invalid_bound_info)


@app.get("/healthz")
def healthz():
    if not DB.exists():
        return JSONResponse({"ok": False, "db": str(DB), "error": "not built"},
                            status_code=503)
    m = meta()
    return JSONResponse({"ok": True, "db": str(DB), "part": m.get("part"),
                         "manifest_version": m.get("manifest_version"),
                         "freshness": m["freshness"]})
