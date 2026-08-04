#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SOL-ExecBench-AMD leaderboard — local web app.

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
  actually passed, always displayed next to its coverage. Useful for reading a
  partial run; meaningless without the coverage figure, so the UI never shows
  one without the other.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HERE = Path(__file__).resolve().parent
DB = Path(os.environ.get("SOLBENCH_DB", HERE / "solbench.db"))

app = FastAPI(title="SOL-ExecBench-AMD leaderboard", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


def db() -> sqlite3.Connection:
    if not DB.exists():
        raise HTTPException(503, f"database not built: run leaderboard/ingest.py ({DB})")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def meta() -> dict:
    with db() as conn:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key,value FROM meta")}


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
            "problems_total": total_problems,
            "problems_complete": complete,
            "benchmark_score": (agg["score_sum"] / total) if total else 0.0,
            "mean_score_attempted": agg["score_mean"],
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
        w["headroom"] = (w["t_b_ms"] / w["t_sol_ms"]) if w["t_sol_ms"] else None
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
                category=category)


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
                categories=[c["category"] for c in cats], category=category, q=q or "")


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


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request):
    with db() as conn:
        bounds = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("bound_sources") or "{}")
        excluded = json.loads(dict(
            (r["key"], r["value"]) for r in conn.execute("SELECT key,value FROM meta")
        ).get("excluded_submissions") or "{}")
        deferred = rows(conn, """SELECT key, deferred_reason FROM problem
                                  WHERE deferred=1 ORDER BY key""")
    return page(request, "methodology.html", bound_sources=bounds,
                deferred=deferred, excluded=excluded)


@app.get("/healthz")
def healthz():
    return JSONResponse({"ok": DB.exists(), "db": str(DB)})
