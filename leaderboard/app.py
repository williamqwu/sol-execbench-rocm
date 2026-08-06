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
import submit

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models import (Health, LeaderboardRow, ProblemDetail, ProblemSummary,
                    RunDetail, Stats, SubmissionDetail)

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
               SUM(r.flagged) AS flagged,
               e.cost_usd, e.wall_seconds, e.turns, e.harness_evals
          FROM result r
          JOIN problem p ON p.key=r.problem_key
          LEFT JOIN run_effort e ON e.submission_id=r.submission_id
                                AND e.problem_key=r.problem_key
         WHERE r.submission_id=?
         GROUP BY p.key
         ORDER BY (mean_score IS NULL), mean_score DESC""", (s["id"],))

    # Kernels that produced no results at all. Without this they are absent
    # from the submission page entirely -- the agent wrote a kernel, the
    # re-time failed, and the run looks like it never touched the problem.
    unmeasured = rows(conn, """
        SELECT k.problem_key AS key, p.category, p.name, p.n_scoreable,
               k.n_lines, k.retime_error
          FROM run_kernel k JOIN problem p ON p.key = k.problem_key
         WHERE k.submission_id = ?
           AND NOT EXISTS (SELECT 1 FROM result r
                            WHERE r.submission_id = k.submission_id
                              AND r.problem_key = k.problem_key)
         ORDER BY k.problem_key""", (s["id"],))

    return {"submission": s, "problems": per_problem, "unmeasured": unmeasured}


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

    kernel = conn.execute(
        """SELECT source, n_lines, sha256, retime_ok, retime_error
             FROM run_kernel WHERE submission_id=? AND problem_key=?""",
        (s["id"], key)).fetchone()

    # Which T_b formulations actually anchored this problem, and on how many
    # workloads. Not "the baseline", plural: T_b is chosen per workload, so a
    # single problem is routinely anchored by two or three different variants
    # and a page that names only one is showing the wrong side of the diff for
    # the rest.
    won = {r["t_b_variant"]: r["n"] for r in conn.execute(
        """SELECT t_b_variant, COUNT(*) AS n FROM workload
            WHERE problem_key=? AND t_b_variant IS NOT NULL
            GROUP BY t_b_variant""", (key,))}
    variants = []
    for vname, n in sorted(won.items(), key=lambda kv: -kv[1]):
        row = conn.execute(
            "SELECT source, n_lines FROM variant_source WHERE problem_key=? AND variant=?",
            (key, vname)).fetchone()
        if row:
            variants.append({"variant": vname, "source": row["source"],
                             "n_lines": row["n_lines"], "won_workloads": n})

    traj = rows(conn, """
        SELECT n, utc, minutes_in, ok, all_passed, passed, workloads,
               geomean_speedup, mean_score, kernel_sha, kernel_lines
          FROM trajectory_eval
         WHERE submission_id=? AND problem_key=? ORDER BY n""", (s["id"], key))
    _mark_regressions(traj)

    effort = conn.execute(
        "SELECT * FROM run_effort WHERE submission_id=? AND problem_key=?",
        (s["id"], key)).fetchone()
    tr = conn.execute(
        "SELECT bytes, n_lines, n_turns, tools_json FROM transcript "
        "WHERE submission_id=? AND problem_key=?", (s["id"], key)).fetchone()

    return {
        "submission": s, "problem": p, "workloads": wls,
        "summary": summary, "peers": peers,
        "kernel": dict(kernel) if kernel else None,
        "variants": variants,
        "reference": p.get("reference"),
        "trajectory": traj,
        "effort": {k: v for k, v in dict(effort).items()
                   if k not in ("submission_id", "problem_key")} if effort else None,
        "transcript": ({"bytes": tr["bytes"], "n_lines": tr["n_lines"],
                        "n_turns": tr["n_turns"],
                        "tools": json.loads(tr["tools_json"] or "{}"),
                        "url": f"/api/v1/submissions/{slug}/problems/{key}/transcript"}
                       if tr else None),
        "depth_note": s.get("depth_note"),
    }


def _mark_regressions(traj: list[dict]) -> None:
    """Classify each eval against the best seen so far in the same problem.

    `regression` is reserved for **correctness**: an eval that passes fewer
    workloads than an earlier one broke something that worked, and that needs
    no threshold to assert.

    Score movement deliberately gets no boolean. The first version flagged any
    eval below the running best, which marked ten of L1__030's fifteen evals as
    regressions -- including 0.5652 against a best of 0.5665, a 0.2% difference
    on a plateau. Calling that a regression is a claim about measurement noise,
    and this repo has no figure for the noise of an agent-side eval (STATE.md
    D20 records that the matmul timing spread on this part is bimodal and that
    no defensible constant could be derived). Inventing a threshold to make the
    flag look reasonable would be inventing a measurement. So `delta_vs_best`
    is reported as a number and the reader judges it.

    An eval where the harness itself did not run is neither: `workloads == 0`
    means nothing was measured, so it cannot have regressed. It is called out
    as a harness error instead, which is what it is.
    """
    best_score = None
    best_passed = 0
    for e in traj:
        measured = bool(e.get("ok")) and (e.get("workloads") or 0) > 0
        e["harness_error"] = not measured
        e["regression"] = bool(
            measured and e.get("passed") is not None and e["passed"] < best_passed)
        e["delta_vs_best"] = (
            (e["mean_score"] - best_score)
            if (measured and e.get("mean_score") is not None and best_score is not None)
            else None)
        if not measured:
            continue
        if e.get("mean_score") is not None:
            best_score = (e["mean_score"] if best_score is None
                          else max(best_score, e["mean_score"]))
        if e.get("passed") is not None:
            best_passed = max(best_passed, e["passed"])


# --------------------------------------------------------------------------
# JSON API — /api/v1 is the contract; bare /api/* is kept as a legacy alias
#
# Every v1 route declares a response_model, so `/openapi.json` describes the
# shapes and a client can be generated from it. The unversioned routes below
# predate that and are left in place only because things already call them;
# they return the same objects with no schema attached.
# --------------------------------------------------------------------------

V1 = APIRouter(prefix="/api/v1", tags=["v1"])


@V1.get("/stats", response_model=Stats)
def v1_stats():
    return api_stats()


@V1.get("/leaderboard", response_model=list[LeaderboardRow])
def v1_leaderboard(category: str | None = None):
    with db() as conn:
        return leaderboard_rows(conn, category)


@V1.get("/problems", response_model=list[ProblemSummary])
def v1_problems(category: str | None = None):
    with db() as conn:
        return problem_rows(conn, category)


@V1.get("/problems/{key}", response_model=ProblemDetail)
def v1_problem(key: str):
    with db() as conn:
        return problem_detail(conn, key)


@V1.get("/submissions/{slug}", response_model=SubmissionDetail)
def v1_submission(slug: str):
    with db() as conn:
        return submission_detail(conn, slug)


@V1.get("/submissions/{slug}/problems/{key}", response_model=RunDetail)
def v1_run(slug: str, key: str):
    with db() as conn:
        return run_detail(conn, slug, key)


@V1.get("/submissions/{slug}/problems/{key}/kernel",
        response_class=PlainTextResponse)
def v1_kernel(slug: str, key: str):
    """The submitted kernel as source, for diffing without unwrapping JSON."""
    with db() as conn:
        row = conn.execute(
            """SELECT k.source FROM run_kernel k
                 JOIN submission s ON s.id = k.submission_id
                WHERE s.slug=? AND k.problem_key=?""", (slug, key)).fetchone()
    if row is None:
        raise HTTPException(404, f"no kernel recorded for {slug} on {key}")
    return PlainTextResponse(row["source"], media_type="text/x-python")


@V1.get("/submissions/{slug}/problems/{key}/transcript")
def v1_transcript(slug: str, key: str):
    """Stream the agent transcript from disk.

    Transcripts are 2 MB of JSONL each and are deliberately NOT in the
    database. The path is looked up in `transcript`, never taken from the
    request, so a caller cannot name a file the ingest did not index -- the
    slug and key are only ever used as lookup keys, and the resolved path is
    re-checked against the indexed value before the file is opened.
    """
    with db() as conn:
        row = conn.execute(
            """SELECT t.path FROM transcript t
                 JOIN submission s ON s.id = t.submission_id
                WHERE s.slug=? AND t.problem_key=?""", (slug, key)).fetchone()
    if row is None:
        raise HTTPException(404, f"no transcript recorded for {slug} on {key}")
    path = Path(row["path"])
    if not path.is_file():
        raise HTTPException(
            410, f"transcript was indexed but is no longer on disk: {path.name}")
    return FileResponse(path, media_type="application/x-ndjson",
                        filename=f"{slug}__{key}.jsonl")


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


def trajectory_chart(traj: list[dict], w: int = 720, h: int = 190) -> dict | None:
    """Lay out the trajectory as SVG coordinates.

    Server-side so the page stays dependency-free — no chart library, no build
    step, and the same numbers the API returns are the ones plotted.

    Plotted against S, not against speedup. Speedup is relative to whatever the
    agent's own harness measured as the reference on its own GPU, which is not
    the anchor the leaderboard scores against; S is. That also puts T_b at a
    fixed y = 0.5 gridline, so "crossed the anchor" is visible as a line
    crossing rather than something the reader has to work out.
    """
    pts = [e for e in traj if e.get("mean_score") is not None]
    if len(pts) < 2:
        return None
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 26
    xs = [e.get("minutes_in") or 0.0 for e in pts]
    x_max = max(xs) or 1.0
    ys = [e["mean_score"] for e in pts]
    # Always include the 0.5 anchor in the visible range, or a run that never
    # reaches it renders as a confident climb to nowhere in particular.
    y_lo, y_hi = min(min(ys), 0.5), max(max(ys), 0.5)
    span = (y_hi - y_lo) or 0.1
    y_lo, y_hi = y_lo - span * 0.15, y_hi + span * 0.15

    def px(m):
        return pad_l + (m / x_max) * (w - pad_l - pad_r)

    def py(s):
        return pad_t + (1 - (s - y_lo) / (y_hi - y_lo)) * (h - pad_t - pad_b)

    points = [{"x": round(px(e.get("minutes_in") or 0.0), 1),
               "y": round(py(e["mean_score"]), 1), **e} for e in pts]
    # Evals with no score still happened, and hiding them would make a run that
    # broke twice look monotonic. Drawn on the axis as marks, not as points.
    marks = [{"x": round(px(e.get("minutes_in") or 0.0), 1),
              "n": e["n"], "harness_error": e.get("harness_error"),
              "regression": e.get("regression")}
             for e in traj if e.get("mean_score") is None]
    return {"w": w, "h": h, "points": points, "marks": marks,
            "path": " ".join(f"{'M' if i == 0 else 'L'}{p['x']},{p['y']}"
                             for i, p in enumerate(points)),
            "y_anchor": round(py(0.5), 1) if y_lo <= 0.5 <= y_hi else None,
            "x_axis": round(h - pad_b, 1), "pad_l": pad_l,
            "x_max_min": round(x_max, 1),
            "y_lo": round(y_lo, 3), "y_hi": round(y_hi, 3)}


@app.get("/submissions/{slug}/problems/{key}", response_class=HTMLResponse)
def run(request: Request, slug: str, key: str):
    with db() as conn:
        d = run_detail(conn, slug, key)
    return page(request, "run.html", chart=trajectory_chart(d["trajectory"]), **d)


app.include_router(V1)
app.include_router(submit.router)


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


@app.get("/healthz", response_model=Health)
def healthz():
    if not DB.exists():
        return JSONResponse({"ok": False, "db": str(DB), "error": "not built"},
                            status_code=503)
    m = meta()
    return JSONResponse({"ok": True, "db": str(DB), "part": m.get("part"),
                         "manifest_version": m.get("manifest_version"),
                         "freshness": m["freshness"]})
