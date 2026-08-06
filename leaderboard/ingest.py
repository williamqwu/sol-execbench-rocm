#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the leaderboard database from the artifacts.

    python leaderboard/ingest.py --db leaderboard/solbench.db

The database is disposable. Everything in it comes from
`artifacts/09/manifest-v1.json`, the dataset definitions, the task 06 variant
sweep and any agent runs under `artifacts/10/`. Rerun this after any artifact
changes; never edit the database by hand.

Scores are computed with the repo's own `sol_score`, not a reimplementation,
so the leaderboard cannot drift from the scoring the harness applies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import inputs  # noqa: E402
from sol_execbench.sol_score import sol_score  # noqa: E402

DATASET = ROOT / "data" / "SOL-ExecBench" / "benchmark"
MANIFEST = ROOT / "artifacts" / "09" / "manifest-v1.json"
DEFERRED = ROOT / "artifacts" / "deferred.json"
CANDIDATES = ROOT / "artifacts" / "06" / "candidates"
AUTHORITATIVE = ROOT / "artifacts" / "06" / "authoritative"
AGENT_RUNS = ROOT / "artifacts" / "10"

VARIANT_LABELS = {
    "v1_eager": "PyTorch eager",
    "v2_compile": "torch.compile",
    "v3_compile_max_autotune": "torch.compile max-autotune",
    "v4_contiguous": "PyTorch eager + contiguous",
}

# Runs that stay on disk as artifacts but off the board. Deleting them would be
# the wrong fix -- they were really run and their transcripts are evidence --
# but a truncated pilot next to finished work invites a comparison that is not
# there to be made. Excluded here, listed with the reason on /methodology.
BOARD_EXCLUSIONS = {
    "pilot8": "Smoke test, not a result. The $8/problem cap stopped all eight "
              "sessions mid-work (8/8 `budget_exhausted`), so none of them "
              "chose when to stop, and three of the eight submitted a kernel "
              "that does not pass. Its mean of 0.776 is survivorship over the "
              "five problems where anything passed at all. Artifacts and "
              "transcripts are kept under artifacts/10/pilot8/.",
}


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def git_sha(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        return None


def part_of(device: str | None) -> str | None:
    """'AMD Instinct MI350X' -> 'MI350X'.

    The part is not decoration. T_SOL in milliseconds, T_b and F_LOCK all differ
    between MI350X and MI355X, and `scripts/score_solutions.py` refuses an
    artifact whose part is not the live node's rather than rescaling it. A board
    that does not name its part invites exactly the comparison that refusal
    exists to prevent.
    """
    for token in (device or "").split():
        if token.upper().startswith("MI"):
            return token
    return None


def ingest_meta(conn, manifest: dict, extra_roots: list[Path] | None = None) -> None:
    prov = manifest.get("_provenance", {})
    device = ((prov.get("torch") or {}).get("devices") or [None])[0]
    rows = {
        "manifest_version": manifest.get("manifest_version"),
        "methodology": manifest.get("methodology"),
        "score_formula": manifest.get("score_formula"),
        "git_sha": prov.get("git_sha"),
        "manifest_utc": prov.get("utc"),
        "host": prov.get("host"),
        "f_lock_mhz": prov.get("f_lock_mhz"),
        "rocm_version": (prov.get("rocm") or {}).get("version"),
        "driver_version": (prov.get("rocm") or {}).get("driver"),
        "torch_version": (prov.get("torch") or {}).get("version"),
        "device": device,
        "part": prov.get("part") or part_of(device),
        "n_devices": (prov.get("torch") or {}).get("device_count"),
        # Freshness. The database is a view of the artifacts and goes stale the
        # moment they move; without this it goes stale *silently*, which is the
        # one failure a leaderboard cannot afford. `app.py` re-derives the
        # signature and compares. The git SHA is kept for provenance only --
        # comparing it was the bug, since it flagged every unrelated commit and
        # missed untracked artifacts entirely.
        "db_built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_git_sha": git_sha(ROOT),
        "total_problems": manifest["problem_set"]["total_in_dataset"],
        "scoreable_problems": manifest["problem_set"]["scoreable_problems"],
        "scoreable_workloads": manifest["stats"]["scoreable_workloads"],
        "expected_by_category": json.dumps(manifest["problem_set"]["expected_by_category"]),
        "bound_sources": json.dumps(manifest.get("bound_sources", {})),
        # The extra roots go in too: without them `app.py` would enumerate a
        # different input set than the build did and report a phantom "files
        # removed" on every request.
        "input_signature": json.dumps(inputs.signature(extra_roots)),
        "input_extra_roots": json.dumps([str(p) for p in (extra_roots or [])]),
    }
    conn.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     [(k, None if v is None else str(v)) for k, v in rows.items()])


def ingest_problems(conn, manifest: dict) -> None:
    # `deferred.json` keys its entries under "problems", as a dict. An earlier
    # version of this read `d.get("deferred", ...)`, which is absent, so every
    # reason silently resolved to None and all 15 NVFP4 problems rendered as an
    # unexplained "0 scoreable" -- indistinguishable, to a reader, from a sweep
    # that never ran. Missing an explanation is how a documented decision comes
    # to look like a bug, so this asserts the shape rather than defaulting.
    deferred_info: dict[str, dict] = {}
    if DEFERRED.exists():
        d = json.loads(DEFERRED.read_text())
        entries = d.get("problems")
        if not isinstance(entries, dict):
            raise SystemExit(
                f"{DEFERRED}: expected a 'problems' object mapping problem key -> "
                f"reason; found keys {sorted(d)}. Refusing to ingest deferrals "
                f"as unexplained zeros.")
        deferred_info = {k: v for k, v in entries.items() if isinstance(v, dict)}
    deferred_set = set(manifest["problem_set"]["deferred_problems"])
    if deferred_set - set(deferred_info):
        raise SystemExit(
            "manifest defers problems that carry no entry in deferred.json: "
            f"{sorted(deferred_set - set(deferred_info))}")

    for key, p in manifest["problems"].items():
        category, name = key.split("__", 1)
        defn_path = DATASET / category / name / "definition.json"
        defn = json.loads(defn_path.read_text()) if defn_path.exists() else {}

        wls = p.get("workloads", {})
        heads = [w["t_b_ms"] / w["t_sol_ms"] for w in wls.values()
                 if w.get("scoreable") and w.get("t_sol_ms") and w.get("t_b_ms")]

        info = deferred_info.get(key) or {}
        conn.execute(
            """INSERT OR REPLACE INTO problem
               (key,category,name,description,hf_id,reference,axes_json,
                inputs_json,outputs_json,n_workloads,n_scoreable,deferred,
                deferred_reason,deferred_mechanism,deferred_error,median_headroom)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, category, name, defn.get("description"), defn.get("hf_id"),
             defn.get("reference"), json.dumps(defn.get("axes") or {}),
             json.dumps(defn.get("inputs") or {}),
             json.dumps(defn.get("outputs") or {}),
             p.get("n_workloads", len(wls)), p.get("n_scoreable", 0),
             1 if key in deferred_set else 0,
             info.get("reason"), info.get("mechanism"), info.get("error"),
             statistics.median(heads) if heads else None))

        for uuid, w in wls.items():
            tol = w.get("tolerance") or {}
            conn.execute(
                """INSERT OR REPLACE INTO workload
                   (problem_key,uuid,axes_json,t_sol_cycles,t_sol_ms,t_sol_source,
                    t_sol_cycles_solar,t_sol_cycles_traffic,sol_bottleneck,
                    t_b_ms,t_b_variant,tol_atol,tol_rtol,tol_ratio,
                    tol_derivation,scoreable)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (key, uuid, json.dumps(w.get("axes") or {}),
                 w.get("t_sol_cycles"), w.get("t_sol_ms"), w.get("t_sol_source"),
                 w.get("t_sol_cycles_solar"), w.get("t_sol_cycles_traffic"),
                 w.get("sol_bottleneck"), w.get("t_b_ms"), w.get("t_b_variant"),
                 tol.get("max_atol"), tol.get("max_rtol"),
                 tol.get("required_matched_ratio"), tol.get("_derivation"),
                 1 if w.get("scoreable") else 0))


def bounds(manifest: dict) -> dict:
    """(problem, uuid) -> (t_sol_ms, t_b_ms) for every scoreable workload."""
    out = {}
    for key, p in manifest["problems"].items():
        for uuid, w in p.get("workloads", {}).items():
            if w.get("scoreable") and w.get("t_sol_ms") and w.get("t_b_ms"):
                out[(key, uuid)] = (w["t_sol_ms"], w["t_b_ms"])
    return out


def add_submission(conn, **kw) -> int:
    cur = conn.execute(
        """INSERT OR REPLACE INTO submission
           (slug,name,kind,author,model,created_utc,notes,provenance_json,
            cost_usd,wall_seconds,gpu)
           VALUES (:slug,:name,:kind,:author,:model,:created_utc,:notes,
                   :provenance_json,:cost_usd,:wall_seconds,:gpu)""",
        {"author": None, "model": None, "created_utc": None, "notes": None,
         "provenance_json": None, "cost_usd": None, "wall_seconds": None,
         "gpu": None, **kw})
    return cur.lastrowid


def ingest_variants(conn, manifest: dict) -> dict:
    """The four PyTorch variants, from the task 06 sweep.

    Authoritative (GPU 0) timings win where they exist; the wider sweep on GPUs
    1-7 fills the rest. Which one produced each number goes in `note`, because
    mixing them silently is exactly the kind of thing that makes a leaderboard
    unfalsifiable.
    """
    b = bounds(manifest)
    prov = manifest.get("_provenance", {})

    per_variant: dict[str, dict] = {}
    for src, label in ((CANDIDATES, "sweep gpu1-7"), (AUTHORITATIVE, "authoritative gpu0")):
        if not src.exists():
            continue
        for f in sorted(src.glob("*.json")):
            doc = json.loads(f.read_text())
            pkey = doc.get("problem") or f.stem
            for vname, v in (doc.get("variants") or {}).items():
                lat = v.get("latency_ms_by_workload") or {}
                store = per_variant.setdefault(vname, {})
                for uuid, ms in lat.items():
                    # authoritative overwrites sweep, by iteration order
                    store[(pkey, uuid)] = (ms, label, bool(v.get("all_passed")))

    excluded: list[str] = []
    for vname, entries in sorted(per_variant.items()):
        # v5_compile_contiguous ran on all 235 problems and passed zero
        # workloads (torch.compile raises during tracing). A row with 0%
        # coverage and no score is noise on a leaderboard, but dropping it
        # silently would hide a variant that was actually run -- so it is
        # excluded here and recorded in `meta`.
        if not entries:
            excluded.append(vname)
            print(f"  {vname}: 0 scored workloads -- excluded, recorded in meta")
            continue
        sub_id = add_submission(
            conn,
            slug=f"baseline-{vname.replace('_','-')}",
            name=VARIANT_LABELS.get(vname, vname),
            kind="reference_variant",
            author="sol-execbench-amd",
            model=None,
            created_utc=prov.get("utc"),
            notes=("One of the four PyTorch formulations that T_b is derived from. "
                   "T_b is the fastest of them per workload, so the winning variant "
                   "scores exactly 0.5 there by construction. These are not agent "
                   "submissions and are not comparable to one."),
            provenance_json=json.dumps(prov),
            gpu="0 (authoritative) / 1-7 (sweep)")
        rows = []
        for (pkey, uuid), (ms, label, all_passed) in entries.items():
            bound = b.get((pkey, uuid))
            if not bound or ms is None:
                continue
            t_sol, t_b = bound
            # Score ONLY on a pass. A variant that failed the correctness check
            # still has a latency -- and `sol_score` will happily turn it into a
            # 0.4956 -- but that number is the speed of computing the wrong
            # answer, and it is not comparable to anything. This path used to
            # store it, which diverged from the agent path (`agent_score.py`
            # leaves score None unless PASSED); the aggregate queries all filter
            # on status='PASSED' so no ranking was ever affected, but every page
            # that displayed a raw per-workload score showed it, and the
            # submission x problem view put "FAILED ... S=0.4956" on one row.
            # Fixed here rather than in the templates so that every consumer,
            # including /api, gets the same answer.
            rows.append((sub_id, pkey, uuid, "PASSED" if all_passed else "FAILED",
                         ms, sol_score(ms, t_b, t_sol) if all_passed else None,
                         0, label))
        conn.executemany(
            """INSERT OR REPLACE INTO result
               (submission_id,problem_key,workload_uuid,status,latency_ms,
                score,flagged,note) VALUES (?,?,?,?,?,?,?,?)""", rows)
        print(f"  {vname}: {len(rows)} scored workloads")

    return {v: "Ran on all 235 problems and passed 0 workloads "
               "(torch.compile raises during tracing)." for v in excluded}


def ingest_agent_runs(conn, extra_roots: list[Path] | None = None) -> dict:
    """Any `<root>/<run>/scored.json` written by `agent_score.py`.

    Extra roots matter because a run directory does not have to live in the
    repo -- a scratch experiment under $HOME is a legitimate place for one, and
    globbing only `artifacts/10` silently omits it from the board rather than
    reporting that it was skipped.
    """
    roots = [AGENT_RUNS, *(extra_roots or [])]
    scored_files = []
    for root in roots:
        if root.exists():
            scored_files += sorted(root.glob("*/scored.json"))
            if (root / "scored.json").exists():      # root IS a run directory
                scored_files.append(root / "scored.json")
        else:
            print(f"  (agent-run root not found, skipped: {root})")

    excluded: dict[str, str] = {}
    # A bound a real kernel beat is a fact about the BOUND, not about the run
    # that happened to expose it. Two ways that got lost: `INSERT OR REPLACE`
    # let each run overwrite the previous run's list, and excluding a run threw
    # its findings away with it -- which is how FlashInfer-Bench__019, found by
    # the pilot and central to D18, disappeared from /methodology the moment
    # the pilot came off the board. Accumulated across every run that was read,
    # excluded or not, and written once.
    invalid_bounds: set[str] = set()

    def note_bound_violations(doc: dict) -> None:
        invalid_bounds.update(r["problem"] for r in doc.get("results", [])
                              if r.get("bound_violation"))

    for scored in scored_files:
        doc = json.loads(scored.read_text())
        run_id = doc.get("run_id", scored.parent.name)
        note_bound_violations(doc)
        # Validation runs are scored the same way and kept as artifacts, but a
        # one-problem smoke test on the board is noise, not information.
        if doc.get("leaderboard") is False:
            excluded[run_id] = doc.get("excluded_reason") or "validation run"
            print(f"  agent {run_id}: excluded from board ({excluded[run_id]})")
            continue
        if run_id in BOARD_EXCLUSIONS:
            excluded[run_id] = BOARD_EXCLUSIONS[run_id]
            print(f"  agent {run_id}: excluded from board (smoke test), "
                  f"recorded in meta")
            continue
        sub_id = add_submission(
            conn,
            slug=f"agent-{run_id}",
            name=doc.get("display_name") or f"Claude Code agent ({run_id})",
            kind="agent",
            author=doc.get("author", "claude-code"),
            model=doc.get("model"),
            created_utc=(doc.get("_provenance") or {}).get("utc"),
            notes=doc.get("notes"),
            provenance_json=json.dumps(doc.get("_provenance") or {}),
            # A run reporting exactly 0 alongside a null wall time did not cost
            # nothing -- nothing recorded it. Storing 0 asserts "free" and puts
            # a $0 in a column next to a real $250; NULL asserts "unknown",
            # which is what is true. glm-run1 is the case in point.
            cost_usd=(doc.get("total_cost_usd") or None),
            wall_seconds=doc.get("wall_seconds_total"),
            gpu="0 (authoritative re-time)")
        rows = []
        for r in doc.get("results", []):
            # A workload measured faster than its own speed-of-light bound is
            # evidence the bound is wrong, not that the kernel is exceptional.
            # It is stored with score NULL so it cannot lift any ranking, and
            # the note says why rather than the row silently vanishing.
            violated = bool(r.get("bound_violation"))
            rows.append((sub_id, r["problem"], r["workload_uuid"], r.get("status"),
                         r.get("latency_ms"),
                         None if violated else r.get("score"),
                         1 if r.get("flagged") else 0, r.get("note")))
        conn.executemany(
            """INSERT OR REPLACE INTO result
               (submission_id,problem_key,workload_uuid,status,latency_ms,
                score,flagged,note) VALUES (?,?,?,?,?,?,?,?)""", rows)
        # Depth: kernels, trajectory, effort, transcripts. Keyed off the
        # problems this run actually has results for, so a stale file left in
        # a run directory for a problem that was never scored cannot invent a
        # row on the board.
        run_problems = sorted({r["problem"] for r in doc.get("results", [])})
        found = ingest_depth(conn, sub_id, scored.parent, run_problems)
        conn.execute("UPDATE submission SET depth_note=?, depth_json=? WHERE id=?",
                     (depth_note(found, len(run_problems)), json.dumps(found), sub_id))
        print(f"    depth: {found['kernels']} kernels, {found['trajectory']} "
              f"trajectories, {found['effort']} effort, "
              f"{found['transcripts']} transcripts")

        bad = sorted({r["problem"] for r in doc.get("results", [])
                      if r.get("bound_violation")})
        if bad:
            print(f"  agent {run_id}: {len(rows)} workloads; "
                  f"{len(bad)} problem(s) have a bound a real kernel beat: {bad}")
        else:
            print(f"  agent {run_id}: {len(rows)} scored workloads")

    if invalid_bounds:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     ("problems_with_invalid_bound", json.dumps(sorted(invalid_bounds))))
        print(f"  bounds beaten by a real kernel, across all runs read: "
              f"{len(invalid_bounds)}")
    return excluded


def ingest_depth(conn, sub_id: int, run_dir: Path, problems: list[str]) -> dict:
    """Kernels, trajectory, per-problem effort and transcripts for one run.

    All of it optional. Different harnesses wrote different subsets -- the
    amdpilot fleet run recorded no trajectory and no transcripts at all -- so
    this reports what it found rather than assuming a layout, and the caller
    stores that report on the submission.
    """
    found = {"kernels": 0, "trajectory": 0, "effort": 0, "transcripts": 0,
             "unmeasured": 0}

    # ---- the kernel that was submitted -------------------------------------
    # Driven by the kernels ON DISK, unioned with the problems that produced
    # results -- not by results alone. A kernel whose re-time timed out has no
    # result rows at all, and keying off results would have dropped it here
    # too, so the one problem where the harness failed to measure a real
    # attempt would look exactly like a problem nobody opened.
    kernel_dir = run_dir / "kernels"
    on_disk = sorted(f.stem for f in kernel_dir.glob("*.py")) if kernel_dir.is_dir() else []
    for key in sorted(set(problems) | set(on_disk)):
        f = kernel_dir / f"{key}.py"
        if not f.exists():
            continue
        src = f.read_text(errors="replace")
        retimed = run_dir / "retimed" / f"{key}.json"
        ok, err = None, None
        if retimed.exists():
            try:
                doc = json.loads(retimed.read_text())
                ok = _int_or_none(doc.get("ok"))
                err = doc.get("error")
            except Exception:
                pass
        conn.execute(
            """INSERT OR REPLACE INTO run_kernel
               (submission_id,problem_key,source,n_lines,sha256,retime_ok,retime_error)
               VALUES (?,?,?,?,?,?,?)""",
            (sub_id, key, src, src.count("\n") + 1,
             hashlib.sha256(src.encode()).hexdigest(), ok, err))
        found["kernels"] += 1
        if key not in problems:
            found["unmeasured"] += 1

    # ---- per-problem cost and effort ---------------------------------------
    cost_report = run_dir / "cost-report.json"
    if cost_report.exists():
        doc = json.loads(cost_report.read_text())
        for p in doc.get("per_problem") or []:
            conn.execute(
                """INSERT OR REPLACE INTO run_effort
                   (submission_id,problem_key,cost_usd,wall_seconds,api_seconds,
                    turns,input_tokens,output_tokens,cache_write_tokens,
                    cache_read_tokens,harness_evals,kernel_changed,capped,
                    timed_out,gpu)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sub_id, p.get("problem"), p.get("cost_usd"), p.get("wall_seconds"),
                 p.get("api_seconds"), p.get("turns"), p.get("input_tokens"),
                 p.get("output_tokens"), p.get("cache_write_tokens"),
                 p.get("cache_read_tokens"), p.get("harness_evals"),
                 _int_or_none(p.get("kernel_changed")), _int_or_none(p.get("capped")),
                 _int_or_none(p.get("timed_out")),
                 None if p.get("gpu") is None else str(p.get("gpu"))))
            found["effort"] += 1

    # ---- the trajectory ----------------------------------------------------
    # Two sources, and they disagree in useful ways. `cost-report.json` holds a
    # ready-made series with `mean_score` already on the SOL scale; the
    # `eval-<ts>.json` files hold the timestamps and the kernel snapshots. The
    # eval files are authoritative for ORDER (they carry the clock), and the
    # cost report is merged in for the score, matched by position.
    scored_series = ((json.loads(cost_report.read_text()).get("trajectory") or {})
                     .get("per_problem") or {}) if cost_report.exists() else {}
    traj_root = run_dir / "trajectory"
    for key in problems:
        d = traj_root / key
        if not d.is_dir():
            continue
        evals = []
        for f in d.glob("eval-*.json"):
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            stamp = f.stem.split("-", 1)[1]
            evals.append((doc.get("_provenance", {}).get("utc") or "", stamp, doc))
        if not evals:
            continue
        evals.sort()
        t0 = _parse_utc(evals[0][0])
        series = scored_series.get(key) or []
        for i, (utc, stamp, doc) in enumerate(evals, 1):
            snap = d / f"kernel-{stamp}.py"
            src = snap.read_text(errors="replace") if snap.exists() else None
            t = _parse_utc(utc)
            scored = series[i - 1] if i - 1 < len(series) else {}
            conn.execute(
                """INSERT OR REPLACE INTO trajectory_eval
                   (submission_id,problem_key,n,utc,minutes_in,ok,all_passed,
                    passed,workloads,geomean_speedup,mean_score,kernel_sha,
                    kernel_source,kernel_lines)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sub_id, key, i, utc or None,
                 (t - t0).total_seconds() / 60 if (t and t0) else None,
                 _int_or_none(doc.get("ok")), _int_or_none(doc.get("all_passed")),
                 doc.get("passed"), doc.get("workloads"),
                 doc.get("geomean_speedup"), scored.get("mean_score"),
                 hashlib.sha256(src.encode()).hexdigest()[:12] if src else None,
                 src, (src.count("\n") + 1) if src else None))
        found["trajectory"] += 1

    # ---- transcripts: indexed, not inlined ---------------------------------
    for key in problems:
        f = run_dir / "transcripts" / f"{key}.jsonl"
        if not f.exists():
            continue
        n_lines = n_turns = 0
        tools: dict[str, int] = {}
        try:
            with f.open(errors="replace") as fh:
                for line in fh:
                    n_lines += 1
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if j.get("type") == "assistant":
                        n_turns += 1
                    for c in (j.get("message") or {}).get("content") or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            tools[c.get("name")] = tools.get(c.get("name"), 0) + 1
        except OSError:
            continue
        conn.execute(
            """INSERT OR REPLACE INTO transcript
               (submission_id,problem_key,path,bytes,n_lines,n_turns,tools_json)
               VALUES (?,?,?,?,?,?,?)""",
            (sub_id, key, str(f.resolve()), f.stat().st_size, n_lines, n_turns,
             json.dumps(tools)))
        found["transcripts"] += 1

    return found


def depth_note(found: dict, n_problems: int) -> str | None:   # noqa: D401
    """One sentence naming what this harness did not write down.

    Stated positively as a limitation of the RUN, because that is what it is.
    The alternative -- an empty panel -- reads as a broken page, and the run
    with the most problems is the one with the least depth.
    """
    missing = [name for name, n in (("a trajectory", found["trajectory"]),
                                    ("transcripts", found["transcripts"]),
                                    ("per-problem cost", found["effort"]))
               if n == 0]
    parts = []
    if missing:
        parts.append("This harness did not record " + ", ".join(missing[:-1])
                     + (" or " if len(missing) > 1 else "") + missing[-1] + ".")
    parts.append(f"{found['kernels']} submitted kernel"
                 f"{'s' if found['kernels'] != 1 else ''} kept.")
    # Not "N of M": the kernel count can legitimately EXCEED the number of
    # problems with results, which is the interesting case and which an
    # "x of y" phrasing renders as the nonsense "24 of 23".
    if found["unmeasured"]:
        n = found["unmeasured"]
        parts.append(f"{n} of them produced no measurement at all and so "
                     f"{'appear' if n != 1 else 'appears'} nowhere in the score.")
    return " ".join(parts) if (missing or found["unmeasured"]) else None


def ingest_variant_sources(conn, problems: dict[str, str]) -> int:
    """Regenerate the T_b formulations from each problem's own reference.

    They are source-to-source transforms (`reference/tb-candidates/variants.py`)
    with no torch import, so this is a pure text operation -- and it produces
    exactly the code task 06 compiled and timed, which is the point: a diff
    against a reconstruction nobody measured would be decoration.
    """
    spec = importlib.util.spec_from_file_location(
        "_tb_variants", ROOT / "reference" / "tb-candidates" / "variants.py")
    if spec is None or spec.loader is None:
        return 0
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n = 0
    for key, reference in problems.items():
        if not reference:
            continue
        for vname, transform in mod.VARIANTS.items():
            try:
                src = transform(reference)
            except Exception:
                continue          # a transform that cannot apply is not a row
            conn.execute(
                """INSERT OR REPLACE INTO variant_source
                   (problem_key,variant,source,n_lines) VALUES (?,?,?,?)""",
                (key, vname, src, src.count("\n") + 1))
            n += 1
    return n


def _int_or_none(v):
    return None if v is None else int(bool(v)) if isinstance(v, bool) else int(v)


def _parse_utc(s: str):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path(__file__).parent / "solbench.db")
    ap.add_argument("--agent-runs", type=Path, nargs="*", default=None,
                    help="extra directories holding <run>/scored.json, for runs "
                         "kept outside the repo. Also read from "
                         "SOLEXBENCH_AGENT_RUNS (colon-separated).")
    a = ap.parse_args()

    extra = list(a.agent_runs or [])
    env_roots = os.environ.get("SOLEXBENCH_AGENT_RUNS", "")
    extra += [Path(p) for p in env_roots.split(":") if p.strip()]

    manifest = json.loads(MANIFEST.read_text())

    # Build into a side file and swap it in at the end. The old path deleted the
    # live database and rebuilt in place, which gave every reader a window --
    # measured at 0.30s, and observed returning `rows=0` on the first attempt --
    # where the API answered 200 with an EMPTY leaderboard. Not an error a
    # client could detect: indistinguishable from "no submissions exist". The
    # window is short but it is a wrong answer, and it grows with every run
    # added. `os.replace` is atomic on POSIX, so a reader sees the whole old
    # database or the whole new one and never a half-built one.
    tmp = a.db.with_name(a.db.name + f".build-{os.getpid()}")
    for stale in _db_files(tmp):
        stale.unlink(missing_ok=True)
    conn = connect(tmp)
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    ingest_meta(conn, manifest, extra)
    ingest_problems(conn, manifest)
    n_vs = ingest_variant_sources(
        conn, {r["key"]: r["reference"]
               for r in conn.execute("SELECT key,reference FROM problem")})
    print(f"variant sources: {n_vs} regenerated from problem references")
    print("variants:")
    excluded = ingest_variants(conn, manifest)
    print("agent runs:")
    excluded.update(ingest_agent_runs(conn, extra))
    if excluded:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     ("excluded_submissions", json.dumps(excluded)))
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("problem", "workload", "submission", "result",
                        "run_kernel", "run_effort", "trajectory_eval", "transcript")}

    # Fold the WAL back into the single file before swapping. `os.replace` moves
    # one inode; a `-wal` sidecar left under the build name would strand
    # committed pages outside the file that gets published. Nothing writes to
    # the served database -- it is rebuilt and swapped, never updated in place --
    # so WAL buys it nothing anyway.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()

    os.replace(tmp, a.db)
    # SQLite keys sidecars to the FILE NAME, not the inode, so a `-wal` left by
    # the previous database now sits next to a new one that has nothing to do
    # with it. The new header says DELETE so it would be ignored, but leaving it
    # to be ignored forever is worse than removing it. Readers still holding the
    # old inode keep their open fds; unlink does not break them.
    for stale in _db_files(a.db)[1:]:
        stale.unlink(missing_ok=True)

    print(f"\n{a.db}: " + ", ".join(f"{n} {t}s" for t, n in counts.items() if n))
    return 0


def _db_files(db: Path) -> list[Path]:
    """The database and the two sidecars SQLite may create beside it."""
    return [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]


if __name__ == "__main__":
    raise SystemExit(main())
