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


def ingest_meta(conn, manifest: dict) -> None:
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
        # moment they move; without these it goes stale *silently*, which is the
        # one failure a leaderboard cannot afford. `app.py` compares
        # `repo_git_sha` against the working tree and says so in the header.
        "db_built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_git_sha": git_sha(ROOT),
        "total_problems": manifest["problem_set"]["total_in_dataset"],
        "scoreable_problems": manifest["problem_set"]["scoreable_problems"],
        "scoreable_workloads": manifest["stats"]["scoreable_workloads"],
        "expected_by_category": json.dumps(manifest["problem_set"]["expected_by_category"]),
        "bound_sources": json.dumps(manifest.get("bound_sources", {})),
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
            rows.append((sub_id, pkey, uuid, "PASSED" if all_passed else "FAILED",
                         ms, sol_score(ms, t_b, t_sol), 0, label))
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
    for scored in scored_files:
        doc = json.loads(scored.read_text())
        run_id = doc.get("run_id", scored.parent.name)
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
            cost_usd=doc.get("total_cost_usd"),
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
        bad = sorted({r["problem"] for r in doc.get("results", [])
                      if r.get("bound_violation")})
        if bad:
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                         ("problems_with_invalid_bound", json.dumps(bad)))
            print(f"  agent {run_id}: {len(rows)} workloads; "
                  f"{len(bad)} problem(s) have a bound a real kernel beat: {bad}")
        else:
            print(f"  agent {run_id}: {len(rows)} scored workloads")
    return excluded


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
    if a.db.exists():
        a.db.unlink()
    conn = connect(a.db)
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    ingest_meta(conn, manifest)
    ingest_problems(conn, manifest)
    print("variants:")
    excluded = ingest_variants(conn, manifest)
    print("agent runs:")
    excluded.update(ingest_agent_runs(conn, extra))
    if excluded:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     ("excluded_submissions", json.dumps(excluded)))
    conn.commit()

    n_p = conn.execute("SELECT COUNT(*) FROM problem").fetchone()[0]
    n_w = conn.execute("SELECT COUNT(*) FROM workload").fetchone()[0]
    n_s = conn.execute("SELECT COUNT(*) FROM submission").fetchone()[0]
    n_r = conn.execute("SELECT COUNT(*) FROM result").fetchone()[0]
    print(f"\n{a.db}: {n_p} problems, {n_w} workloads, {n_s} submissions, {n_r} results")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
