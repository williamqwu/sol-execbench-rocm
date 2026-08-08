#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the leaderboard database from the artifacts.

    python leaderboard/ingest.py            # -> leaderboard/db/solbench-<PART>.db

One database per part, named after the part the manifest was measured on. A
score from MI350X and one from MI355X are not comparable, and keeping them in
separate files means no query can mix them by accident. `--db` still names an
explicit path for a scratch build or the legacy `leaderboard/solbench.db`.

The database is disposable. Everything in it comes from
`artifacts/09/manifest-v1.json`, the dataset definitions, the task 06 variant
sweep, any agent runs under `artifacts/10/`, and any further run roots listed in
`leaderboard/sources.json` (read by default -- see `sources.json.example`).
Rerun this after any artifact changes; never edit the database by hand.

Two things it will refuse to do, both loudly:

* ingest a run measured on a part other than the manifest's, because scoring an
  MI355X latency against MI350X bounds is wrong in a way no reader can see; and
* publish a board that has lost a submission the previous one had, unless
  `--allow-drop` says the retirement is deliberate.

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
# Machine-local config, read BY DEFAULT, listing the agent-run roots that do not
# live in the repo. It is the fix for a defect that has now been introduced four
# times (STATE.md D24): every caller that shelled out to a bare `ingest.py` --
# the staleness banner, `worker.py`, a human following the README -- silently
# dropped every out-of-repo run off the board. Patching call sites does not
# work, because the next caller is not written yet. Making the default
# non-lossy does: after this, `ingest.py` and `ingest.py --agent-runs <the same
# roots>` are the same build, so "rebuild" cannot mean two different things.
# Not tracked (paths are machine-local); `sources.json.example` is.
SOURCES = Path(__file__).parent / "sources.json"

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
    "submitted-apitest": "Not a result. Two problems, and the 'kernel' "
              "submitted for each is that problem's own reference "
              "implementation -- it exists to exercise the write path end to "
              "end (POST -> queue -> worker -> GPU 0 -> score -> rebuild) and "
              "it scores below 0.5 by construction, since T_b is the fastest "
              "of four formulations of the same code. Kept as artifacts under "
              "artifacts/10/submitted-apitest/ and as jobs 1-2 in the queue. "
              "Delete this entry to put it on the board.",
}

# Of the exclusions above, the ones that are nevertheless INGESTED and merely
# hidden (`submission.board_visible = 0`). Their results, kernels, trajectory,
# effort and transcripts all land; only `leaderboard_rows()` skips them.
#
# Excluding a run from the ranking and deleting its evidence are two decisions
# and only the first was ever made. `docs/agent-baseline.md` walks a reader to
# pilot8's trajectory and its 25 bound violations, and until this the link led
# to a database that had never heard of it. Nothing on the board moves: the
# flag is read by the board query, not by the ingest of results.
#
# `submitted-apitest` stays fully out. It is not a run of the benchmark -- the
# "kernel" for each of its two problems is that problem's own reference -- so
# there is no evidence in it for a reader to follow, only a write-path check.
INGESTED_BUT_HIDDEN = {"pilot8"}

# The setup a run belongs to, hardcoded because inferring "same setup" from a
# model name is a guess: two runs of one model under different harnesses are
# not trials of the same thing. A run absent from this table is a group of one
# and gets no switcher.
TRIAL_GROUPS = {
    "pilot8":            ("agent-opus5", "Claude-Opus-5 · agent sandbox harness"),
    "opus5-budget100":   ("agent-opus5", "Claude-Opus-5 · agent sandbox harness"),
}


def load_sources(path: Path) -> list[Path]:
    """The extra agent-run roots from `sources.json`. Absent file -> no roots.

    Malformed is a hard error, never an empty list. A config that silently
    resolves to nothing is worse than no config at all: it looks like it is
    being honoured, and the symptom -- a run missing from the board -- is the
    exact symptom this file exists to prevent.

    Relative paths resolve against the repo root, not the CWD, because the
    worker, the tests and a human all invoke this from different directories
    and a root that means three things is not a config.
    """
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON ({exc}). Refusing to build a "
                         f"board from a config that cannot be read; fix it, or "
                         f"pass a bare --agent-runs to build from artifacts/10 "
                         f"alone.")
    roots = doc.get("agent_run_roots")
    if roots is None:
        raise SystemExit(
            f"{path}: no 'agent_run_roots' key; found {sorted(doc)}. See "
            f"{SOURCES.name}.example.")
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        raise SystemExit(f"{path}: 'agent_run_roots' must be a list of path "
                         f"strings, found {type(roots).__name__}.")
    return [Path(r) if Path(r).is_absolute() else (ROOT / r) for r in roots]


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


def manifest_part(manifest: dict) -> str | None:
    """The part the manifest itself was measured on.

    v1 carries no explicit `_provenance.part`, so it comes from the device
    string; a future manifest that states it wins. This is what `--part` is
    checked against and what names the database file, so an unresolvable part
    is a hard stop rather than a database called `solbench-None.db`.
    """
    prov = manifest.get("_provenance", {})
    device = ((prov.get("torch") or {}).get("devices") or [None])[0]
    return prov.get("part") or part_of(device)


def ingest_meta(conn, manifest: dict, part: str,
                extra_roots: list[Path] | None = None) -> None:
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
        # The part this whole database is about. One database per part, because
        # the two parts' numbers are not comparable and a filter over one table
        # is one bad WHERE clause away from mixing them.
        "part": part,
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
            cost_usd,wall_seconds,gpu,group_slug,group_name,trial_label,
            constraint_json,board_visible,exclusion_reason,part,variant,
            depth_note)
           VALUES (:slug,:name,:kind,:author,:model,:created_utc,:notes,
                   :provenance_json,:cost_usd,:wall_seconds,:gpu,:group_slug,
                   :group_name,:trial_label,:constraint_json,:board_visible,
                   :exclusion_reason,:part,:variant,:depth_note)""",
        # `trial_n` is deliberately not settable here: it is a position within
        # a group and cannot be known until every member of the group has been
        # read. `assign_trial_numbers()` fills it once, at the end.
        {"author": None, "model": None, "created_utc": None, "notes": None,
         "provenance_json": None, "cost_usd": None, "wall_seconds": None,
         "gpu": None, "group_slug": None, "group_name": None,
         "trial_label": None, "constraint_json": None, "board_visible": 1,
         "exclusion_reason": None, "part": None, "variant": None,
         "depth_note": None, **kw})
    return cur.lastrowid


def assign_trial_numbers(conn) -> None:
    """`trial_n`: 1-based within the group, in `created_utc` order.

    Ordered in SQL on the ISO-8601 string, which is chronological because every
    stamp is UTC in the same format (`provenance.py` writes no other). `id`
    breaks ties so the numbering is stable across rebuilds rather than
    depending on directory iteration order.
    """
    groups = [r[0] for r in conn.execute(
        "SELECT DISTINCT group_slug FROM submission WHERE group_slug IS NOT NULL")]
    for group in groups:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM submission WHERE group_slug=? ORDER BY created_utc, id",
            (group,))]
        conn.executemany("UPDATE submission SET trial_n=? WHERE id=?",
                         [(n, i) for n, i in enumerate(ids, 1)])


def run_constraint(run_dir: Path, sessions: dict | None) -> tuple[str | None, str | None]:
    """The constraint that distinguishes one trial of a setup from another.

    Read from the run's own artifact, never from the group table: pilot8 is
    "$8 / problem" because its `run.json` says `budget_usd_per_session: 8.0`,
    and if a future rerun changes that number the label changes with it.

    `agent_baseline.py` also takes a `--timeout` and writes it nowhere, so a
    run of that harness still gets no cap here. An explicit null would read as
    "no timeout", which is a different claim from "not recorded".

    A *fleet* run is the other case, and it is capped in seconds rather than
    dollars: `scripts/import_fleet_depth.py` copies `spec.eta_s` out of the J2
    job spec into `wall_cap_seconds`, and the daemon SIGTERMs at exactly that.
    It is the same fact as pilot8's dollar budget said in the units it was
    imposed in, and it earns a label for the same reason -- a session that was
    stopped did not choose when to stop.
    """
    doc: dict = {}
    for name in ("cost-report.json", "run.json"):
        f = run_dir / name
        if f.exists():
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            if (doc.get("budget_usd_per_session") is not None
                    or doc.get("wall_cap_seconds") is not None):
                break
    budget = doc.get("budget_usd_per_session")
    wall_cap = doc.get("wall_cap_seconds")
    if budget is None and wall_cap is not None:
        hours = wall_cap / 3600
        label = (f"{hours:g} h / problem" if wall_cap % 3600 == 0
                 else f"{wall_cap:g} s / problem")
        return label, json.dumps({"wall_cap_seconds": wall_cap})
    if budget is None:
        return None, None
    # The field says "per session". These harnesses open exactly one session
    # per problem -- `run.json.sessions` is keyed by problem key -- so "per
    # problem" is the same cap said in the reader's units. A harness that ever
    # runs two sessions on one problem gets the field's own wording instead.
    per = "problem" if sessions and set(sessions) == {
        s.get("problem") for s in sessions.values()} else "session"
    return f"${budget:g} / {per}", json.dumps({"budget_usd_per_session": budget})


def run_part(run_dir: Path) -> str | None:
    """The part a run was measured on, from the re-time that produced its scores.

    The `retimed/*.json` files are the only source: they are written by the
    authoritative GPU-0 evaluation, so their `torch.devices` is the device the
    numbers came off. `scored.json`'s own provenance is written by the driver
    process, which frequently has no torch at all (pilot8, glm-run1 and
    opus5-budget100 all record `torch.available false`), and the manifest's part
    is a fact about the bounds, not about this run -- taking either as a
    stand-in is how a run acquires a part it was never measured on.

    That is not hypothetical: this function used to end by reading
    `scored.json`'s provenance, which is exactly what the paragraph above
    forbids. It never fired on any run in the tree -- all four carry re-time
    provenance -- but it was one torch-bearing driver process away from
    labelling a run with the host it was *scored on* instead of the host it was
    *measured on*. Removed rather than documented, since a rule that has to be
    remembered at the call site is not a rule.

    None means "this run does not say", which is not the same as "MI350X" and
    the caller must refuse it rather than assume.
    """
    parts = set()
    for f in sorted((run_dir / "retimed").glob("*.json")):
        try:
            prov = (json.loads(f.read_text()).get("_provenance") or {})
        except Exception:
            continue
        for dev in (prov.get("torch") or {}).get("devices") or []:
            parts.add(part_of(dev))
    parts.discard(None)
    if len(parts) > 1:
        raise SystemExit(
            f"{run_dir}: re-timed on more than one part {sorted(parts)}. The "
            f"workloads in this run are not comparable to each other, let "
            f"alone to the board; refusing to ingest it under either part.")
    return parts.pop() if parts else None


def ingest_variants(conn, manifest: dict, part: str) -> dict:
    """The four PyTorch variants, from the task 06 sweep.

    Authoritative (GPU 0) timings win where they exist; the wider sweep on GPUs
    1-7 fills the rest. Which one produced each number goes in `note`, because
    mixing them silently is exactly the kind of thing that makes a leaderboard
    unfalsifiable.
    """
    b = bounds(manifest)
    prov = manifest.get("_provenance", {})
    # Every scoreable workload of each problem, so a variant that produced no
    # per-workload record at all can still be shown as having been run on them.
    by_problem: dict[str, list[str]] = {}
    for pkey, uuid in b:
        by_problem.setdefault(pkey, []).append(uuid)

    # PER WORKLOAD, from both halves of the artifact.
    #
    # `latency_ms_by_workload` holds exactly the workloads that PASSED; the ones
    # that did not are in `failures`, with a status each. This used to read only
    # the first and stamp the whole problem with the variant's `all_passed`
    # flag, which got two separate things wrong and neither was visible:
    #
    #   * 1,239 workloads that passed were published as FAILED with no score,
    #     because some OTHER workload of the same problem failed (D28);
    #   * a problem where the variant passed nothing produced no rows at all,
    #     so it read as NEVER ATTEMPTED. That is the whole of the board's
    #     "torch.compile: 213/220 problems" -- it ran all 220 and passed
    #     nothing on 7 of them.
    #
    # Failed workloads now get a row carrying the status the harness recorded
    # and no latency, because none was measured for them.
    per_variant: dict[str, dict] = {}
    for src, label in ((CANDIDATES, "sweep gpu1-7"), (AUTHORITATIVE, "authoritative gpu0")):
        if not src.exists():
            continue
        for f in sorted(src.glob("*.json")):
            doc = json.loads(f.read_text())
            pkey = doc.get("problem") or f.stem
            for vname, v in (doc.get("variants") or {}).items():
                store = per_variant.setdefault(vname, {})
                for fail in v.get("failures") or []:
                    uuid = fail.get("workload_uuid")
                    if uuid:
                        store[(pkey, uuid)] = (
                            None, label, fail.get("status") or "FAILED")
                for uuid, ms in (v.get("latency_ms_by_workload") or {}).items():
                    # authoritative overwrites sweep, by iteration order
                    store[(pkey, uuid)] = (ms, label, "PASSED")
                # A variant that died before measuring anything -- a timeout, a
                # crashed driver -- records an `error` and neither list. It was
                # still run on every workload of the problem, and saying so is
                # the difference between "this variant cannot do it" and "this
                # variant was never tried", which are not the same claim.
                if (v.get("error") and not v.get("failures")
                        and not v.get("latency_ms_by_workload")):
                    for uuid in by_problem.get(pkey, []):
                        store.setdefault((pkey, uuid), (None, label, "ERROR"))

    excluded: list[str] = []
    for vname, entries in sorted(per_variant.items()):
        # v5_compile_contiguous ran on all 235 problems and passed zero
        # workloads (torch.compile raises during tracing). A row with 0%
        # coverage and no score is noise on a leaderboard, but dropping it
        # silently would hide a variant that was actually run -- so it is
        # excluded here and recorded in `meta`.
        #
        # The test is "passed nothing", not "produced no rows". It used to be
        # the latter and that was the same statement only because failures were
        # being discarded; the moment they were kept, v5 acquired 3,717 rows and
        # walked back onto the board with a score of 0.0000.
        if not any(st == "PASSED" for _ms, _lab, st in entries.values()):
            excluded.append(vname)
            print(f"  {vname}: 0 scored workloads -- excluded, recorded in meta")
            continue
        sub_id = add_submission(
            conn,
            slug=f"baseline-{vname.replace('_','-')}",
            name=VARIANT_LABELS.get(vname, vname),
            kind="reference_variant",
            author="sol-execbench-rocm",
            model=None,
            created_utc=prov.get("utc"),
            notes=("One of the four PyTorch formulations that T_b is derived from. "
                   "T_b is the fastest of them per workload, so the winning variant "
                   "scores exactly 0.5 there by construction. These are not agent "
                   "submissions and are not comparable to one."),
            provenance_json=json.dumps(prov),
            variant=vname,
            # Not "we failed to record this". A variant is one deterministic
            # source transform, compiled and timed once by task 06: there are no
            # rounds to plot and no tokens to bill. The generic fallbacks say
            # "no trajectory was recorded", which reads as a gap in the harness
            # and invites someone to go looking for the file. There is no file.
            depth_note=("A reference variant is a single deterministic source "
                        "transform, compiled and timed once by the task 06 sweep. "
                        "There is no trajectory because nothing iterated, and no "
                        "cost because no model was called — both are zero by "
                        "construction, not unrecorded."),
            gpu="0 (authoritative) / 1-7 (sweep)",
            # The variants are the manifest's own timings, so their part is the
            # manifest's part by definition, not by inference.
            part=part)
        rows = []
        for (pkey, uuid), (ms, label, status) in entries.items():
            bound = b.get((pkey, uuid))
            if not bound:
                continue
            t_sol, t_b = bound
            passed = status == "PASSED"
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
            rows.append((sub_id, pkey, uuid, status,
                         ms if passed else None,
                         sol_score(ms, t_b, t_sol) if (passed and ms) else None,
                         0, label))
        conn.executemany(
            """INSERT OR REPLACE INTO result
               (submission_id,problem_key,workload_uuid,status,latency_ms,
                score,flagged,note) VALUES (?,?,?,?,?,?,?,?)""", rows)
        print(f"  {vname}: {len(rows)} scored workloads")

    return {v: "Ran on all 235 problems and passed 0 workloads "
               "(torch.compile raises during tracing)." for v in excluded}


def check_run_part(run_dir: Path, run_id: str, measured: str | None,
                   part: str) -> str:
    """Refuse a run that was not measured on this database's part.

    The most damaging error this project can make, because it is invisible
    downstream: MI355X has a different power cap, so a different F_LOCK, so a
    different T_SOL and a different T_b. An MI355X latency scored against
    MI350X bounds produces a plausible number on every row and a rank nobody
    can tell is wrong. A reviewer demonstrated it -- relabelling glm-run1 as
    MI355X left it sitting at #5 on the MI350X board, unmarked.

    So it fails here, at the earliest point that knows both parts, and it fails
    hard. A warning would be read once and scrolled past; a skip would remove
    the run from the board without removing it from the reader's expectations,
    which is D24 wearing a different hat. Build the other part's database from
    the other part's manifest instead -- that is what one-database-per-part is
    for.
    """
    if measured is None:
        raise SystemExit(
            f"{run_dir}: run '{run_id}' does not say which part it was measured "
            f"on -- no `retimed/*.json` carries a torch device name. Its scores "
            f"cannot be shown to belong on the {part} board, and assuming they "
            f"do is the error this check exists to prevent. Re-time it on the "
            f"target part, or drop the run directory.")
    if measured != part:
        raise SystemExit(
            f"{run_dir}: run '{run_id}' was measured on {measured}, but this "
            f"database is built from a {part} manifest ({MANIFEST.name}). "
            f"Different part -> different F_LOCK -> different T_SOL and T_b, so "
            f"its latencies are not scoreable against these bounds and every "
            f"number it produced here would look plausible. Build the {measured} "
            f"database from a {measured} manifest instead.")
    return measured


def ingest_agent_runs(conn, part: str, extra_roots: list[Path] | None = None) -> dict:
    """Any `<root>/<run>/scored.json` written by `agent_score.py`.

    Extra roots matter because a run directory does not have to live in the
    repo -- a scratch experiment under $HOME is a legitimate place for one, and
    globbing only `artifacts/10` silently omits it from the board rather than
    reporting that it was skipped.

    `part` is the manifest's part, and every run read here must match it; see
    `check_run_part`.
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
        hidden = run_id in BOARD_EXCLUSIONS
        if hidden:
            excluded[run_id] = BOARD_EXCLUSIONS[run_id]
            if run_id not in INGESTED_BUT_HIDDEN:
                print(f"  agent {run_id}: excluded from board (smoke test), "
                      f"recorded in meta")
                continue
            print(f"  agent {run_id}: off the board (board_visible=0), "
                  f"ingested in full; reason recorded in meta")
        # Before anything is written for this run, not after: a part mismatch
        # invalidates every row the run would produce, so there is no partial
        # ingest worth keeping.
        measured_part = check_run_part(
            scored.parent, run_id, run_part(scored.parent), part)
        group_slug, group_name = TRIAL_GROUPS.get(run_id, (None, None))
        trial_label, constraint_json = run_constraint(
            scored.parent, (run_json(scored.parent) or {}).get("sessions"))
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
            group_slug=group_slug, group_name=group_name,
            trial_label=trial_label, constraint_json=constraint_json,
            board_visible=0 if hidden else 1,
            exclusion_reason=BOARD_EXCLUSIONS[run_id] if hidden else None,
            part=measured_part,
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
        windows = ingest_run_window(conn, sub_id, scored.parent)
        print("    window: " + (", ".join(f"{n} {src}" for src, n
                                          in sorted(windows.items()))
                                or "no timestamp evidence, no rows"))

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
                    # Two harnesses, two record shapes, counted into the same
                    # two numbers. Neither file is rewritten to match the
                    # other: the transcript is served verbatim and a
                    # translation shown under that word would be a different
                    # artifact wearing its name.
                    #
                    #   claude-code : {"type":"assistant",
                    #                  "message":{"content":[{"type":"tool_use",
                    #                                         "name":...}]}}
                    #   codex       : {"type":"item.completed",
                    #                  "item":{"type":"assistant_message"
                    #                          | "command_execution"
                    #                          | "file_change" | ...}}
                    if j.get("type") == "assistant":
                        n_turns += 1
                    # `message` is a dict in the claude-code record and a bare
                    # string in some codex ones (`{"type":"error","message":
                    # "Model metadata for `GLM-5.2` not found..."}`). Reading
                    # `.get` off it raised AttributeError mid-ingest and took
                    # the whole board rebuild down with it -- one malformed
                    # line in one transcript, and nothing publishes.
                    msg = j.get("message")
                    for c in (msg.get("content") if isinstance(msg, dict) else None) or []:
                        if isinstance(c, dict) and c.get("type") == "tool_use":
                            tools[c.get("name")] = tools.get(c.get("name"), 0) + 1
                    item = j.get("item")
                    if j.get("type") == "item.completed" and isinstance(item, dict):
                        kind = item.get("type")
                        if kind in ("assistant_message", "agent_message",
                                    "reasoning"):
                            n_turns += int(kind != "reasoning")
                        elif kind:
                            tools[kind] = tools.get(kind, 0) + 1
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


def ingest_run_window(conn, sub_id: int, run_dir: Path) -> dict[str, int]:
    """When this run worked on each problem, and on whose clock.

    Two sources, in precedence order, and never anything else:

    1. `first_last_eval` -- MIN/MAX over the trajectory rows JUST ingested, so
       the window can never disagree with the series the page plots next to it.
       It is a window inside the session: the agent was working before its
       first eval and after its last, which is why the source travels with the
       number instead of being resolved here.
    2. `retime_only` -- the authoritative GPU-0 re-time's provenance stamp,
       written when the re-time FINISHED (`agent_eval.py` stamps on write), so
       it is a finish with no start. It says when the kernel was scored, not
       when it was worked on. `INSERT OR IGNORE` makes it a fallback: a problem
       that has evals keeps them.

    A harness with neither gets no row. `run_window` has no "unknown" source
    because there is nothing for one to mean: absence already says it.
    """
    conn.execute(
        """INSERT OR REPLACE INTO run_window
             (submission_id,problem_key,started_utc,finished_utc,source)
           SELECT submission_id, problem_key, MIN(utc), MAX(utc), 'first_last_eval'
             FROM trajectory_eval
            WHERE submission_id=? AND utc IS NOT NULL
            GROUP BY problem_key""", (sub_id,))

    # Driven by the kernels this run left behind rather than by the results,
    # for the reason `ingest_depth` unions them: glm-run1's FlashInfer-Bench__014
    # produced no result rows at all because its re-time timed out, and the
    # timestamp of that failed attempt is exactly the evidence that it WAS
    # attempted. A retimed file for a problem with no kernel would be a stale
    # leftover and gets no row.
    attempted = [r[0] for r in conn.execute(
        "SELECT problem_key FROM run_kernel WHERE submission_id=?", (sub_id,))]
    for key in attempted:
        f = run_dir / "retimed" / f"{key}.json"
        if not f.exists():
            continue
        try:
            utc = (json.loads(f.read_text()).get("_provenance") or {}).get("utc")
        except Exception:
            continue
        if not utc:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO run_window
               (submission_id,problem_key,started_utc,finished_utc,source)
               VALUES (?,?,NULL,?,'retime_only')""", (sub_id, key, utc))

    return {r[0]: r[1] for r in conn.execute(
        """SELECT source, COUNT(*) FROM run_window
            WHERE submission_id=? GROUP BY source""", (sub_id,))}


def run_json(run_dir: Path) -> dict | None:
    f = run_dir / "run.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


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
    ap.add_argument("--db", type=Path, default=None,
                    help="explicit output path. Defaults to "
                         "leaderboard/db/solbench-<PART>.db.")
    ap.add_argument("--part", default=None,
                    help="assert the part this manifest was measured on. Not an "
                         "override: a value that disagrees with the manifest is "
                         "an error.")
    ap.add_argument("--agent-runs", type=Path, nargs="*", default=None,
                    help="extra directories holding <run>/scored.json, for runs "
                         "kept outside the repo. OVERRIDES leaderboard/"
                         "sources.json. Also read from SOLEXBENCH_AGENT_RUNS "
                         "(colon-separated), which is additive.")
    ap.add_argument("--sources", type=Path, default=SOURCES,
                    help=f"config listing the extra agent-run roots. Read by "
                         f"default from {SOURCES}; a missing file is fine, a "
                         f"malformed one is an error.")
    ap.add_argument("--allow-drop", action="store_true",
                    help="permit the rebuild to publish a board with fewer "
                         "submissions than the one it replaces. Required to "
                         "retire a run on purpose; without it, a rebuild that "
                         "would lose a submission refuses.")
    a = ap.parse_args()

    # Precedence: an explicit --agent-runs replaces the config (it is the
    # narrower, more deliberate statement), the environment adds to whichever
    # won. Every root is printed with where it came from, because "which roots
    # did this build read" is the question behind every instance of D24.
    if a.agent_runs is not None:
        extra = [(p, "--agent-runs") for p in a.agent_runs]
    else:
        extra = [(p, str(a.sources)) for p in load_sources(a.sources)]
    extra += [(Path(p), "SOLEXBENCH_AGENT_RUNS")
              for p in os.environ.get("SOLEXBENCH_AGENT_RUNS", "").split(":")
              if p.strip()]
    for p, src in extra:
        print(f"extra agent-run root: {p}  (from {src})")
    extra = [p for p, _ in extra]

    manifest = json.loads(MANIFEST.read_text())

    # One database per part, named after the part, because the two parts'
    # numbers are not comparable and the safest place to enforce that is the
    # filesystem: a query cannot accidentally join across two files.
    #
    # `--part` asserts rather than sets. Letting it relabel the manifest would
    # produce `solbench-MI355X.db` full of MI350X timings -- every number
    # plausible, nothing detectably wrong, which is precisely the failure the
    # per-part split exists to prevent.
    part = manifest_part(manifest)
    if part is None:
        raise SystemExit(
            f"{MANIFEST}: cannot tell which part this was measured on "
            f"(_provenance.part absent and no MI* device name). Refusing to "
            f"build a database that cannot name its part.")
    if a.part and a.part != part:
        raise SystemExit(
            f"--part {a.part} disagrees with the manifest, which was measured "
            f"on {part}. --part asserts, it does not relabel; measure on "
            f"{a.part} and build from that manifest instead.")
    db = a.db or Path(__file__).parent / "db" / f"solbench-{part}.db"

    # Build into a side file and swap it in at the end. The old path deleted the
    # live database and rebuilt in place, which gave every reader a window --
    # measured at 0.30s, and observed returning `rows=0` on the first attempt --
    # where the API answered 200 with an EMPTY leaderboard. Not an error a
    # client could detect: indistinguishable from "no submissions exist". The
    # window is short but it is a wrong answer, and it grows with every run
    # added. `os.replace` is atomic on POSIX, so a reader sees the whole old
    # database or the whole new one and never a half-built one.
    tmp = db.with_name(db.name + f".build-{os.getpid()}")
    for stale in _db_files(tmp):
        stale.unlink(missing_ok=True)
    try:
        return build(a, manifest, part, extra, db, tmp)
    except BaseException:
        # A build that refuses -- or crashes -- must leave nothing behind. The
        # half-built file is never published, but `solbench-MI350X.db.build-123`
        # sitting next to the live board is one `mv` away from being, and it
        # contains exactly the rows the refusal said must not be published.
        for stale in _db_files(tmp):
            stale.unlink(missing_ok=True)
        raise


def build(a, manifest: dict, part: str, extra: list[Path],
          db: Path, tmp: Path) -> int:
    conn = connect(tmp)
    conn.executescript((Path(__file__).parent / "schema.sql").read_text())

    ingest_meta(conn, manifest, part, extra)
    ingest_problems(conn, manifest)
    n_vs = ingest_variant_sources(
        conn, {r["key"]: r["reference"]
               for r in conn.execute("SELECT key,reference FROM problem")})
    print(f"variant sources: {n_vs} regenerated from problem references")
    print("variants:")
    excluded = ingest_variants(conn, manifest, part)
    print("agent runs:")
    excluded.update(ingest_agent_runs(conn, part, extra))
    assign_trial_numbers(conn)
    for g in conn.execute("""SELECT group_slug, COUNT(*) FROM submission
                              WHERE group_slug IS NOT NULL
                              GROUP BY group_slug""").fetchall():
        print(f"  trial group {g[0]}: {g[1]} trial(s) -- " + ", ".join(
            f"#{r[0]} {r[1]} ({r[2] or 'constraint not recorded'})"
            for r in conn.execute(
                """SELECT trial_n, slug, trial_label FROM submission
                    WHERE group_slug=? ORDER BY trial_n""", (g[0],))))
    if excluded:
        conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     ("excluded_submissions", json.dumps(excluded)))
    conn.commit()

    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("problem", "workload", "submission", "result",
                        "run_kernel", "run_effort", "trajectory_eval", "transcript",
                        "run_window")}
    built_slugs = {r[0] for r in conn.execute("SELECT slug FROM submission")}

    # Fold the WAL back into the single file before swapping. `os.replace` moves
    # one inode; a `-wal` sidecar left under the build name would strand
    # committed pages outside the file that gets published. Nothing writes to
    # the served database -- it is rebuilt and swapped, never updated in place --
    # so WAL buys it nothing anyway.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()

    lost = published_slugs(db) - built_slugs
    if lost and not a.allow_drop:
        for stale in _db_files(tmp):
            stale.unlink(missing_ok=True)
        raise SystemExit(
            f"refusing to publish {db}: this build has no {sorted(lost)}, which "
            f"the board it would replace does have. The usual cause is a rebuild "
            f"that did not read the roots the last one did -- this build read "
            f"{[str(p) for p in extra] or 'no extra roots'}; the published board "
            f"was built from "
            f"{json.loads(published_meta(db).get('input_extra_roots') or '[]')}. "
            f"Add them to {SOURCES} (see {SOURCES.name}.example), or pass "
            f"--allow-drop if the run is being retired on purpose. The existing "
            f"board is untouched.")
    if lost:
        print(f"\n--allow-drop: dropping {sorted(lost)} from the board")

    os.replace(tmp, db)
    # SQLite keys sidecars to the FILE NAME, not the inode, so a `-wal` left by
    # the previous database now sits next to a new one that has nothing to do
    # with it. The new header says DELETE so it would be ignored, but leaving it
    # to be ignored forever is worse than removing it. Readers still holding the
    # old inode keep their open fds; unlink does not break them.
    for stale in _db_files(db)[1:]:
        stale.unlink(missing_ok=True)

    print(f"\n{db}: " + ", ".join(f"{n} {t}s" for t, n in counts.items() if n))
    return 0


def published_slugs(db: Path) -> set[str]:
    """The submissions on the board being replaced. Missing board -> empty set.

    Read read-only and defensively: this guard must never be the reason a first
    build, or a build over a truncated file, fails. It exists to catch the
    opposite case -- a complete board quietly becoming a smaller one.
    """
    if not db.is_file():
        return set()
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return {r[0] for r in conn.execute("SELECT slug FROM submission")}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


def published_meta(db: Path) -> dict:
    """`meta` of the board being replaced, for the refusal message only."""
    if not db.is_file():
        return {}
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return {r[0]: r[1] for r in conn.execute("SELECT key,value FROM meta")}
        finally:
            conn.close()
    except sqlite3.Error:
        return {}


def _db_files(db: Path) -> list[Path]:
    """The database and the two sidecars SQLite may create beside it."""
    return [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]


if __name__ == "__main__":
    raise SystemExit(main())
