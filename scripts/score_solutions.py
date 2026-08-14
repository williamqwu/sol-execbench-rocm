#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — score harvested agent solutions from a pristine tree.

    python scripts/score_solutions.py --run-id pilot-01

For every session in a run: load the solution the agent left behind, screen it,
re-evaluate it on the **authoritative GPU** one problem at a time, and write one
score record per workload.

Why re-evaluate at all, when ``./verify`` already ran it? Three reasons, and each
of them has bitten this project or its upstream:

1. ``./verify`` ran on a pool GPU with seven busy neighbours. Task 01 measures
   how much that matters; whatever the answer, authoritative timing is pinned to
   one GPU and every timing artifact records which (STATE.md, *Decisions taken*).
2. The agent could have influenced its own verification. It runs as a local
   process with a writable filesystem, so the only defensible position is that
   nothing it produced is trusted to score itself.
3. ``./verify`` is capped at a handful of attempts, so the last thing an agent
   ran is often not the last thing it *wrote*.

Scoring is deliberately serial. Two evaluations sharing a GPU inflate each other
and, as deviation D11 records, nothing in the output would say so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents.gpu_pool import AUTHORITATIVE_GPU  # noqa: E402
from solexbench_agents.scoring import (  # noqa: E402
    ScoreBasis,
    clock_lock_state,
    compare_digests,
    headroom_fraction,
    reference_copy_verdict,
    resolve_basis,
    sol_score,
    tree_digest,
)


def load_manifest_bounds(path: Path) -> tuple[dict, dict, dict]:
    """(t_sol, t_b, meta) from a frozen scoring manifest.

    Preferred over the raw artifacts, because a score is only meaningful inside
    one manifest version: the README says so, and the manifest is the thing that
    pins `T_SOL`, `T_b` and the tolerances together with the stack that produced
    them. Reading the raw artifacts instead lets a score be assembled from a
    `T_SOL` measured at one clock and a `T_b` measured at another, with nothing
    recording that it happened.

    Returned in the shape ``workload_bound`` already expects, so the caller does
    not care which source it got.
    """
    if not path.exists():
        return {}, {}, {}
    doc = json.loads(path.read_text())
    problems = doc.get("problems") or {}
    t_sol: dict = {}
    t_b: dict = {}
    for key, entry in problems.items():
        sol_wl, b_wl = {}, {}
        for uuid, w in (entry.get("workloads") or {}).items():
            if w.get("t_sol_ms") is not None:
                sol_wl[uuid] = {"t_sol_ms": w["t_sol_ms"],
                                "t_sol_cycles": w.get("t_sol_cycles")}
            if w.get("t_b_ms") is not None:
                b_wl[uuid] = {"t_b_ms": w["t_b_ms"],
                              "t_b_variant": w.get("t_b_variant")}
        if sol_wl:
            t_sol[key] = {"workloads": sol_wl}
        if b_wl:
            t_b[key] = {"workloads": b_wl}
    meta = {
        "manifest_version": doc.get("manifest_version"),
        "path": str(path),
        "stats": doc.get("stats"),
        "problems_with_t_sol": len(t_sol),
        "problems_with_t_b": len(t_b),
    }
    return t_sol, t_b, meta


def load_bounds(path: Path, part: str | None) -> tuple[dict, dict]:
    """(problems-by-key, meta) from a T_SOL or T_b artifact, or ({}, {}).

    The part check is the whole reason this is a function rather than a
    ``json.load``. ``artifacts/03/t_sol.json`` carries ``part`` in its header,
    and a bound derived at MI350X's F_LOCK of 1300 MHz applied to an MI355X
    measurement would rescale every score by the clock ratio -- plausibly,
    invisibly, and in the direction that flatters the kernel. Prime directive 2.
    """
    if not path.exists():
        return {}, {}
    doc = json.loads(path.read_text())
    meta = {k: v for k, v in doc.items() if k != "problems"}
    if part and doc.get("part") and doc["part"] != part:
        meta["rejected"] = (
            f"artifact was derived on {doc['part']} but this node is {part}; "
            f"not used"
        )
        return {}, meta
    return doc.get("problems", {}) or {}, meta


def workload_bound(bounds: dict, problem_key: str, uuid: str, field: str):
    entry = (bounds.get(problem_key) or {}).get("workloads") or {}
    return (entry.get(uuid) or {}).get(field)


def _spec_divergence(packet: Path, authoritative_def: Path,
                     authoritative_wl: Path) -> list[str]:
    """What, if anything, the packet's copy of the problem no longer matches.

    Reported per field rather than as one boolean, because the fields are not
    equally serious: a changed ``reference`` redefines what "correct" means, while
    a changed ``description`` is cosmetic.
    """
    divergence: list[str] = []
    try:
        want = json.loads(authoritative_def.read_text())
        have = json.loads((packet / "definition.json").read_text())
    except Exception as exc:  # noqa: BLE001
        return [f"definition.json unreadable: {type(exc).__name__}: {exc}"]

    for field in sorted(set(want) | set(have)):
        if want.get(field) != have.get(field):
            divergence.append(f"definition.{field}")

    try:
        want_lines = [ln for ln in authoritative_wl.read_text().splitlines()
                      if ln.strip()]
        have_lines = [ln for ln in (packet / "workload.jsonl").read_text().splitlines()
                      if ln.strip()]
        if want_lines != have_lines:
            divergence.append("workload.jsonl")
    except Exception as exc:  # noqa: BLE001
        divergence.append(f"workload.jsonl unreadable: {type(exc).__name__}: {exc}")
    return divergence


def score_one(session_dir: Path, problem_dir: Path, workloads_root: Path | None,
              *, t_sol: dict, t_b: dict, part: str | None, timeout: int,
              gpu: int) -> dict:
    """Evaluate one harvested solution against the AUTHORITATIVE problem spec.

    The problem — ``definition.json`` and ``workload.jsonl`` — is read from the
    dataset and the tolerance tree, **never from the packet**, and the packet's
    copies are diffed against them and any difference reported.

    This is not defensive decoration. An earlier version loaded the definition
    from the packet, and a submission was found having rewritten the reference
    inside it to work around a broken interpreter (STATE.md D24). Editing the
    reference edits the definition of correctness: in the limit a submission can
    replace the reference with its own kernel and score 100% on every workload.
    Only ``solution.json`` and its sources may come from the packet.
    """
    from _common import evaluate, summarize  # noqa: E402

    from sol_execbench.core import BenchmarkConfig, Definition, Workload  # noqa: E402
    from sol_execbench.core.bench.reward_hack import static_source_screen  # noqa: E402

    from agent_verify import load_solution  # noqa: E402

    # `benchmark_reference=True` because without it reference_latency_ms stays 0.0
    # and there is no speed axis at all -- not even speedup-vs-reference, which is
    # the strongest basis available until T_b exists.
    #
    # `lock_clocks=True` asserts the clock rather than assuming it. Every latency
    # here is quoted at F_LOCK, and an unlocked GPU would produce numbers that
    # look identical and mean something else.
    config = BenchmarkConfig(benchmark_reference=True, lock_clocks=True)

    packet = session_dir / "packet"
    session = json.loads((session_dir / "session.json").read_text())
    manifest = json.loads((packet / ".packet.json").read_text())
    problem_key = manifest["problem"].replace("/", "__")

    result: dict = {
        "problem": problem_key,
        "category": problem_key.split("__", 1)[0],
        "harness": session["harness"],
        "model": session.get("model"),
        "gpu": gpu,
        "agent": {
            "wallclock_s": session.get("wallclock_s"),
            "cost_usd": session.get("cost_usd"),
            "cost_source": session.get("cost_source"),
            "input_tokens": session.get("input_tokens"),
            "output_tokens": session.get("output_tokens"),
            "reasoning_tokens": session.get("reasoning_tokens"),
            "num_turns": session.get("num_turns"),
            "verify_attempts": session.get("verify_attempts"),
            "timed_out": session.get("timed_out"),
            "harness_error": session.get("error"),
        },
        "records": [],
    }

    # Resolved before the early return, so tampering is reported even for a
    # session that produced nothing.
    category, name = problem_key.split("__", 1)
    authoritative_def = problem_dir / "definition.json"
    authoritative_wl = (
        workloads_root / category / name / "workload.jsonl"
        if workloads_root else problem_dir / "workload.jsonl"
    )
    result["spec_source"] = {
        "definition": str(authoritative_def),
        "workloads": str(authoritative_wl),
    }
    result["packet_spec_divergence"] = _spec_divergence(
        packet, authoritative_def, authoritative_wl
    )
    if result["packet_spec_divergence"]:
        # Loud, and it does not stop the scoring: the authoritative spec is what
        # gets used either way, so the honest thing is to score against it and
        # record that the packet disagreed.
        result["spec_tampered"] = True

    if not session.get("produced_solution"):
        result["outcome"] = "no_solution"
        result["note"] = "the agent left no solution.json"
        return result

    definition = Definition(**json.loads(authoritative_def.read_text()))
    raw_workloads = [json.loads(ln) for ln
                     in authoritative_wl.read_text().splitlines()
                     if ln.strip()]
    workloads = [Workload(**w) for w in raw_workloads]

    try:
        solution = load_solution(packet)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = "invalid_solution"
        result["note"] = f"{type(exc).__name__}: {exc}"
        return result

    result["languages"] = [x.value for x in solution.spec.languages]
    result["n_sources"] = len(solution.sources)
    result["source_bytes"] = sum(len(s.content or "") for s in solution.sources)

    # Reported, not raised: a finding is a result to record, and the sweep must
    # not die on one submission.
    findings = static_source_screen(solution.sources)
    result["static_screen"] = [
        {"path": p, "pattern": pat, "why": why} for p, pat, why in findings
    ]
    copy = reference_copy_verdict(solution.sources, definition.reference)
    result["reference_copy"] = {"kind": copy.kind, "similarity": copy.similarity,
                                "detail": copy.detail}

    if findings:
        # Zero regardless of what it measures. A submission that trips the screen
        # is not a slow kernel, it is a kernel measured under conditions the
        # benchmark does not permit.
        result["outcome"] = "rejected_static_screen"
        return result

    started = time.time()
    try:
        traces = evaluate(definition, workloads, solution, config=config,
                          timeout=timeout)
        summary = summarize(traces)
    except Exception as exc:  # noqa: BLE001
        result["outcome"] = "eval_failed"
        result["note"] = f"{type(exc).__name__}: {exc}"
        result["eval_wallclock_s"] = time.time() - started
        return result

    result["eval_wallclock_s"] = time.time() - started
    result["outcome"] = "evaluated"

    for i, w in enumerate(summary["per_workload"]):
        uuid = w.get("workload_uuid") or (
            raw_workloads[i].get("uuid") if i < len(raw_workloads) else None
        )
        t_k = w.get("latency_ms")
        t_ref = w.get("reference_latency_ms")
        t_sol_ms = workload_bound(t_sol, problem_key, uuid, "t_sol_ms")
        t_b_ms = workload_bound(t_b, problem_key, uuid, "t_b_ms")
        correct = w["status"] == "PASSED"

        basis = resolve_basis(correct=correct, t_k_ms=t_k, t_ref_ms=t_ref,
                              t_sol_ms=t_sol_ms, t_b_ms=t_b_ms)
        s = sol_score(t_k, t_sol_ms, t_b_ms) if correct else None
        hf = headroom_fraction(t_k, t_ref, t_sol_ms) if correct else None

        tol = raw_workloads[i].get("tolerance", {}) if i < len(raw_workloads) else {}
        record = {
            "workload_index": i,
            "workload_uuid": uuid,
            "status": w["status"],
            "correct": correct,
            "t_k_ms": t_k,
            "t_ref_ms": t_ref,
            "speedup_vs_reference": (t_ref / t_k) if (t_k and t_ref) else None,
            "t_sol_ms": t_sol_ms,
            "t_sol_cycles": workload_bound(t_sol, problem_key, uuid, "t_sol_cycles"),
            "t_b_ms": t_b_ms,
            "sol_score": s,
            "headroom_fraction": hf,
            "score_basis": basis.value,
            "max_absolute_error": w.get("max_absolute_error"),
            "max_relative_error": w.get("max_relative_error"),
            "allowed_atol": tol.get("max_atol"),
            "allowed_rtol": tol.get("max_rtol"),
            "has_nan": w.get("has_nan"),
            "has_inf": w.get("has_inf"),
            "methodology": w.get("methodology"),
            "failure_log": w.get("log") or "",
        }
        # A kernel faster than its own analytic lower bound is impossible; the
        # bound is wrong. Surfaced per record rather than clamped, because
        # clamping would hide exactly the thing worth fixing (cf. D12).
        if correct and t_k and t_sol_ms and t_k < t_sol_ms:
            record["bound_violation"] = (
                f"t_k {t_k:.6g} ms is below t_sol {t_sol_ms:.6g} ms"
            )
        result["records"].append(record)

    result["workloads"] = summary["workloads"]
    result["passed"] = summary["passed"]
    result["all_passed"] = summary["all_passed"]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--gpu", type=int, default=AUTHORITATIVE_GPU,
                    help="authoritative timing GPU")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--benchmark-dir",
                    default=str(ROOT / "data" / "SOL-ExecBench" / "benchmark"),
                    type=Path)
    ap.add_argument("--t-sol", default=str(ROOT / "artifacts" / "03" / "t_sol.json"),
                    type=Path)
    ap.add_argument("--t-b", default=str(ROOT / "artifacts" / "06" / "t_b.json"),
                    type=Path)
    ap.add_argument("--manifest",
                    help="frozen scoring manifest; supersedes --t-sol/--t-b. "
                         "Preferred, because a score is only meaningful inside "
                         "one manifest version")
    ap.add_argument("--workloads-root",
                    default=str(ROOT / "artifacts" / "05" / "workloads"), type=Path,
                    help="AMD-derived tolerances; the authoritative workload "
                         "source. 'none' falls back to the dataset's B200 ones")
    ap.add_argument("--force", action="store_true",
                    help="re-score sessions that already have a score record")
    args = ap.parse_args()

    workloads_root = None if str(args.workloads_root).lower() == "none" \
        else args.workloads_root

    run_root = ROOT / "artifacts" / "10" / "runs" / args.run_id
    if not run_root.exists():
        sys.exit(f"no such run: {run_root}")
    out_dir = ROOT / "artifacts" / "10" / "scores" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    cfg_path = run_root / "config.json"
    if cfg_path.exists():
        config = json.loads(cfg_path.read_text())

    prov = stamp("10-scoring")["_provenance"]
    part = prov.get("part")

    integrity = compare_digests(config.get("harness_tree_digest"), tree_digest(ROOT))
    if integrity.get("comparable") and not integrity.get("match"):
        print("WARNING: the scoring-critical source tree changed since the sweep "
              "started.", file=sys.stderr)
        for p in integrity.get("changed", []):
            print(f"  changed: {p}", file=sys.stderr)
        print("  This is recorded on the summary. It does not prove tampering -- an "
              "operator edit looks the same -- but a score whose harness moved "
              "mid-run is not comparable with one whose did not.", file=sys.stderr)

    manifest_meta: dict = {}
    if args.manifest and Path(args.manifest).exists():
        t_sol, t_b, manifest_meta = load_manifest_bounds(Path(args.manifest))
        t_sol_meta = t_b_meta = {"from_manifest": manifest_meta.get("manifest_version")}
        print(f"  bounds from manifest {manifest_meta.get('manifest_version')}: "
              f"T_SOL for {manifest_meta['problems_with_t_sol']} problems, "
              f"T_b for {manifest_meta['problems_with_t_b']}", file=sys.stderr)
    else:
        t_sol, t_sol_meta = load_bounds(args.t_sol, part)
        t_b, t_b_meta = load_bounds(args.t_b, part)
    for label, meta in (("T_SOL", t_sol_meta), ("T_b", t_b_meta)):
        if meta.get("rejected"):
            print(f"  {label}: {meta['rejected']}", file=sys.stderr)
        elif not meta:
            print(f"  {label}: absent — scores fall back to a weaker basis",
                  file=sys.stderr)

    # Probed before anything is timed, and the harness's env flag is set only if
    # the probe agrees. Refusing here costs a re-run; not refusing produces a
    # scoreboard of numbers taken at an unknown clock, which is unrecoverable.
    lock = clock_lock_state(ROOT, args.gpu)
    if not lock.get("locked"):
        sys.exit(
            f"GPU {args.gpu} is not clock-locked: {lock}\n"
            f"Every latency here is quoted at F_LOCK. Lock it first:\n"
            f"  python scripts/clock_calibrate.py lock --freq-mhz <F_LOCK> --all-gpus\n"
            f"  python scripts/clock_calibrate.py verify --freq-mhz <F_LOCK> --under-load"
        )
    print(f"  clock lock: GPU {args.gpu} at {lock['performance_level']} "
          f"({lock.get('drm_card')})")

    sessions = sorted(run_root.glob("*/*/session.json"))
    print(f"{len(sessions)} session(s) to score on GPU {args.gpu} (serial)")

    import os
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SOL_EXECBENCH_CLOCKS_LOCKED"] = "1"

    results = []
    for n, sess_path in enumerate(sessions, 1):
        session_dir = sess_path.parent
        harness, problem_key = session_dir.parent.name, session_dir.name
        dest = out_dir / harness / f"{problem_key}.json"
        if dest.exists() and not args.force:
            results.append(json.loads(dest.read_text()))
            print(f"  [{n}/{len(sessions)}] {harness}/{problem_key}: already scored")
            continue

        category, name = problem_key.split("__", 1)
        problem_dir = args.benchmark_dir / category / name
        try:
            result = score_one(session_dir, problem_dir, workloads_root,
                               t_sol=t_sol, t_b=t_b, part=part,
                               timeout=args.timeout, gpu=args.gpu)
        except Exception as exc:  # noqa: BLE001
            result = {"problem": problem_key, "harness": harness,
                      "outcome": "scorer_error",
                      "note": f"{type(exc).__name__}: {exc}", "records": []}

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({**stamp("10-scoring"), **result},
                                   indent=2, default=str))
        results.append(result)

        detail = result.get("outcome")
        if detail == "evaluated":
            detail = f"{result['passed']}/{result['workloads']} workloads passed"
            if result["reference_copy"]["kind"] != "distinct":
                detail += f" (reference {result['reference_copy']['kind']} copy)"
        print(f"  [{n}/{len(sessions)}] {harness}/{problem_key}: {detail}")

    summary = {
        "run_id": args.run_id,
        "part": part,
        "authoritative_gpu": args.gpu,
        "clock_lock": lock,
        "f_lock_mhz": prov.get("f_lock_mhz"),
        "sessions_scored": len(results),
        "manifest": manifest_meta or None,
        "t_sol": {"path": str(args.t_sol), "used": bool(t_sol), **t_sol_meta},
        "t_b": {"path": str(args.t_b), "used": bool(t_b), **t_b_meta},
        "harness_integrity": integrity,
        "workloads_root": str(workloads_root) if workloads_root else None,
        "spec_tampered": [
            {"harness": r.get("harness"), "problem": r.get("problem"),
             "divergence": r.get("packet_spec_divergence")}
            for r in results if r.get("spec_tampered")
        ],
        "outcomes": _count(results, "outcome"),
        "score_bases": _count(
            [r for res in results for r in res.get("records", [])], "score_basis"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({**stamp("10-scoring"), **summary}, indent=2, default=str)
    )
    print(f"\nwrote {out_dir}/summary.json")
    print(json.dumps(summary["outcomes"], indent=2))
    print(f"next: python scripts/build_scoreboard.py --run-id {args.run_id}")
    return 0


def _count(items: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    sys.exit(main())
