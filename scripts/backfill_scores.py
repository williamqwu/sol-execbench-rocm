#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 10 — recompute scores from a newer manifest without re-running any GPU work.

    python scripts/backfill_scores.py --run-id pilot-01 \
        --manifest artifacts/09/manifest-MI355X-v1.json

The two-phase plan this repo is following measures agents *before* `T_b` exists,
under a weaker `score_basis`, and layers the real SOL score in when the anchor
lands. That layering must not re-time anything: `T_k`, `T_ref` and the pass/fail
verdicts were measured on the authoritative GPU at F_LOCK and are still valid.
Only the bounds changed.

Re-running the evaluation instead would cost hours of GPU time to produce the
same latencies, and — worse — the new latencies would be measured on a different
day under different node conditions, so a record's basis and its timing would
come from different runs. Recomputing arithmetic is both cheaper and more
correct.

Every record keeps its previous basis under ``score_basis_history``, so a
strengthened score is visibly a strengthened score rather than a silently
different number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402
from solexbench_agents.scoring import (  # noqa: E402
    headroom_fraction,
    resolve_basis,
    sol_score,
)

from score_solutions import load_manifest_bounds, workload_bound  # noqa: E402


def backfill_record(record: dict, problem_key: str, t_sol: dict,
                    t_b: dict) -> tuple[dict, bool]:
    """Return (record, changed). Timings are never touched, only the bounds."""
    uuid = record.get("workload_uuid")
    new_sol = workload_bound(t_sol, problem_key, uuid, "t_sol_ms")
    new_b = workload_bound(t_b, problem_key, uuid, "t_b_ms")

    old_basis = record.get("score_basis")
    t_k = record.get("t_k_ms")
    t_ref = record.get("t_ref_ms")
    correct = bool(record.get("correct"))

    basis = resolve_basis(correct=correct, t_k_ms=t_k, t_ref_ms=t_ref,
                          t_sol_ms=new_sol, t_b_ms=new_b)
    s = sol_score(t_k, new_sol, new_b) if correct else None
    hf = headroom_fraction(t_k, t_ref, new_sol) if correct else None

    changed = (
        new_sol != record.get("t_sol_ms")
        or new_b != record.get("t_b_ms")
        or basis.value != old_basis
    )
    if not changed:
        return record, False

    history = list(record.get("score_basis_history") or [])
    if old_basis and old_basis != basis.value:
        history.append(old_basis)

    record.update(
        t_sol_ms=new_sol,
        t_sol_cycles=workload_bound(t_sol, problem_key, uuid, "t_sol_cycles"),
        t_b_ms=new_b,
        t_b_variant=workload_bound(t_b, problem_key, uuid, "t_b_variant"),
        sol_score=s,
        headroom_fraction=hf,
        score_basis=basis.value,
        score_basis_history=history,
    )
    # Re-derived rather than carried over: a bound that just changed may now be
    # the one being violated, or may have stopped being violated.
    record.pop("bound_violation", None)
    if correct and t_k and new_sol and t_k < new_sol:
        record["bound_violation"] = (
            f"t_k {t_k:.6g} ms is below t_sol {new_sol:.6g} ms"
        )
    return record, True


def _assert_comparable(score_files: list[Path], manifest: Path) -> None:
    """Refuse to pair a T_b with a T_k that is not comparable to it.

    The docstring above claims T_k "was measured on the authoritative GPU at F_LOCK
    and is still valid, only the bounds changed". S divides one by the other, so that
    claim is load-bearing, and it is only true while both sides come from the same
    card at the same clock. Nothing checked it.

    It has since stopped being true here. The pilot's T_k was measured on **GPU 0 at
    F_LOCK 1640**, before D30 established that GPU 0 does not hold a setpoint under
    load -- it runs 1657 MHz alone and 1410 with the node busy. The T_b in
    manifest-MI355X-v1 was measured on **GPU 1 at 1650**. Backfilling one into the
    other silently rescales every score by the ratio between two cards' real clocks,
    and the output looks entirely plausible: this is D26 again, one layer further in.

    So the guard `collect_t_b()` applies when *gathering* T_b is applied here when
    *pairing* it. Refuses rather than warns, because the failure is invisible
    downstream.
    """
    m = json.loads(manifest.read_text())
    m_prov = m.get("_provenance") or {}
    m_lock = m_prov.get("f_lock_mhz")
    m_gpu = m_prov.get("authoritative_gpu")

    seen: dict[tuple, int] = {}
    for p in score_files:
        prov = (json.loads(p.read_text()).get("_provenance") or {})
        seen[(prov.get("f_lock_mhz"), prov.get("authoritative_gpu"))] = \
            seen.get((prov.get("f_lock_mhz"), prov.get("authoritative_gpu")), 0) + 1

    bad = {k: n for k, n in seen.items()
           if (k[0] is not None and m_lock is not None and k[0] != m_lock)
           or (k[1] is not None and m_gpu is not None and k[1] != m_gpu)}
    if bad:
        lines = "\n".join(
            f"    {n} record file(s) measured at F_LOCK {lk} on GPU {g}"
            for (lk, g), n in sorted(bad.items(), key=lambda kv: -kv[1]))
        sys.exit(
            f"REFUSING to backfill: these scores were not measured under the "
            f"conditions this manifest's T_b was.\n"
            f"  manifest T_b : F_LOCK {m_lock} MHz on GPU {m_gpu}\n"
            f"{lines}\n"
            f"  S = f(T_k, T_b, T_SOL) divides one timing by another, so pairing "
            f"them rescales every score by the ratio between two different clocks "
            f"-- silently, and the result looks plausible (STATE.md D26).\n"
            f"  Re-run scoring for this run against the current node, or backfill "
            f"from a manifest measured under the same conditions.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scores_dir = ROOT / "artifacts" / "10" / "scores" / args.run_id
    if not scores_dir.is_dir():
        sys.exit(f"no scores for run {args.run_id!r} at {scores_dir}")

    t_sol, t_b, meta = load_manifest_bounds(args.manifest)
    if not meta:
        sys.exit(f"manifest not readable: {args.manifest}")
    print(f"manifest {meta['manifest_version']}: T_SOL for "
          f"{meta['problems_with_t_sol']} problems, T_b for "
          f"{meta['problems_with_t_b']}")

    files = sorted(p for p in scores_dir.glob("*/*.json") if p.name != "summary.json")
    _assert_comparable(files, args.manifest)
    changed_files = 0
    changed_records = 0
    basis_counts: dict[str, int] = {}
    violations = 0

    for path in files:
        doc = json.loads(path.read_text())
        problem_key = doc.get("problem")
        touched = False
        for record in doc.get("records", []):
            record, changed = backfill_record(record, problem_key, t_sol, t_b)
            if changed:
                changed_records += 1
                touched = True
            key = record.get("score_basis") or "none"
            basis_counts[key] = basis_counts.get(key, 0) + 1
            if record.get("bound_violation"):
                violations += 1
        if touched:
            changed_files += 1
            doc["backfilled_from_manifest"] = meta["manifest_version"]
            if not args.dry_run:
                path.write_text(json.dumps({**stamp("10-backfill"), **doc},
                                           indent=2, default=str))

    # The run summary carries the basis census, and `verify_artifacts --task 10`
    # reads it. Leaving it stale after a rebase would make the acceptance check
    # describe a distribution that no longer exists on disk -- the same class of
    # drift as a document and a constant disagreeing about F_LOCK (F17).
    summary_path = scores_dir / "summary.json"
    if summary_path.exists() and not args.dry_run:
        summary = json.loads(summary_path.read_text())
        summary["score_bases"] = dict(sorted(basis_counts.items()))
        summary["backfilled_from_manifest"] = meta["manifest_version"]
        summary["bound_violations"] = violations
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"{changed_files} of {len(files)} score files updated, "
          f"{changed_records} records rebased"
          + ("  (dry run, nothing written)" if args.dry_run else ""))
    print("score bases now:")
    for basis, n in sorted(basis_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {basis:<24} {n}")
    if violations:
        print(f"\n{violations} record(s) remain faster than their T_SOL bound. "
              f"Those bounds are wrong, not those kernels; S must not be "
              f"published for them (STATE.md D25).")
    print(f"\nnext: python scripts/build_scoreboard.py --run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
