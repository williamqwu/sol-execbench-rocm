#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 06 runner — time every T_b candidate variant for one problem.

T_b is the optimized-PyTorch anchor that sets the entire score scale
(S = 0.5 at T_k = T_b), so it is measured, per platform, under the harness's
own conditions — not ported from B200 and not hand-tuned to make scores land
somewhere pleasing.

Variants come from `reference/tb-candidates/variants.py` (generic transforms of
the problem's own reference) plus any per-problem overrides in
`reference/tb-candidates/<Category>__<problem>/vN_*.py`. The winner is the
fastest variant that is also CORRECT: a variant that is fast because it is
wrong is not a baseline.

    python scripts/runners/time_tb_candidates.py --problem <dir> --out <file>

**Unlocked parts** (``SOLEXBENCH_CLOCK_BASIS=unlocked``): every timed window is
bracketed by a clock sample either side, and a measurement whose two samples
disagree by more than the threshold is **refused** -- recorded, counted, and not
eligible to be an anchor. The threshold, the refusal count and the per-workload
brackets are fields on the artifact, not log lines, because task 01's acceptance
on such a part is "the refusal rate is below a stated bound".

That bracket bounds the *clock* error and nothing else. It does not touch the
short-window timing bias in ``docs/methodology.md`` §7, which is larger, is
measured not to be a clock effect, and is carried by every T_b here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    PASSED,
    ROOT,
    evaluate,
    load_problem,
    problem_key,
    reference_solution,
    run_guarded,
    summarize,
    workloads_path,
)

CANDIDATE_DIR = ROOT / "reference" / "tb-candidates"


def _load_variants() -> dict:
    spec = importlib.util.spec_from_file_location(
        "_tb_variants", CANDIDATE_DIR / "variants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.VARIANTS)


def _overrides(key: str, reference_src: str) -> dict[str, str]:
    """Hand-written per-problem variants, if any.

    The generic set is the default because task 06 is explicitly a batch job,
    not an authoring one. An override exists only where the pre-authored set
    was clearly missing an obvious formulation, and its presence in the
    artifact is how that addition gets recorded.
    """
    d = CANDIDATE_DIR / key
    if not d.is_dir():
        return {}
    return {p.stem: p.read_text() for p in sorted(d.glob("v*_*.py"))}


def select_winners(results: dict, bracketing: bool) -> tuple[dict, dict]:
    """Pick T_b per workload, and say which measurements were refused.

    Per workload, not per problem: the fastest formulation genuinely differs with
    shape (compile wins on large shapes, eager on tiny ones where compilation
    guards dominate), and T_b is defined per workload instance. Picking one
    variant for the whole problem would inflate T_b on every shape it does not
    suit, which would make every score on that shape look better than it is.

    **Unlocked basis: a refused measurement is not eligible to be an anchor.** Its
    latency is real, but the clock it was taken at is not pinned down well enough
    to divide a bound by, and T_b sets the entire score scale -- a T_b measured
    while a card was moving between power states rescales every score on that
    workload, silently and forever.

    Refused measurements are *recorded*, not discarded: the workload then appears
    as missing T_b and reaches triage. A refusal that vanished would be
    indistinguishable from a workload that was never run, which is precisely how
    scope shrinks by omission rather than by decision (CLAUDE.md §0).

    A module-level function rather than a block inside `body()` so it can be
    tested without a GPU, a dataset or a subprocess.
    """
    from sol_execbench.core.bench.clock_bracket import has_clock_evidence

    winners: dict[str, dict] = {}
    refused: dict[str, list[dict]] = {}
    # EVERY bracket this function looked at, refused or not, in one list.
    #
    # The summary used to be built from `winners` alone, which is exactly
    # backwards: a refused measurement never becomes a winner, so the artifact
    # reported `n_bracketed: 0, n_refused: 0, refused_by_reason: {}` at the very
    # moment all 16 workloads had been refused -- three lines below a
    # `n_workloads_refused_on_clock: 16` that was right. Two counters, two
    # sources, and the one a reader would check went blind precisely when it
    # mattered. Both now come from here, so they cannot disagree again.
    considered: list[dict] = []
    for name, r in results.items():
        if not r.get("ok") or not r.get("all_passed"):
            continue
        brackets = r.get("clock_bracket_by_workload") or {}
        ref_brackets = r.get("reference_clock_bracket_by_workload") or {}
        for uuid, ms in r["latency_ms_by_workload"].items():
            br = brackets.get(uuid)
            if br:
                considered.append(br)
            if bracketing and not has_clock_evidence(br):
                refused.setdefault(uuid, []).append({
                    "variant": name, "t_b_ms": ms,
                    "clock_bracket_refused_reason": (br or {}).get(
                        "clock_bracket_refused_reason", "no_bracket_recorded"),
                    "clock_bracket_spread": (br or {}).get("clock_bracket_spread"),
                })
                continue
            if uuid not in winners or ms < winners[uuid]["t_b_ms"]:
                winners[uuid] = {
                    "variant": name, "t_b_ms": ms,
                    **(br or {}),
                    # §4.4: the reference arm's clock, timed back to back with
                    # this one on the same card. Carried beside T_b rather than
                    # summarised, so a two-clock `S` is visible as a two-clock
                    # quantity instead of being read as a one-clock one.
                    "reference_clock_bracket": ref_brackets.get(uuid),
                }
    return winners, refused, considered


def clock_fatalities(bracket_summary: dict, considered: list[dict],
                     attempted: int, bracketing: bool) -> list[str]:
    """Conditions under which this run must fail closed and say so.

    Before this existed, a total sampling failure and a clean run under a strict
    threshold were indistinguishable: both produced zero anchors, exit 0, and an
    artifact that read like a result. The first bracketed T_b sweep refused 100%
    of its workloads and reported success. **A benchmark that refuses everything
    has to say so in its exit status.**

    *attempted* counts measurements that reached selection, not workloads, so a
    problem where every variant legitimately failed correctness is not a clock
    failure and is not reported as one.
    """
    if not bracketing or not attempted:
        return []
    fatal: list[str] = []
    n_err = bracket_summary["refused_by_reason"].get("sampler_error", 0)
    if n_err:
        first = next((b.get("clock_bracket_sampler_error") for b in considered
                      if b.get("clock_bracket_sampler_error")), None)
        fatal.append(
            f"the clock sampler RAISED on {n_err} of {attempted} measurements — "
            f"a defect in the sampling path, not a property of the node. "
            f"First error: {first}")
    if bracket_summary["n_bracketed"] == 0:
        fatal.append(
            f"{attempted} measurement(s) were taken and NOT ONE carries a clock "
            f"bracket. Under SOLEXBENCH_CLOCK_BASIS=unlocked the eval driver "
            f"must bracket every timed window; a run with no brackets at all is "
            f"a plumbing failure, not a strict threshold.")
    elif bracket_summary["refusal_rate"] == 1.0:
        fatal.append(
            f"every one of {bracket_summary['n_bracketed']} bracketed "
            f"measurements was refused ({bracket_summary['refused_by_reason']}). "
            f"Do not raise the threshold to make this pass: at a 100% rate the "
            f"threshold is not the finding, whatever produced the spreads is.")
    return fatal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only-variant", action="append", default=None,
                    help="restrict to named variants (the authoritative pass "
                         "re-times only the winner)")
    a = ap.parse_args()

    def body() -> dict:
        from sol_execbench.core import BenchmarkConfig
        from sol_execbench.core.bench.clock_bracket import (
            bracket_threshold,
            bracketing_enabled,
            checked_clock_basis,
            summarize_brackets,
        )

        bracketing = bracketing_enabled()
        # Resolved BEFORE any timing, so a basis this part cannot support costs
        # nothing rather than a full sweep whose artifacts are all unusable.
        import torch
        basis = checked_clock_basis(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

        definition, workloads = load_problem(a.problem)
        key = problem_key(a.problem)

        sources = {
            name: transform(definition.reference)
            for name, transform in _load_variants().items()
        }
        overrides = _overrides(key, definition.reference)
        sources.update(overrides)
        if a.only_variant:
            sources = {k: v for k, v in sources.items() if k in set(a.only_variant)}

        config = BenchmarkConfig(
            warmup_runs=a.warmup,
            iterations=a.iterations,
            benchmark_reference=False,
        )

        results: dict[str, dict] = {}
        for name, src in sources.items():
            try:
                traces = evaluate(
                    definition,
                    workloads,
                    reference_solution(definition, name_suffix=name, source=src),
                    config,
                    timeout=a.timeout,
                )
                summary = summarize(traces)
            except Exception as e:                  # noqa: BLE001
                # A variant that fails to compile or run is a RESULT: some
                # formulations legitimately do not work for some problems
                # (torch.compile falls over on a few). Recording it keeps the
                # remaining variants' timings, which is the point of doing
                # this per variant rather than per problem.
                results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                continue

            passing = [w for w in summary["per_workload"]
                       if w["status"] == PASSED and w["latency_ms"]]
            per_wl = {w["workload_uuid"]: w["latency_ms"] for w in passing}
            # AMD, unlocked basis: one bracket per passing workload, keyed the
            # same way as the latencies so the two cannot be paired up wrong.
            # All None under the locked basis, where the driver does not bracket.
            brackets = {w["workload_uuid"]: w.get("clock_bracket") for w in passing}
            ref_brackets = {
                w["workload_uuid"]: w["reference_clock_bracket"]
                for w in passing if w.get("reference_clock_bracket")
            }
            results[name] = {
                "ok": True,
                "workloads": summary["workloads"],
                "passed": summary["passed"],
                "all_passed": summary["all_passed"],
                "latency_ms_by_workload": per_wl,
                "clock_bracket_by_workload": brackets,
                # §4.4: the T_b arm and the T_k arm timed back to back on one
                # card, both clocks recorded rather than assumed equal.
                "reference_clock_bracket_by_workload": ref_brackets,
                "clock_bracket_summary": summarize_brackets(list(brackets.values())),
                "is_override": name in overrides,
                "failures": [
                    {"workload_uuid": w["workload_uuid"], "status": w["status"],
                     "log": w["log"][:1000]}
                    for w in summary["per_workload"] if w["status"] != PASSED
                ],
            }

        winners, refused, considered = select_winners(results, bracketing)
        bracket_summary = summarize_brackets(considered)

        attempted = sum(
            len(r.get("latency_ms_by_workload") or {})
            for r in results.values() if r.get("ok") and r.get("all_passed")
        )
        fatal = clock_fatalities(bracket_summary, considered, attempted, bracketing)
        for line in fatal:
            print(f"  FATAL: {line}", file=sys.stderr)

        return {
            # `ok` False propagates to a non-zero exit through run_guarded, while
            # the artifact below is still written in full -- a recorded failure
            # is a result, an unrecorded one is a gap.
            "ok": not fatal,
            "clock_bracket_fatal": fatal or None,
            "problem": key,
            "definition": definition.name,
            # Which tolerances decided `all_passed`, and therefore which
            # variant was eligible to become the anchor.
            "workloads_from": str(workloads_path(a.problem)),
            "gpu": os.environ.get("HIP_VISIBLE_DEVICES"),
            "variants": results,
            "winner_by_workload": winners,
            # First-class, not a log line: task 01's acceptance on an unlocked
            # part is "the bracket refusal rate is below a stated bound", and a
            # rate nobody can read from the artifact cannot gate anything.
            "clock_basis": basis,
            "clock_bracket_threshold": bracket_threshold() if bracketing else None,
            # Over every bracket considered, winners AND refusals, from the one
            # list `select_winners` built. Not over winners: see its comment.
            "clock_bracket_summary": bracket_summary,
            "clock_bracket_refused_by_workload": refused,
            # Two different quantities, and conflating them is how the first
            # artifact read "16 refused" beside "12 anchored" on 16 workloads.
            # A workload can have one variant refused and still be anchored by
            # another, which is the system working, not a loss.
            "n_workloads_with_a_refusal": len(refused),
            "n_workloads_with_no_anchor_due_to_clock": len(
                set(refused) - set(winners)),
            "n_workloads": len(workloads),
            "n_workloads_with_tb": len(winners),
            # Loud rather than silent: a problem where no variant passed every
            # workload has no anchor, and must reach triage instead of being
            # quietly absent from the manifest.
            "complete": len(winners) == len(workloads),
        }

    return run_guarded(a.out, "06-tb-candidates", body)


if __name__ == "__main__":
    raise SystemExit(main())
