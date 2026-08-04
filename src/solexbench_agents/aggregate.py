# SPDX-License-Identifier: Apache-2.0
"""Turn per-problem score files into the numbers a scoreboard shows.

One rule runs through all of it: **never average across score bases.** A record
scored ``correctness_only`` has no timing claim, one scored ``sol_headroom`` has
a bound but no anchor, and one scored ``sol_score_v1`` has both. Pooling them
would produce a mean that moves when the *bounds* land rather than when the
kernels improve, and nothing in the output would say which happened. Every
timing aggregate here therefore carries the basis it was computed over and the
count of records that qualified.

The second rule: report the denominator. "Solved 30" means nothing without "of
how many attempted", and the two differ whenever a harness crashed or a packet
could not be built.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Iterable

from .scoring import ScoreBasis

CATEGORIES = ("L1", "L2", "Quant", "FlashInfer-Bench")

# Outcomes that mean "the agent did not deliver a scoreable artifact", as
# distinct from "it delivered one and it was wrong". Keeping them apart matters:
# the first is a harness or budget story, the second is a model story.
NON_DELIVERY = ("no_solution", "invalid_solution", "scorer_error")


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _pct(numerator: int, denominator: int) -> float | None:
    return (100.0 * numerator / denominator) if denominator else None


def failure_kind(result: dict, record: dict | None = None) -> str:
    """A coarse taxonomy of why something did not score.

    Coarse on purpose. The useful question is which *stage* failed -- did the
    agent produce nothing, did it fail to build, did it run and disagree
    numerically -- because those point at different fixes. Finer detail lives in
    the per-record log.
    """
    outcome = result.get("outcome")
    if outcome in NON_DELIVERY:
        return outcome
    if outcome == "rejected_static_screen":
        return "rejected_static_screen"
    if outcome == "eval_failed":
        return "harness_error"
    if record is None:
        return "unknown"
    status = (record.get("status") or "").upper()
    if status == "PASSED":
        return "passed"
    if "COMPIL" in status or "BUILD" in status:
        return "compile_error"
    if "NUMERIC" in status or "INCORRECT" in status:
        return "incorrect_numerical"
    if "TIMEOUT" in status:
        return "timeout"
    if "RUNTIME" in status or "ERROR" in status or "FAIL" in status:
        return "runtime_error"
    return status.lower() or "unknown"


def summarize_results(results: Iterable[dict]) -> dict:
    """Aggregate a flat list of per-problem score files."""
    results = list(results)
    by_harness: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_harness[r.get("harness") or "unknown"].append(r)

    harnesses = {h: _summarize_harness(rs) for h, rs in sorted(by_harness.items())}
    return {
        "harnesses": harnesses,
        "problems": _per_problem(results),
        "totals": {
            "score_files": len(results),
            "harnesses": sorted(by_harness),
        },
    }


def _summarize_harness(results: list[dict]) -> dict:
    attempted = len(results)
    delivered = [r for r in results if r.get("outcome") == "evaluated"]
    solved = [r for r in delivered if r.get("all_passed")]

    records = [rec for r in delivered for rec in r.get("records", [])]
    correct = [rec for rec in records if rec.get("correct")]

    # Timing aggregates, each over exactly the records whose basis supports it.
    speedups = [rec["speedup_vs_reference"] for rec in correct
                if rec.get("speedup_vs_reference")]
    headroom = [rec["headroom_fraction"] for rec in correct
                if rec.get("headroom_fraction") is not None
                and rec.get("score_basis") in (ScoreBasis.SOL_HEADROOM.value,
                                               ScoreBasis.SOL_SCORE_V1.value)]
    sol_scores = [rec["sol_score"] for rec in correct
                  if rec.get("sol_score") is not None
                  and rec.get("score_basis") == ScoreBasis.SOL_SCORE_V1.value]

    costs = [r["agent"]["cost_usd"] for r in results
             if (r.get("agent") or {}).get("cost_usd") is not None]
    tokens_in = sum((r.get("agent") or {}).get("input_tokens") or 0 for r in results)
    tokens_out = sum((r.get("agent") or {}).get("output_tokens") or 0 for r in results)
    token_sessions = sum(
        1 for r in results if (r.get("agent") or {}).get("input_tokens") is not None
    )
    wallclocks = [r["agent"]["wallclock_s"] for r in results
                  if (r.get("agent") or {}).get("wallclock_s") is not None]
    attempts = [r["agent"]["verify_attempts"] for r in results
                if (r.get("agent") or {}).get("verify_attempts") is not None]

    languages: dict[str, int] = defaultdict(int)
    for r in delivered:
        for lang in r.get("languages") or []:
            languages[lang] += 1

    copies = defaultdict(int)
    for r in delivered:
        copies[(r.get("reference_copy") or {}).get("kind", "unknown")] += 1

    failures: dict[str, int] = defaultdict(int)
    for r in results:
        if r.get("outcome") != "evaluated":
            failures[failure_kind(r)] += 1
            continue
        for rec in r.get("records", []):
            failures[failure_kind(r, rec)] += 1

    bound_violations = [
        {"problem": r["problem"], "workload_uuid": rec.get("workload_uuid"),
         "detail": rec["bound_violation"]}
        for r in delivered for rec in r.get("records", [])
        if rec.get("bound_violation")
    ]

    total_cost = sum(costs) if costs else None
    return {
        "model": next((r.get("model") for r in results if r.get("model")), None),
        "attempted": attempted,
        "delivered": len(delivered),
        "solved": len(solved),
        "solve_rate_pct": _pct(len(solved), attempted),
        "workloads": len(records),
        "workloads_passed": len(correct),
        "workload_pass_rate_pct": _pct(len(correct), len(records)),
        "timing": {
            "speedup_vs_reference": {
                "n": len(speedups), "mean": _mean(speedups),
                "median": _median(speedups),
                "basis": ScoreBasis.SPEEDUP_VS_REFERENCE.value,
            },
            "headroom_fraction": {
                "n": len(headroom), "mean": _mean(headroom),
                "median": _median(headroom),
                "basis": ScoreBasis.SOL_HEADROOM.value,
            },
            "sol_score": {
                "n": len(sol_scores), "mean": _mean(sol_scores),
                "median": _median(sol_scores),
                "basis": ScoreBasis.SOL_SCORE_V1.value,
            },
        },
        "cost": {
            "total_usd": total_cost,
            "priced_sessions": len(costs),
            "unpriced_sessions": attempted - len(costs),
            # The number that makes two harnesses comparable at different
            # budgets. None when the harness reports no cost, rather than a
            # token-derived guess.
            "usd_per_solved": (total_cost / len(solved))
            if (total_cost is not None and solved) else None,
        },
        # Reported alongside cost, not instead of it. Tokens survive a session
        # that was killed at the wallclock cap; the price does not, so on a run
        # where most sessions time out this is the only comparable effort measure
        # available -- and it is a count, not a currency, so nobody can mistake
        # it for a measured price.
        "tokens": {
            "input": tokens_in,
            "output": tokens_out,
            "sessions_with_tokens": token_sessions,
            "input_per_solved": (tokens_in / len(solved)) if solved else None,
        },
        "time": {
            "total_min": (sum(wallclocks) / 60) if wallclocks else None,
            "mean_min": (_mean(wallclocks) / 60) if wallclocks else None,
            "min_per_solved": (sum(wallclocks) / 60 / len(solved))
            if (wallclocks and solved) else None,
        },
        "verify_attempts": {"mean": _mean(attempts), "median": _median(attempts)},
        "languages": dict(sorted(languages.items(), key=lambda kv: -kv[1])),
        "reference_copies": dict(sorted(copies.items())),
        "failures": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "bound_violations": bound_violations,
        "by_category": _by_category(results),
    }


def _by_category(results: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for cat in CATEGORIES:
        rs = [r for r in results if r.get("category") == cat]
        if not rs:
            continue
        delivered = [r for r in rs if r.get("outcome") == "evaluated"]
        solved = [r for r in delivered if r.get("all_passed")]
        recs = [rec for r in delivered for rec in r.get("records", [])]
        passed = [rec for rec in recs if rec.get("correct")]
        out[cat] = {
            "attempted": len(rs),
            "solved": len(solved),
            "solve_rate_pct": _pct(len(solved), len(rs)),
            "workloads": len(recs),
            "workloads_passed": len(passed),
            "workload_pass_rate_pct": _pct(len(passed), len(recs)),
        }
    return out


def _per_problem(results: list[dict]) -> list[dict]:
    """One row per problem, showing which harnesses solved it.

    The point of this view is the problems *nobody* solved: they are either the
    genuinely hard kernels or a port defect, and the two are worth telling apart
    before anybody reads the headline rate.
    """
    grouped: dict[str, dict] = {}
    for r in results:
        key = r.get("problem")
        if key is None:
            continue
        row = grouped.setdefault(key, {
            "problem": key,
            "category": r.get("category"),
            "harnesses": {},
        })
        recs = r.get("records", [])
        passed = sum(1 for rec in recs if rec.get("correct"))
        speedups = [rec["speedup_vs_reference"] for rec in recs
                    if rec.get("correct") and rec.get("speedup_vs_reference")]
        row["harnesses"][r.get("harness") or "unknown"] = {
            "outcome": r.get("outcome"),
            "solved": bool(r.get("all_passed")),
            "passed": passed,
            "workloads": r.get("workloads") or len(recs),
            "languages": r.get("languages") or [],
            "reference_copy": (r.get("reference_copy") or {}).get("kind"),
            "median_speedup": _median(speedups),
            "cost_usd": (r.get("agent") or {}).get("cost_usd"),
            "wallclock_min": ((r.get("agent") or {}).get("wallclock_s") or 0) / 60,
        }
    rows = sorted(grouped.values(), key=lambda r: (r["category"] or "", r["problem"]))
    for row in rows:
        row["solved_by"] = sorted(h for h, v in row["harnesses"].items() if v["solved"])
        row["solved_by_none"] = not row["solved_by"]
    return rows
