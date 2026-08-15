#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 09 — freeze the scoring manifest.

A SOL score is only meaningful *inside* a manifest version. The manifest is the
complete statement of what a score means: the bound it is measured against
(T_SOL), the anchor that puts S=0.5 somewhere (T_b), the tolerances a
submission must satisfy, and the exact hardware and software the two reference
numbers were produced on.

    python scripts/build_manifest.py --out artifacts/09/manifest-v1.json

Rules this script enforces rather than assumes:

* **Never edit a manifest in place.** Any stack change that moves T_b needs a
  new version. The script refuses to overwrite an existing file without
  --force, and records the git SHA it was built from.
* **Count honestly.** Every problem is either in the manifest or in
  `artifacts/deferred.json` with a reason. The totals printed here are the
  numbers that must appear in the README, the paper, and any leaderboard --
  if it is 220 and not 235, it is 220 everywhere.
* **A workload with a T_SOL but no T_b is not scoreable** and is reported as
  such rather than shipped with a guessed anchor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import write_artifact  # noqa: E402
# Both are pure-python and import no torch, so they are safe at module scope.
from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
    clock_interval,
    has_clock_interval,
)
from solexbench_rocm.t_sol_at import (  # noqa: E402
    INTERVAL_FIELDS,
    MissingBoundTerms,
    t_sol_interval,
)

EXPECTED = {"L1": 94, "L2": 82, "Quant": 33, "FlashInfer-Bench": 26}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def collect_t_sol(path: Path) -> dict[str, dict]:
    """{problem: {workload_uuid: {...}}} from artifacts/03/t_sol.json."""
    doc = _load(path)
    if not doc:
        return {}
    return {k: (v.get("workloads") or {}) for k, v in doc.get("problems", {}).items()}


#: The terms `solexbench_rocm.t_sol_at` needs to re-evaluate a bound at the clock
#: a measurement actually ran at. `mac_per_cycle` is carried alongside for
#: provenance; the arithmetic uses the other three.
RECLOCK_FIELDS = ("compute_cycles", "memory_bytes", "dram_byte_per_sec",
                  "mac_per_cycle")

#: The bracket a T_b measurement carries into the manifest. `clock_mhz` is the
#: mean of the two samples and the only one a bound may be divided by; the rest
#: are what makes that number auditable rather than asserted. The window is
#: published so a reader can see how long the bracketed region was -- a 1 ms
#: window bracketed by two multi-millisecond SMI reads is a weaker bracket than
#: the same spread across a 13 ms one, and only `window_ns` says which it was.
CLOCK_FIELDS = ("clock_before_mhz", "clock_after_mhz", "clock_mhz",
                "clock_bracket_spread", "clock_bracket_threshold",
                "clock_bracket_refused", "clock_bracket_refused_reason",
                "clock_bracket_sampler_error",
                "window_ns", "window_ms", "reference_clock_bracket",
                # Why this anchor is here at all when the sweep-time gate dropped
                # it. Absent on every anchor the sweep kept, so its presence is
                # itself the flag.
                "t_b_admitted_by_interval")


def _recover_interval_anchors(doc: dict, already: set) -> dict:
    """Anchors the sweep-time gate discarded, recovered from the same artifact.

    **Why this exists.** ``time_tb_candidates.select_winners`` refuses a candidate
    whose bracket spread is above threshold, so a workload whose every variant read
    wide gets no winner and reaches the manifest as "missing T_b". Under the
    interval methodology that is no longer the right disposal: a wide bracket makes
    a measurement *uncertain*, not *absent*, and the timing was really taken. On the
    MI355X corpus this is about 10% of problems severely affected and five that
    cannot be anchored at all.

    **Why the recovery is here and not in the runner.** Doing it at sweep time would
    mean re-running the sweeps, and would split the corpus in half: problems timed
    before the change selected under the gate, problems timed after it under the
    label, with nothing in either artifact saying which rule applied. Recovering at
    manifest-build time applies one rule to every artifact that already exists,
    including the ones written before this was decided. It also costs no GPU.

    **Nothing is invented.** Every number returned is read out of the artifact's own
    ``variants`` block, which records each variant's per-workload latency beside the
    bracket it was measured under. The selection rule is the runner's: the fastest
    passing variant wins. What differs is only which candidates are eligible.
    """
    recovered: dict[str, dict] = {}
    for name, r in (doc.get("variants") or {}).items():
        if not (r.get("ok") and r.get("all_passed")):
            continue
        brackets = r.get("clock_bracket_by_workload") or {}
        ref_brackets = r.get("reference_clock_bracket_by_workload") or {}
        for uuid, ms in (r.get("latency_ms_by_workload") or {}).items():
            if uuid in already or ms is None:
                continue
            br = brackets.get(uuid)
            if not has_clock_interval(br):
                continue
            if uuid not in recovered or ms < recovered[uuid]["t_b_ms"]:
                recovered[uuid] = {
                    "variant": name, "t_b_ms": ms, **br,
                    "reference_clock_bracket": ref_brackets.get(uuid),
                    # The label, on the record, permanently. This anchor rests on a
                    # bracket the threshold refused; its T_SOL interval will be
                    # wide and that width is the honest statement of what it is
                    # worth. A reader filtering the corpus down to tight
                    # measurements filters on this and on the width, and both are
                    # columns rather than something to recompute.
                    "t_b_admitted_by_interval": True,
                }
    return recovered


#: How close two tiers' DRAM bandwidths must be to count as one number.
#:
#: 1e-9 relative is roughly six orders of magnitude tighter than any real
#: difference between two arch configurations and six orders looser than float
#: round-trip noise, so it cannot admit a genuine conflict and cannot reject a
#: representational one. It is NOT a tolerance on the physics: the merge identity
#: in `_reclock_terms` requires the two tiers to share a bandwidth exactly, and
#: this says when they do.
BANDWIDTH_IDENTICAL_REL = 1e-9


def _one_bandwidth(values: set) -> bool:
    """Are these all the same bandwidth, printed differently?"""
    lo, hi = min(values), max(values)
    return lo > 0 and (hi - lo) / hi <= BANDWIDTH_IDENTICAL_REL


def _reclock_terms(s: dict, t: dict, source: str, stats: dict) -> dict:
    """The re-clocking terms for a merged bound, correct for BOTH tiers.

    docs/TODO-MI355X.md §4.2(b) names this as a real gap and it is: taking the
    winning tier's terms is wrong exactly where `max_of_both` won. The two tiers
    are separate lower bounds and the manifest ships their max, so re-clocking
    has to re-max BOTH -- and the declared-traffic tier carries no compute term,
    so a `max_of_both` record that inherited only the traffic tier's terms would
    re-clock as if the workload had no arithmetic at all. On MI350X's manifest
    that is the tier under 328 workloads across 38 problems.

    The union is exact, not an approximation, and the algebra is short enough to
    check here. Write B_s(F) and B_t(F) for the two tiers' bounds in cycles:

        B_s(F) = max(compute_cycles_s, memory_bytes_s * F / bytes_per_sec)
        B_t(F) = max(0,                memory_bytes_t * F / bytes_per_sec)

    The shipped bound is max(B_s, B_t), and because both are a max over the same
    two shapes with the SAME `dram_byte_per_sec`,

        max(B_s(F), B_t(F)) = max(compute_cycles_s,
                                  max(memory_bytes_s, memory_bytes_t) * F / bps)

    -- i.e. one record with the larger memory term and the compute term that
    exists reproduces the two-tier max at every clock. That identity depends on
    the two tiers sharing a bandwidth figure; they are generated from the same
    arch YAML, but if they ever disagree the merge is unsound, so the disagreement
    is detected and the record is left un-re-clockable rather than silently wrong.
    """
    have = [d for d in (s, t) if d]
    bps = {d.get("dram_byte_per_sec") for d in have
           if d.get("dram_byte_per_sec") is not None}
    if len(bps) > 1 and _one_bandwidth(bps):
        # One bandwidth, two float printings of it. The MI355X tiers emit
        # 7999919999999.999 and 7999920000000.0 -- the same 7.99992e12 from the same
        # arch YAML, differing by 1 part in 8e12 because one script reached it by a
        # multiplication and the other by a division. Exact set equality read that
        # as two arch configs and refused to merge EVERY two-tier record on that
        # corpus, which is not what the guard is for and is not a failure it can
        # ever be right about: no two arch configs differ by 2e-16 relative.
        # Collapsed to the larger, which is the pre-rounding value.
        bps = {max(bps)}
    if len(bps) > 1:
        # Two bandwidths means two arch configs produced these tiers. Refusing to
        # merge leaves `t_sol_at` raising MissingBoundTerms, which is visible;
        # picking one would produce a plausible bound at every other clock.
        stats["reclock_terms_conflicting_bandwidth"] = (
            stats.get("reclock_terms_conflicting_bandwidth", 0) + 1)
        return {}

    compute = [d.get("compute_cycles") for d in have
               if d.get("compute_cycles") is not None]
    membytes = [d.get("memory_bytes") for d in have
                if d.get("memory_bytes") is not None]
    if not bps or not compute or not membytes:
        stats["reclock_terms_missing"] = stats.get("reclock_terms_missing", 0) + 1
        return {}

    stats["reclock_terms_present"] = stats.get("reclock_terms_present", 0) + 1
    if source == "max_of_both":
        stats["reclock_terms_unioned"] = stats.get("reclock_terms_unioned", 0) + 1
    return {
        "compute_cycles": max(compute),
        "memory_bytes": max(membytes),
        "dram_byte_per_sec": next(iter(bps)),
        # Provenance only. The winning tier's rate, or SOLAR's where the traffic
        # tier (which has none) won, so a reader can see which arithmetic model
        # the compute term came from.
        "mac_per_cycle": (s.get("mac_per_cycle")
                          if s.get("mac_per_cycle") is not None
                          else t.get("mac_per_cycle")),
    }


def combine_bounds(solar: dict, traffic: dict, tb: dict) -> tuple[dict, dict]:
    """One T_SOL per workload, from two derivations, with the source recorded.

    Both are lower bounds on the same quantity and neither dominates:

      solar_fused       accounts for the arithmetic, but only for the graph
                        SOLAR managed to extract. On 48 problems that graph is
                        missing tensors the definition itself declares, so the
                        bound is real but loose.
      declared_traffic  every declared input read once and every declared
                        output written once, over DRAM bandwidth. Accounts for
                        no arithmetic at all, but it is complete.

    The larger of two valid lower bounds is the better lower bound, so the rule
    is `max`, with one exception that is not optional: where a problem declares
    a tensor it *indexes* rather than streams -- a 131072-position KV cache --
    the declared total is above any real kernel's traffic and the "bound" would
    sit above the measured time. Those are caught by comparing against T_b and
    fall back to SOLAR's value, with the fallback recorded.
    """
    out: dict[str, dict] = {}
    stats = {"solar_fused": 0, "declared_traffic": 0, "max_of_both": 0,
             "traffic_rejected_above_t_b": 0, "solar_rejected_above_t_b": 0,
             "no_valid_bound": 0}
    for key in set(solar) | set(traffic):
        s_w, t_w = solar.get(key, {}), traffic.get(key, {})
        merged: dict[str, dict] = {}
        for u in set(s_w) | set(t_w):
            s, t = s_w.get(u) or {}, t_w.get(u) or {}
            s_cyc, t_cyc = s.get("t_sol_cycles"), t.get("t_sol_cycles")
            measured = ((tb.get(key) or {}).get(u) or {}).get("t_b_ms")
            # A candidate bound above the measured time is not a loose lower
            # bound, it is not a lower bound at all -- it would make
            # (T_b - T_SOL) negative and push scores past 1. The rule is
            # symmetric: reject any candidate that fails, take the max of what
            # survives, and if nothing survives the workload is not scoreable
            # and is counted as such rather than shipped with a bad anchor.
            if measured is not None:
                if t_cyc is not None and t.get("t_sol_ms", 0) > measured:
                    stats["traffic_rejected_above_t_b"] += 1
                    t_cyc = None
                if s_cyc is not None and s.get("t_sol_ms", 0) > measured:
                    stats["solar_rejected_above_t_b"] += 1
                    s_cyc = None
            if s_cyc is not None and t_cyc is not None:
                source = "max_of_both" if t_cyc > s_cyc else "solar_fused"
                chosen = t if t_cyc > s_cyc else s
            elif s_cyc is not None:
                source, chosen = "solar_fused", s
            elif t_cyc is not None:
                source, chosen = "declared_traffic", t
            else:
                stats["no_valid_bound"] += 1
                continue
            stats[source] += 1
            # The re-clocking terms are recomputed from BOTH tiers, never
            # inherited from the winner: `chosen` is one tier's record, and for
            # `max_of_both` the winner is often the traffic tier, whose
            # compute_cycles is 0 by construction. Dropping them first means a
            # record `_reclock_terms` declines to merge is left visibly
            # un-re-clockable (t_sol_at raises) rather than quietly carrying one
            # tier's half of the answer.
            base = {k: v for k, v in chosen.items() if k not in RECLOCK_FIELDS}
            merged[u] = {**base, "t_sol_source": source,
                         "t_sol_cycles_solar": s_cyc,
                         "t_sol_cycles_traffic": t.get("t_sol_cycles"),
                         **_reclock_terms(s, t, source, stats)}
        if merged:
            out[key] = merged
    return out, stats


def collect_t_b(directory: Path, f_lock_mhz: int | None = None,
                clock_basis: str = "locked") -> dict[str, dict]:
    """{problem: {workload_uuid: {variant, t_b_ms}}} from artifacts/06.

    ``f_lock_mhz`` refuses any artifact whose *stamped* clock differs. T_b is a
    wall-clock time, so mixing two clocks rescales those problems' scores by the
    ratio between them — silently, and per problem.

    This is not hypothetical. Merging two ports of this benchmark added 87 T_b
    artifacts stamped F_LOCK 1300 into a directory of artifacts stamped 1640, and
    no conflict was raised, because a three-way merge does not conflict on a file
    present on only one side. The manifest then built from the mixture without
    complaint. Every one of those files was internally correct and correctly
    stamped; the *directory* was wrong, and a directory has no provenance of its
    own — which is exactly why every artifact here carries one.

    Rejecting at the point of consumption rather than asking the merger to be
    careful is the only version of this that stays true: any directory that
    accumulates per-problem artifacts across two machines has the same hazard.

    **What this check does NOT do, stated because an earlier version of this
    docstring claimed otherwise.** It compares the stamp against the preset table.
    Both come from the same place, so it catches artifacts from *another clock* and
    is blind to an artifact whose stamp is simply **wrong** — and that happens: an
    unreset determinism sweep once left a node at a 1900 MHz setpoint while
    ``provenance.f_lock_mhz()`` returned the preset's 1640 without reading a
    device, so 143 artifacts measured at ~1860 MHz were stamped 1640, and 1640 was
    checked against 1640 and passed. The table is not the hardware.

    Closing that requires reading the setpoint back off the GPUs before measuring
    and stamping the clock actually observed, which is a change to the timing
    runners rather than to this function. This check remains necessary and is not
    sufficient.

    **``clock_basis="unlocked"``** replaces the F_LOCK comparison rather than
    relaxing it. There is no single clock to compare a stamp against on that
    part, so the guard moves down a level: every winner record must carry its own
    clock bracket. Records that do not are dropped and counted; an artifact where
    *no* record carries clock evidence is rejected whole, loudly. The reading is
    the same one the F_LOCK guard applies -- an unknown clock is not a permissive
    one -- moved from once per artifact to once per measurement, which is where
    the clock actually varies.

    **What the guard now asks is ``has_clock_interval``, not ``has_clock_evidence``.**
    A bracket refused for spread has two real samples and supports an interval-valued
    T_SOL; it is admitted, labelled, and published with its width. A bracket with no
    samples at all -- ``sampler_error``, ``no_clock_evidence`` -- is still refused
    here, because no width can be stated for a window nobody sampled. The refusal
    counts are untouched by this and are still reported (``_bracket_summary``); what
    changed is what a refusal *does*, not whether it is recorded.
    """
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    unlocked = clock_basis == "unlocked"
    foreign: list[tuple[str, object]] = []
    no_evidence: list[str] = []
    dropped = 0
    recovered_total = 0
    for f in sorted(directory.glob("*.json")):
        doc = _load(f)
        if not doc:
            continue
        if unlocked:
            winners = doc.get("winner_by_workload") or {}
            kept = {u: w for u, w in winners.items() if has_clock_interval(w)}
            dropped += len(winners) - len(kept)
            # An artifact with no winners at all is not skipped any more: under the
            # gate that was the signature of a problem where every bracket read
            # wide, which is exactly the population the interval methodology
            # exists to recover. Five problems on this corpus are in it.
            recovered = _recover_interval_anchors(doc, set(kept))
            recovered_total += len(recovered)
            kept.update(recovered)
            if not kept:
                # Not "zero scoreable workloads" -- a rejected artifact. A file
                # of anchors with no clock evidence at all is indistinguishable
                # from one measured on another node at another clock, and that
                # is the exact failure F18 was.
                no_evidence.append(f.name)
                continue
            out[doc.get("problem", f.stem)] = kept
            continue
        # -- Locked basis below. Byte-identical to what it was before the interval
        # methodology existed, including this skip, which the unlocked branch above
        # no longer shares. The MI350X corpus is frozen and must not move.
        if not doc.get("winner_by_workload"):
            continue
        # None means the artifact predates provenance stamping of F_LOCK, which is
        # a different problem from being measured at the wrong clock; those are
        # admitted, and check_06 already requires provenance separately.
        measured_at = (doc.get("_provenance") or {}).get("f_lock_mhz")
        if f_lock_mhz is not None and measured_at not in (None, f_lock_mhz):
            foreign.append((f.name, measured_at))
            continue
        out[doc.get("problem", f.stem)] = doc["winner_by_workload"]
    if recovered_total:
        print(f"\n  UNLOCKED BASIS: admitted {recovered_total} T_b measurement(s) "
              f"whose bracket the sweep-time threshold refused. Each is published "
              f"with an interval-valued T_SOL and carries "
              f"`t_b_admitted_by_interval`; the width is the statement of what it "
              f"is worth. The refusal counts are unchanged.\n", file=sys.stderr)
    if no_evidence or dropped:
        print(f"\n  UNLOCKED BASIS: dropped {dropped} T_b measurement(s) with no "
              f"usable clock bracket, and REJECTED {len(no_evidence)} artifact(s) "
              f"carrying none at all:", file=sys.stderr)
        for name in no_evidence[:5]:
            print(f"    {name}", file=sys.stderr)
        if len(no_evidence) > 5:
            print(f"    ... and {len(no_evidence) - 5} more", file=sys.stderr)
        print("  Unlocked, T_b's clock is per measurement; without one there is "
              "nothing to divide a bound by. An unknown clock is not a "
              "permissive one.\n", file=sys.stderr)
    if foreign:
        print(f"\n  REJECTED {len(foreign)} T_b artifact(s) measured at a different "
              f"clock than F_LOCK={f_lock_mhz}:", file=sys.stderr)
        for name, at in foreign[:5]:
            print(f"    {name} (F_LOCK {at})", file=sys.stderr)
        if len(foreign) > 5:
            print(f"    ... and {len(foreign) - 5} more", file=sys.stderr)
        print("  T_b is a wall-clock time; mixing clocks rescales those problems' "
              "scores.\n", file=sys.stderr)
    return out


#: Emitted in place of the interval when one cannot be computed, so a consumer that
#: reads `t_sol_ms_published` off every record gets a None rather than a
#: KeyError-shaped hole, and a reader gets the reason without cross-referencing.
_NO_INTERVAL = {k: None for k in INTERVAL_FIELDS}


def _interval_fields(s: dict, b: dict, clock_basis: str, stats: dict) -> dict:
    """The interval-valued T_SOL for one workload, or a stated absence.

    Three things have to be true before an interval exists, and each failure is
    counted separately rather than collapsed into "no interval":

    * the basis is ``unlocked`` -- under ``locked`` there is one F_LOCK, the bound is
      a point, and this returns ``{}`` so the record is byte-identical to what the
      frozen MI350X manifest carries;
    * the T_b measurement carries two clock samples (``clock_interval``);
    * the bound carries both roofline terms, without which it cannot be re-evaluated
      at any clock at all (``MissingBoundTerms``, which is raised rather than
      guessed around -- see ``t_sol_at``).
    """
    if clock_basis != "unlocked":
        return {}
    interval = clock_interval(b)
    if interval is None:
        stats["workloads_without_clock_interval"] = (
            stats.get("workloads_without_clock_interval", 0) + 1)
        return {**_NO_INTERVAL, "t_sol_interval_absent": "no_clock_samples"}
    try:
        fields = t_sol_interval(s, *interval)
    except MissingBoundTerms:
        # Not inferred from `bottleneck`. A record that kept only the max of the two
        # terms cannot be re-clocked, and pretending otherwise would produce a
        # plausible bound at every clock but the reference one.
        stats["workloads_without_reclock_terms"] = (
            stats.get("workloads_without_reclock_terms", 0) + 1)
        return {**_NO_INTERVAL, "t_sol_interval_absent": "no_reclock_terms"}
    stats["workloads_with_t_sol_interval"] = (
        stats.get("workloads_with_t_sol_interval", 0) + 1)
    if fields["t_sol_bottleneck_flips"]:
        stats["workloads_with_bottleneck_flip"] = (
            stats.get("workloads_with_bottleneck_flip", 0) + 1)
    return {**fields, "t_sol_interval_absent": None}


def _problem_interval_summary(entries: dict[str, dict]) -> dict:
    """Per-problem roll-up of the per-workload interval widths.

    Reportable and sortable without reprocessing: that is the requirement, and it is
    why these are stored rather than derived on read. `None` throughout when no
    workload in the problem has an interval, which is a different statement from a
    width of zero -- zero means "measured, and the bound does not move"; None means
    "not established".
    """
    widths = [e["t_sol_interval_halfwidth_rel"] for e in entries.values()
              if e.get("t_sol_interval_halfwidth_rel") is not None]
    if not widths:
        return {}
    widths.sort()
    return {
        "t_sol_interval_halfwidth_max": widths[-1],
        "t_sol_interval_halfwidth_median": widths[len(widths) // 2],
        "n_workloads_with_t_sol_interval": len(widths),
        "n_workloads_with_bottleneck_flip": sum(
            1 for e in entries.values() if e.get("t_sol_bottleneck_flips")),
        # An anchor admitted only because refusal was demoted to a label. Counted
        # per problem because "10% of problems severely affected" has to be a
        # number someone can recompute from the manifest.
        "n_workloads_admitted_by_interval": sum(
            1 for e in entries.values() if e.get("t_b_admitted_by_interval")),
    }


#: A workload whose bound moves by more than this across its own bracket is called
#: out by name in the corpus summary. 0.05 is a reporting cut, not a gate: nothing is
#: dropped for exceeding it and no score changes at it. It is set here rather than
#: derived from the distribution because the first question anyone asks of this
#: manifest is "which problems are the uncertain ones", and that needs a list, not a
#: histogram. The measured corpus splits far either side of it -- clean problems read
#: ~0.016 and the worst read ~0.15 -- so the exact cut does not decide membership.
WIDE_INTERVAL_HALFWIDTH = 0.05


def _corpus_interval_summary(problems: dict[str, dict]) -> dict:
    """Every problem's interval width in one sortable place, plus the wide ones.

    The published bound is the minimum-clock end everywhere (``t_sol_at`` explains
    why), so this summary is not a spread of published values -- it is a statement of
    how far the *other* admissible end sits from each one.
    """
    per_problem = {
        k: v["t_sol_interval_halfwidth_max"] for k, v in problems.items()
        if v.get("t_sol_interval_halfwidth_max") is not None
    }
    widths = sorted(per_problem.values())
    flips = sum(v.get("n_workloads_with_bottleneck_flip") or 0
                for v in problems.values())
    admitted = sum(v.get("n_workloads_admitted_by_interval") or 0
                   for v in problems.values())
    return {
        "note": "T_SOL is published at the MINIMUM clock of each measurement's "
                "bracket: the largest T_SOL, hence the tightest bound. Wrong in "
                "that direction is detectable -- a measurement beats its own bound "
                "and the bound check fires. Published at the maximum clock it "
                "would be undetectable (CLAUDE.md §6). "
                "`t_sol_ms_at_clock_min` / `t_sol_ms_at_clock_max` are the two "
                "ends, `t_sol_interval_halfwidth_rel` the +- around their midpoint.",
        "published_at": "clock_min",
        "n_problems_with_interval": len(per_problem),
        "halfwidth_median": widths[len(widths) // 2] if widths else None,
        "halfwidth_max": widths[-1] if widths else None,
        "wide_threshold": WIDE_INTERVAL_HALFWIDTH,
        "problems_wide": sorted(
            (k for k, v in per_problem.items() if v > WIDE_INTERVAL_HALFWIDTH),
            key=lambda k: -per_problem[k]),
        # Both counted at the top level because both are claims the release notes
        # make, and a number quoted in prose that nobody can recompute from the
        # artifact is how the count of scoreable problems drifted last time.
        "n_workloads_with_bottleneck_flip": flips,
        "n_workloads_admitted_by_interval": admitted,
    }


def _bracket_summary(t_b: dict[str, dict]) -> dict:
    """Refusal statistics over every T_b that made it into the manifest.

    Note what this can and cannot count. Measurements refused at *sweep* time
    never became winners and so are not here; `time_tb_candidates` records those
    per problem in `clock_bracket_refused_by_workload`, and they show up in this
    manifest as workloads missing T_b. What this counts is what survived --
    which is the number a reader needs in order to know whether "the clock is
    characterised" is a claim about the whole corpus or about a filtered part of
    it.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from sol_execbench.core.bench.clock_bracket import summarize_brackets

    records = [w for wl in t_b.values() for w in wl.values()]
    summary = summarize_brackets(records)
    summary["n_with_clock_evidence"] = sum(
        1 for r in records
        if isinstance(r.get("clock_mhz"), (int, float)) and r["clock_mhz"] > 0
    )
    summary["n_t_b_total"] = len(records)
    return summary


def _methodology_of(directory: Path) -> str:
    """Which timing methodology produced the T_b measurements.

    Read from the artifacts rather than assumed, and a mixture is reported as
    a mixture instead of being collapsed to whichever came first.
    """
    seen = set()
    for f in sorted(directory.glob("*.json")):
        doc = _load(f) or {}
        prov = doc.get("_provenance") or {}
        m = prov.get("methodology") or (doc.get("environment") or {}).get("methodology")
        if m:
            seen.add(m)
    if not seen:
        return "hip_events"        # the harness default; see device.py
    return "+".join(sorted(seen))


def collect_tolerances(directory: Path) -> dict[str, dict]:
    """{problem: {workload_uuid: tolerance}} from artifacts/05."""
    out: dict[str, dict] = {}
    if not directory.exists():
        return out
    for f in sorted(directory.glob("*.json")):
        doc = _load(f)
        if not doc:
            continue
        per = {}
        for w in doc.get("per_workload", []):
            if w.get("tolerance"):
                per[w["workload_uuid"]] = w["tolerance"]
        if per:
            out[doc.get("problem", f.stem)] = per
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/09/manifest-v1.json")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--t-sol-traffic", default="artifacts/03/t_sol_traffic.json")
    ap.add_argument("--t-b", default="artifacts/06/authoritative")
    ap.add_argument("--tolerances", default="artifacts/05")
    ap.add_argument("--deferred", default="artifacts/deferred.json")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest (do not do this to a "
                         "published one -- cut a new version instead)")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        sys.exit(
            f"{out} already exists. A manifest is frozen once published: "
            f"scores are only comparable within a version. Cut a new version "
            f"instead, or pass --force if this one was never published."
        )

    methodology = _methodology_of(Path(a.t_b))
    # The clock every T_b in this manifest must have been measured at, read from
    # the same table lock_clocks() applies from, so the manifest and the hardware
    # cannot disagree about it.
    from provenance import f_lock_mhz as _f_lock

    # AMD, docs/TODO-MI355X.md §3.4: which clock basis is this manifest built on?
    #
    # `locked`   (default) — one F_LOCK for the whole manifest, the MI350X case.
    # `unlocked` — no F_LOCK exists; each measurement carries its own bracketed
    #              clock and the guard moves into `collect_t_b`.
    #
    # Read from the environment rather than a flag so it travels with the sweep
    # that produced the artifacts: a manifest built on the other basis from the
    # one the T_b runs used is not a build error, it is a wrong manifest.
    from sol_execbench.core.bench.clock_bracket import clock_basis as _basis

    clock_basis = _basis()
    if clock_basis not in ("locked", "unlocked"):
        raise SystemExit(
            f"SOLEXBENCH_CLOCK_BASIS={clock_basis!r} is not a basis this build "
            f"knows. Use 'locked' (one F_LOCK) or 'unlocked' (per-measurement "
            f"bracketed clocks).")

    expected_f_lock = _f_lock()
    if expected_f_lock is None and clock_basis != "unlocked":
        # The guard below is only a guard when it has a number to compare
        # against. `f_lock_mhz()` returns None off-GPU with no override set, and
        # `collect_t_b(..., None)` then admits artifacts from any clock -- so
        # building the manifest in the wrong environment would silently restore
        # exactly the defect F18 fixed, and the output would look normal.
        # Refusing is the only safe reading: an unknown clock is not a
        # permissive one.
        #
        # Note the two F_LOCK guards in this file treat None in OPPOSITE ways
        # and both are deliberate: HERE a None means "we do not know the clock,
        # refuse"; inside `collect_t_b` a None *stamp on an artifact* means "this
        # artifact predates F_LOCK stamping", which is a different problem and is
        # admitted. Do not unify them.
        raise SystemExit(
            "cannot resolve F_LOCK: no GPU preset and no SOLEXBENCH_F_LOCK_MHZ.\n"
            "  The T_b clock guard cannot run without it, and building without "
            "the guard is how a two-clock T_b directory got into a manifest in "
            "the first place (STATE.md F18).\n"
            "  Set SOLEXBENCH_F_LOCK_MHZ=<achieved MHz> to build off-GPU, or "
            "SOLEXBENCH_CLOCK_BASIS=unlocked to score from per-measurement "
            "bracketed clocks (docs/TODO-MI355X.md §4.3 option 2). The unlocked "
            "basis is NOT the permissive option: it refuses every measurement "
            "that carries no clock evidence of its own.")

    t_b = collect_t_b(Path(a.t_b), expected_f_lock, clock_basis)
    if clock_basis == "unlocked" and not t_b:
        raise SystemExit(
            f"unlocked basis: no T_b artifact in {a.t_b} carries a usable clock "
            f"bracket, so nothing is scoreable.\n"
            f"  Re-run the task 06 sweep with SOLEXBENCH_CLOCK_BASIS=unlocked so "
            f"the eval driver brackets each timed window, or build on the locked "
            f"basis with a measured SOLEXBENCH_F_LOCK_MHZ.")
    t_sol, bound_sources = combine_bounds(
        collect_t_sol(Path(a.t_sol)),
        collect_t_sol(Path(a.t_sol_traffic)),
        t_b,
    )
    tolerances = collect_tolerances(Path(a.tolerances))
    # The ledger is `{_note, dataset_total, ..., problems: {key: reason}}`.
    # Read the mapping out of it rather than iterating the whole document --
    # `sorted(doc)` over the outer dict would list "_note" as a deferred
    # problem and inflate every count that quotes this file.
    deferred_doc = _load(Path(a.deferred)) or {}
    deferred = deferred_doc.get("problems", {})
    if not isinstance(deferred, dict):
        sys.exit(f"{a.deferred}: 'problems' must map problem key -> reason")

    data = Path(a.data)
    census = {
        f"{cat}__{p.name}": cat
        for cat in EXPECTED
        for p in sorted((data / cat).glob("*"))
        if (p / "definition.json").exists()
    }

    problems: dict[str, dict] = {}
    stats = {"scoreable_workloads": 0, "workloads_missing_t_sol": 0,
             "workloads_missing_t_b": 0, "workloads_missing_tolerance": 0}

    for key, category in sorted(census.items()):
        sol = t_sol.get(key, {})
        tb = t_b.get(key, {})
        tol = tolerances.get(key, {})
        uuids = sorted(set(sol) | set(tb) | set(tol))
        entries = {}
        for u in uuids:
            s, b = sol.get(u, {}), tb.get(u, {})
            has_sol = "t_sol_cycles" in s
            has_tb = "t_b_ms" in b
            if not has_sol:
                stats["workloads_missing_t_sol"] += 1
            if not has_tb:
                stats["workloads_missing_t_b"] += 1
            if u not in tol:
                stats["workloads_missing_tolerance"] += 1
            if has_sol and has_tb:
                stats["scoreable_workloads"] += 1
            entries[u] = {
                # Cycles first: it is the F_LOCK-invariant figure, so a future
                # re-lock rescales the ms column by one division instead of
                # invalidating the manifest's analytic half.
                "t_sol_cycles": s.get("t_sol_cycles"),
                "t_sol_ms": s.get("t_sol_ms"),
                # Which derivation produced the bound, and what the other one
                # said. Two lower bounds on the same quantity, neither of them
                # dominating -- a consumer that cares can filter on this.
                "t_sol_source": s.get("t_sol_source"),
                "t_sol_cycles_solar": s.get("t_sol_cycles_solar"),
                "t_sol_cycles_traffic": s.get("t_sol_cycles_traffic"),
                "sol_bottleneck": s.get("bottleneck"),
                "t_b_ms": b.get("t_b_ms"),
                # "Optimized PyTorch" is not reproducible; a named variant is.
                "t_b_variant": b.get("variant"),
                # -- The four terms that let a consumer re-evaluate this bound at
                # the clock the measurement ran at (§4.2(c)). `t_sol_at` cannot
                # see them any other way; without them it raises
                # MissingBoundTerms on every record, however correct it is.
                **{k: s.get(k) for k in RECLOCK_FIELDS},
                # -- The clock this T_b was measured at, as evidence rather than
                # as a single number: both samples, their spread, the threshold
                # in force, the verdict, and the window they bracket. Under the
                # locked basis every one of these is None and `t_sol_ms` above
                # remains the bound at F_LOCK.
                **{k: b.get(k) for k in CLOCK_FIELDS},
                # -- T_SOL as an INTERVAL over the bracket's two clocks (§2 of the
                # approved unlocked methodology). Empty under the locked basis, and
                # empty for any record that cannot support it, in which case the
                # reason is on the record rather than absent.
                **_interval_fields(s, b, clock_basis, stats),
                "tolerance": tol.get(u),
                "scoreable": has_sol and has_tb,
            }
        problems[key] = {
            "category": category,
            "n_workloads": len(entries),
            "n_scoreable": sum(1 for e in entries.values() if e["scoreable"]),
            # Per problem, aggregated from its own workloads so a reader can sort
            # 235 problems by how much clock ambiguity their bound carries without
            # opening any of them. `max` is the headline because a problem is only
            # as well-determined as its worst workload.
            **_problem_interval_summary(entries),
            "workloads": entries,
            "deferred": deferred.get(key),
        }

    scoreable_problems = [k for k, v in problems.items() if v["n_scoreable"]]
    payload = {
        "manifest_version": a.version,
        # Stated at the top level, not buried in provenance: a manifest built
        # from hip_events traces and one built from rocprof traces are not
        # comparable, and the whole point of recording the methodology per
        # trace is lost if the manifest that aggregates them does not say
        # which one it aggregated.
        "methodology": methodology,
        "score_formula": "S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))",
        "problem_set": {
            "total_in_dataset": len(census),
            "expected_by_category": EXPECTED,
            "scoreable_problems": len(scoreable_problems),
            "deferred_problems": sorted(deferred),
            # Stated once, here, so every other document can quote one number
            # rather than each computing its own and drifting.
            "headline_count": len(scoreable_problems),
        },
        "stats": stats,
        # Which trees this manifest was actually built from. Not decoration:
        # task 06's gate used to read `artifacts/06-MI355X/authoritative` by
        # assumption while the manifest was built from `authoritative-merged`,
        # so the gate reported "T_b covers only 208 of 220" against a manifest
        # that had 219. The gate was auditing a tree nobody published. A gate
        # must verify what was shipped, and it can only do that if what was
        # shipped says so.
        "sources": {
            "t_b": str(a.t_b),
            "t_sol": str(a.t_sol),
            "t_sol_traffic": str(a.t_sol_traffic),
            "tolerances": str(a.tolerances),
        },
        "bound_sources": bound_sources,
        # AMD: what a T_SOL in this manifest is expressed at.
        #
        # `locked`   — every t_sol_ms is at f_lock_mhz and is directly usable.
        # `unlocked` — t_sol_ms is at the REFERENCE clock sol_bounds.py was run
        #              with, and is a reference value, not the bound the score
        #              should use. The bound for a measurement is
        #              `t_sol_ms_published` -- T_SOL at the MINIMUM clock of that
        #              measurement's own bracket -- and the two ends of the
        #              interval are `t_sol_ms_at_clock_min` /
        #              `t_sol_ms_at_clock_max`, which is why the four re-clocking
        #              terms and the bracket travel in the same record.
        #
        # `t_sol_ms` is deliberately NOT overwritten with the published value: it
        # is the reference-clock figure `sol_bounds.py` derived, it is what the
        # cycle column corresponds to, and silently redefining a field that the
        # frozen MI350X manifest also carries would make the two versions of the
        # same key mean different things.
        #
        # Stated at the top level for the same reason `methodology` is: a
        # consumer that reads `t_sol_ms` off an unlocked manifest without
        # re-clocking gets a plausible wrong number, and nothing else in the file
        # would tell it so.
        "clock_basis": clock_basis,
        # AMD, unlocked basis: T_SOL is an interval, and this is the corpus-level
        # view of it. Absent under the locked basis, where the bound is a point.
        **({"t_sol_interval": _corpus_interval_summary(problems)}
           if clock_basis == "unlocked" else {}),
        "clock_bracket": {
            "note": "Bounds how wrong assuming one clock for the timed window "
                    "is; does NOT recover the window's clock, and does NOT "
                    "address the short-window timing bias "
                    "(docs/methodology.md §7), which is a separate and larger "
                    "effect and is not a clock effect.",
            **_bracket_summary(t_b),
        },
        "problems": problems,
    }
    write_artifact(out, f"09-manifest-{a.version}", payload)

    print(f"manifest {a.version} -> {out}")
    print(f"  problems scoreable   {len(scoreable_problems)}/{len(census)}")
    print(f"  workloads scoreable  {stats['scoreable_workloads']}")
    for k in ("workloads_missing_t_sol", "workloads_missing_t_b",
              "workloads_missing_tolerance"):
        print(f"  {k:<28} {stats[k]}")
    if clock_basis == "unlocked":
        iv = payload["t_sol_interval"]
        print(f"  T_SOL interval, published at {iv['published_at']}:")
        print(f"    problems with an interval  {iv['n_problems_with_interval']}")
        print(f"    halfwidth median / max     "
              f"{_pct(iv['halfwidth_median'])} / {_pct(iv['halfwidth_max'])}")
        print(f"    wide (> {_pct(iv['wide_threshold'])})              "
              f"{len(iv['problems_wide'])} problems")
        print(f"    bottleneck flips           "
              f"{iv['n_workloads_with_bottleneck_flip']} workloads")
        print(f"    admitted by interval       "
              f"{iv['n_workloads_admitted_by_interval']} workloads "
              f"(bracket refused, measurement kept)")
        for k in ("workloads_without_clock_interval",
                  "workloads_without_reclock_terms"):
            if stats.get(k):
                print(f"    {k:<26} {stats[k]}")
    if len(scoreable_problems) < len(census):
        missing = sorted(set(census) - set(scoreable_problems) - set(deferred))
        print(f"\n{len(missing)} problems are neither scoreable nor recorded in "
              f"{a.deferred}. Each is a gap without a decision behind it:")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
