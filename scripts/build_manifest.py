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
import math
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
    t_sol_ms_at,
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
RECLOCK_TERM_FIELDS = ("compute_cycles", "memory_bytes", "dram_byte_per_sec",
                       "mac_per_cycle")

#: The clock each tier's OWN `t_sol_ms`/`t_sol_cycles` column was converted at.
#: Not an input to the arithmetic -- the four terms above are clock-free, which is
#: the whole reason a bound can be re-evaluated per measurement -- but the only
#: thing that makes those columns readable.
#:
#: D63: `artifacts/03-MI355X/t_sol.json` states `f_lock_mhz: 2400` in its header
#: while 2902 of its 2998 records are converted at 1.8 GHz and 96 at 2.4 GHz, and
#: nothing anywhere said so. A cycle count on this part is meaningless without the
#: clock beside it, so the clock travels in the record, per record.
#:
#: `f_ref_mhz` on a merged record is the CHOSEN tier's -- it describes the
#: `t_sol_ms` column the record actually carries. The two suffixed fields keep both
#: tiers' clocks visible, because a merged record genuinely has two and collapsing
#: them to one is how the mixture hid in the first place.
CLOCK_PROVENANCE_FIELDS = ("f_ref_mhz", "f_ref_mhz_solar", "f_ref_mhz_traffic")

#: What is stripped off the winning tier's record before merge, and what is copied
#: back out of the merged record into the manifest entry. The two lists have to be
#: the same list, or a field is dropped on one side and read as None on the other.
RECLOCK_FIELDS = RECLOCK_TERM_FIELDS + CLOCK_PROVENANCE_FIELDS

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


def _reclock_terms(s: dict, t: dict, source: str, stats: dict,
                   compared_at_mhz: float | None = None,
                   unioned: bool | None = None) -> dict:
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

    **The reference clocks (D63).** The bandwidth guard above provably cannot see
    a clock mismatch, and it is worth being exact about why: `DRAM_byte_per_cycle`
    is *defined* in the arch YAML as `bytes_per_sec / freq`, so the two printings
    of the same 8.0 TB/s are `4444.4 * 1.8e9` and `3333.3 * 2.4e9`, which are the
    same number -- 7,999,920,000,000 -- to the last bit. A tier at f_ref 1.8 GHz
    and a tier at 2.4 GHz agree on `dram_byte_per_sec` exactly, and that is how
    `artifacts/03-MI355X/t_sol.json` (1.8 GHz body, 2400 header) merged with a
    2.4 GHz traffic tier for three manifest versions without a word.

    So `f_ref_mhz` gets its own guard -- but only over the ground it can be right
    about, and *compared_at_mhz* is what says which ground that is:

    * ``compared_at_mhz`` set: `combine_bounds` evaluated BOTH tiers at one clock
      (the T_b measurement's own bracket minimum) before the rejection gate and the
      tier comparison, so no comparison in this build read either tier's stored
      `t_sol_ms`, and a difference between the two stored f_refs entered nothing.
      Refusing the merge there would delete the published bound from **2826 of the
      3717 scoreable MI355X workloads across 181 problems** -- measured, by the
      adversarial review of the D63 write-up, which forced the existing bandwidth
      refusal on that corpus and counted the result -- and would send exactly those
      workloads back to being scored against the mixed-clock legacy column this
      guard exists to condemn. A guard that costs 76% of the corpus to catch a
      difference that changed no number is not a guard.
    * ``compared_at_mhz`` None: there was no measurement clock to evaluate at, so
      the tiers were compared by their own `t_sol_ms` columns. THAT comparison is
      a unit error whenever the two f_refs differ -- it is D63's original defect,
      biased by 2.4/1.8 = 1.3333x in the traffic tier's favour -- and the record is
      left un-re-clockable so that the conflict is visible rather than shipped.

    *unioned* overrides how the `reclock_terms_unioned` count is taken. It defaults
    to `source == "max_of_both"`, which is the same thing everywhere except on the
    gathered labels, where the source is named for the term that binds and both
    tiers still contribute one.
    """
    have = [d for d in (s, t) if d]
    f_refs = {d.get("f_ref_mhz") for d in have if d.get("f_ref_mhz") is not None}
    if len(f_refs) > 1 and compared_at_mhz is None:
        stats["reclock_terms_conflicting_f_ref"] = (
            stats.get("reclock_terms_conflicting_f_ref", 0) + 1)
        return {}
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
    if unioned if unioned is not None else source == "max_of_both":
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
        # Both tiers' reference clocks, side by side and un-collapsed. A merged
        # record has two of them and a single field would have to pick one; the
        # picking is what nobody wrote down for three manifest versions.
        "f_ref_mhz_solar": s.get("f_ref_mhz"),
        "f_ref_mhz_traffic": t.get("f_ref_mhz"),
    }


def _record_f_ref_mhz(rec: dict) -> float | None:
    """The clock *rec*'s own `t_sol_ms` column was converted at.

    Stated by the record when the tier writer emits `f_ref_mhz`. Where it does not
    -- every artifact written before that field existed, which is all of the ones
    on disk today -- it is RECOVERED by algebra over two numbers in the same
    record, `t_sol_cycles / (t_sol_ms * 1e3)`, which is the divisor the writer used
    and nothing else. That is a reading of the artifact, not an assumption about
    it: it is how D63 established that 2902 of `t_sol.json`'s 2998 records are at
    1.8 GHz while the header says 2400.

    It is used only to restate a *display* column. No bound is ever divided by it
    -- bounds come from the clock-free terms at the clock the measurement ran at.
    """
    f = rec.get("f_ref_mhz")
    if f:
        return float(f)
    cycles, ms = rec.get("t_sol_cycles"), rec.get("t_sol_ms")
    if cycles and ms and ms > 0:
        return float(cycles) / (float(ms) * 1e3)
    return None


def _tier_times_ms(s: dict, t: dict, f_mhz: float | None,
                   stats: dict) -> tuple[float | None, float | None,
                                         float | None]:
    """Both tiers' bounds in ms on ONE clock, and which clock that was.

    At *f_mhz* -- the measurement's own clock -- from the clock-free terms, which
    is the form in which the two are actually comparable. Both or neither: 572
    SOLAR records on this corpus predate the term split and cannot be evaluated at
    any clock, and evaluating only the tier that can would compare a stored
    reference-clock column against a re-evaluated one, which is the same unit error
    in new clothes. Those fall back to both tiers' stored columns, exactly as
    before, and the third return value is None to say the comparison was not made
    on one clock.

    **Every outcome is counted, and each counter is named for what it counts.**
    An earlier version incremented a single `tier_compared_at_reference_clock`
    inside the `except` branch alone, so the manifest published 348 -- the number
    of records that could NOT be put on one clock -- under a name asserting they
    had been. The true count was 3369. The inversion was worse than a mislabel:
    those 348 are exactly the records still being compared through the mixed-clock
    stored column that this correction (D63) exists to retire, so the one
    statistic a reader would use to size the remaining exposure named it
    backwards, and 8.8% coverage read as if it were 85%.

    **"Compared" with only one tier present is counted separately.** A record
    with no SOLAR entry at all cannot raise `MissingBoundTerms` -- there is
    nothing to raise about -- so it takes the success branch, while a record with
    a SOLAR *error* entry falls back. Both are SOLAR failures and the split
    between them would otherwise track which flavour occurred rather than what
    the counter names. `tier_compared_one_tier_only` says how much of
    `tier_compared_at_reference_clock` is a comparison of one tier against
    nothing. Measured on MI355X manifest-v4 (instrumented build, all 3957
    records): both tiers 3360, one tier 0, fell back 357, no measurement clock
    240 -- so the main counter is honest on this corpus today, and this exists so
    that stays checkable rather than assumed.
    """
    if f_mhz is not None:
        try:
            on_one_clock = (t_sol_ms_at(s, f_mhz) if s else None,
                            t_sol_ms_at(t, f_mhz) if t else None,
                            f_mhz)
        except (MissingBoundTerms, ValueError):
            # Terms too old to re-evaluate: falls through to the stored columns.
            stats["tier_fell_back_to_stored_clock"] = (
                stats.get("tier_fell_back_to_stored_clock", 0) + 1)
        else:
            stats["tier_compared_at_reference_clock"] = (
                stats.get("tier_compared_at_reference_clock", 0) + 1)
            if s is None or t is None:
                stats["tier_compared_one_tier_only"] = (
                    stats.get("tier_compared_one_tier_only", 0) + 1)
            return on_one_clock
    else:
        # No measurement clock to compare at -- not a defect in the T_SOL record.
        stats["tier_no_measurement_clock"] = (
            stats.get("tier_no_measurement_clock", 0) + 1)
    return (s.get("t_sol_ms") if s else None,
            t.get("t_sol_ms") if t else None,
            None)


#: How far SOLAR's memory term may sit ABOVE the declared allocation before
#: `_solar_arithmetic_only` refuses to discard it.
#:
#: The rule's whole premise is that SOLAR's memory term IS the allocation --
#: measured, on the 63 records where it fires on MI355X manifest-v4, the ratio
#: `solar_memory_bytes / allocation_bytes` spans **0.9609 .. 0.99999995** and never
#: reaches 1. SOLAR lands slightly UNDER because its count is a deduplicated per-op
#: sum at a single `bytes_per_element` (54 B under on `FlashInfer-Bench__018`), so
#: the interesting direction is the other one: a term materially ABOVE the
#: allocation is SOLAR reporting traffic the allocation does not contain -- a real
#: second stream -- and discarding the whole term would delete it.
#:
#: 1% is two orders of magnitude above the largest deviation measured today (5e-8
#: relative, on 018) and far below the smallest thing that could plausibly be an
#: extra stream, so it can neither fire on rounding nor miss a tensor. It is a
#: guard, not a tolerance on the physics.
SOLAR_MEMORY_ABOVE_ALLOCATION_REL = 0.01

#: The declared-traffic tier's own evidence for the gathered correction: what the
#: declaration allocated, which axes the workload indexes rather than streams, and
#: what those indices actually name. Carried onto the merged record whenever the
#: correction fires, including when SOLAR wins the comparison and the traffic tier's
#: record is not the one the manifest entry is built from.
GATHERED_AUDIT_FIELDS = ("allocation_bytes", "gathered_axes", "gathered_bytes")


def _solar_arithmetic_only(s: dict, t: dict, stats: dict) -> dict:
    """SOLAR's record with its MEMORY term discarded and its arithmetic kept.

    D18, one tier over. Where a problem *indexes* a tensor rather than streaming it
    -- a 989,669-page KV cache read at 8 pages -- `sol_gathered_traffic` reprices
    the declared-traffic tier at the rows the workload names. SOLAR's memory term
    is then the only allocation-priced number left, and `max` in time hands the
    bound straight back to it: on `FlashInfer-Bench__018` all 47 workloads publish
    SOLAR's 1,140,133,554 B streaming time, 12.8x to 24,432x (p50 127.4x) above
    their own corrected floor.

    **Why the memory term and not the bytes.** SOLAR is not naively pricing the
    gather -- measured, a bare `table[idx]` is charged the gathered rows exactly
    (8,192 B out of a 1,024,000,000 B table). The allocation enters through the
    reference's own `cache.squeeze(1).to(torch.float32)`, a real full-tensor cast
    that runs BEFORE the gather and that SOLAR traces faithfully. So the bound is a
    correct roofline for the REFERENCE'S ALGORITHM and a wrong one for the PROBLEM,
    and there is no byte subtraction that is a derivation rather than a coincidence:
    SOLAR's count is a deduplicated per-op sum at a single `bytes_per_element`,
    which is why it lands 54 B off the allocation on 018 rather than on it.

    **This is not a new rule.** MI350X v1.1 shipped exactly it --
    `scripts/rebuild_manifest_v11.py:246-270`, whose comment names this very byte
    count -- and MI350X v1.2 publishes all 47 of 018 at `declared_traffic_gathered`.
    The rule was never carried across to this builder, so the MI355X manifest
    reinstated the v1 number. Restoring it is cross-part parity, not a methodology
    change; the durable fix (teach the graph analyzer to push a gather back through
    an elementwise producer) is recommended in `docs/issues/` and not enacted here.

    The arithmetic term is carried rather than assumed to be negligible. On 018 it
    binds on 0 of 47 -- so the corrected bound IS the gathered floor -- but "it was
    small last time" is not a derivation.

    **The premise is now checked rather than assumed.** The rule used to key on the
    mere presence of `gathered_axes` and discard SOLAR's memory term whole, with
    nothing asking whether that term was in fact the allocation. On today's corpus
    it always is -- `solar_memory_bytes / allocation_bytes` is 0.9609..0.99999995
    over all 63 records where this fires, never above 1 -- but that is a property of
    the data, not of the rule, and the uncovered case fails in the direction nothing
    downstream can see: a problem with both a gathered axis and a real streaming
    tensor would have the stream deleted along with the allocation, and the
    resulting bound would be too SMALL, which no measurement can contradict.

    So two things now stop the discard, and each is counted rather than silent:

    * `allocation_bytes` absent -- there is no number to check the premise against.
      Refusing leaves the allocation-priced bound in place, which is the *detectable*
      way to be wrong: a real kernel beats it and task 03's check D fires. Applying
      the correction unverified would be the undetectable way.
    * SOLAR's memory term above the allocation by more than
      `SOLAR_MEMORY_ABOVE_ALLOCATION_REL` -- SOLAR saw traffic the allocation does
      not contain, so the term is not purely the mispricing this rule exists to
      remove. The record falls back to the ordinary max-of-both behaviour.

    Neither guard fires on the MI355X corpus: all 63 gathered records carry
    `allocation_bytes` and none exceeds it. This moves no bound today, which is the
    point -- it is here for the corpus that has not been measured yet.
    """
    allocation = t.get("allocation_bytes")
    solar_memory = s.get("memory_bytes")
    if not allocation or allocation <= 0:
        # The traffic tier repriced a gather but did not say what the declaration
        # allocated, so "SOLAR is pricing the allocation" is unverifiable here.
        stats["gathered_solar_allocation_unknown"] = (
            stats.get("gathered_solar_allocation_unknown", 0) + 1)
        return s
    if (solar_memory is not None
            and solar_memory > allocation * (1.0 + SOLAR_MEMORY_ABOVE_ALLOCATION_REL)):
        stats["gathered_solar_memory_above_allocation"] = (
            stats.get("gathered_solar_memory_above_allocation", 0) + 1)
        return s
    compute = s.get("compute_cycles")
    if compute is None:
        # Nothing to keep and nothing safe to drop: leave the record alone and
        # count it, rather than publishing a bound with no memory term and no
        # arithmetic one either.
        stats["gathered_solar_not_correctable"] = (
            stats.get("gathered_solar_not_correctable", 0) + 1)
        return s
    stats["gathered_solar_memory_discarded"] = (
        stats.get("gathered_solar_memory_discarded", 0) + 1)
    cycles = max(1, math.ceil(float(compute)))
    f_ref = _record_f_ref_mhz(s)
    return {**s,
            # Zero, not absent: `_reclock_terms` maxes the two tiers' memory terms
            # and a missing one would leave the record un-re-clockable instead of
            # leaving the gathered floor to supply it.
            "memory_bytes": 0,
            "memory_cycles_at_f_ref": 0.0,
            "t_sol_cycles": cycles,
            "t_sol_cycles_exact": float(compute),
            "t_sol_ms": (cycles / (f_ref * 1e3)) if f_ref else None,
            "bottleneck": "compute"}


#: The bands `leaderboard/ingest.py::bound_quality` already sorts published bounds
#: into (D39). Duplicated rather than imported because that module is a FastAPI
#: application in its own venv and this script must run without it; the vocabulary
#: is what has to match, and a value written here is read there under the same name.
BOUND_QUALITY_BANDS = ((1000.0, "vacuous"), (100.0, "loose"), (2.0, "ok"),
                       (0.0, "narrow"))


def _bound_quality(t_sol_ms: float | None,
                   t_b_ms: float | None) -> tuple[str | None, float | None]:
    """(band, headroom) for one bound. (None, None) when there is nothing to band."""
    if not t_sol_ms or not t_b_ms or t_sol_ms <= 0:
        return None, None
    h = t_b_ms / t_sol_ms
    for lo, label in BOUND_QUALITY_BANDS:
        if h >= lo:
            return label, h
    return None, h


def _published_bound_ms(terms: dict, chosen: dict, bracket: tuple | None,
                        gate_mhz: float | None) -> float | None:
    """The bound this manifest will PUBLISH for one workload, in ms.

    Not `chosen["t_sol_ms"]`, except in the one case where it is. Which number a
    manifest publishes depends on the clock basis, and this reads the basis off the
    anchor rather than being told it, because the anchor is what decides:

    * **bracketed anchor** (the unlocked basis): the published bound is T_SOL
      re-evaluated at the MINIMUM of the measurement's own bracket -- exactly what
      `_interval_fields` emits as `t_sol_ms_published`, from the same terms at the
      same clock, so the two agree by construction rather than by coincidence.
    * **no bracket** (the locked basis): there is one F_LOCK, `_interval_fields`
      returns nothing, and `t_sol_ms` *is* the published bound.

    The two cases cannot be confused, and that is worth stating because getting it
    wrong is D63 again: under the unlocked basis `t_sol_ms` is a cycle count divided
    by whichever REFERENCE clock its tier happened to use (1.8 GHz for SOLAR, 2.4
    GHz for the traffic tier), and banding that column would compare a 2.4 GHz
    number against a 2.0 GHz measurement. `collect_t_b` is what makes the split
    safe: on the unlocked basis it admits **no** anchor without a clock interval, so
    a workload that has a T_b to band against always has a bracket, and the
    stored-column branch is unreachable there. `tests/scripts/
    test_build_manifest_bound_quality.py` pins that invariant.

    None is returned for a bracketed record whose terms `_reclock_terms` declined to
    merge: it has a published bound in the manifest only as `t_sol_interval_absent`,
    and inventing one from the stored column to get a band is the same unit error.
    """
    if bracket is None:
        return chosen.get("t_sol_ms")
    if not terms or gate_mhz is None:
        return None
    return t_sol_ms_at(terms, gate_mhz)


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

    **Both tiers are evaluated at ONE clock before either comparison** -- the
    minimum of the T_b measurement's own clock bracket, which is the clock the
    published bound is taken at (`t_sol_at`, and `_interval_fields` below). D63's
    two earlier corrections were both "compare in time, not in cycles"; this is the
    same correction one level up, because on this part a time is not a unit either
    until it says which clock it was converted at. The rejection gate is where it
    mattered: SOLAR's tier at a stored f_ref of 1.8 GHz reads 1.333x too slow on a
    compute-bound workload, which pushed it above the measured T_b on 127 workloads
    across 13 problems, the gate dropped the tier, and the published bound fell to
    the declared-traffic floor -- 4.58x to 249x TOO SMALL, median 39.5x, in the
    undetectable direction. 1.8 GHz is refuted on exactly that population by the
    measurements themselves: `compute_cycles / T_b` puts a floor of 1809-2306 MHz
    under the clock the card sustained on all 127.

    Where there is no measurement clock -- the locked basis, a workload with no
    T_b, a bracket with no samples -- the tiers' own `t_sol_ms` columns are the
    only comparands there are, and the build says so by passing
    `compared_at_mhz=None` down to `_reclock_terms`, which then refuses to merge
    two tiers whose reference clocks disagree.
    """
    out: dict[str, dict] = {}
    stats = {"solar_fused": 0, "declared_traffic": 0, "max_of_both": 0,
             "solar_arithmetic_gathered": 0, "declared_traffic_gathered": 0,
             "traffic_rejected_above_t_b": 0, "solar_rejected_above_t_b": 0,
             "no_valid_bound": 0}
    for key in set(solar) | set(traffic):
        s_w, t_w = solar.get(key, {}), traffic.get(key, {})
        merged: dict[str, dict] = {}
        for u in set(s_w) | set(t_w):
            s, t = s_w.get(u) or {}, t_w.get(u) or {}
            anchor = (tb.get(key) or {}).get(u) or {}
            measured = anchor.get("t_b_ms")
            # The clock BOTH tiers are compared at: the minimum of the T_b
            # bracket, i.e. the clock the published bound is taken at, so the gate
            # asks the one question it means to ask -- "is the number this manifest
            # would publish above the time that was measured?"
            bracket = clock_interval(anchor)
            gate_mhz = bracket[0] if bracket else None
            # --- D18, on the SOLAR tier. The signal is the traffic tier's own
            # `gathered_axes`, derived per workload from the problem's reference by
            # `sol_gathered_traffic`; where it is set, SOLAR's memory term prices an
            # allocation the problem only indexes into. Discard that term, keep the
            # arithmetic one. Done BEFORE the gate and the comparison, because it
            # changes both.
            gathered = bool(t.get("gathered_axes"))
            solar_uncorrected = s
            if gathered and s:
                s = _solar_arithmetic_only(s, t, stats)
            gathered_applied = s is not solar_uncorrected
            s_cyc, t_cyc = s.get("t_sol_cycles"), t.get("t_sol_cycles")
            s_time, t_time, compared_at = _tier_times_ms(s, t, gate_mhz, stats)
            # A candidate bound above the measured time is not a loose lower
            # bound, it is not a lower bound at all -- it would make
            # (T_b - T_SOL) negative and push scores past 1. The rule is
            # symmetric: reject any candidate that fails, take the max of what
            # survives, and if nothing survives the workload is not scoreable
            # and is counted as such rather than shipped with a bad anchor.
            t_rejected = s_rejected = False
            if measured is not None:
                if t_cyc is not None and (t_time or 0) > measured:
                    stats["traffic_rejected_above_t_b"] += 1
                    t_cyc, t_rejected = None, True
                if s_cyc is not None and (s_time or 0) > measured:
                    stats["solar_rejected_above_t_b"] += 1
                    s_cyc, s_rejected = None, True
            # A rejected tier is rejected for re-clocking too. Nulling only the
            # cycle count left the tier's `compute_cycles`/`memory_bytes` in the
            # union `_reclock_terms` builds, and under the unlocked basis the
            # PUBLISHED bound is re-evaluated from that union -- so the rejected
            # tier came back as the shipped number and the gate had no effect.
            # On MI355X manifest-v2 that put 41 workloads across 4 problems above
            # their own measured T_b, by up to 4.06x. The terms are a property of
            # the tier that carries them: drop them with it.
            s_terms = {} if s_rejected else s
            t_terms = {} if t_rejected else t
            if s_cyc is not None and t_cyc is not None:
                # In TIME, never in cycles. The two tiers count cycles at
                # DIFFERENT reference clocks -- SOLAR at f_ref 1.8 GHz (4444.4
                # DRAM bytes/cycle), the traffic tier at the arch config's 2.4
                # GHz (3333.3) -- so `t_cyc > s_cyc` is a comparison between
                # two units, biased 2.4/1.8 = 1.3333x in the traffic tier's
                # favour. It picked the SMALLER of the two bounds on 255 of the
                # 2796 MI355X workloads where both tiers survived, across 74
                # problems, by up to 1.3312x: the max was not a max. And the
                # error is in the invisible direction -- a T_SOL below the true
                # bound inflates S for every submission slower than T_b, where
                # a T_SOL above it is caught by the T_b gate right above.
                #
                # `s_time`/`t_time` are both at `gate_mhz` where the measurement
                # supplied one, and each tier's own cycles at its own clock where
                # it did not -- which is the only OTHER form in which they are
                # comparable, and only when the two clocks agree.
                if t_time is None or s_time is None:
                    raise ValueError(
                        f"{key}/{u}: a surviving tier has no t_sol_ms, so the "
                        "two cannot be compared in time; comparing cycles "
                        "across tiers is a unit error, not a fallback")
                traffic_wins = t_time > s_time
                source = "max_of_both" if traffic_wins else "solar_fused"
                chosen = t if traffic_wins else s
            elif s_cyc is not None:
                source, chosen = "solar_fused", s
            elif t_cyc is not None:
                source, chosen = "declared_traffic", t
            else:
                stats["no_valid_bound"] += 1
                continue
            if gathered_applied:
                # Named for the term that binds, as MI350X v1.1/v1.2 named them,
                # so the label says what the bound IS rather than which tier the
                # comparison happened to pick. Only records this build actually
                # corrected are relabelled: a gathered workload with no SOLAR tier
                # (the five paged problems SOLAR timed out on) is unchanged, and
                # saying otherwise would advertise a correction that did not run.
                source = ("solar_arithmetic_gathered" if source == "solar_fused"
                          else "declared_traffic_gathered")
            stats[source] += 1
            # The re-clocking terms are recomputed from BOTH tiers, never
            # inherited from the winner: `chosen` is one tier's record, and for
            # `max_of_both` the winner is often the traffic tier, whose
            # compute_cycles is 0 by construction. Dropping them first means a
            # record `_reclock_terms` declines to merge is left visibly
            # un-re-clockable (t_sol_at raises) rather than quietly carrying one
            # tier's half of the answer.
            base = {k: v for k, v in chosen.items() if k not in RECLOCK_FIELDS}
            if gathered_applied:
                # The D18 audit trail lives on the TRAFFIC tier's record, and `base`
                # is the CHOSEN tier's. Where SOLAR won the comparison -- three
                # L1__009 workloads on MI355X manifest-v4 -- the record shipped
                # labelled `solar_arithmetic_gathered` with `gathered_axes` null and
                # `allocation_bytes`/`gathered_bytes` absent, i.e. a label naming a
                # correction beside no evidence that it happened. The label was
                # right and the evidence was on the tier that lost. Carry it.
                for k in GATHERED_AUDIT_FIELDS:
                    if k not in base and t.get(k) is not None:
                        base[k] = t[k]
            terms = _reclock_terms(s_terms, t_terms, source, stats,
                                   compared_at_mhz=compared_at,
                                   unioned=(bool(s_terms and t_terms)
                                            if gathered_applied else None))
            rec = {**base, "t_sol_source": source,
                   "t_sol_cycles_solar": s_cyc,
                   "t_sol_cycles_traffic": t.get("t_sol_cycles"),
                   "t_sol_tier_rejected_above_t_b": (
                       [n for n, r in (("solar_fused", s_rejected),
                                       ("declared_traffic", t_rejected))
                        if r] or None),
                   # The clock the `t_sol_ms`/`t_sol_cycles` columns just above are
                   # in, taken from the tier they came from. Null where that tier
                   # never stated one -- which is a statement, and the one
                   # `t_sol_at.bound_ms` refuses on.
                   "f_ref_mhz": chosen.get("f_ref_mhz"),
                   **terms}
            # --- D39, on EVERY published bound.
            #
            # Nothing may beat a bound; nothing checks that a bound is tight. This
            # band is the only statement the manifest makes in the loose direction,
            # and it used to be made on the 63 records this build had just corrected
            # and on nothing else -- which read as a census of the loose bounds and
            # was not one. Banding all 3717 scoreable MI355X workloads by
            # `T_b / t_sol_ms_published` under these same bands gives vacuous 398,
            # loose 322, ok 2482, narrow 515; 63 carried a mark. The two populations
            # the same session moved were both outside it:
            #
            # * the 127 records the clock-correct tier comparison rescued from the
            #   rejection gate (D63) land at 0.7592..0.9641 of T_b -- headroom
            #   1.037..1.317, `narrow` on 127 of 127. That band is not a discovery
            #   about those workloads, it is what the acceptance inequality forces:
            #   a record joins that population only if SOLAR's bound at the stored
            #   1.8 GHz sits above T_b while the same bound at the measurement's own
            #   clock does not, which pins it into (1800/f_published, 1.0] of T_b.
            #   For scale, `published/T_b` has p50 0.082 over the whole corpus.
            # * the causal-mask records on FlashInfer-Bench 014/015 reach headroom
            #   1.113e6 -- larger than anything on 018, which WAS marked.
            #
            # Both are visible now for the same reason 018 was: this is one call
            # against `t_sol_ms_published`, so it moves no bound and hides no
            # trade. It is the manifest saying how much room it left, on every row.
            published = _published_bound_ms(terms, chosen, bracket, gate_mhz)
            quality, headroom = _bound_quality(published, measured)
            rec["bound_quality"] = quality
            rec["bound_headroom"] = headroom
            k = f"bound_quality_{quality or 'unbanded'}"
            stats[k] = stats.get(k, 0) + 1
            if gathered_applied:
                # The correction moves these from "bound too large" -- detectable,
                # a real kernel falsifies it -- to "bound very loose", which is the
                # direction nothing downstream can contradict. On 018 the arithmetic
                # term binds on none of the 47, so the corrected bound IS the
                # gathered floor, as low as 5.83e-06 ms, and T_b/T_SOL lands near
                # 1.5e5x. D39's threshold is 100x. The band above says so for every
                # bound; these two fields say what this build did to THIS one, so
                # the trade stays legible after the band stops being unusual.
                rec["solar_memory_bytes_at_allocation"] = (
                    solar_uncorrected.get("memory_bytes"))
                rec["t_sol_cycles_solar_uncorrected"] = (
                    solar_uncorrected.get("t_sol_cycles"))
                if quality:
                    k = f"gathered_bound_quality_{quality}"
                    stats[k] = stats.get(k, 0) + 1
            merged[u] = rec
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


#: Every part this tree has artifacts for, so an inference from a device name is
#: a lookup rather than a substring guess. Kept beside `verify_artifacts.py`'s
#: list of the same name -- duplicated, not imported, because that module is the
#: gate and this one is the builder, and a build must not depend on its own gate.
KNOWN_PARTS = ("MI355X", "MI350X", "MI300X")


def _doc_part(doc: dict | None) -> str | None:
    """The part *doc* is about: what it SAYS first, what it shows second.

    A union of three places, in decreasing order of how much of a statement each
    one is: the top-level `part` key `sol_bounds.py` writes, `_provenance.part`
    which `provenance.stamp()` now emits, and finally the torch device names,
    which are evidence about the node the file was written on rather than a
    claim about what is in it. The last is why this returns something at all for
    `artifacts/03-MI355X/t_sol_traffic.json`, which states nothing.
    """
    if not doc:
        return None
    prov = doc.get("_provenance") or {}
    for named in (doc.get("part"), prov.get("part")):
        if isinstance(named, str) and named:
            return named
    for dev in (prov.get("torch") or {}).get("devices") or []:
        for part in KNOWN_PARTS:
            if part in str(dev):
                return part
    return None


def _inputs_part(*paths: Path) -> str | None:
    """The one part every input agrees on, or a refusal.

    None when no input says -- which is the MI350X release inputs, whose
    `t_sol.json` predates device names in provenance -- and that is a legible
    absence, not a guess. Two different answers is a build error: a manifest
    pairing one part's bounds with another part's anchors would be scored
    against silently, exactly the way MI350X bounds were nearly used to score
    MI355X kernels.
    """
    seen: dict[str, str] = {}
    for p in paths:
        part = _doc_part(_load(p))
        if part:
            seen[part] = str(p)
    if len(seen) > 1:
        raise SystemExit(
            "the inputs to this manifest are not all about the same part:\n  "
            + "\n  ".join(f"{v}: {k}" for k, v in sorted(seen.items()))
            + "\nA manifest is one part's bounds against that part's anchors. "
              "Rebuild the odd one out, or pass --part if you are certain the "
              "attribution rather than the data is wrong.")
    return next(iter(seen), None)


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
    ap.add_argument("--part", default=None,
                    help="e.g. MI355X. Declares what this manifest is about. "
                         "Default: taken from the inputs, which must agree.")
    ap.add_argument("--allow-cross-part", action="store_true",
                    help="permit building a manifest for a part this node does "
                         "not have. Only for re-deriving a frozen release.")
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
    # Which part this manifest is about, DECLARED rather than left to be
    # inferred downstream. manifest-v2 and v3 shipped with no part at all, and
    # every consumer that needed one recovered it from the torch device names in
    # the provenance block -- inference that happens to be right, in a tree that
    # holds two parts' artifacts side by side (Issue 7). The inputs know: take
    # the part from them and refuse if they disagree, in the same shape as the
    # bandwidth-disagreement refusal above, because a manifest built from one
    # part's bounds and another part's anchors is not a manifest.
    part = a.part or _inputs_part(Path(a.t_sol), Path(a.t_sol_traffic))
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
                # -- D39, on every row. `bound_quality` bands `T_b / T_SOL` for the
                # bound this manifest publishes, and `bound_headroom` is that ratio.
                # Present on every workload with a bound and an anchor, not only on
                # the ones this build corrected: a mark that appears solely where
                # something was loosened reads as a census of the loose bounds and
                # is not one (see `combine_bounds`). Null means "no band could be
                # taken" -- no T_b, or a bracketed record whose terms would not
                # merge -- and is a stated absence, not a claim of quality.
                "bound_quality": s.get("bound_quality"),
                "bound_headroom": s.get("bound_headroom"),
                # -- Set only where this build discarded SOLAR's allocation-priced
                # memory term on a gathered workload (D18, `_solar_arithmetic_only`):
                # what that term was, and what the uncorrected bound would have been.
                **{k: s[k] for k in ("solar_memory_bytes_at_allocation",
                                     "t_sol_cycles_solar_uncorrected") if k in s},
                # -- Which tier, if either, `combine_bounds` refused because its
                # candidate sat above the measured T_b. `combine_bounds` has always
                # computed this and a test has always asserted it, but it stopped
                # here: the manifest could not say which workloads publish on one
                # tier because the OTHER was thrown out, as against publishing on
                # one tier because the other was smaller. Those are different
                # claims and half of why "120 vs 193" (Issue 3) was arguable.
                "t_sol_tier_rejected_above_t_b": s.get(
                    "t_sol_tier_rejected_above_t_b"),
                # -- The declared-traffic tier's own audit trail for D18 and its
                # second guise, carried through so the correction is checkable from
                # the published artifact instead of only from the tier file:
                # what the declaration allocated, what the workload's own index
                # vectors name, and which axis a causal mask left partly dead.
                **{k: s[k] for k in ("allocation_bytes", "gathered_axes",
                                     "gathered_bytes", "masked_axis",
                                     "masked_rows") if k in s},
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
        # Which part these bounds are for, as a statement. Every consumer that
        # needs it -- the scorer's part guard, the leaderboard's ingest, task
        # 03's check D -- previously recovered it from the torch device names in
        # the provenance block, i.e. from where the file was WRITTEN rather than
        # from what is in it. `_provenance.part` carries the same value; both,
        # because `verify_artifacts.artifact_part()` reads the provenance one and
        # `agent_score.py` unions all three.
        "part": part,
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
    write_artifact(out, f"09-manifest-{a.version}", payload, part=part,
                   allow_cross_part=a.allow_cross_part)

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
