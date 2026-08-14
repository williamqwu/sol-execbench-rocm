# SPDX-License-Identifier: Apache-2.0
"""Evaluate T_SOL at the clock a measurement actually ran at.

The **MI355X** node this was written for cannot usefully pin a GPU clock.
`--setperfdeterminism` is accepted by every card and read back as
`perf_determinism`, and then 15 of 16 per-card measurements land at 0.795-0.864x
of what was asked while drawing 734-999 W of a 1400 W cap; the one measurement
that reached its setpoint drew 1272 W. So the failure is the card not raising
its power state, after which the clock it can hold follows -- it is not clock
control refusing a number, and it is not a property of particular cards
(docs/methodology.md §3, "That question, asked on a second MI355X node").

  An earlier version of this docstring said the setpoint was a *no-op on the
  cards that hold a steady frequency*. Its author withdrew that in PR #2: the
  reading came from `rocm-smi -d <torch index>`, and rocm-smi orders devices by
  PCI bus while torch does not, so the request landed on a neighbour while the
  measured card was left at its own boost clock. Correctly addressed, those
  cards track a request to within 0.4%. The retraction is kept visible here
  rather than edited away, because this file is the reason someone will go
  looking, and because it is the third finding this repo has lost to that
  ordering (STATE.md D11, D20's clock alignment, and this).

**None of that is true of MI350X**, and the difference is the same mechanism
seen from the other side: STATE.md D55 measured locked vs unlocked vs
cap-raised-to-2200 over twelve loads on `gbt350-odcdh1-a08-1` and found them
indistinguishable, because a 1000 W part is already sitting at the conservative
operating point the MI355X SMU drops *into*. On MI350X the lock is inert; on
MI355X it costs 20% of the clock. One part is not a guide to the other, which
is why the policy is per-part and not a repo-wide constant.

Left unlocked the MI355X cards are far better behaved -- 0.7% drift
over 8 minutes, 1.0% sensitivity to what the neighbours are doing -- but the clock
then depends on the *kernel*: a dense GEMM saturates the 1400 W package budget and
is pushed down to ~1730 MHz, while a memory-bound kernel needs only ~1170 W and
boosts to ~2394 MHz. Across kernel types that is a 28% spread
(`artifacts/01/unlocked-clock.json`).

So a single F_LOCK cannot describe a timing run here, and the fix is to stop
pretending one frequency applies to everything: each measurement records the clock
it observed, and its bound is evaluated at that clock.

That is only sound because the two terms of the roofline scale differently, and
knowing which is which is the whole trick. From the arch YAML's own annotations:

* ``MAC_per_cycle`` is architectural and frequency-independent, so the compute term
  is a fixed number of CYCLES and its TIME scales as 1/F.
* ``DRAM_byte_per_cycle`` is derived as ``bytes_per_sec / freq``, so the memory term
  is a fixed TIME and its cycle count scales with F.

A card that boosts to 2394 MHz on a memory-bound kernel has not moved that kernel's
bound at all -- HBM does not run off the core clock. A card pushed to 1730 MHz on a
dense GEMM genuinely cannot beat ``compute_cycles / 1730``. Both facts are what the
fixed-lock model was approximating, and evaluating per measurement is strictly more
faithful, not a concession.

Because the terms scale oppositely, **the bottleneck can flip as F moves**, so both
terms must be carried and re-maxed rather than scaling whichever one happened to win
at the reference clock. `scripts/sol_bounds.py` emits both.

--------------------------------------------------------------------------------
T_SOL AS AN INTERVAL, AND WHY THE PUBLISHED END IS THE MINIMUM CLOCK
--------------------------------------------------------------------------------

Everything above assumed the window has *a* clock. On this part it does not. The
bracket (``sol_execbench.core.bench.clock_bracket``) samples immediately before and
immediately after the timed region, and on the worst problems in the corpus those two
samples are far apart -- measured, on ``artifacts/06-MI355X``:

===============================  ==============  ==========================
problem                          bracket         compute-term span of T_SOL
===============================  ==============  ==========================
L2__005_swiglu_mlp_backward      1582-2259 MHz   42.8%
L1__003_lm_head_projection       1711-2386 MHz   39.5%
L2__004_fused_residual_rms_mlp   1607-2148 MHz   33.7%
L2__012_moe_expert_batched       1742-2306 MHz   32.4%
L1__013_fused_residual_rms_norm  2314-2390 MHz    3.3%   (clean, for contrast)
===============================  ==============  ==========================

No single frequency describes such a window, so a single T_SOL does not either.
``t_sol_interval`` evaluates the bound at **both** ends and reports the pair, its
width, and the bottleneck at each end.

**The published bound is the one evaluated at the MINIMUM clock**, which is the
LARGEST T_SOL and therefore the TIGHTEST, most demanding bound. That direction is a
deliberate choice about which way to be wrong, and the reasoning is the whole point:

* Publishing at the minimum clock can only make the bound too *strict*. A bound that
  is too strict is **detectable from the outside**: a real measurement beats it, and
  the existing "no measurement beats its T_SOL" check (task 03 check D, and the
  per-record ``bound_violation`` in ``scripts/score_solutions.py``) fires. The error
  announces itself.
* Publishing at the maximum clock would give the *smallest* T_SOL -- a bound nothing
  can violate, because it sits below every achievable time. Nothing downstream can
  ever contradict it. CLAUDE.md §6 names exactly that shape of failure: *"a
  self-consistent bound and anchor cannot detect a shared error"*, which is how all
  three known-bad bounds in this repo survived a frozen manifest.
* Taking the midpoint splits the difference and inherits the worse half of both: it
  is neither the tightest bound nor a detectable one, and it invents a frequency the
  card was never observed at.

So the asymmetry is intentional. Between an error that trips an existing alarm and an
error that silences one, this takes the first every time.

**The interval is not a substitute for the point, it is the honesty attached to it.**
A workload whose interval is +-1.6% and one whose interval is +-17% both publish a
number; only the width says which of the two means anything, which is why the width
is a first-class field (``t_sol_interval_halfwidth_rel``) rather than something a
reader recomputes from the endpoints.

**What the interval does NOT bound.** It bounds the clock ambiguity of the *bound*.
It says nothing about the short-window timing bias on the measurement it will be
compared against (``docs/methodology.md`` §7, up to +106.9%), which is a separate,
larger and non-clock effect. A narrow interval is not a small error bar on the score.
"""

from __future__ import annotations

import math

__all__ = [
    "t_sol_ms_at",
    "t_sol_cycles_at",
    "bottleneck_at",
    "t_sol_interval",
    "INTERVAL_FIELDS",
    "REQUIRED_FIELDS",
]

REQUIRED_FIELDS = ("compute_cycles", "memory_bytes", "dram_byte_per_sec")


class MissingBoundTerms(KeyError):
    """A T_SOL record predates the split into separately-scalable terms."""


def _terms(w: dict) -> tuple[float, int, float]:
    missing = [k for k in REQUIRED_FIELDS if w.get(k) is None]
    if missing:
        raise MissingBoundTerms(
            f"T_SOL record lacks {missing}, so it cannot be evaluated at a "
            f"different clock -- only the max of the two roofline terms was kept. "
            f"Re-run scripts/sol_bounds.py to emit them."
        )
    return float(w["compute_cycles"]), int(w["memory_bytes"]), float(w["dram_byte_per_sec"])


def t_sol_cycles_at(w: dict, f_mhz: float) -> int:
    """Bound in cycles at *f_mhz*, rounded exactly as `sol_bounds.py` rounds it.

    Deliberately mirrors that function's ``max(1, ceil(...))``: truncation once gave
    eight workloads a bound of zero cycles, which no kernel can approach and which
    puts a division by zero into the score. The smallest workloads here really are
    sub-cycle, so the floor is the normal case at the small end, not an edge case.
    """
    if f_mhz <= 0:
        raise ValueError(f"f_mhz must be positive, got {f_mhz}")
    compute_cycles, memory_bytes, dram_byte_per_sec = _terms(w)
    # memory_cycles(F) = bytes / (bytes_per_sec / F) = bytes * F / bytes_per_sec
    memory_cycles = (memory_bytes * f_mhz * 1e6 / dram_byte_per_sec
                     if dram_byte_per_sec > 0 else 0.0)
    return max(1, math.ceil(max(compute_cycles, memory_cycles)))


def t_sol_ms_at(w: dict, f_mhz: float) -> float:
    """Bound in milliseconds at *f_mhz*."""
    return t_sol_cycles_at(w, f_mhz) / (f_mhz * 1e3)


def bottleneck_at(w: dict, f_mhz: float) -> str:
    """Which term binds at *f_mhz*. May differ from the record's own `bottleneck`.

    Reported rather than assumed: a workload that is compute-bound at 1650 MHz can
    be memory-bound at 2394 MHz, and a score that silently changed regime is worth
    being able to see.
    """
    compute_cycles, memory_bytes, dram_byte_per_sec = _terms(w)
    memory_cycles = (memory_bytes * f_mhz * 1e6 / dram_byte_per_sec
                     if dram_byte_per_sec > 0 else 0.0)
    return "memory" if memory_cycles >= compute_cycles else "compute"


#: Every field ``t_sol_interval`` emits, listed once so the manifest builder, the
#: scorer and the tests cannot drift apart about what an interval record carries.
#: Flat, not nested, and each name is its own column: the width has to be sortable
#: across 3717 workloads without reprocessing anything.
INTERVAL_FIELDS: tuple[str, ...] = (
    "t_sol_clock_min_mhz",
    "t_sol_clock_max_mhz",
    "t_sol_ms_at_clock_min",
    "t_sol_ms_at_clock_max",
    "t_sol_cycles_at_clock_min",
    "t_sol_cycles_at_clock_max",
    "t_sol_ms_published",
    "t_sol_cycles_published",
    "t_sol_published_at_mhz",
    "t_sol_published_end",
    "t_sol_interval_width_rel",
    "t_sol_interval_halfwidth_rel",
    "t_sol_bottleneck_at_clock_min",
    "t_sol_bottleneck_at_clock_max",
    "t_sol_bottleneck_flips",
)

#: Which end of the bracket the published bound is taken at. Named rather than
#: implied, because the choice is the methodology (see the module docstring) and a
#: reader must be able to confirm it from the artifact instead of trusting a
#: convention.
PUBLISHED_END = "clock_min"


def t_sol_interval(w: dict, f_before_mhz: float, f_after_mhz: float) -> dict:
    """T_SOL at both ends of a clock bracket, with the width and the bottleneck.

    *f_before_mhz* and *f_after_mhz* are the two bracket samples in either order;
    only their min and max matter, because the bracket is evidence about a range and
    carries no information about which end the window spent more time at. Ordering
    them here rather than at each call site is deliberate: a caller that passed them
    the wrong way round would otherwise publish the loosest bound instead of the
    tightest, silently.

    The published value is the **minimum-clock** end -- the largest T_SOL, the
    tightest bound, and the one whose failure mode is detectable. That reasoning is
    in the module docstring and is not repeated here, but it is the reason this
    function returns ``t_sol_ms_published`` at all rather than leaving the caller to
    pick an end.

    A degenerate bracket (both samples equal) collapses to a point: the two ends are
    identical and the width is exactly 0.0. So is a **memory-bound** workload's
    interval at *any* bracket width, because the memory term is a fixed time and does
    not move with the clock -- that zero is a correctness property of the whole
    scheme, not a special case, and ``tests/scripts/test_t_sol_at.py`` pins it.

    ``t_sol_bottleneck_flips`` is reported rather than resolved. Across the 42.8%
    spans measured on this corpus some workloads are compute-bound at one end and
    memory-bound at the other, and a record that named a single bottleneck would be
    making a claim the bracket does not support.

    Raises ``MissingBoundTerms`` for a record that predates the split, and
    ``ValueError`` for a non-positive clock -- both by way of ``t_sol_cycles_at``,
    so there is one definition of each failure.
    """
    f_lo = float(min(f_before_mhz, f_after_mhz))
    f_hi = float(max(f_before_mhz, f_after_mhz))

    # At the LOW clock the bound is the LARGE one: the compute term is a fixed cycle
    # count, so less clock buys more milliseconds. The naming below is by CLOCK, not
    # by magnitude, and the two are inverted -- worth reading twice.
    cyc_lo, cyc_hi = t_sol_cycles_at(w, f_lo), t_sol_cycles_at(w, f_hi)
    t_at_lo, t_at_hi = t_sol_ms_at(w, f_lo), t_sol_ms_at(w, f_hi)

    t_max, t_min = max(t_at_lo, t_at_hi), min(t_at_lo, t_at_hi)
    total = t_max + t_min
    # Two widths, because two questions get asked of this number and answering
    # both from one column has produced arguments before:
    #   width_rel     "how much bigger is the strict end than the loose one?"
    #                 -- (max-min)/min, the span, comparable with the clock span.
    #   halfwidth_rel "what is the +- on this bound?" -- half the span over the
    #                 midpoint, which is the form a reader means by "+-35%".
    width_rel = ((t_max - t_min) / t_min) if t_min > 0 else None
    halfwidth_rel = ((t_max - t_min) / total) if total > 0 else 0.0

    b_lo, b_hi = bottleneck_at(w, f_lo), bottleneck_at(w, f_hi)
    return {
        "t_sol_clock_min_mhz": f_lo,
        "t_sol_clock_max_mhz": f_hi,
        "t_sol_ms_at_clock_min": t_at_lo,
        "t_sol_ms_at_clock_max": t_at_hi,
        "t_sol_cycles_at_clock_min": cyc_lo,
        "t_sol_cycles_at_clock_max": cyc_hi,
        # The published bound, restated under its own name rather than left as
        # "whichever of the two above you are supposed to know to take". A consumer
        # that reads only this key gets the methodology by default.
        "t_sol_ms_published": t_at_lo,
        "t_sol_cycles_published": cyc_lo,
        "t_sol_published_at_mhz": f_lo,
        "t_sol_published_end": PUBLISHED_END,
        "t_sol_interval_width_rel": width_rel,
        "t_sol_interval_halfwidth_rel": halfwidth_rel,
        "t_sol_bottleneck_at_clock_min": b_lo,
        "t_sol_bottleneck_at_clock_max": b_hi,
        "t_sol_bottleneck_flips": b_lo != b_hi,
    }
