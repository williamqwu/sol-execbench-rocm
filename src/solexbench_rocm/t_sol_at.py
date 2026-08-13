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
"""

from __future__ import annotations

import math

__all__ = ["t_sol_ms_at", "t_sol_cycles_at", "bottleneck_at", "REQUIRED_FIELDS"]

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
