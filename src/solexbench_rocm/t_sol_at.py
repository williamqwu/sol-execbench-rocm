# SPDX-License-Identifier: Apache-2.0
"""Evaluate T_SOL at the clock a measurement actually ran at.

This node cannot pin a GPU clock. `--setperfdeterminism` is a no-op on the cards
that hold a steady frequency and applies a 0.80-0.85 scale error on the cards that
respond to it (STATE.md D30), so there is no setting that makes every timing happen
at one known frequency. Left unlocked the cards are far better behaved -- 0.7% drift
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
