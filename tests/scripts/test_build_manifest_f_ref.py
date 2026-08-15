# SPDX-License-Identifier: Apache-2.0
"""D63: two T_SOL tiers, two reference clocks, and one comparison between them.

`artifacts/03-MI355X/t_sol.json` converts 2902 of its 2998 records at 1.8 GHz and
96 at 2.4 GHz under a header that says `f_lock_mhz: 2400`; the declared-traffic
tier is uniformly at 2.4 GHz. Two things in `combine_bounds` used to read those
columns: the `T_SOL <= T_b` rejection gate and the tier comparison. Both were
therefore biased by 2.4/1.8 = 1.3333x.

The tier comparison was fixed by comparing in time. The gate was not, and it is
where the damage was: SOLAR's compute term reads 1.333x too slow at 1.8 GHz, which
pushed it above the measured T_b on 127 workloads across 13 problems; the gate
dropped the tier; and the published bound collapsed to the declared-traffic floor,
4.58x to 249x TOO SMALL, median 39.5x -- the undetectable direction. On exactly
that population 1.8 GHz is refuted by measurement: `compute_cycles / T_b` puts a
floor of 1809-2306 MHz under the clock the card sustained on all 127.

So the correction is the same one, one level up: evaluate BOTH tiers at the T_b
measurement's own bracket-minimum clock -- the clock the published bound is taken
at -- before the gate and before the comparison.

CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_manifest as bm  # noqa: E402
from solexbench_rocm.t_sol_at import t_sol_ms_at  # noqa: E402

BPS = 7999920000000.0
KEY = "L2__002_decoder_layer_full_block"
UUID = "u0"

#: A compute-bound SOLAR tier that straddles T_b across the two clocks: 2,000,000
#: cycles is 1.1111 ms at 1.8 GHz and 0.8333 ms at 2.4 GHz, against a measured
#: T_b of 1.0 ms. At its stored f_ref it is "impossible"; at the clock the anchor
#: was actually measured at it is a valid, and much tighter, lower bound.
COMPUTE_CYCLES = 2_000_000.0
T_B_MS = 1.0

#: The bracket. Both samples above 1.8 GHz, as every one of the 127 reads.
BRACKET = {"clock_before_mhz": 2400.0, "clock_after_mhz": 2410.0}


def _solar(f_ref_mhz=1800.0):
    cycles = int(COMPUTE_CYCLES)
    return {KEY: {UUID: {
        "t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
        "bottleneck": "compute", "compute_cycles": COMPUTE_CYCLES,
        "memory_bytes": 4096, "mac_per_cycle": 524288.0,
        "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref_mhz}}}


def _traffic(f_ref_mhz=2400.0, memory_bytes=1_000_000):
    cycles = max(1, int(memory_bytes * f_ref_mhz * 1e6 / BPS) + 1)
    return {KEY: {UUID: {
        "t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
        "bottleneck": "memory", "compute_cycles": 0.0,
        "memory_bytes": memory_bytes, "mac_per_cycle": None,
        "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref_mhz}}}


def _tb(bracket=True, t_b_ms=T_B_MS):
    return {KEY: {UUID: {"variant": "v2_compile", "t_b_ms": t_b_ms,
                         **(BRACKET if bracket else {})}}}


# ------------------------------------------------- the gate, at which clock


def test_the_gate_reads_the_tier_at_the_measurement_clock():
    """The 127. A tier that is only impossible at a clock nobody measured stays."""
    merged, stats = bm.combine_bounds(_solar(), _traffic(), _tb())
    rec = merged[KEY][UUID]
    assert stats["solar_rejected_above_t_b"] == 0
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["compute_cycles"] == COMPUTE_CYCLES
    published = t_sol_ms_at(rec, 2400.0)
    assert published == COMPUTE_CYCLES / (2400.0 * 1e3)
    assert published < T_B_MS, "and it is still a lower bound on the measurement"


def test_the_same_tier_at_its_stored_f_ref_is_the_defect_being_fixed():
    """Without a measurement clock the stored column is all there is -- and it is
    what rejected the tier and collapsed the bound onto the traffic floor."""
    merged, stats = bm.combine_bounds(_solar(), _traffic(), _tb(bracket=False))
    rec = merged[KEY][UUID]
    assert stats["solar_rejected_above_t_b"] == 1
    assert rec["t_sol_source"] == "declared_traffic"
    assert rec["compute_cycles"] == 0.0, "the rejected tier's terms go with it"
    # The size of the defect, on this record: the floor is ~2000x looser than the
    # bound the tier would have supplied. On the real corpus: 4.58x to 249x.
    floor = t_sol_ms_at(rec, 2400.0)
    assert (COMPUTE_CYCLES / (2400.0 * 1e3)) / floor > 1000


def test_the_gate_still_rejects_a_tier_that_is_impossible_at_the_real_clock():
    """The gate is corrected, not disabled. A bound above the measured time at the
    clock the measurement ran at is still not a lower bound."""
    merged, stats = bm.combine_bounds(
        _solar(), _traffic(), {KEY: {UUID: {"variant": "v1", "t_b_ms": 0.5,
                                            **BRACKET}}})
    assert stats["solar_rejected_above_t_b"] == 1
    assert merged[KEY][UUID]["t_sol_source"] == "declared_traffic"


# ------------------------------------------------- the f_ref field itself


def test_the_bandwidth_guard_provably_cannot_see_a_clock_mismatch():
    """Why `f_ref_mhz` needs a guard of its own.

    `DRAM_byte_per_cycle` is defined in the arch YAML as `bytes_per_sec / freq`,
    so 4444.4 B/cycle at 1.8 GHz and 3333.3 B/cycle at 2.4 GHz are the same
    7,999,920,000,000 B/s -- 7999919999999.999 and 7999920000000.0, one part in
    8e12 apart because one was reached by a multiplication and the other by a
    division. `_one_bandwidth` exists precisely to call those one number, and it
    is right to: no two arch configs differ by 2e-16. So the bandwidth guard
    agrees perfectly across a 1.8/2.4 GHz split, which is how the mixture merged
    for three manifest versions in silence.
    """
    a, b = 4444.4 * 1.8e9, 3333.3 * 2.4e9
    assert a != b, "two float printings, not one float"
    assert bm._one_bandwidth({a, b}), "and the guard collapses them, correctly"
    _, stats = bm.combine_bounds(_solar(), _traffic(), _tb(bracket=False))
    assert not stats.get("reclock_terms_conflicting_bandwidth")


def test_two_f_refs_do_not_merge_when_that_is_how_the_tiers_were_compared():
    """No measurement clock means the stored columns WERE the comparands, and a
    comparison between two clocks is a unit error. Refuse, visibly.

    T_b is 5 ms here so that both tiers survive the gate: a rejected tier takes
    its terms with it, and one surviving tier has only one clock to disagree with.
    """
    unbracketed = _tb(bracket=False, t_b_ms=5.0)
    merged, stats = bm.combine_bounds(_solar(f_ref_mhz=1800.0),
                                      _traffic(f_ref_mhz=2400.0), unbracketed)
    assert stats["reclock_terms_conflicting_f_ref"] == 1
    assert "compute_cycles" not in merged[KEY][UUID], (
        "a record nobody can re-clock must say so by raising, not by carrying "
        "one tier's half of the answer")
    # One clock, same corpus: no conflict, terms merge.
    _, same = bm.combine_bounds(_solar(f_ref_mhz=2400.0),
                                _traffic(f_ref_mhz=2400.0), unbracketed)
    assert not same.get("reclock_terms_conflicting_f_ref")


def test_a_measurement_clock_reconciles_the_two_f_refs():
    """And the refusal must NOT fire where the clocks entered nothing.

    Measured by the adversarial review of the D63 write-up: refusing every
    two-tier merge on an f_ref difference strips the published bound from 2826 of
    the 3717 scoreable MI355X workloads across 181 problems, and sends exactly
    those workloads back to being scored against the mixed-clock legacy column
    this guard exists to condemn. Where both tiers were evaluated at the
    measurement's own clock, the stored f_refs entered no comparison and no
    arithmetic -- the four merged terms are clock-free -- so there is nothing to
    refuse.
    """
    merged, stats = bm.combine_bounds(_solar(f_ref_mhz=1800.0),
                                      _traffic(f_ref_mhz=2400.0), _tb())
    assert not stats.get("reclock_terms_conflicting_f_ref")
    assert merged[KEY][UUID]["compute_cycles"] == COMPUTE_CYCLES


def test_the_merged_record_names_the_clock_its_ms_column_is_on():
    """A cycle count on this part is meaningless without the clock beside it.

    `f_ref_mhz` is the CHOSEN tier's, because that is the tier `t_sol_ms` and
    `t_sol_cycles` were copied from. Both tiers' clocks stay visible under their
    own names, because a merged record genuinely has two.
    """
    merged, _ = bm.combine_bounds(_solar(), _traffic(), _tb())
    rec = merged[KEY][UUID]
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["f_ref_mhz"] == 1800.0
    assert rec["f_ref_mhz_solar"] == 1800.0
    assert rec["f_ref_mhz_traffic"] == 2400.0
    assert rec["t_sol_ms"] == rec["t_sol_cycles"] / (rec["f_ref_mhz"] * 1e3)

    # ...and where the traffic tier wins, the record is on the traffic clock.
    # 7.2e9 B at 8 TB/s is 0.9 ms: above SOLAR's 0.833 ms at 2400 MHz and still
    # below the 1.0 ms anchor, so it wins the comparison without being rejected.
    merged, _ = bm.combine_bounds(_solar(), _traffic(memory_bytes=7_200_000_000),
                                  _tb())
    rec = merged[KEY][UUID]
    assert rec["t_sol_source"] == "max_of_both"
    assert rec["f_ref_mhz"] == 2400.0


def test_a_tier_that_never_stated_its_clock_leaves_the_field_null():
    """Every artifact on disk today predates the field. Null is the honest value
    -- it is what `t_sol_at.bound_ms` refuses on -- and the field must be present
    to be refused."""
    solar = _solar()
    del solar[KEY][UUID]["f_ref_mhz"]
    traffic = _traffic()
    del traffic[KEY][UUID]["f_ref_mhz"]
    merged, stats = bm.combine_bounds(solar, traffic, _tb(bracket=False))
    rec = merged[KEY][UUID]
    assert rec["f_ref_mhz"] is None and "f_ref_mhz" in rec
    assert not stats.get("reclock_terms_conflicting_f_ref"), (
        "two unstated clocks are not two different clocks")
