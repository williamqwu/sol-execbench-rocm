# SPDX-License-Identifier: Apache-2.0
"""D18, on the SOLAR tier: a gathered input must not be priced at its allocation.

`sol_gathered_traffic` reprices the declared-traffic tier at the rows a paged
workload actually names, and `combine_bounds` then took `max` in time against a
SOLAR tier that still carried the whole allocation -- so the correction moved the
binding tier onto an uncorrected one and the pre-correction number came straight
back. On `FlashInfer-Bench__018` that is all 47 workloads, published 12.8x to
24,432x (p50 127.4x) above their own corrected floor.

MI350X v1.1 already shipped the rule these tests pin (`rebuild_manifest_v11.py`
:246-270): where the traffic tier records `gathered_axes`, discard SOLAR's MEMORY
term, keep its ARITHMETIC term, publish `max(solar_arithmetic, gathered_traffic)`.
This is cross-part parity, not a new methodology.

The numbers below are the measured ones, from `/var/tmp/solbench/investigate/`:
018's shipped SOLAR `memory_bytes` of 1,140,133,554 against an allocation of
1,140,133,608 and a gathered floor of 44,136, and 014's probe at 32,012,552
against 32,012,568 and 37,144.

CPU-only, no artifact read: every record here is a literal.
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
KEY = "FlashInfer-Bench__018_mla_paged_decode_h16_ckv512_kpe64_ps1"
UUID = "00cb2bc2-c7c7-43a1-b857-b516eb2ce061"

#: The bracket a real MI355X anchor carries. Two samples, near the top of the
#: part's unlocked range, as the 018 anchors read.
BRACKET = {"clock_before_mhz": 2380.0, "clock_after_mhz": 2390.0}


def _solar(memory_bytes=1_140_133_554, compute_cycles=0.265625,
           f_ref_mhz=1800.0):
    cycles = max(1, int(memory_bytes * f_ref_mhz * 1e6 / BPS) + 1)
    return {KEY: {UUID: {
        "t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
        "bottleneck": "memory", "compute_cycles": compute_cycles,
        "memory_bytes": memory_bytes, "mac_per_cycle": 524288.0,
        "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref_mhz}}}


def _traffic(memory_bytes=44_136, gathered=True, f_ref_mhz=2400.0):
    cycles = max(1, int(memory_bytes * f_ref_mhz * 1e6 / BPS) + 1)
    rec = {"t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
           "bottleneck": "memory", "compute_cycles": 0.0,
           "memory_bytes": memory_bytes, "allocation_bytes": 1_140_133_608,
           "mac_per_cycle": None, "dram_byte_per_sec": BPS,
           "f_ref_mhz": f_ref_mhz}
    if gathered:
        rec["gathered_axes"] = {"num_pages": "num_kv_indices"}
    return {KEY: {UUID: rec}}


def _tb(t_b_ms=0.8935, bracket=True):
    return {KEY: {UUID: {"variant": "v2_compile", "t_b_ms": t_b_ms,
                         **(BRACKET if bracket else {})}}}


def test_the_allocation_priced_memory_term_is_discarded():
    """The bound must stop being the allocation-streaming time.

    SOLAR's 1,140,133,554 B is not a mispriced gather -- measured, a bare
    `table[idx]` is charged the gathered rows exactly. It is the reference's own
    `ckv_cache.squeeze(1).to(torch.float32)`, a full-tensor cast that really does
    run before the gather. That makes it a correct roofline for the reference's
    algorithm and a wrong lower bound for the problem, which any correct kernel
    beats by reading 44,136 B.
    """
    merged, stats = bm.combine_bounds(_solar(), _traffic(), _tb())
    rec = merged[KEY][UUID]
    assert rec["memory_bytes"] == 44_136, "the gathered floor, not the allocation"
    assert stats["gathered_solar_memory_discarded"] == 1
    assert rec["solar_memory_bytes_at_allocation"] == 1_140_133_554
    # The published bound is now the floor: 44,136 B at 8 TB/s, at the clock the
    # anchor was measured at.
    assert t_sol_ms_at(rec, 2380.0) < 1e-5


def test_solar_arithmetic_survives_the_discard():
    """The arithmetic term is carried, not assumed negligible.

    On 018 it binds on 0 of 47 workloads, which is exactly why it has to be
    checked rather than dropped with the memory term: "it was small last time" is
    not a derivation. Given a workload where it does bind, it must win.
    """
    merged, stats = bm.combine_bounds(
        _solar(compute_cycles=2_000_000.0), _traffic(), _tb(t_b_ms=5.0))
    rec = merged[KEY][UUID]
    assert rec["t_sol_source"] == "solar_arithmetic_gathered"
    assert rec["compute_cycles"] == 2_000_000.0
    assert rec["memory_bytes"] == 44_136
    assert stats["solar_arithmetic_gathered"] == 1
    # 2e6 cycles at 2380 MHz beats the 44,136 B floor by five orders of magnitude.
    assert t_sol_ms_at(rec, 2380.0) == 2_000_000.0 / (2380.0 * 1e3)


def test_a_workload_with_no_gathered_axes_is_untouched():
    """The trigger is the traffic tier's own derived signal, nothing else.

    Without it SOLAR's memory term is a normal roofline term and must keep
    binding -- this is the negative control that stops the rule from quietly
    deleting memory terms across the corpus.
    """
    merged, stats = bm.combine_bounds(
        _solar(), _traffic(gathered=False), _tb(t_b_ms=5.0))
    rec = merged[KEY][UUID]
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["memory_bytes"] == 1_140_133_554
    assert "gathered_solar_memory_discarded" not in stats
    # What must be absent is the EVIDENCE OF A CORRECTION, not the D39 band.
    # `bound_quality` is now taken on every published bound (see
    # test_build_manifest_bound_quality.py); marking only the corrected records
    # is what made 63 of 3717 look like the manifest's whole loose population.
    assert "solar_memory_bytes_at_allocation" not in rec
    assert "t_sol_cycles_solar_uncorrected" not in rec
    assert not any(k.startswith("gathered_bound_quality_") for k in stats)


def test_the_correction_arrives_marked_bound_quality():
    """It trades a detectable error for an undetectable one, and says so.

    Before: too LARGE -- a real kernel falsifies it and check D fires. After: the
    gathered floor, with `T_b / T_SOL` around 1.5e5x, far past D39's 100x. Nothing
    downstream can contradict a bound nothing can reach, so the marking is the
    only thing that keeps the trade visible.
    """
    merged, stats = bm.combine_bounds(_solar(), _traffic(), _tb())
    rec = merged[KEY][UUID]
    assert rec["bound_quality"] == "vacuous"
    assert rec["bound_headroom"] > 1000.0
    assert stats["gathered_bound_quality_vacuous"] == 1


def test_it_is_the_MI350X_v11_rule_and_lands_on_the_same_number():
    """Cross-part parity, checked as arithmetic rather than asserted.

    MI350X v1.2 publishes 018 at `declared_traffic_gathered`, 6.153846e-06 ms --
    44,136 B at 8 TB/s expressed at F_LOCK 1300, i.e. 8 cycles / 1300e3. The same
    record evaluated at 1300 MHz must reproduce it, which is what "the same rule"
    means when one part is locked and the other is not.
    """
    merged, _ = bm.combine_bounds(_solar(), _traffic(), _tb())
    assert t_sol_ms_at(merged[KEY][UUID], 1300.0) == 8 / (1300.0 * 1e3)


# --------------------------------------------------------------- the latent 202


def test_the_rule_covers_a_paged_problem_the_moment_solar_produces_a_bound():
    """The SOLAR timeout is load-bearing today, and must stop being so.

    Five more paged problems -- FlashInfer-Bench 012 (48 workloads), 013 (48),
    014 (30), 015 (38), 019 (38) = 202 -- have the identical cast-before-gather
    reference and identical measured SOLAR behaviour, and carry no SOLAR tier only
    because SOLAR timed out at 900 s on them. Raising that timeout would hand all
    202 an allocation-priced bound and silently undo the D18 tier fix on five
    problems, two of which (014, 015) already have kernels beating their bounds.

    So the rule is pinned against a record SOLAR has not yet produced on this
    part, using the bytes a probe measured for 014's smallest workload
    (`a31a212f`, SOLAR 32,012,552 B against an allocation of 32,012,568 and a
    gathered floor of 37,144). The correction must fire on it unprompted.
    """
    key = "FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1"
    uuid = "a31a212f-5e08-4961-879d-781f5886cf06"
    solar = {key: {uuid: {
        "t_sol_cycles": 7204, "t_sol_ms": 7204 / 1.8e6, "bottleneck": "memory",
        "compute_cycles": 12.0, "memory_bytes": 32_012_552,
        "mac_per_cycle": 524288.0, "dram_byte_per_sec": BPS,
        "f_ref_mhz": 1800.0}}}
    traffic = {key: {uuid: {
        "t_sol_cycles": 12, "t_sol_ms": 12 / 2.4e6, "bottleneck": "memory",
        "compute_cycles": 0.0, "memory_bytes": 37_144,
        "allocation_bytes": 32_012_568, "mac_per_cycle": None,
        "dram_byte_per_sec": BPS, "f_ref_mhz": 2400.0,
        "gathered_axes": {"num_pages": "num_kv_indices"}}}}
    tb = {key: {uuid: {"variant": "v1_eager", "t_b_ms": 6.837, **BRACKET}}}

    rec = bm.combine_bounds(solar, traffic, tb)[0][key][uuid]
    assert rec["memory_bytes"] == 37_144, (
        "a SOLAR bound appearing on a timed-out paged problem must be corrected "
        "on arrival, not after someone notices the bounds moved")
    assert rec["bound_quality"] is not None
