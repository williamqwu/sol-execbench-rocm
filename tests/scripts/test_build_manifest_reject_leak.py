# SPDX-License-Identifier: Apache-2.0
"""A tier rejected for sitting above T_b must not come back through re-clocking.

`combine_bounds` nulls the rejected tier's cycle count, but the published bound
under the unlocked basis is re-evaluated from the `compute_cycles` /
`memory_bytes` union `_reclock_terms` builds. While that union was taken over
both tiers unconditionally, the rejected tier supplied the winning term and the
gate had no effect: 41 workloads across 4 problems shipped in MI355X
manifest-v2 above their own measured T_b, by up to 4.06x.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_manifest import combine_bounds  # noqa: E402

KEY = "L1__029_mamba_conv1d_with_gating"
UUID = "u0"
BPS = 7999920000000.0


def _corpus(solar_ms: float, traffic_ms: float, t_b_ms: float):
    solar = {KEY: {UUID: {
        "t_sol_cycles": 20971520, "t_sol_ms": solar_ms, "bottleneck": "compute",
        "compute_cycles": 20971520.0, "memory_bytes": 402849792,
        "mac_per_cycle": 524288.0, "dram_byte_per_sec": BPS}}}
    traffic = {KEY: {UUID: {
        "t_sol_cycles": 362466, "t_sol_ms": traffic_ms, "bottleneck": "memory",
        "compute_cycles": 0.0, "memory_bytes": 1208205312,
        "mac_per_cycle": None, "dram_byte_per_sec": BPS}}}
    tb = {KEY: {UUID: {"variant": "v2_compile", "t_b_ms": t_b_ms}}}
    return solar, traffic, tb


def test_rejected_solar_tier_does_not_supply_the_compute_term():
    # SOLAR is 11.65 ms against a 1.21 ms measurement: not a lower bound.
    out, stats = combine_bounds(*_corpus(11.65, 0.1510, 1.2140))
    rec = out[KEY][UUID]
    assert stats["solar_rejected_above_t_b"] == 1
    assert rec["t_sol_source"] == "declared_traffic"
    assert rec["t_sol_cycles_solar"] is None
    # The whole point: the rejected tier's terms are gone, so re-clocking cannot
    # reproduce its bound at any clock.
    assert rec["compute_cycles"] == 0.0
    assert rec["memory_bytes"] == 1208205312
    assert rec["t_sol_tier_rejected_above_t_b"] == ["solar_fused"]


def test_rejected_traffic_tier_does_not_supply_the_memory_term():
    out, stats = combine_bounds(*_corpus(0.0116, 0.1510, 0.1134))
    rec = out[KEY][UUID]
    assert stats["traffic_rejected_above_t_b"] == 1
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["memory_bytes"] == 402849792
    assert rec["t_sol_tier_rejected_above_t_b"] == ["declared_traffic"]


def test_surviving_tiers_still_union_their_terms():
    out, stats = combine_bounds(*_corpus(0.0116, 0.1510, 9.0))
    rec = out[KEY][UUID]
    assert stats["traffic_rejected_above_t_b"] == 0
    assert stats["solar_rejected_above_t_b"] == 0
    assert rec["t_sol_source"] == "max_of_both"
    assert rec["compute_cycles"] == 20971520.0
    assert rec["memory_bytes"] == 1208205312
    assert rec["t_sol_tier_rejected_above_t_b"] is None
