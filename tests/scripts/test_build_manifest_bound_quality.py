# SPDX-License-Identifier: Apache-2.0
"""D39, on every published bound -- and the guard under the D18 gathered rule.

`bound_quality` is the manifest's only statement in the LOOSE direction. The
bound check is one-sided: nothing may beat a bound, nothing checks that a bound is
tight, and a bound with 1.1e6x of headroom is a `S` that is a PyTorch comparison
with no roofline content in it. The band is what makes that visible.

It was being emitted on 63 of 3957 records -- exactly the ones the same build had
just loosened -- which reads as a census of the manifest's loose bounds and is not
one. Banding all 3717 scoreable MI355X workloads by `T_b / t_sol_ms_published`
under the manifest's own `BOUND_QUALITY_BANDS` gives **vacuous 398, loose 322, ok
2482, narrow 515**, and the two populations that session moved were both outside
the marked 63:

* the **127** records the clock-correct tier comparison rescued from the rejection
  gate (D63) are `narrow` on 127 of 127 -- headroom 1.0372..1.3171, published/T_b
  0.7592..0.9641 -- which the acceptance inequality forces rather than discovers.
  Corpus p50 of published/T_b is 0.082, so those are the tightest bounds in the
  manifest and none of them said so.
* the causal-mask records on `FlashInfer-Bench__014`/`__015` reach headroom
  1.113e6 and 9.73e5, larger than anything on `__018` -- which WAS marked.

The second half of this file pins the guard on `_solar_arithmetic_only`. That rule
discards SOLAR's whole memory term on the presence of `gathered_axes`; it is right
only while that term IS the allocation the traffic tier repriced, which on today's
corpus it always is (`solar_memory_bytes / allocation_bytes` 0.9609..0.99999995
over all 63 records where it fires, never above 1) and which nothing was checking.

CPU-only, no artifact read: every record here is a literal, and the numbers in the
docstrings are the measured ones from `artifacts/09-MI355X/manifest-v4.json`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

import build_manifest as bm  # noqa: E402

BPS = 7999920000000.0
KEY = "L2__002_decoder_layer_full_block"
UUID = "11111111-2222-3333-4444-555555555555"

#: A real MI355X bracket, from the L2__002 record that motivates half of this file.
BRACKET = {"clock_before_mhz": 2369.0, "clock_after_mhz": 2386.0}


def _solar(compute_cycles=29_360_128.0, memory_bytes=1_044_414_464,
           f_ref_mhz=1800.0):
    """A SOLAR tier record, stored at *f_ref_mhz* exactly as `sol_bounds.py` writes it."""
    cycles = max(1, int(max(compute_cycles,
                            memory_bytes * f_ref_mhz * 1e6 / BPS)))
    return {KEY: {UUID: {
        "t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
        "bottleneck": "compute", "compute_cycles": compute_cycles,
        "memory_bytes": memory_bytes, "mac_per_cycle": 32768.0,
        "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref_mhz}}}


def _traffic(memory_bytes=313_328, f_ref_mhz=2400.0, gathered=None,
             allocation_bytes=None, gathered_bytes=None):
    cycles = max(1, int(memory_bytes * f_ref_mhz * 1e6 / BPS) + 1)
    rec = {"t_sol_cycles": cycles, "t_sol_ms": cycles / (f_ref_mhz * 1e3),
           "bottleneck": "memory", "compute_cycles": 0.0,
           "memory_bytes": memory_bytes, "mac_per_cycle": None,
           "dram_byte_per_sec": BPS, "f_ref_mhz": f_ref_mhz}
    if gathered is not None:
        rec["gathered_axes"] = gathered
    if allocation_bytes is not None:
        rec["allocation_bytes"] = allocation_bytes
    if gathered_bytes is not None:
        rec["gathered_bytes"] = gathered_bytes
    return {KEY: {UUID: rec}}


def _tb(t_b_ms=14.748950958251953, bracket=True):
    return {KEY: {UUID: {"variant": "v4_contiguous", "t_b_ms": t_b_ms,
                         **(BRACKET if bracket else {})}}}


def _one(solar, traffic, tb):
    merged, stats = bm.combine_bounds(solar, traffic, tb)
    return merged[KEY][UUID], stats


# ------------------------------------------------------- every bound gets a band


def test_an_ordinary_uncorrected_bound_is_banded():
    """The plain case, which is 3894 of the manifest's 3957 records.

    Nothing was corrected here and nothing is unusual about the workload. That is
    the point: the band is a property of every published bound, not a footnote on
    the ones a build touched.
    """
    rec, stats = _one(_solar(), _traffic(), _tb())
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["bound_quality"] == "narrow"
    assert stats["bound_quality_narrow"] == 1
    # No correction fired, so none of the correction's own evidence is present.
    assert "solar_memory_bytes_at_allocation" not in rec


def test_the_band_is_taken_on_the_PUBLISHED_bound_not_the_stored_column():
    """The whole of D63 in one assertion.

    The stored `t_sol_ms` is a cycle count over whichever REFERENCE clock its tier
    used -- 1.8 GHz for SOLAR here -- while the bound this manifest publishes is
    the same cycles at the MINIMUM of the measurement's own bracket, 2369 MHz. The
    two differ by 2369/1800 = 1.3161x, and banding the wrong one is the same unit
    error that put 127 correct bounds under the rejection gate.

    The record is the real L2__002 one, and its headroom must reproduce
    manifest-v4's published 1.190058327405755 rather than the 0.9042 the stored
    column gives -- a ratio of exactly 2369/1800 between the two answers.
    """
    rec, _ = _one(_solar(), _traffic(), _tb())
    assert rec["t_sol_ms"] == 29_360_128 / 1.8e6          # the stored column
    published = 29_360_128 / (2369.0 * 1e3)               # what is published
    assert rec["bound_headroom"] == 14.748950958251953 / published
    assert rec["bound_headroom"] == 1.190058327405755     # manifest-v4, this row
    stored_h = bm._bound_quality(rec["t_sol_ms"], 14.748950958251953)[1]
    assert abs(rec["bound_headroom"] / stored_h - 2369.0 / 1800.0) < 1e-12

    # And the 1.3161x is enough to move the BAND, not only the number: at this
    # anchor the published bound is `ok` and the stored column would say `narrow`.
    rec, _ = _one(_solar(), _traffic(), _tb(t_b_ms=26.0))
    assert rec["bound_quality"] == "ok"
    assert bm._bound_quality(rec["t_sol_ms"], 26.0)[0] == "narrow"


def test_every_band_in_the_vocabulary_is_reachable_and_counted():
    """One workload per band, with the anchor doing all the moving.

    The bands are `leaderboard/ingest.py`'s, duplicated here because that module
    lives in its own venv; a value written under a name in this manifest is read
    there under the same name, so the vocabulary has to be pinned on both sides.
    """
    published = 29_360_128 / (2369.0 * 1e3)               # 12.39346897 ms
    for factor, expected in ((1.5, "narrow"), (50.0, "ok"),
                             (500.0, "loose"), (5000.0, "vacuous")):
        rec, stats = _one(_solar(), _traffic(),
                          _tb(t_b_ms=published * factor))
        assert rec["bound_quality"] == expected, factor
        assert abs(rec["bound_headroom"] - factor) < 1e-9
        assert stats[f"bound_quality_{expected}"] == 1


def test_a_workload_with_no_t_b_is_unbanded_and_says_so():
    """`None` is a stated absence, not a claim of quality.

    A bound with no anchor has no headroom -- there is no `T_b` to be loose
    against. 240 of the manifest's 3957 records are in this state and they are
    counted under their own name so the five counters sum to the record count.
    """
    rec, stats = _one(_solar(), _traffic(), {})
    assert rec["bound_quality"] is None and rec["bound_headroom"] is None
    assert stats["bound_quality_unbanded"] == 1
    assert not any(k.startswith("bound_quality_") and k.endswith(
        ("narrow", "ok", "loose", "vacuous")) for k in stats)


def test_the_counters_partition_the_records():
    """Five counters, every record in exactly one, so the census is checkable.

    On manifest-v4 that is vacuous 398 + loose 322 + ok 2482 + narrow 515 +
    unbanded 240 = 3957, which is also the sum of `bound_sources`' tier counts. A
    census a reader cannot recompute from the artifact is how the count of
    scoreable problems drifted last time.
    """
    published = 29_360_128 / (2369.0 * 1e3)
    solar, traffic = {KEY: {}}, {KEY: {}}
    tb: dict = {KEY: {}}
    for i, factor in enumerate((1.5, 50.0, 500.0, 5000.0, None)):
        u = f"{i}0000000-0000-0000-0000-000000000000"
        solar[KEY][u] = _solar()[KEY][UUID]
        traffic[KEY][u] = _traffic()[KEY][UUID]
        if factor is not None:
            tb[KEY][u] = _tb(t_b_ms=published * factor)[KEY][UUID]
    merged, stats = bm.combine_bounds(solar, traffic, tb)
    n = len(merged[KEY])
    assert n == 5
    assert sum(v for k, v in stats.items()
               if k.startswith("bound_quality_")) == n
    tiers = ("solar_fused", "declared_traffic", "max_of_both",
             "solar_arithmetic_gathered", "declared_traffic_gathered")
    assert sum(stats[k] for k in tiers) == n


def test_an_anchor_with_no_bracket_bands_the_locked_columns():
    """The locked basis, where `t_sol_ms` IS the published bound.

    MI350X publishes one T_SOL at one F_LOCK, `_interval_fields` emits nothing,
    and there is no bracket to re-evaluate at -- so the stored column is the right
    thing to band, and banding it there is not the D63 error but the correct read.

    The anchor is 20 ms rather than the 14.75 ms above for a reason worth stating:
    with no bracket there is no clock to compare the two tiers at, so a SOLAR bound
    stored at 1.8 GHz is measured against T_b as-is and rejected the moment it
    reads above it. That is cause C, in the form it had before the clock-correct
    comparison -- and the record then publishes the declared-traffic floor, whose
    headroom here is 376,569x. Banding is downstream of that, not a cure for it.
    """
    rec, _ = _one(_solar(), _traffic(), _tb(t_b_ms=20.0, bracket=False))
    assert rec.get("t_sol_ms_published") is None
    assert rec["t_sol_source"] == "solar_fused"
    assert rec["bound_headroom"] == 20.0 / (29_360_128 / 1.8e6)
    assert rec["bound_quality"] == "narrow"

    # The rejected-tier form, for the contrast: same fixture, tighter anchor.
    rejected, _ = _one(_solar(), _traffic(), _tb(bracket=False))
    assert rejected["t_sol_tier_rejected_above_t_b"] == ["solar_fused"]
    assert rejected["bound_quality"] == "vacuous"
    assert rejected["bound_headroom"] > 1e5


def test_the_unlocked_basis_cannot_reach_the_stored_column_branch(tmp_path):
    """Why the two branches above cannot be confused, checked rather than argued.

    `_published_bound_ms` falls back to the stored column only when the anchor
    carries no clock bracket -- and on the unlocked basis `collect_t_b` admits NO
    anchor without one. So a workload that has a `T_b` to be banded against always
    has a bracket, and the stored-column branch is unreachable exactly where
    reading it would be the D63 unit error.
    """
    (tmp_path / "p.json").write_text(json.dumps({
        "problem": KEY,
        "winner_by_workload": {UUID: {"variant": "v1_eager", "t_b_ms": 1.0}},
    }))
    assert bm.collect_t_b(tmp_path, None, "unlocked") == {}
    # The same artifact WITH a bracket is admitted, so the emptiness above is the
    # bracket filter and not the fixture being malformed.
    (tmp_path / "p.json").write_text(json.dumps({
        "problem": KEY,
        "winner_by_workload": {UUID: {"variant": "v1_eager", "t_b_ms": 1.0,
                                      **BRACKET}},
    }))
    assert list(bm.collect_t_b(tmp_path, None, "unlocked")) == [KEY]


def test_the_gathered_correction_still_marks_and_still_counts_separately():
    """The 63 records keep everything they had; 3894 more join them.

    `gathered_bound_quality_*` is what this build LOOSENED. `bound_quality_*` is
    what the manifest publishes. Collapsing the two would lose the trade, which is
    the thing the correction was required to state.
    """
    solar = _solar(compute_cycles=1.0, memory_bytes=1_140_133_554)
    traffic = _traffic(memory_bytes=44_136, gathered={"num_pages": "num_kv_indices"},
                       allocation_bytes=1_140_133_608, gathered_bytes=44_136)
    rec, stats = _one(solar, traffic, _tb(t_b_ms=0.8935))
    assert rec["t_sol_source"] == "declared_traffic_gathered"
    assert rec["bound_quality"] == "vacuous"
    assert stats["gathered_bound_quality_vacuous"] == 1
    assert stats["bound_quality_vacuous"] == 1
    assert rec["solar_memory_bytes_at_allocation"] == 1_140_133_554


# --------------------------------------- the gathered rule's premise is checked


def test_the_correction_carries_the_traffic_tiers_evidence_when_solar_wins():
    """A label naming a correction must not ship beside no evidence of it.

    The D18 audit trail -- `allocation_bytes`, `gathered_axes`, `gathered_bytes` --
    is written by the traffic tier, and the manifest entry is built from whichever
    tier WON. On three `L1__009` workloads SOLAR's arithmetic won, so manifest-v4
    shipped them labelled `solar_arithmetic_gathered` with `gathered_axes` null and
    the other two absent: the correction fired and the record could not show it.
    """
    solar = _solar(compute_cycles=2_000_000.0, memory_bytes=806_217_664)
    traffic = _traffic(memory_bytes=806_228_574,
                       gathered={"batch_seq_len": "num_tokens"},
                       allocation_bytes=834_663_006, gathered_bytes=806_228_574)
    rec, _ = _one(solar, traffic, _tb(t_b_ms=5.0))
    assert rec["t_sol_source"] == "solar_arithmetic_gathered"
    assert rec["gathered_axes"] == {"batch_seq_len": "num_tokens"}
    assert rec["allocation_bytes"] == 834_663_006
    assert rec["gathered_bytes"] == 806_228_574


def test_the_rule_still_fires_across_the_whole_measured_allocation_ratio():
    """The guard must be inert on every record it fires on today.

    Measured over the 63: `solar_memory_bytes / allocation_bytes` spans 0.9609
    (L1__009) to 0.99999995 (FlashInfer-Bench__018, 54 B under 1,140,133,608).
    Both ends, and a hair over 1, must still be corrected -- a guard that clips the
    top of the observed range would silently undo D18 on part of the corpus.
    """
    allocation = 1_140_133_608
    for ratio in (0.9609, 0.99999995, 1.0, 1.0 + bm.SOLAR_MEMORY_ABOVE_ALLOCATION_REL):
        solar = _solar(compute_cycles=1.0, memory_bytes=int(allocation * ratio))
        traffic = _traffic(memory_bytes=44_136,
                           gathered={"num_pages": "num_kv_indices"},
                           allocation_bytes=allocation, gathered_bytes=44_136)
        rec, stats = _one(solar, traffic, _tb(t_b_ms=0.8935))
        assert rec["memory_bytes"] == 44_136, ratio
        assert stats["gathered_solar_memory_discarded"] == 1, ratio


def test_solar_memory_above_the_allocation_keeps_its_memory_term():
    """The failure mode the guard exists for, which today's corpus cannot show.

    A problem with a gathered axis AND a real streaming tensor has a SOLAR memory
    term that is the allocation PLUS the stream. Discarding it whole would delete
    the stream, and the resulting bound would be too SMALL -- the direction no
    measurement can contradict, because nothing downstream checks that a bound is
    tight. So the record falls back to the ordinary max-of-both behaviour: SOLAR's
    term survives, binds, and the bound stays large enough for a real kernel to
    falsify if it is wrong.
    """
    allocation = 1_140_133_608
    solar = _solar(compute_cycles=1.0, memory_bytes=allocation * 2)
    traffic = _traffic(memory_bytes=44_136,
                       gathered={"num_pages": "num_kv_indices"},
                       allocation_bytes=allocation, gathered_bytes=44_136)
    rec, stats = _one(solar, traffic, _tb(t_b_ms=5.0))
    assert rec["memory_bytes"] == allocation * 2, "the stream was not deleted"
    assert stats["gathered_solar_memory_above_allocation"] == 1
    assert "gathered_solar_memory_discarded" not in stats
    # No correction fired, so no correction is claimed.
    assert rec["t_sol_source"] == "solar_fused"
    assert "solar_memory_bytes_at_allocation" not in rec


def test_a_gathered_record_with_no_allocation_bytes_is_not_corrected():
    """No allocation, no way to check the premise -- so refuse, and count.

    Refusing leaves the allocation-priced bound in place, which is the DETECTABLE
    way to be wrong: a real kernel beats it and task 03's check D fires. Applying
    an unverified correction is the undetectable way. All 63 records that fire the
    rule today carry `allocation_bytes`, so this changes nothing on this corpus.
    """
    solar = _solar(compute_cycles=1.0, memory_bytes=1_140_133_554)
    traffic = _traffic(memory_bytes=44_136,
                       gathered={"num_pages": "num_kv_indices"})
    rec, stats = _one(solar, traffic, _tb(t_b_ms=5.0))
    assert rec["memory_bytes"] == 1_140_133_554
    assert stats["gathered_solar_allocation_unknown"] == 1
    assert "gathered_solar_memory_discarded" not in stats
    assert rec["t_sol_source"] == "solar_fused"
