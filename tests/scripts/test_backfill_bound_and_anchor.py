# SPDX-License-Identifier: Apache-2.0
"""Two things `backfill_scores.py` was reading from the wrong place.

**The bound.** It took `t_sol_ms` straight out of the manifest. Under the
unlocked basis that column is T_SOL at the *reference* clock `sol_bounds.py` was
run with -- the manifest says so at its own top level -- and the bound a T_k is
judged against is T_SOL over T_K'S OWN clock window. `score_solutions.py` has
done that since the unlocked basis landed. On MI355X manifest-v4 the two columns
differ by more than 1% on 1622 of 3717 scoreable workloads and by more than 30%
on 1084, in both directions, so it is not a conservative simplification. Measured
end to end: the same backfill of `full-01` against manifest-v3 reports 12
records faster than their own bound before this fix and 5 after -- the same 5
that task 03's check D reports once IT re-derives at the measurement's bracket.
Two consumers agreeing is the point; they now share `_interval_score`.

**The anchor tree.** The card check compared each record's T_k card against
`artifacts/06-MI355X/authoritative` while the manifest was built from
`authoritative-merged`. A problem is anchored on several cards across three
nodes and the merge publishes one of them, so that compared one measurement's
card against a different measurement of the same problem. 126 records keep
`sol_score_v1` under the wrong tree, 594 under the manifest's own.

CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import backfill_scores as bf  # noqa: E402

BPS = 7999920000000.0
KEY = "L1__001_attention_softmax_dropout_value_matmul_backward"
UUID = "u0"

#: Compute-bound, so the bound scales with the clock and the two columns differ.
#: 2,400,000 cycles is 1.0 ms at 2.4 GHz and 1.3333 ms at 1.8 GHz.
CYCLES = 2_400_000.0


def _bounds(f_ref_mhz=1800.0):
    """A manifest T_SOL entry in `load_manifest_bounds`' shape."""
    return {KEY: {"workloads": {UUID: {
        "t_sol_ms": CYCLES / (f_ref_mhz * 1e3),
        "t_sol_cycles": int(CYCLES),
        "compute_cycles": CYCLES, "memory_bytes": 4096,
        "dram_byte_per_sec": BPS, "mac_per_cycle": 524288.0}}}}


def _t_b(ms=4.0):
    return {KEY: {"workloads": {UUID: {"t_b_ms": ms, "t_b_variant": "torch"}}}}


def _record(bracket=True):
    r = {"workload_uuid": UUID, "t_k_ms": 2.0, "t_ref_ms": 5.0,
         "correct": True, "t_sol_ms": None, "t_b_ms": None}
    if bracket:
        r["clock_bracket"] = {"clock_before_mhz": 2400.0,
                              "clock_after_mhz": 2400.0}
    return r


def test_the_bound_is_taken_at_the_measurements_own_clock():
    """Not the manifest's reference-clock column. 2,400,000 cycles over a 2.4 GHz
    bracket is 1.0 ms; the stored column, at f_ref 1.8 GHz, says 1.3333."""
    rec, changed = bf.backfill_record(_record(), KEY, _bounds(), _t_b())
    assert changed
    assert rec["t_sol_ms"] == 1.0
    assert rec["t_sol_ms_published"] == 1.0
    assert rec["t_sol_published_at_mhz"] == 2400.0
    # The reference-clock column is what the old code would have used, and it is
    # 1.333x away -- the D63 ratio, in the direction that deflates S.
    assert abs(_bounds()[KEY]["workloads"][UUID]["t_sol_ms"] - 4 / 3) < 1e-12


def test_the_score_follows_the_bound_it_was_computed_against():
    """S is not just re-labelled: the number moves, and it must move with the
    bound rather than with whichever column happened to be read."""
    rec, _ = bf.backfill_record(_record(), KEY, _bounds(), _t_b())
    # S = 1 / (1 + (T_k - T_SOL)/(T_b - T_SOL)) with T_k 2.0, T_b 4.0, T_SOL 1.0
    assert abs(rec["sol_score"] - 1 / (1 + (2.0 - 1.0) / (4.0 - 1.0))) < 1e-12


def test_a_record_with_no_bracket_keeps_the_reference_clock_column():
    """The locked basis and every pre-bracket record. Refused, not guessed: the
    bound is what the manifest says, exactly as before, and 0 of MI350X's 3717
    scoreable workloads take any other path."""
    rec, changed = bf.backfill_record(_record(bracket=False), KEY,
                                      _bounds(), _t_b())
    assert changed
    assert abs(rec["t_sol_ms"] - 4 / 3) < 1e-12
    assert "t_sol_ms_published" not in rec


def test_a_bound_that_cannot_be_re_clocked_is_not_invented():
    """A pre-split manifest record carries no terms, so no interval can be formed
    even with a perfectly good bracket. `_interval_score` returns {} and the
    stored column stands -- a guessed bound would be wrong only at clocks nobody
    would think to check."""
    b = _bounds()
    del b[KEY]["workloads"][UUID]["compute_cycles"]
    rec, _ = bf.backfill_record(_record(), KEY, b, _t_b())
    assert abs(rec["t_sol_ms"] - 4 / 3) < 1e-12


def test_the_anchor_tree_defaults_to_the_one_the_manifest_names():
    class _Tree:
        def dir(self, task):
            return Path("/nowhere") / task

    trees = bf.anchor_trees(
        None, {"sources": {"t_b": "artifacts/06-MI355X/authoritative-merged"}},
        _Tree())
    assert trees == [ROOT / "artifacts/06-MI355X/authoritative-merged"]


def test_a_manifest_with_no_sources_keeps_the_old_default():
    """Both frozen MI350X manifests have no `sources` block, so their backfills
    take exactly the path they took before."""
    class _Tree:
        def dir(self, task):
            return Path("/six") / task

    assert bf.anchor_trees(None, {}, _Tree()) == [Path("/six/06/authoritative")]


def test_an_explicit_tree_still_wins():
    class _Tree:
        def dir(self, task):
            return Path("/six") / task

    explicit = [Path("/a"), Path("/b")]
    assert bf.anchor_trees(explicit, {"sources": {"t_b": "x"}},
                           _Tree()) == explicit
