# SPDX-License-Identifier: Apache-2.0
"""The anchor property can only be adjudicated where the score scale has room.

S runs from 0.5 at T_b to 1.0 at T_SOL, so asking that a re-timed T_b score
0.5 ± 3% demands a timing precision that depends on the gap between the two.
Where T_b is already within a few percent of the speed of light, that demand falls
below any precision achievable here and the check fails on arithmetic rather than on
anything being wrong.

Measured on this node: all 171 workloads with headroom ≥ 25% passed, while all 13
failures had headroom ≤ 16% (median 3.2%) with indistinguishable timing reproduction
error (0.51% vs 0.75%).

The danger in exempting anything is obvious, so these tests are mostly about what
must NOT be exempted: a real failure at healthy headroom, and a run with no
well-conditioned sample to calibrate against.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_anchor import _classify_headroom  # noqa: E402

TOL = 0.03


def _check(headroom: float, retime_err: float, anchor_ok: bool = True) -> dict:
    """A check with the given headroom and re-timing error. t_b fixed at 100 ms."""
    t_b = 100.0
    return {
        "t_b_ms": t_b,
        "t_sol_ms": t_b * (1 - headroom),
        "t_k_ms": t_b * (1 + retime_err),
        "anchor_ok": anchor_ok,
    }


def _healthy(n: int, err: float = 0.005) -> list[dict]:
    """Well-conditioned workloads, which is what the precision is estimated from."""
    return [_check(0.80, err) for _ in range(n)]


def test_tiny_headroom_is_undecidable():
    checks = _healthy(20) + [_check(0.03, 0.007, anchor_ok=False)]
    h_min = _classify_headroom(checks, TOL)
    assert checks[-1]["headroom_sufficient"] is False
    assert "headroom" in checks[-1]["undecidable_reason"]
    assert h_min is not None and 0.03 < h_min


def test_healthy_headroom_failure_is_still_a_failure():
    """The gate exists to catch this. Exempting it would defeat the whole check."""
    checks = _healthy(20) + [_check(0.80, 0.20, anchor_ok=False)]
    _classify_headroom(checks, TOL)
    assert checks[-1]["headroom_sufficient"] is True


def test_a_noisy_workload_cannot_excuse_itself():
    """Precision is estimated from the well-conditioned sample, not per workload.
    Using each workload's own error would let a broken measurement be exempted for
    being broken."""
    checks = _healthy(20, err=0.005) + [_check(0.50, 0.40, anchor_ok=False)]
    _classify_headroom(checks, TOL)
    assert checks[-1]["headroom_sufficient"] is True, \
        "50% headroom is ample; a huge error there is a real failure"


def test_threshold_follows_the_measured_precision():
    """h_min = 0.5 * eps / tol, so a noisier run exempts more, a cleaner run less."""
    clean = _healthy(20, err=0.002)
    noisy = _healthy(20, err=0.02)
    h_clean = _classify_headroom(clean, TOL)
    h_noisy = _classify_headroom(noisy, TOL)
    assert h_noisy > h_clean
    assert abs(h_clean - 0.5 * 0.002 / TOL) < 1e-6


def test_a_few_noisy_outliers_do_not_widen_the_exemption():
    """The median is used precisely for this. A high percentile was tried first and
    gave eps=4.0%, h_min=67%, exempting 89 of 219 workloads including ones at 60%
    headroom that were passing -- an exemption wide enough to hide anything. A
    pessimistic precision estimate widens the exemption, which is the unsafe
    direction."""
    checks = _healthy(18, err=0.005) + _healthy(4, err=0.08)   # 18% wild outliers
    h_min = _classify_headroom(checks, TOL)
    assert h_min < 0.15, f"a handful of outliers must not blow up h_min, got {h_min}"


def test_exemption_stays_narrow_on_a_realistic_headroom_spread():
    """Mirrors the measured distribution: most workloads have ample headroom and only
    a small degenerate tail should be exempted."""
    checks = ([_check(0.83, 0.005) for _ in range(171)]      # ample
              + [_check(0.15, 0.005) for _ in range(23)]     # comfortable
              + [_check(0.03, 0.007, anchor_ok=False) for _ in range(16)])  # degenerate
    _classify_headroom(checks, TOL)
    exempt = sum(1 for c in checks if not c["headroom_sufficient"])
    assert exempt == 16, f"only the degenerate tail should be exempt, got {exempt}"


def test_no_well_conditioned_sample_adjudicates_everything():
    """With nothing to calibrate against, exempting the whole run would be the worst
    outcome available -- it would silently declare an unmeasured run publishable."""
    checks = [_check(0.02, 0.01, anchor_ok=False) for _ in range(5)]
    assert _classify_headroom(checks, TOL) is None
    assert all(c["headroom_sufficient"] for c in checks)


def test_undecidable_is_not_counted_as_passing():
    """The distinction that keeps this honest: excluded from the gate, not passed."""
    checks = _healthy(20) + [_check(0.03, 0.007, anchor_ok=False)]
    _classify_headroom(checks, TOL)
    passing = sum(1 for c in checks if c["anchor_ok"] and c["headroom_sufficient"])
    undecidable = sum(1 for c in checks if not c["headroom_sufficient"])
    assert passing == 20 and undecidable == 1
    assert passing + undecidable == len(checks)


def test_missing_bound_is_adjudicated_not_exempted():
    checks = _healthy(20) + [{"t_b_ms": 100.0, "t_sol_ms": None,
                              "t_k_ms": 100.0, "anchor_ok": False}]
    _classify_headroom(checks, TOL)
    assert checks[-1]["headroom_sufficient"] is True
