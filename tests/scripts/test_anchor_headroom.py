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

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

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
    """h_min is linear in eps, so a noisier run exempts more, a cleaner run less."""
    clean = _healthy(20, err=0.002)
    noisy = _healthy(20, err=0.02)
    h_clean = _classify_headroom(clean, TOL)
    h_noisy = _classify_headroom(noisy, TOL)
    assert h_noisy > h_clean
    assert abs(h_clean - 0.002 / (2 - 1 / (0.5 + TOL))) < 1e-12


def test_h_min_is_the_exact_constant_not_the_linearisation():
    """S = 1/(2 + d/h) exactly, so h_min/eps = 1 / (2 - 1/(0.5+tol)) = 53/6 at 3%.

    The linearisation |dS| = 0.5*eps/h gives 0.5/tol = 50/3 instead, larger by
    exactly 1/(0.5+tol) = 100/53. A larger h_min exempts MORE workloads from the
    gate, so the old constant never caused a false failure -- it silently
    excused workloads that were adjudicable.
    """
    eps = 0.002
    h_min = _classify_headroom(_healthy(20, err=eps), TOL)
    assert h_min == pytest.approx(eps * 53.0 / 6.0, rel=1e-12)
    assert h_min == pytest.approx(eps * (0.5 + TOL) / (2 * TOL), rel=1e-12)
    # ...and it is strictly tighter than what the linearisation asked for
    assert h_min == pytest.approx((0.5 * eps / TOL) * (0.5 + TOL), rel=1e-12)


@pytest.mark.parametrize("tol", [0.01, 0.03, 0.05, 0.10])
def test_the_fast_arm_is_the_binding_one(tol):
    """A kernel re-timed FASTER than T_b leaves the tolerance band first.

    S(x) = 1/(2+x) is decreasing and convex, so -x and +x are not equivalent:
    at the fast arm's limit S sits exactly on 0.5+tol, while the same |x| on the
    slow side is still comfortably inside 0.5-tol. h_min must come from the
    fast arm.
    """
    fast = 2 - 1 / (0.5 + tol)          # |d/h| allowed when d < 0
    slow = 1 / (0.5 - tol) - 2          # |d/h| allowed when d > 0
    assert fast < slow, "if this flips, h_min must be derived from the other arm"
    assert 1 / (2 - fast) == pytest.approx(0.5 + tol, rel=1e-12)
    assert 1 / (2 + slow) == pytest.approx(0.5 - tol, rel=1e-12)
    # the fast arm's limit applied on the slow side is strictly inside the band
    assert 1 / (2 + fast) > 0.5 - tol


def test_at_h_min_a_fast_eps_error_lands_on_the_tolerance_edge():
    """The round trip the linearised derivation would have failed.

    Build T_b and T_SOL at exactly h = h_min, re-time the anchor eps FASTER (the
    binding arm), and score it through the real scoring function. S must land on
    0.5 + tol to within floating point -- which is what makes h_min the smallest
    headroom at which the anchor property is still testable.
    """
    from sol_execbench.sol_score import sol_score

    eps = 0.002
    h_min = _classify_headroom(_healthy(20, err=eps), TOL)
    t_b = 100.0
    t_k = t_b * (1 - eps)                       # fast arm
    s = sol_score(t_k, t_b, t_b * (1 - h_min))
    assert s == pytest.approx(0.5 + TOL, abs=1e-12)
    assert abs(s - 0.5) <= TOL + 1e-12
    # a hair less headroom and the property is genuinely untestable
    assert abs(sol_score(t_k, t_b, t_b * (1 - 0.99 * h_min)) - 0.5) > TOL


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


# ---------------------------------------------------------------------------
# Over-exempting is itself a failure
#
# `_classify_headroom` is self-widening in the dangerous direction: a noisier run
# raises `eps`, which raises `h_min`, which exempts more workloads, which raises
# the PASS RATE, because the rate is computed over adjudicable workloads only. The
# exemption therefore needs a bound of its own, or "more workloads passed" and
# "fewer workloads were judged" are indistinguishable in the artifact.
# `verify_artifacts._check_headroom_exemption` is that bound.
# ---------------------------------------------------------------------------

from verify_artifacts import (  # noqa: E402
    FAIL, MAX_EXEMPT_FRACTION, MAX_H_MIN, PASS, WARN, Checks,
    _check_headroom_exemption,
)


def _statuses(ap: dict) -> list[str]:
    c = Checks()
    _check_headroom_exemption(c, ap)
    return [s for s, _, _ in c.results]


def test_an_exemption_within_the_measured_bound_passes():
    """10.02% is the worst 20-problem draw manifest v1 can produce — every
    problem containing a low-headroom workload, 57 of 569. It must not fail."""
    assert _statuses({"undecidable_insufficient_headroom": 57, "checked": 569,
                      "min_headroom_for_tolerance": 0.05}) == [PASS, PASS]


def test_over_exempting_is_a_failure_not_a_higher_pass_rate():
    ap = {"undecidable_insufficient_headroom": 100, "checked": 349,
          "min_headroom_for_tolerance": 0.05}
    assert FAIL in _statuses(ap)


def test_the_bound_is_on_the_fraction_not_the_count():
    """A larger sample may exempt more workloads and still be adjudicating the
    same share of them."""
    small = {"undecidable_insufficient_headroom": 10, "checked": 100,
             "min_headroom_for_tolerance": 0.05}
    big = {"undecidable_insufficient_headroom": 100, "checked": 1000,
           "min_headroom_for_tolerance": 0.05}
    assert _statuses(small) == _statuses(big) == [PASS, PASS]


def test_a_widened_threshold_fails_even_when_it_catches_nothing():
    """The tighter of the two statements. h_min above the measured precision band
    means the run's own re-timing got noisier, whether or not the wider exemption
    happened to catch any workload on this particular sample."""
    ap = {"undecidable_insufficient_headroom": 0, "checked": 349,
          "min_headroom_for_tolerance": MAX_H_MIN * 1.5}
    assert _statuses(ap) == [PASS, FAIL]


def test_the_bounds_are_the_measured_ones():
    """Pinned so a future edit is deliberate. Both come from the headroom
    distribution of the manifest the gate reads, not from a preference."""
    assert MAX_EXEMPT_FRACTION == 0.12 and MAX_H_MIN == 0.066


def test_an_artifact_without_the_fields_warns_rather_than_passing():
    """A check keyed on a field that does not exist always passes, and this file
    has been bitten by exactly that before (the `n_failed` guess)."""
    assert _statuses({"passing": 336, "total": 349}) == [WARN]
    assert _statuses({"undecidable_insufficient_headroom": 0}) == [WARN]
