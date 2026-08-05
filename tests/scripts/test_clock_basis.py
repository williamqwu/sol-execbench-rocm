# SPDX-License-Identifier: Apache-2.0
"""The unlocked measurement basis: what makes a measurement's clock trustworthy.

Under a fixed F_LOCK a measurement is void if its clock deviates from the constant.
Unlocked there is no constant -- the clock is part of the result and the bound is
evaluated at it -- so what voids a measurement is a clock that cannot be pinned down
at all: too few samples for a median, or a window spanning two regimes. These are
gates, never adjustments; the failure mode being defended against is a timing scored
against a guessed frequency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import provenance  # noqa: E402
from provenance import ClockMonitor, clock_basis  # noqa: E402


@pytest.fixture
def unlocked(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")


@pytest.fixture
def fixed(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "fixed_f_lock")


def _monitor_with(samples: list[int]) -> ClockMonitor:
    """A monitor holding pre-baked samples, so the gates can be tested without a GPU."""
    m = ClockMonitor()
    m._samples = [(mhz, 1200.0, 1) for mhz in samples]
    m._busy_gpus = {1}
    return m


def test_basis_defaults_to_fixed(monkeypatch):
    """Every artifact written before this change was on the fixed basis; the default
    must keep meaning that, so nothing is silently reinterpreted."""
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    assert clock_basis() == "fixed_f_lock"


def test_unknown_basis_is_refused(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "whatever")
    with pytest.raises(ValueError, match="SOLEXBENCH_CLOCK_BASIS"):
        clock_basis()


def test_stable_window_yields_a_bound_frequency(unlocked):
    """A tight, well-sampled window is exactly what per-measurement F needs."""
    s = _monitor_with([1740, 1742, 1743, 1743, 1744, 1745, 1746, 1747, 1748, 1750])
    out = s.summary()
    assert out["clock_basis"] == "unlocked"
    assert out["clock_stable"] is True
    assert out["unstable_reason"] is None
    assert out["f_for_bound_mhz"] == out["median_mhz"]
    assert out["clock_spread"] < 0.01


def test_too_few_samples_is_not_scoreable(unlocked):
    """Two samples cannot establish a median. Real case: the four clock violations in
    the locked run had n_busy_samples of 1-2, because a sub-millisecond kernel is
    invisible to a 5 Hz sampler."""
    out = _monitor_with([1740, 2300]).summary()
    assert out["clock_stable"] is False
    assert "busy samples" in out["unstable_reason"]
    assert out["f_for_bound_mhz"] is None


def test_window_spanning_two_regimes_is_not_scoreable(unlocked):
    """Autotuning bursty kernels boosts to ~2394 MHz while the timed dense kernel sits
    at ~1730. A window covering both has a median that describes neither, and unlocked
    that median would otherwise become the frequency the bound is placed at."""
    out = _monitor_with([1730] * 6 + [2390] * 6).summary()
    assert out["clock_stable"] is False
    assert "clock moved" in out["unstable_reason"]
    assert out["f_for_bound_mhz"] is None


def test_no_busy_samples_is_not_scoreable(unlocked):
    """Under the fixed basis an unsampled window is merely unverified, because
    assert_clock_lock() still covers the systematic case. Unlocked there is no
    systematic case to fall back on."""
    out = _monitor_with([]).summary()
    assert out["clock_stable"] is False
    assert "no busy samples" in out["unstable_reason"]
    assert out["f_for_bound_mhz"] is None


def test_fixed_basis_is_unchanged_by_any_of_this(fixed):
    """Regression guard. The fixed basis must not acquire stability gating, or
    previously-valid artifacts would start being rejected for a new reason."""
    out = _monitor_with([1730] * 6 + [2390] * 6).summary()
    assert out["clock_basis"] == "fixed_f_lock"
    assert "clock_stable" not in out
    assert "f_for_bound_mhz" not in out
    assert "within_tolerance" in out


def test_fixed_basis_still_flags_deviation_from_f_lock(fixed, monkeypatch):
    monkeypatch.setattr(provenance, "f_lock_mhz", lambda: 1650)
    assert _monitor_with([1650] * 10).summary()["within_tolerance"] is True
    assert _monitor_with([1320] * 10).summary()["within_tolerance"] is False


def test_unlocked_does_not_reject_for_deviating_from_f_lock(unlocked, monkeypatch):
    """The point of the exercise: 1743 MHz is not a violation just because the preset
    table says 1650. It is the frequency this measurement is scored at."""
    monkeypatch.setattr(provenance, "f_lock_mhz", lambda: 1650)
    out = _monitor_with([1743] * 12).summary()
    assert out["within_tolerance"] is False        # still reported, for diagnosis
    assert out["clock_stable"] is True             # but not what decides validity
    assert out["f_for_bound_mhz"] == 1743
