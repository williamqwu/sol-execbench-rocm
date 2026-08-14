# SPDX-License-Identifier: Apache-2.0
"""The clock bracket must refuse, and must refuse for the right reasons.

`docs/TODO-MI355X.md` §4.3 option 2: sample the clock either side of the timed
window, record both, and refuse the measurement when they disagree by more than a
stated threshold. The value of that is entirely in the refusal — a bracket that
records two numbers and admits everything is a decoration — so most of what is
tested here is *what does not get through*.

Three things are load-bearing and each has a test:

* the refusal fires above the threshold and not below, at the boundary;
* an unreadable clock is refused as **absent evidence**, not admitted as a
  permissive unknown;
* the bracketed window contains the thunk and nothing else, which is the whole
  reason the driver brackets `time_runnable` rather than `evaluate()`.

CPU-only. Nothing here touches a GPU or amdsmi.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
    DEFAULT_BRACKET_THRESHOLD,
    bracket_threshold,
    bracketed,
    bracketing_enabled,
    clock_basis,
    has_clock_evidence,
    make_bracket,
    summarize_brackets,
)

# A clock pair whose relative spread is exactly `s`, centred on 1800 MHz --
# roughly where a dense GEMM sits on this part (1800 MHz @1383 W, mia1-p02-g46).
def _pair(spread: float, centre: float = 1800.0) -> tuple[float, float]:
    half = centre * spread / 2.0
    return centre - half, centre + half


# ---------------------------------------------------------------- the refusal


@pytest.mark.parametrize("spread,refused", [
    (0.0, False),                                   # a perfectly steady card
    (0.00111, False),                               # the measured MEDIAN spread
    (DEFAULT_BRACKET_THRESHOLD * 0.999, False),     # just inside
    (DEFAULT_BRACKET_THRESHOLD * 1.001, True),      # just outside
    (0.082, True),                                  # smallest measured excursion
    (0.264, True),                                  # largest measured excursion
])
def test_refusal_fires_above_the_threshold_and_not_below(spread, refused):
    before, after = _pair(spread)
    b = make_bracket(before, after, threshold=DEFAULT_BRACKET_THRESHOLD)
    assert b.clock_bracket_refused is refused
    assert b.clock_bracket_spread == pytest.approx(spread, rel=1e-9)


def test_the_threshold_boundary_is_exclusive_not_inclusive():
    """Exactly at the threshold is admitted; strictly above is refused.

    Pinned because "> vs >=" is the kind of edge that gets flipped during a
    refactor and shifts the refusal rate that task 01's acceptance reads.
    """
    thr = 0.01
    at = make_bracket(*_pair(thr), threshold=thr)
    assert at.clock_bracket_spread == pytest.approx(thr, rel=1e-12)
    assert at.clock_bracket_refused is False
    over = make_bracket(*_pair(thr * (1 + 1e-9)), threshold=thr)
    assert over.clock_bracket_refused is True


def test_refusal_is_symmetric_in_direction():
    """A card that slowed down and one that sped up are equally unusable.

    The spread is a magnitude: which sample was larger says something about the
    card but nothing about how much the bound can be trusted.
    """
    up = make_bracket(1700.0, 1900.0, threshold=0.01)
    down = make_bracket(1900.0, 1700.0, threshold=0.01)
    assert up.clock_bracket_spread == pytest.approx(down.clock_bracket_spread)
    assert up.clock_bracket_refused and down.clock_bracket_refused
    assert up.clock_mhz == down.clock_mhz == 1800.0


def test_a_refused_bracket_still_records_both_samples():
    """Refusing is a result, not an erasure. The evidence has to survive it, or
    nobody can tell a card that transitioned power state from one that was never
    read."""
    b = make_bracket(1739.0, 2392.0, threshold=DEFAULT_BRACKET_THRESHOLD)
    assert b.clock_bracket_refused is True
    assert b.clock_before_mhz == 1739.0 and b.clock_after_mhz == 2392.0
    assert b.clock_bracket_refused_reason == "bracket_spread_above_threshold"


# ------------------------------------------------------- absent evidence


@pytest.mark.parametrize("before,after", [(None, 1800.0), (1800.0, None),
                                          (None, None)])
def test_an_unreadable_clock_is_refused_not_admitted(before, after):
    """An unknown clock is not a permissive one — the same reading
    `build_manifest.py` applies to a missing F_LOCK."""
    b = make_bracket(before, after, threshold=DEFAULT_BRACKET_THRESHOLD)
    assert b.clock_bracket_refused is True
    assert b.clock_bracket_refused_reason == "no_clock_evidence"
    assert b.clock_mhz is None, "no clock may be synthesised from a missing one"
    assert b.clock_bracket_source == "unavailable"
    assert has_clock_evidence(b.as_dict()) is False


def test_no_clock_evidence_is_distinguishable_from_a_wide_bracket():
    """Two different failures. One says the card moved; the other says nobody
    looked. Collapsing them would hide a broken sampler behind a busy node."""
    absent = make_bracket(None, None, threshold=0.01)
    wide = make_bracket(*_pair(0.5), threshold=0.01)
    assert absent.clock_bracket_refused_reason != wide.clock_bracket_refused_reason


def test_zero_or_negative_clocks_are_absent_evidence():
    """A 0 MHz reading is a failed read, not a stopped card, and dividing a
    bound by it is a ZeroDivisionError at best."""
    assert make_bracket(0.0, 0.0).clock_bracket_refused_reason == "no_clock_evidence"
    assert has_clock_evidence({"clock_bracket_refused": False, "clock_mhz": 0}) is False


@pytest.mark.parametrize("record", [
    None,
    {},
    {"clock_mhz": 1800.0},                                   # no verdict
    {"clock_bracket_refused": True, "clock_mhz": 1800.0},    # refused
    {"clock_bracket_refused": False},                        # no clock
    {"clock_bracket_refused": False, "clock_mhz": None},
])
def test_has_clock_evidence_rejects_everything_incomplete(record):
    assert has_clock_evidence(record) is False


def test_has_clock_evidence_accepts_a_complete_bracket():
    assert has_clock_evidence(make_bracket(1800.0, 1802.0).as_dict()) is True


# ------------------------------------------------------------- the window


def test_the_window_contains_the_thunk_and_excludes_work_around_it():
    """The window is the timed region, not the process.

    This is the property the prior attempt got wrong by bracketing `evaluate()`:
    packaging, compilation and max_autotune landed inside the window, the kernel
    was 0.8-55% of it, and 85% of measurements were refused on a clock spread
    that was really a compilation spread. Here the "compilation" is a sleep
    before the bracket and the "teardown" a sleep after it; neither may appear
    in [t0, t1].
    """
    sleep_ms = 20.0

    def sampler(_device):
        time.sleep(sleep_ms / 1000.0)      # an SMI read is not free
        return 1800

    time.sleep(sleep_ms / 1000.0)          # stands in for compilation
    _, b = bracketed(lambda: time.sleep(sleep_ms / 1000.0), sampler=sampler)
    time.sleep(sleep_ms / 1000.0)          # stands in for teardown

    assert b.window_ms == pytest.approx(sleep_ms, rel=0.5)
    # The sampler's own cost sits OUTSIDE the window and is reported, not hidden:
    # on a short window the bracket can span several times the region it brackets
    # and a reader has to be able to see that.
    assert all(lag >= sleep_ms * 1e6 * 0.5 for lag in b.clock_bracket_lag_ns)
    t0, t1 = b.window_ns
    assert t1 > t0


def test_the_thunks_return_value_is_passed_through():
    result, b = bracketed(lambda: 3.14, sampler=lambda _d: 1800)
    assert result == 3.14 and b.clock_bracket_refused is False


def test_a_failing_thunk_propagates_rather_than_producing_a_bracket():
    """A timing that raised has no measurement to attach a clock to. Swallowing
    it here would turn a runtime error into a plausible number."""
    def boom():
        raise RuntimeError("timing failed")

    with pytest.raises(RuntimeError, match="timing failed"):
        bracketed(boom, sampler=lambda _d: 1800)


def test_the_samples_are_taken_in_order_around_the_window():
    """before -> t0 -> thunk -> t1 -> after. If the ordering were ever inverted
    the bracket would describe the wrong interval and nothing would raise."""
    events: list[str] = []

    def sampler(_device):
        events.append("sample")
        return 1800

    bracketed(lambda: events.append("thunk"), sampler=sampler)
    assert events == ["sample", "thunk", "sample"]


# ------------------------------------------------------ basis and threshold


def test_bracketing_is_off_unless_the_basis_is_unlocked(monkeypatch):
    """Default off. Switching it on for the locked MI350X corpus would change how
    every number is taken, which is a methodology change to already-measured
    work (prime directive 7)."""
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    assert clock_basis() == "locked" and bracketing_enabled() is False
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")
    assert bracketing_enabled() is True
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "UNLOCKED")
    assert bracketing_enabled() is True, "the basis is not case-sensitive"
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "locked")
    assert bracketing_enabled() is False


def test_the_default_threshold_is_the_measured_p99(monkeypatch):
    """Pinned to the number, not to a range.

    0.0078 is the 99th percentile of 6544 consecutive-sample clock spreads in
    `artifacts/01/unlocked-clock.json`. Changing it changes the refusal rate that
    task 01's acceptance reads, so it should be a deliberate edit with a
    re-derivation behind it — not a drift.
    """
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BRACKET_THRESHOLD", raising=False)
    assert DEFAULT_BRACKET_THRESHOLD == 0.0078
    assert bracket_threshold() == 0.0078
    # ... and it sits between ordinary jitter and the smallest measured excursion
    assert 0.00111 < DEFAULT_BRACKET_THRESHOLD < 0.082


def test_an_override_is_honoured_and_a_bad_one_refuses(monkeypatch):
    """A typo must not silently fall back to the default: that would loosen or
    tighten every refusal in the run with nothing in the artifact to show it."""
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BRACKET_THRESHOLD", "0.02")
    assert bracket_threshold() == 0.02
    for bad in ("0.o2", "", "  ", "0", "-0.01"):
        monkeypatch.setenv("SOLEXBENCH_CLOCK_BRACKET_THRESHOLD", bad)
        if bad.strip() == "":
            assert bracket_threshold() == DEFAULT_BRACKET_THRESHOLD
        else:
            with pytest.raises(ValueError):
                bracket_threshold()


def test_the_threshold_in_force_lands_on_every_bracket(monkeypatch):
    """A refusal is only auditable if the rule it was refused under is recorded
    beside it."""
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BRACKET_THRESHOLD", "0.05")
    b = make_bracket(*_pair(0.03))
    assert b.clock_bracket_threshold == 0.05 and b.clock_bracket_refused is False


# ----------------------------------------------------------- the summary


def test_the_refusal_rate_is_a_field_not_a_log_line():
    """Task 01's acceptance on an unlocked part is "the refusal rate is below a
    stated bound". A rate a reader has to grep for cannot gate anything."""
    records = (
        [make_bracket(*_pair(0.001)).as_dict() for _ in range(97)]
        + [make_bracket(*_pair(0.2)).as_dict() for _ in range(2)]
        + [make_bracket(None, None).as_dict()]
    )
    s = summarize_brackets(records)
    assert s["n_bracketed"] == 100 and s["n_refused"] == 3
    assert s["refusal_rate"] == pytest.approx(0.03)
    assert s["refused_by_reason"] == {"bracket_spread_above_threshold": 2,
                                      "no_clock_evidence": 1}
    assert s["spread_max"] == pytest.approx(0.2)


def test_a_mixture_of_thresholds_is_reported_as_a_mixture():
    """Two thresholds in one artifact means two policies produced it, which is a
    fact about the artifact and not something to collapse to the first one."""
    s = summarize_brackets([make_bracket(1800.0, 1801.0, threshold=0.01).as_dict(),
                            make_bracket(1800.0, 1801.0, threshold=0.02).as_dict()])
    assert s["thresholds_in_force"] == [0.01, 0.02]


def test_an_empty_set_reports_no_rate_rather_than_a_perfect_one():
    """0/0 must not read as "nothing was refused"."""
    s = summarize_brackets([])
    assert s["n_bracketed"] == 0 and s["refusal_rate"] is None


# ------------------------------------------- what the bracket does NOT claim


def test_the_bracket_says_nothing_about_the_short_window_bias():
    """The one misreading that would matter.

    On mia1-p02-g46 the worst shape reads +106.9% per iteration at
    `time_runnable`'s own burst length against a 50,000-iteration loop, and the
    per-iteration cost attributes as 21.1 / 12.6 / 1.2 us across shapes -- an 18x
    spread, where a depressed clock would slow all of them alike. So the bias is
    not a clock effect and a tight bracket does not shrink it.

    This test states that as code: a perfectly steady clock admits the
    measurement, and the bracket carries nothing that could be read as a
    correction, bound or estimate for the window bias.
    """
    b = make_bracket(1800.0, 1800.0).as_dict()
    assert b["clock_bracket_refused"] is False
    assert b["clock_bracket_spread"] == 0.0
    assert not [k for k in b if "bias" in k or "correct" in k]


# ---------------------------------------------------------------------------
# The pre-window settle.
#
# It runs the real kernel until the clock stops climbing, BEFORE the window
# opens. It changes the state the card is in, never the measured quantity --
# that distinction is what makes it a fix for a duty-cycle artifact rather than
# §4.3 option 1, which was declined because lengthening the window changes what
# is measured and breaks comparability with upstream.
# ---------------------------------------------------------------------------

from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
    SETTLE_STABLE_SAMPLES,
    settle_clock,
    settle_enabled,
)


def _ramp(values):
    """A sampler replaying a fixed clock trajectory, holding the last value."""
    seq = list(values)
    state = {"i": 0}

    def sampler(_device):
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]

    return sampler


#: A stand-in kernel with a real, known cost. The settle sizes its batches from
#: the measured per-iteration time and judges stability over `window_iters` of
#: them, so a zero-cost kernel would collapse the horizon to nothing and test
#: none of the logic that matters.
_KERNEL_MS = 0.001


def _settle(sampler, **kw):
    # 30 iterations x 1 ms = a 30 ms horizon, against ~10 ms batches: three
    # samples land inside it, which is what SETTLE_STABLE_SAMPLES asks for.
    kw.setdefault("window_iters", 30)
    kw.setdefault("band", 0.0078)
    return settle_clock(lambda: time.sleep(_KERNEL_MS), device=0,
                        sampler=sampler, synchronize=lambda: None, **kw)


def test_a_card_already_flat_settles_immediately():
    r = _settle(_ramp([2390] * 400), max_ms=5000.0)
    assert r["settled"] is True and r["settle_capped"] is None


def test_a_climbing_card_keeps_going_until_it_flattens():
    r = _settle(_ramp([1600, 1800, 2000, 2200, 2380] + [2390] * 400),
                max_ms=5000.0)
    assert r["settled"] is True
    assert r["settle_entry_mhz"] == 1600 and r["settle_exit_mhz"] == 2390
    assert r["settle_min_mhz"] == 1600 and r["settle_max_mhz"] == 2390


def test_a_slow_ramp_does_not_read_as_settled():
    """THE regression. A monotonic climb of ~0.3% per sample keeps every
    ADJACENT pair inside the band while the total move over the window's own
    duration is many times it. The first implementation exited on exactly this,
    after a median of 12 ms and 6 iterations, and the refusal rate got WORSE --
    53.1% to 78.1%. Stability must be judged over the horizon the bracket will
    judge the measurement over, never a shorter one."""
    creep = [2000 + 6 * i for i in range(400)]       # +0.3%/sample, monotonic
    r = _settle(_ramp(creep), max_ms=300.0)
    assert r["settled"] is False and r["settle_capped"] == "max_ms"


def test_the_caps_bound_a_card_that_never_settles():
    for kw, expect in (({"max_ms": 200.0}, "max_ms"),
                       ({"max_iters": 3, "max_ms": 1e9}, "max_iters")):
        r = _settle(_ramp([2000 + 60 * i for i in range(500)]), **kw)
        assert r["settled"] is False and r["settle_capped"] == expect


def test_a_capped_settle_is_visible_not_silently_accepted():
    """A measurement taken after a settle that timed out was NOT taken under the
    condition the settle claims to establish, and averaging it in silently would
    hide the exact population this exists to fix."""
    r = _settle(_ramp([2000 + 60 * i for i in range(500)]), max_ms=200.0)
    assert r["settled"] is False
    assert r["settle_capped"] == "max_ms"
    assert r["settle_ms"] > 0 and r["settle_iterations"] > 0


def test_the_stability_horizon_is_recorded():
    """Without it a reader cannot tell a settle that held for 300 ms from one
    that held for 12 ms, and that difference is the entire fix."""
    r = _settle(_ramp([2390] * 400), window_iters=60, max_ms=5000.0)
    assert r["settle_stability_horizon_ms"] is not None
    assert r["settle_stability_horizon_ms"] > 0


def test_a_longer_window_demands_a_longer_settle():
    """The horizon scales with the window being protected, because that is the
    interval the bracket will measure across."""
    short = _settle(_ramp([2390] * 400), window_iters=30, max_ms=5000.0)
    long_ = _settle(_ramp([2390] * 400), window_iters=200, max_ms=5000.0)
    assert long_["settle_stability_horizon_ms"] > short["settle_stability_horizon_ms"]


def test_the_settle_record_reaches_the_bracket():
    r, br = bracketed(lambda: None, device=0, sampler=_ramp([2390] * 400),
                      settle=lambda: time.sleep(_KERNEL_MS), window_iters=30,
                      threshold=0.0078)
    assert br.settle is not None and br.settle["settled"] is True
    assert br.clock_bracket_refused is False


def test_no_settle_means_no_settle_field():
    """The locked basis and the NVIDIA path must be byte-identical to before."""
    _, br = bracketed(lambda: None, device=0, sampler=_ramp([2390] * 5))
    assert br.settle is None


def test_settling_is_off_unless_bracketing_is(monkeypatch):
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    monkeypatch.delenv("SOLEXBENCH_CLOCK_SETTLE", raising=False)
    assert settle_enabled() is False
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")
    assert settle_enabled() is True
    monkeypatch.setenv("SOLEXBENCH_CLOCK_SETTLE", "0")
    assert settle_enabled() is False, "the opt-out exists to measure the settle "\
        "against its own absence in one session"
    monkeypatch.setenv("SOLEXBENCH_CLOCK_SETTLE", "1")
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "locked")
    assert settle_enabled() is False, "never on the locked basis, whatever the knob says"


# --- checked_clock_basis: the "locked" claim must be supportable -------------
#
# Regression for a defect that cost a full re-measurement sweep. An
# authoritative T_b pass was launched without SOLEXBENCH_CLOCK_BASIS=unlocked;
# clock_basis() defaulted to "locked", the runner stamped that on six MI355X
# artifacts, and the run reported "6 ok, 0 failed". The manifest then dropped
# all six for having no per-measurement clock, and the only visible symptom was
# the scoreable problem count going DOWN. Nothing failed anywhere.

import pytest  # noqa: E402

from sol_execbench.core.bench.clock_bracket import (  # noqa: E402
    ClockBasisUnsupported,
    checked_clock_basis,
)


def test_unlocked_is_returned_and_never_checked(monkeypatch):
    """The unlocked basis needs no preset -- it carries its own clock evidence."""
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "unlocked")
    assert checked_clock_basis("AMD Instinct MI355X") == "unlocked"
    assert checked_clock_basis(None) == "unlocked"


def test_locked_survives_on_a_part_with_an_achieved_f_lock(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "locked")
    assert checked_clock_basis("NVIDIA B200") == "locked"


def test_unset_env_raises_on_a_part_with_no_achieved_f_lock(monkeypatch):
    """The silent default is the whole defect: no answer is better than a wrong one."""
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    with pytest.raises(ClockBasisUnsupported, match="unlocked"):
        checked_clock_basis("AMD Instinct MI355X")


def test_explicitly_saying_locked_does_not_make_it_true(monkeypatch):
    monkeypatch.setenv("SOLEXBENCH_CLOCK_BASIS", "locked")
    with pytest.raises(ClockBasisUnsupported):
        checked_clock_basis("AMD Instinct MI355X")


def test_unknown_device_is_refused_rather_than_assumed_locked(monkeypatch):
    monkeypatch.delenv("SOLEXBENCH_CLOCK_BASIS", raising=False)
    with pytest.raises(ClockBasisUnsupported):
        checked_clock_basis("Some Accelerator Nobody Has Calibrated")
