# SPDX-License-Identifier: Apache-2.0
"""T_b selection must not anchor the score scale on an unclocked measurement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "runners"))

_spec = importlib.util.spec_from_file_location(
    "_time_tb_candidates", ROOT / "scripts" / "runners" / "time_tb_candidates.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
select_winners = _mod.select_winners


def _bracket(refused=False, reason=None, mhz=1803.0):
    return {"clock_before_mhz": 1800.0, "clock_after_mhz": 1806.0,
            "clock_mhz": None if refused and reason == "no_clock_evidence" else mhz,
            "clock_bracket_spread": 0.0033, "clock_bracket_threshold": 0.0078,
            "clock_bracket_refused": refused,
            "clock_bracket_refused_reason": reason}


def _variant(latencies, brackets=None, ok=True, all_passed=True, refs=None):
    return {"ok": ok, "all_passed": all_passed,
            "latency_ms_by_workload": latencies,
            "clock_bracket_by_workload": brackets or {},
            "reference_clock_bracket_by_workload": refs or {}}


def test_the_fastest_bracketed_variant_wins():
    results = {"v1": _variant({"a": 2.0}, {"a": _bracket()}),
               "v2": _variant({"a": 1.0}, {"a": _bracket()})}
    winners, refused, considered = select_winners(results, bracketing=True)
    assert winners["a"]["variant"] == "v2" and winners["a"]["t_b_ms"] == 1.0
    assert winners["a"]["clock_mhz"] == 1803.0
    assert refused == {}


def test_a_refused_measurement_cannot_become_the_anchor():
    """The one that matters. The refused variant is FASTER, so a selector that
    ignored the verdict would pick exactly it -- and T_b sets the whole scale."""
    results = {"slow_ok": _variant({"a": 2.0}, {"a": _bracket()}),
               "fast_refused": _variant(
                   {"a": 1.0},
                   {"a": _bracket(refused=True,
                                  reason="bracket_spread_above_threshold")})}
    winners, refused, considered = select_winners(results, bracketing=True)
    assert winners["a"]["variant"] == "slow_ok"
    assert refused["a"][0]["variant"] == "fast_refused"
    assert refused["a"][0]["clock_bracket_refused_reason"] == \
        "bracket_spread_above_threshold"


def test_a_workload_whose_every_variant_was_refused_gets_no_anchor():
    """Loud, not quiet: the problem then reports incomplete and reaches triage
    instead of shipping an anchor measured at an unknown clock."""
    results = {"v1": _variant({"a": 1.0}, {"a": _bracket(refused=True,
                                                         reason="no_clock_evidence")})}
    winners, refused, considered = select_winners(results, bracketing=True)
    assert winners == {} and list(refused) == ["a"]


def test_a_missing_bracket_is_refused_under_the_unlocked_basis():
    results = {"v1": _variant({"a": 1.0}, {})}
    winners, refused, considered = select_winners(results, bracketing=True)
    assert winners == {}
    assert refused["a"][0]["clock_bracket_refused_reason"] == "no_bracket_recorded"


def test_the_locked_basis_selects_exactly_as_it_always_did():
    """No bracket, no refusal, same winner. The MI350X corpus must not move."""
    results = {"v1": _variant({"a": 2.0}), "v2": _variant({"a": 1.0})}
    winners, refused, considered = select_winners(results, bracketing=False)
    assert winners == {"a": {"variant": "v2", "t_b_ms": 1.0,
                             "reference_clock_bracket": None}}
    assert refused == {}


def test_a_variant_that_failed_a_workload_is_still_ineligible():
    """Fast-but-wrong is not a baseline, bracket or no bracket."""
    results = {"wrong": _variant({"a": 0.1}, {"a": _bracket()}, all_passed=False),
               "right": _variant({"a": 1.0}, {"a": _bracket()})}
    winners, _, _ = select_winners(results, bracketing=True)
    assert winners["a"]["variant"] == "right"


def test_the_reference_arm_bracket_travels_with_the_winner():
    """§4.4: T_b and T_k timed back to back, BOTH clocks recorded."""
    ref = _bracket(mhz=1795.0)
    results = {"v1": _variant({"a": 1.0}, {"a": _bracket()}, refs={"a": ref})}
    winners, _, _ = select_winners(results, bracketing=True)
    assert winners["a"]["reference_clock_bracket"]["clock_mhz"] == 1795.0


def test_refusal_is_per_workload_not_per_problem():
    """A clock excursion on one shape must not cost the problem its other
    anchors."""
    results = {"v1": _variant(
        {"a": 1.0, "b": 2.0},
        {"a": _bracket(), "b": _bracket(refused=True, reason="no_clock_evidence")})}
    winners, refused, considered = select_winners(results, bracketing=True)
    assert set(winners) == {"a"} and set(refused) == {"b"}


# ---------------------------------------------------------------------------
# The summary and the refusal counter must come from ONE place.
#
# They did not, and the consequence was a live sweep that reported
# `n_workloads_refused_on_clock: 16` beside `n_bracketed: 0, n_refused: 0,
# refused_by_reason: {}`. The summary was built from the WINNERS, and a refused
# measurement never becomes a winner — so the field a reader would check to
# discover a 100% refusal rate was empty exactly because the rate was 100%.
# ---------------------------------------------------------------------------


def test_the_summary_sees_refusals_the_counter_sees():
    """The regression, stated directly."""
    from sol_execbench.core.bench.clock_bracket import summarize_brackets

    results = {"v1": _variant(
        {f"w{i}": 1.0 for i in range(16)},
        {f"w{i}": _bracket(refused=True, reason="no_clock_evidence")
         for i in range(16)})}
    winners, refused, considered = select_winners(results, bracketing=True)
    s = summarize_brackets(considered)

    assert winners == {}
    assert len(refused) == 16, "the per-workload counter still sees them"
    assert s["n_bracketed"] == 16, "and now so does the summary"
    assert s["n_refused"] == 16
    assert s["refusal_rate"] == 1.0
    assert s["refused_by_reason"] == {"no_clock_evidence": 16}


def test_considered_holds_winners_and_refusals_together():
    results = {"v1": _variant({"a": 1.0, "b": 2.0},
                              {"a": _bracket(),
                               "b": _bracket(refused=True,
                                             reason="no_clock_evidence")})}
    _, _, considered = select_winners(results, bracketing=True)
    assert len(considered) == 2
    assert sum(1 for b in considered if b["clock_bracket_refused"]) == 1


def test_a_sampler_error_is_not_reported_as_absent_telemetry():
    """"Nobody could read the clock" and "we called the sampler wrong" must not
    collapse into one reason — that collapse is what hid the string-device bug
    for a whole sweep."""
    from sol_execbench.core.bench.clock_bracket import summarize_brackets

    br = _bracket(refused=True, reason="sampler_error")
    _, _, considered = select_winners(
        {"v1": _variant({"a": 1.0}, {"a": br})}, bracketing=True)
    assert summarize_brackets(considered)["refused_by_reason"] == \
        {"sampler_error": 1}


# ---------------------------------------------------------------------------
# Failing closed must be visible in the exit status.
#
# The first bracketed sweep refused 100% of its workloads and exited 0. A total
# sampling failure and a clean run under a strict threshold produced the same
# artifact shape: zero anchors, no error.
# ---------------------------------------------------------------------------

clock_fatalities = _mod.clock_fatalities


def _summary(n_bracketed, n_refused, reasons):
    return {"n_bracketed": n_bracketed, "n_refused": n_refused,
            "refusal_rate": (n_refused / n_bracketed) if n_bracketed else None,
            "refused_by_reason": reasons}


def test_no_bracket_at_all_is_fatal():
    """The exact signature of the live failure: measurements taken, none
    bracketed."""
    f = clock_fatalities(_summary(0, 0, {}), [], attempted=16, bracketing=True)
    assert len(f) == 1 and "NOT ONE carries a clock bracket" in f[0]


def test_total_refusal_is_fatal():
    f = clock_fatalities(_summary(16, 16, {"bracket_spread_above_threshold": 16}),
                         [], attempted=16, bracketing=True)
    assert len(f) == 1 and "Do not raise the threshold" in f[0]


def test_a_sampler_error_is_fatal_even_at_a_low_rate():
    """One raise is a defect in our code, not a property of the node, so it does
    not get to hide behind a healthy majority."""
    considered = [{"clock_bracket_sampler_error": "TypeError: int() argument..."}]
    f = clock_fatalities(_summary(64, 1, {"sampler_error": 1}), considered,
                         attempted=64, bracketing=True)
    assert len(f) == 1 and "RAISED" in f[0] and "TypeError" in f[0]


def test_a_high_but_partial_refusal_rate_is_not_fatal():
    """57.8% was the real measured rate on g46 once the bug was fixed. It is a
    finding to report, not a crash: 12 of 16 workloads still got an anchor."""
    f = clock_fatalities(_summary(64, 37, {"bracket_spread_above_threshold": 37}),
                         [], attempted=64, bracketing=True)
    assert f == []


def test_nothing_is_fatal_on_the_locked_basis():
    assert clock_fatalities(_summary(0, 0, {}), [], attempted=16,
                            bracketing=False) == []


def test_a_problem_where_every_variant_failed_correctness_is_not_a_clock_failure():
    """attempted == 0. Blaming the clock for a correctness failure would send
    triage to the wrong place."""
    assert clock_fatalities(_summary(0, 0, {}), [], attempted=0,
                            bracketing=True) == []
