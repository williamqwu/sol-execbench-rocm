# SPDX-License-Identifier: Apache-2.0
"""Where the clock bracket is, structurally, in the eval driver.

The bracket is only worth anything if it surrounds the timed window and nothing
else. A prior attempt bracketed `evaluate()`, which spawns the eval-driver
subprocess and therefore contains packaging, compilation, `max_autotune` and the
whole correctness pass: the kernel was 0.8-55% of the bracketed span, the
observed clock spread was 36%, and 85% of measurements were refused. That is a
measurement of the compiler's clock, not the kernel's.

These tests read the driver template with `ast` rather than running it, because
running it needs a GPU and the property is structural: *what is inside the
bracket*. A test that ran the driver on a GPU would tell us the numbers came out;
only this tells us they came from the right region.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sol_execbench.driver as _driver_pkg

SOURCE = (Path(_driver_pkg.__file__).parent / "templates" / "eval_driver.py").read_text()
TREE = ast.parse(SOURCE)


def _calls(tree, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_every_time_runnable_call_is_bracketed():
    """Both arms — the solution's and the reference's — or the two clocks §4.4
    wants compared are not comparable."""
    bracketed = _calls(TREE, "_bracketed_timing")
    assert len(bracketed) == 2, "solution arm and reference arm, no more, no less"

    inner = [c for b in bracketed for c in _calls(b, "time_runnable")]
    assert len(inner) == 2
    assert len(_calls(TREE, "time_runnable")) == 2, \
        "a time_runnable call outside the bracket would be an unclocked measurement"


def test_the_bracket_contains_a_time_runnable_call_and_nothing_else():
    """The window is the timed region. Anything else inside it — an allocation,
    a correctness check, a compile — is time the clock is being averaged over
    that no measurement depends on."""
    for call in _calls(TREE, "_bracketed_timing"):
        thunk = call.args[0]
        assert isinstance(thunk, ast.Lambda), "the thunk must be a bare lambda"
        assert isinstance(thunk.body, ast.Call), \
            "the lambda's body must be a single call, not a block of work"
        assert isinstance(thunk.body.func, ast.Name)
        assert thunk.body.func.id == "time_runnable"


def test_allocation_happens_outside_the_bracket():
    """`allocate_outputs` for the DPS timing buffers precedes the bracket. Inside
    it, a large allocation would sit in the window on exactly the workloads whose
    windows are shortest."""
    for call in _calls(TREE, "_bracketed_timing"):
        assert not _calls(call, "allocate_outputs")


@pytest.mark.parametrize("forbidden", [
    "evaluate", "compile", "_call_and_collect_outputs", "compute_error_stats",
    "gen_inputs", "check_monkey_patch", "check_thread_injection",
])
def test_nothing_that_compiles_or_checks_is_inside_the_bracket(forbidden):
    """The named failure. Each of these is either compilation, input generation
    or a correctness/reward-hack check, and each takes far longer than the 1-13 ms
    window it would be averaged into."""
    for call in _calls(TREE, "_bracketed_timing"):
        assert not _calls(call, forbidden)


def test_the_bracket_helper_is_in_the_integrity_snapshot():
    """It decides which clock a bound is evaluated at, so a submission that
    replaced it could report a depressed clock, loosen its own compute bound and
    score higher with nothing in the trace to show for it."""
    names = [n for n in ast.walk(TREE)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_CRITICAL_NAMES"
                     for t in n.targets)]
    assert len(names) == 1
    listed = {e.value for e in names[0].value.elts if isinstance(e, ast.Constant)}
    assert "_bracketed_timing" in listed and "time_runnable" in listed


def test_the_bracket_helper_is_defined_before_the_snapshot_is_taken():
    """A function defined after `snapshot_critical_functions` cannot be in the
    snapshot, and the name would then be silently absent from it — a guard that
    guards nothing."""
    def _lineno(pred):
        return next(n.lineno for n in ast.walk(TREE) if pred(n))

    defined = _lineno(lambda n: isinstance(n, ast.FunctionDef)
                      and n.name == "_bracketed_timing")
    snapshotted = _lineno(lambda n: isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name)
                          and n.func.id == "snapshot_critical_functions")
    assert defined < snapshotted


def test_both_brackets_reach_the_emitted_trace():
    """A bracket that is measured and then dropped on the floor is worse than no
    bracket: it costs SMI traffic and produces no evidence."""
    perf = [c for c in ast.walk(TREE)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            and c.func.id == "Performance"]
    assert len(perf) == 1
    kwargs = {k.arg for k in perf[0].keywords}
    assert {"clock_bracket", "reference_clock_bracket"} <= kwargs


def test_both_arms_settle_with_their_own_real_kernel():
    """The settle must run the SAME callable the window is about to time.

    This part's clock is workload-dependent — 1800 MHz at 1383 W on a dense GEMM
    against 2392 MHz at 673 W on a small one, 36.8% apart on this node. A
    synthetic settle, or the solution's kernel used to settle for the reference
    arm, would leave the card at the wrong operating point and the ramp would
    simply happen inside the window instead.
    """
    calls = _calls(TREE, "_bracketed_timing")
    assert len(calls) == 2
    settles = {}
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert "settle" in kw, "every bracketed arm must settle"
        lam = kw["settle"]
        assert isinstance(lam, ast.Lambda)
        assert isinstance(lam.body, ast.Call)
        assert isinstance(lam.body.func, ast.Name)
        timed = kw.get("settle") and c.args[0].body        # the time_runnable call
        settles[lam.body.func.id] = timed.args[0].id
    # solution arm settles with user_fn and times user_fn; reference arm with ref_fn
    assert settles == {"user_fn": "user_fn", "ref_fn": "ref_fn"}


def test_the_settle_horizon_is_the_real_window_length():
    """Stability judged over a shorter horizon than the window is the bug that
    made the first settle exit after 12 ms and make things worse."""
    for c in _calls(TREE, "_bracketed_timing"):
        kw = {k.arg: k.value for k in c.keywords}
        assert "window_iters" in kw
        src = ast.unparse(kw["window_iters"])
        assert "warmup_runs" in src and "iterations" in src
