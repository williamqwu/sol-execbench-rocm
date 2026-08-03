# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the GPU timing attribution path.

No GPU, no torch, no vendor runtime. Run: pytest -q test_gpu_activity.py
"""

from __future__ import annotations

import pytest

from activity_sources import ReplayActivitySource, verify_clock_domain
from gpu_activity import (
    ActivityKind,
    ActivitySequenceNotFound,
    GpuActivity,
    activity_span,
    measure_iterations,
    select_activity_sequence,
    sort_activities,
)
from trace_fixtures import TraceBuilder, expected_identities

KERNELS = [("rmsnorm_kernel", 40_000), ("gemm_bf16_kernel", 250_000), ("swiglu_kernel", 30_000)]
EXPECTED = expected_identities(KERNELS)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_clean_trace_selects_user_kernels_only():
    b = TraceBuilder()
    b.iteration(KERNELS)
    picked = select_activity_sequence(sort_activities(b.activities), EXPECTED)
    assert [a.name for a in picked] == [n for n, _ in KERNELS]
    # setup copies and the flush memset must not appear
    assert all(a.kind is ActivityKind.KERNEL for a in picked)


def test_span_excludes_setup_and_flush():
    b = TraceBuilder()
    b.iteration(KERNELS, setup_copies=3, flush=True)
    picked = select_activity_sequence(sort_activities(b.activities), EXPECTED)
    span = activity_span(picked)
    # 320us of kernels + 2 inter-kernel gaps of 200ns
    assert 320_000 <= span <= 321_000
    # the flush alone is 12us; if it leaked in, span would exceed 330us
    assert span < 330_000


def test_multi_iteration_attribution_end_to_end():
    b = TraceBuilder()
    for _ in range(50):
        b.iteration(KERNELS)
    times = measure_iterations(b.activities, EXPECTED, b.windows)
    assert len(times) == 50
    assert all(0.320 <= t <= 0.321 for t in times)


def test_buffer_arrival_order_is_irrelevant():
    b = TraceBuilder()
    for _ in range(10):
        b.iteration(KERNELS)
    ordered = measure_iterations(b.activities, EXPECTED, b.windows)
    shuffled = measure_iterations(b.shuffled(), EXPECTED, b.windows)
    assert ordered == shuffled


def test_jitter_is_reflected_not_smoothed():
    b = TraceBuilder(seed=7)
    for _ in range(30):
        b.iteration(KERNELS, jitter=5_000)
    times = measure_iterations(b.activities, EXPECTED, b.windows)
    assert len(set(times)) > 1                 # real variance survives
    assert max(times) - min(times) < 0.05      # but stays bounded


# ---------------------------------------------------------------------------
# Repeated identities -- the case most likely to break a naive matcher
# ---------------------------------------------------------------------------

def test_repeated_kernel_identity_in_sequence():
    kernels = [("attn_kernel", 100_000), ("attn_kernel", 100_000), ("proj_kernel", 50_000)]
    expected = expected_identities(kernels)
    b = TraceBuilder()
    for _ in range(5):
        b.iteration(kernels)
    times = measure_iterations(b.activities, expected, b.windows)
    assert len(times) == 5
    assert all(0.250 <= t <= 0.2506 for t in times)


# ---------------------------------------------------------------------------
# Anti-reward-hacking properties
# ---------------------------------------------------------------------------

def test_foreign_kernel_between_user_kernels_still_counted_in_span():
    """Work hidden under an unrecognized kernel name is filtered from the
    *identity* match but remains inside the measured span. Renaming a kernel
    must not make its cost disappear."""
    b_clean = TraceBuilder()
    b_clean.iteration(KERNELS)
    clean = activity_span(select_activity_sequence(sort_activities(b_clean.activities), EXPECTED))

    b_hack = TraceBuilder()
    b_hack.iteration(KERNELS, interleave=[(0, "sneaky_extra_work", 500_000)])
    hacked = activity_span(select_activity_sequence(sort_activities(b_hack.activities), EXPECTED))

    assert hacked > clean + 490_000, "hidden interleaved work escaped the span"


def test_work_after_last_user_kernel_escapes_the_span():
    """Known limitation, asserted so it is a documented property rather than a
    surprise: activity that starts after the final measured kernel ends is
    outside max(end)-min(start) and is NOT attributed.

    Upstream mitigates this by taking the host end-stamp only after a full
    device synchronize, so such work still delays the *loop*; but it does not
    land in the reported span. The AMD port inherits this and should keep the
    synchronize, plus the thread/stream checks that make it hard to exploit."""
    b = TraceBuilder()
    b.iteration(KERNELS, interleave=[(2, "trailing_work", 500_000)])
    span = activity_span(select_activity_sequence(sort_activities(b.activities), EXPECTED))
    assert span < 330_000


# ---------------------------------------------------------------------------
# Degenerate / failure cases
# ---------------------------------------------------------------------------

def test_missing_kernel_raises():
    b = TraceBuilder()
    b.iteration(KERNELS[:2])
    with pytest.raises(ActivitySequenceNotFound):
        select_activity_sequence(sort_activities(b.activities), EXPECTED)


def test_empty_expected_raises():
    with pytest.raises(ActivitySequenceNotFound):
        select_activity_sequence([], [])


def test_exactly_enough_activities_is_not_off_by_one():
    b = TraceBuilder()
    b.iteration(KERNELS, setup_copies=0, flush=False)
    picked = select_activity_sequence(sort_activities(b.activities), EXPECTED)
    assert len(picked) == len(KERNELS)


def test_reordered_dispatch_falls_back_to_best_window():
    """When dispatch order differs from discovery order, stage 2 fails and
    stage 3 must still find the multiset-matching window."""
    b = TraceBuilder()
    reordered = [KERNELS[1], KERNELS[0], KERNELS[2]]
    b.iteration(reordered)
    picked = select_activity_sequence(sort_activities(b.activities), EXPECTED)
    assert sorted(a.name for a in picked) == sorted(n for n, _ in KERNELS)


def test_duplicate_full_sequence_in_window_picks_the_tighter_one():
    """Two complete copies of the sequence inside one window (e.g. a leftover
    warmup dispatch). Stage 2 returns the first exact contiguous match."""
    b = TraceBuilder()
    b.iteration(KERNELS, setup_copies=0, flush=False)
    b.iteration(KERNELS, setup_copies=0, flush=False)
    merged_window = [(b.windows[0][0], b.windows[1][1])]
    times = measure_iterations(b.activities, EXPECTED, merged_window)
    assert 0.320 <= times[0] <= 0.321   # one sequence's worth, not both


# ---------------------------------------------------------------------------
# Stage-3 scoring. These exist because mutation testing showed the suite
# otherwise passes with the (LCS, -span) tiebreak and the exact-match fast path
# both neutered -- i.e. the trickiest logic was untested.
# ---------------------------------------------------------------------------

def _k(name: str, start: int, end: int, cid: int) -> GpuActivity:
    return GpuActivity(name=name, start=start, end=end, correlation_id=cid)


def test_exact_match_fast_path_wins_over_tighter_later_window():
    """Two exact-order copies of the sequence in one window; the first is
    slow, the second tight. Stage 2 must return the FIRST occurrence.

    If the fast path were removed, stage 3's -span tiebreak would prefer the
    tight one, silently under-reporting the iteration."""
    expected = expected_identities([("A", 0), ("B", 0)])
    acts = [
        _k("A", 1_000, 2_000, 1),
        _k("B", 900_000, 901_000, 2),      # copy 1: ~900us span
        _k("A", 1_000_000, 1_001_000, 3),
        _k("B", 1_002_000, 1_003_000, 4),  # copy 2: ~3us span
    ]
    picked = select_activity_sequence(acts, expected)
    assert [a.correlation_id for a in picked] == [1, 2]
    assert activity_span(picked) == 900_000


def test_stage3_prefers_better_dispatch_order_over_shorter_span():
    """No exact-order window exists, so stage 3 arbitrates between:
      window A = [C,B,A]  LCS 1 vs expected, tightly packed (small span)
      window B = [A,C,B]  LCS 2 vs expected, widely spaced  (large span)
    Order fidelity must outrank span, so window B wins despite being longer.
    Neutering the tiebreak flips this to window A."""
    expected = expected_identities([("A", 0), ("B", 0), ("C", 0)])
    acts = [
        _k("C", 1_000, 1_100, 1),
        _k("B", 1_200, 1_300, 2),
        _k("A", 1_400, 1_500, 3),          # window A: span 500ns, LCS 1
        _k("A", 100_000, 100_100, 4),
        _k("C", 300_000, 300_100, 5),
        _k("B", 600_000, 600_100, 6),      # window B: span 500.1us, LCS 2
    ]
    picked = select_activity_sequence(acts, expected)
    assert [a.correlation_id for a in picked] == [4, 5, 6]


def test_stage3_breaks_lcs_ties_by_shortest_span():
    """Two windows with equal LCS; the tighter one must win."""
    expected = expected_identities([("A", 0), ("B", 0), ("C", 0)])
    acts = [
        _k("A", 1_000, 1_100, 1),
        _k("C", 500_000, 500_100, 2),
        _k("B", 900_000, 900_100, 3),      # [A,C,B] LCS 2, span ~899us
        _k("A", 1_000_000, 1_000_100, 4),
        _k("C", 1_000_200, 1_000_300, 5),
        _k("B", 1_000_400, 1_000_500, 6),  # [A,C,B] LCS 2, span 500ns
    ]
    picked = select_activity_sequence(acts, expected)
    assert [a.correlation_id for a in picked] == [4, 5, 6]


# ---------------------------------------------------------------------------
# Source protocol + clock-domain guard
# ---------------------------------------------------------------------------

def test_replay_source_round_trips():
    b = TraceBuilder()
    b.iteration(KERNELS)
    with ReplayActivitySource(b.activities) as src:
        pass
    assert src.drain() == b.activities


def test_clock_domain_guard_accepts_consistent_stamps():
    b = TraceBuilder()
    for _ in range(3):
        b.iteration(KERNELS)
    verify_clock_domain(b.activities, b.windows)   # must not raise


def test_clock_domain_guard_catches_wrong_domain():
    """Simulates ROCM CONTRACT #1 being violated: records stamped from a
    different clock than timestamp(). Without this guard the symptom is
    silently wrong numbers, not an exception."""
    b = TraceBuilder()
    b.iteration(KERNELS)
    offset = 10**12   # a different epoch
    skewed = [
        GpuActivity(a.name, a.start + offset, a.end + offset, a.correlation_id, a.kind)
        for a in b.activities
    ]
    with pytest.raises(RuntimeError, match="Clock-domain mismatch"):
        verify_clock_domain(skewed, b.windows)
