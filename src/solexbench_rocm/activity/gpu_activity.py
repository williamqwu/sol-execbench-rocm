# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral GPU activity records and timing-window selection.

This is a behaviour-preserving extraction of the selection logic currently in
``sol_execbench/core/bench/cupti_utils.py``. The original module cannot even be
IMPORTED without CUDA present, because it does ``from cupti import cupti`` at
module scope -- yet roughly 70 of its lines are pure logic over strings and
floats with no vendor coupling at all.

Splitting that logic out buys three things:

  1. The subtle part of the timing path (which GPU activities belong to *this*
     iteration of the benchmark loop) becomes testable on CPU, against
     fabricated traces, with no GPU of any kind.
  2. The ROCm port only has to supply a *record source* -- something that emits
     ``GpuActivity`` tuples -- rather than re-implementing selection.
  3. Adversarial cases (hidden work, reordered dispatch, duplicated kernels)
     can be regression-tested deterministically instead of being discovered on
     scarce hardware.

Nothing here imports torch, cupti, rocprofiler, or any GPU runtime.

Fidelity note: ``identity()`` mirrors the original ``kernel_string()``. The
original interpolated a ``cupti.ActivityKind`` enum member directly; here the
kind is a plain ``str`` enum. The rendered text therefore differs, which is
harmless because the identity string is only ever compared against other
identity strings produced in the same process during the same benchmark run.
It is never persisted or compared across vendors.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class ActivityKind(str, Enum):
    """Vendor-neutral GPU activity kinds.

    Deliberately minimal: these are the only three kinds the timing path
    consumes. CUPTI's CONCURRENT_KERNEL / MEMCPY / MEMSET and
    rocprofiler-sdk's kernel-dispatch / memory-copy / memory-set records both
    map onto this set.
    """

    KERNEL = "KERNEL"
    MEMCPY = "MEMCPY"
    MEMSET = "MEMSET"


# ---------------------------------------------------------------------------
# Symbol demangling
#
# Kept here rather than in the vendor adapters because it is genuinely shared:
# HIP kernel symbols are Itanium-ABI mangled exactly as CUDA ones are, so the
# same libstdc++ __cxa_demangle call serves both. Falls back to the raw name if
# libstdc++ is unavailable (e.g. a slim container) instead of exploding at
# import time -- a failure mode the original module has.
# ---------------------------------------------------------------------------

def _load_demangler():
    try:
        lib = ctypes.CDLL(ctypes.util.find_library("stdc++"))
        lib.__cxa_demangle.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.__cxa_demangle.restype = ctypes.c_char_p
        return lib
    except Exception:
        return None


_LIBSTDCXX = _load_demangler()


@lru_cache(maxsize=4096)
def demangle(name: str) -> str:
    """Return the demangled C++ symbol name, or *name* if it cannot be decoded."""
    if _LIBSTDCXX is None:
        return name
    status = ctypes.c_int()
    try:
        result = _LIBSTDCXX.__cxa_demangle(
            name.encode(), None, None, ctypes.byref(status)
        )
    except Exception:
        return name
    if status.value != 0 or result is None:
        return name
    return result.decode().replace(" >", ">")


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuActivity:
    """One GPU-side activity: a kernel launch, a memcpy, or a memset.

    Timestamps are integer nanoseconds in the *device activity clock domain*.
    Which domain that is depends on the source, and mixing domains is the
    single easiest way to produce silently wrong measurements -- see the
    contract notes in ``activity_sources.py``.
    """

    name: str
    start: int
    end: int
    correlation_id: int
    kind: ActivityKind = ActivityKind.KERNEL
    copy_kind: int = 0
    bytes: int = 0
    value: int = 0

    @property
    def duration(self) -> int:
        return self.end - self.start

    def identity(self) -> str:
        """Stable identity used to match activities across benchmark iterations.

        Two activities with the same identity are considered interchangeable
        instances of the same logical operation. Note this intentionally does
        NOT include timestamps or correlation id.
        """
        return f"{self.name}_{self.copy_kind}_{self.bytes}_{self.value}_{self.kind.value}"


def activity_sequence(activities: list[GpuActivity]) -> list[str]:
    return [a.identity() for a in activities]


def activity_counts(activities: list[GpuActivity]) -> Counter:
    return Counter(activity_sequence(activities))


def activity_span(activities: list[GpuActivity]) -> int:
    """Wall-clock span covered by *activities*: max(end) - min(start).

    This is the quantity reported as the iteration's runtime. Note it is a
    span, not a sum: gaps between activities ARE included. That is deliberate
    and is load-bearing for anti-reward-hacking -- work interleaved between two
    measured kernels still lands inside the span even if the interleaved
    activity itself is filtered out by name.
    """
    return max(a.end for a in activities) - min(a.start for a in activities)


def sort_activities(activities: list[GpuActivity]) -> list[GpuActivity]:
    """Canonical ordering. Activity buffers are NOT guaranteed to arrive sorted."""
    return sorted(activities, key=lambda a: (a.start, a.end, a.correlation_id))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def relative_order_score(candidate: list[str], expected: list[str]) -> int:
    """Longest-common-subsequence length between two identity lists.

    Used to prefer windows whose dispatch order best matches the order observed
    during discovery, when an exact contiguous match is unavailable.
    """
    scores = [0] * (len(expected) + 1)
    for candidate_name in candidate:
        previous_diagonal = 0
        for idx, expected_name in enumerate(expected, start=1):
            previous_score = scores[idx]
            if candidate_name == expected_name:
                scores[idx] = max(scores[idx], previous_diagonal + 1)
            else:
                scores[idx] = max(scores[idx], scores[idx - 1])
            previous_diagonal = previous_score
    return scores[-1]


class ActivitySequenceNotFound(ValueError):
    """Raised when an iteration's expected activity sequence cannot be located."""


def select_activity_sequence(
    activities: list[GpuActivity],
    expected: list[str],
    *,
    iteration: int = 0,
) -> list[GpuActivity]:
    """Pick the user's activity sequence out of a noisy timing window.

    Three-stage strategy, preserved exactly from the upstream implementation:

      1. Filter to activities whose identity appears in *expected*. Setup work,
         cache-flush memsets, and allocator traffic vanish here.
      2. Scan for the first contiguous window whose identities equal *expected*
         exactly. This is the overwhelmingly common case.
      3. Otherwise consider every window that is a multiset match, and pick the
         best by (longest-common-subsequence score, then shortest span).

    Raises ActivitySequenceNotFound if no window qualifies.
    """
    expected_count = len(expected)
    if expected_count == 0:
        raise ActivitySequenceNotFound("No expected activities recorded")

    expected_set = set(expected)
    expected_counts = Counter(expected)
    candidates = [a for a in activities if a.identity() in expected_set]
    candidate_names = activity_sequence(candidates)

    # Stage 2: exact contiguous match.
    for start in range(len(candidates) - expected_count + 1):
        end = start + expected_count
        if candidate_names[start:end] == expected:
            return candidates[start:end]

    # Stage 3: best multiset-matching window.
    best: list[GpuActivity] | None = None
    best_score: tuple[int, int] | None = None
    for start in range(len(candidates) - expected_count + 1):
        end = start + expected_count
        window_names = candidate_names[start:end]
        if Counter(window_names) != expected_counts:
            continue
        window = candidates[start:end]
        score = (relative_order_score(window_names, expected), -activity_span(window))
        if best_score is None or score > best_score:
            best_score = score
            best = window

    if best is not None:
        return best

    raise ActivitySequenceNotFound(
        f"Expected activity sequence not found at iteration {iteration}: "
        f"{expected} not in {candidate_names}"
    )


def measure_iterations(
    activities: list[GpuActivity],
    expected: list[str],
    windows: list[tuple[int, int]],
) -> list[float]:
    """Attribute a runtime in milliseconds to each timed iteration.

    *windows* are the (cpu_start, cpu_end) bracket timestamps recorded around
    each iteration on the host. Activities are bisected into each window, then
    the expected sequence is selected within it.

    This mirrors the loop in ``timing.py`` and is included so the whole
    attribution path -- not just selection -- is CPU-testable.
    """
    import bisect

    ordered = sort_activities(activities)
    starts = [a.start for a in ordered]
    out: list[float] = []
    for idx, (cpu_start, cpu_end) in enumerate(windows):
        left = bisect.bisect_left(starts, cpu_start)
        right = bisect.bisect_right(starts, cpu_end)
        window_activities = ordered[left:right]
        selected = select_activity_sequence(window_activities, expected, iteration=idx)
        if activity_counts(selected) != Counter(expected):
            raise ActivitySequenceNotFound(
                f"Activity count mismatch at iteration {idx}"
            )
        out.append(activity_span(selected) / 1e6)
    return out
