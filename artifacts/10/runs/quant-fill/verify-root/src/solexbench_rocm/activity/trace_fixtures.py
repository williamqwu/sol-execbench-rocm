# SPDX-License-Identifier: Apache-2.0
"""Synthetic GPU activity traces for testing the timing path without hardware.

Models what the benchmark loop actually produces per iteration:

    [ allocator copies ]  [ LLC flush memset ]  [ ... user kernels ... ]
    <------------- filtered out ------------->  <---- measured span ---->

plus the failure modes that matter: jitter, out-of-order buffer arrival,
foreign kernels interleaved with user work, repeated kernel identities, and
dispatch reordering.

Deterministic by construction -- a seeded Random, never the global one -- so
failures reproduce exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from gpu_activity import ActivityKind, GpuActivity


@dataclass
class TraceBuilder:
    """Builds a synthetic activity stream one iteration at a time."""

    clock: int = 1_000_000_000          # ns; arbitrary non-zero origin
    seed: int = 0
    activities: list[GpuActivity] = field(default_factory=list)
    windows: list[tuple[int, int]] = field(default_factory=list)
    _cid: int = 0

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _next_cid(self) -> int:
        self._cid += 1
        return self._cid

    def _emit(self, name, duration, kind=ActivityKind.KERNEL, gap=200, **kw):
        self.clock += gap
        act = GpuActivity(
            name=name,
            start=self.clock,
            end=self.clock + duration,
            correlation_id=self._next_cid(),
            kind=kind,
            **kw,
        )
        self.clock += duration
        self.activities.append(act)
        return act

    def iteration(
        self,
        kernels: list[tuple[str, int]],
        *,
        setup_copies: int = 2,
        flush: bool = True,
        jitter: int = 0,
        interleave: list[tuple[int, str, int]] | None = None,
        host_slack: int = 5_000,
    ) -> tuple[int, int]:
        """Emit one benchmark iteration; return its (cpu_start, cpu_end) window.

        kernels    -- [(name, duration_ns), ...] the user's measured sequence
        interleave -- [(after_index, name, duration_ns), ...] foreign activity
                      injected *between* user kernels (i.e. inside the span)
        """
        cpu_start = self.clock - host_slack

        # Setup: allocator staging copies. Filtered out by identity.
        for _ in range(setup_copies):
            self._emit("MEMCPY", 800, kind=ActivityKind.MEMCPY, copy_kind=1, bytes=4096)

        # Cache flush: the 2x-LLC zeroing memset. Also filtered out.
        if flush:
            self._emit("MEMSET", 12_000, kind=ActivityKind.MEMSET, bytes=512 << 20, value=0)

        interleave = interleave or []
        by_index: dict[int, list[tuple[str, int]]] = {}
        for idx, name, dur in interleave:
            by_index.setdefault(idx, []).append((name, dur))

        for i, (name, dur) in enumerate(kernels):
            if jitter:
                dur += self._rng.randint(-jitter, jitter)
            self._emit(name, dur)
            for foreign_name, foreign_dur in by_index.get(i, []):
                self._emit(foreign_name, foreign_dur)

        cpu_end = self.clock + host_slack
        self.clock = cpu_end
        self.windows.append((cpu_start, cpu_end))
        return cpu_start, cpu_end

    def shuffled(self) -> list[GpuActivity]:
        """Activities in buffer-arrival order, i.e. NOT sorted by timestamp.

        Real activity buffers make no ordering guarantee; the pure layer sorts
        defensively. This exercises that.
        """
        out = list(self.activities)
        random.Random(self.seed + 1).shuffle(out)
        return out


def expected_identities(kernels: list[tuple[str, int]]) -> list[str]:
    """Identity list for a user kernel sequence, as discovery would produce it."""
    return [
        GpuActivity(name=n, start=0, end=0, correlation_id=0).identity()
        for n, _ in kernels
    ]
