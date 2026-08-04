# SPDX-License-Identifier: Apache-2.0
"""A pool of GPUs that workers borrow and return.

Deliberately a queue rather than ``gpus[i % len(gpus)]``. Deviation D11 in
``STATE.md`` records what the modular form costs: assigning a GPU by task index
at *submission* time, while running ``len(gpus)`` workers, lets two tasks whose
indices are congruent mod ``len(gpus)`` be in flight at once. They then share a
GPU while another sits idle, each inflating the other's timings, and **nothing
in the output says so** -- the artifact records the device it was told to use,
which is the same either way. It cost an unknown subset of 176 artifacts.

Borrowing from a queue makes concurrency bounded by the pool itself, so the bug
is not merely fixed but unexpressible.
"""

from __future__ import annotations

import queue
from contextlib import contextmanager
from typing import Iterable, Iterator

# The GPU authoritative timing is pinned to. Task 01 measured a 65 MHz spread
# in achieved clock across the eight GPUs of the MI350X node -- larger than most
# of the optimization differences this benchmark exists to measure -- so timings
# from different GPUs are not interchangeable. Agents therefore never get GPU 0:
# their exploratory load would perturb whatever is being timed on it.
AUTHORITATIVE_GPU = 0


class GpuPool:
    """Hands out GPU indices, one holder at a time.

    Indices are *torch* device indices. They are passed to workers through
    ``HIP_VISIBLE_DEVICES``, which renumbers the visible device to 0, so a
    worker's own code always addresses ``cuda:0`` regardless of which physical
    GPU it received.
    """

    def __init__(self, gpus: Iterable[int]) -> None:
        gpus = list(gpus)
        if not gpus:
            raise ValueError("GpuPool needs at least one GPU")
        if AUTHORITATIVE_GPU in gpus:
            raise ValueError(
                f"GPU {AUTHORITATIVE_GPU} is reserved for authoritative timing and "
                f"must not be in an agent pool; got {gpus}. Scoring runs there, and "
                f"agent load on the same device would silently inflate every "
                f"latency it measures."
            )
        self._q: queue.Queue[int] = queue.Queue()
        for g in gpus:
            self._q.put(g)
        self.gpus = gpus

    @property
    def size(self) -> int:
        return len(self.gpus)

    @contextmanager
    def lease(self, timeout: float | None = None) -> Iterator[int]:
        """Borrow one GPU for the duration of the block."""
        gpu = self._q.get(timeout=timeout)
        try:
            yield gpu
        finally:
            self._q.put(gpu)


def default_agent_gpus(device_count: int) -> list[int]:
    """Every GPU except the authoritative one."""
    return [g for g in range(device_count) if g != AUTHORITATIVE_GPU]
