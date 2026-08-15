# SPDX-License-Identifier: Apache-2.0
"""Activity sources: the one thing a vendor backend must supply.

An ``ActivitySource`` is a context manager that, while active, records GPU
activities, and afterwards hands back a list of ``GpuActivity``. Everything
downstream -- window bisection, sequence selection, span attribution -- is
vendor-neutral and lives in ``gpu_activity.py``.

Three implementations:

  * ``CuptiActivitySource``  -- NVIDIA, wraps the existing cupti-python path.
  * ``RocprofActivitySource`` -- AMD, wraps a small C++ shim over
                                 rocprofiler-sdk. Stubbed here; the CONTRACT it
                                 must satisfy is specified in detail below,
                                 because getting that contract wrong is the
                                 most likely source of silently-wrong timings.
  * ``ReplayActivitySource``  -- CPU-only, replays a canned or synthetic trace.
                                 This is what makes the port testable without
                                 hardware, and doubles as a way to freeze real
                                 captured traces as regression fixtures.

Only the last one is exercised in this sandbox. The first two import their
vendor runtimes lazily, inside ``__enter__``, so that merely importing this
module never requires a GPU.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gpu_activity import ActivityKind, GpuActivity, demangle


@runtime_checkable
class ActivitySource(Protocol):
    """Records GPU activities over a scope and returns normalized records."""

    def __enter__(self) -> "ActivitySource": ...
    def __exit__(self, *exc) -> bool | None: ...

    def drain(self) -> list[GpuActivity]:
        """Return everything recorded since ``__enter__``. Called after exit."""
        ...

    def timestamp(self) -> int:
        """Return a host-side timestamp in the SAME clock domain as record
        timestamps, in nanoseconds.

        This is used to bracket each timed iteration. If this domain does not
        match the domain of the ``start``/``end`` fields on emitted records,
        window bisection silently selects the wrong activities and every
        measurement is wrong -- usually without raising. See ROCM CONTRACT #1.
        """
        ...


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------

class CuptiActivitySource:
    """NVIDIA CUPTI-backed source. Behaviourally identical to upstream."""

    def __init__(self, buffer_bytes: int = 8 * 1024 * 1024):
        self._buffer_bytes = buffer_bytes
        self._records: list[GpuActivity] = []
        self._enabled: list = []

    def __enter__(self):
        from cupti import cupti  # lazy: import only when actually used

        self._cupti = cupti
        kinds = (
            cupti.ActivityKind.CONCURRENT_KERNEL,
            cupti.ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET,
        )
        self._kind_map = {
            cupti.ActivityKind.CONCURRENT_KERNEL: ActivityKind.KERNEL,
            cupti.ActivityKind.MEMCPY: ActivityKind.MEMCPY,
            cupti.ActivityKind.MEMSET: ActivityKind.MEMSET,
        }
        for kind in kinds:
            cupti.activity_enable(kind)
            self._enabled.append(kind)
        cupti.activity_register_callbacks(
            lambda: (self._buffer_bytes, 0), self._on_records
        )
        return self

    def _on_records(self, activities) -> None:
        for a in activities:
            kind = self._kind_map.get(a.kind)
            if kind is None:
                continue
            self._records.append(
                GpuActivity(
                    name=demangle(a.name) if kind is ActivityKind.KERNEL else kind.value,
                    start=a.start,
                    end=a.end,
                    correlation_id=a.correlation_id,
                    kind=kind,
                    copy_kind=getattr(a, "copy_kind", 0) if kind is ActivityKind.MEMCPY else 0,
                    bytes=getattr(a, "bytes", 0) if kind is not ActivityKind.KERNEL else 0,
                    value=getattr(a, "value", 0) if kind is ActivityKind.MEMSET else 0,
                )
            )

    def __exit__(self, *exc):
        cupti = self._cupti
        try:
            cupti.activity_flush_all(0)
        finally:
            for kind in self._enabled:
                cupti.activity_disable(kind)
            cupti.finalize()
        return False

    def drain(self) -> list[GpuActivity]:
        return list(self._records)

    def timestamp(self) -> int:
        return self._cupti.get_timestamp()


# ---------------------------------------------------------------------------
# AMD
# ---------------------------------------------------------------------------

class RocprofActivitySource:
    """AMD rocprofiler-sdk-backed source. Requires the ``_rocprof_shim`` module.

    =====================================================================
    CONTRACT the C++ shim must satisfy  (spec for whoever writes it)
    =====================================================================

    The shim is a small pybind11/torch extension exposing four symbols:

        start()      -> None   configure + start a buffered tracing session
        stop()       -> None   stop and flush the session
        drain()      -> list   list of tuples, one per activity (below)
        timestamp()  -> int    host timestamp, SAME domain as record stamps

    ``drain()`` tuples are:
        (kind:str, name:str, start_ns:int, end_ns:int,
         correlation_id:int, copy_kind:int, nbytes:int, value:int)
    with kind in {"KERNEL", "MEMCPY", "MEMSET"}.

    ---------------------------------------------------------------------
    #1  CLOCK DOMAIN -- the highest-risk item.
        CUPTI's get_timestamp() and its activity start/end stamps share a
        normalized domain. rocprofiler-sdk timestamps come from the HSA clock.
        ``timestamp()`` MUST therefore call rocprofiler's own timestamp entry
        point, not clock_gettime(CLOCK_MONOTONIC), even though the two look
        interchangeable and will produce plausible-looking numbers if confused.
        Symptom of getting this wrong: bisection selects zero or all
        activities, so iterations either raise ActivitySequenceNotFound or all
        report the same implausible span.
        CPU-side guard for this: assert every record's [start,end] falls inside
        the union of the host windows. Cheap, catches domain mismatch at once.

    #2  NAME RESOLUTION.
        Kernel-dispatch records carry a kernel_id, not a string. The shim must
        resolve ids to symbol names via the code-object callback and cache the
        mapping, because ids are only valid for the loaded code object. Names
        arrive Itanium-mangled, exactly as on NVIDIA -- reuse demangle().

    #3  BUFFER FLUSH ORDERING.
        Records must be flushed before drain() returns, and drain() must be
        callable after stop(). Do not assume callback ordering: activities may
        arrive out of timestamp order, which is why the pure layer sorts
        defensively in sort_activities().

    #4  DISPATCH-LEVEL, NOT API-LEVEL.
        Trace the kernel-dispatch and memory-copy buffered categories, NOT the
        HIP API trace. API-level records include host launch overhead, which is
        precisely what this timing methodology exists to exclude.

    #5  MEMSET REPRESENTATION.
        The benchmark's own L2/LLC flush is a memset, and it must remain
        distinguishable so it can be filtered. If rocprofiler reports fills as
        a memory-copy variant rather than a distinct set operation, map it to
        MEMSET and carry the fill value so identity() stays discriminating.
    =====================================================================
    """

    _KINDS = {"KERNEL": ActivityKind.KERNEL,
              "MEMCPY": ActivityKind.MEMCPY,
              "MEMSET": ActivityKind.MEMSET}

    def __enter__(self):
        import _rocprof_shim  # lazy

        self._shim = _rocprof_shim
        self._shim.start()
        return self

    def __exit__(self, *exc):
        self._shim.stop()
        return False

    def drain(self) -> list[GpuActivity]:
        out = []
        for kind, name, start, end, cid, copy_kind, nbytes, value in self._shim.drain():
            k = self._KINDS[kind]
            out.append(
                GpuActivity(
                    name=demangle(name) if k is ActivityKind.KERNEL else k.value,
                    start=start,
                    end=end,
                    correlation_id=cid,
                    kind=k,
                    copy_kind=copy_kind,
                    bytes=nbytes,
                    value=value,
                )
            )
        return out

    def timestamp(self) -> int:
        return self._shim.timestamp()


# ---------------------------------------------------------------------------
# CPU-only
# ---------------------------------------------------------------------------

class ReplayActivitySource:
    """Replays a fixed activity list. No GPU, no vendor runtime.

    Two uses:
      * synthetic traces in tests (see trace_fixtures.py)
      * frozen captures from real runs, checked in as regression fixtures, so
        a real-world trace that once broke selection can never regress
    """

    def __init__(self, activities: list[GpuActivity], clock_origin: int = 0):
        self._activities = list(activities)
        self._clock = clock_origin

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def drain(self) -> list[GpuActivity]:
        return list(self._activities)

    def timestamp(self) -> int:
        return self._clock


def verify_clock_domain(
    activities: list[GpuActivity], windows: list[tuple[int, int]]
) -> None:
    """Guard for ROCM CONTRACT #1: record stamps must live inside host windows.

    Call this once during bring-up on real hardware. It turns a silent
    wrong-domain bug into a loud, immediate failure.
    """
    if not activities or not windows:
        return
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    stray = [a for a in activities if a.start < lo or a.end > hi]
    if stray:
        raise RuntimeError(
            f"Clock-domain mismatch: {len(stray)}/{len(activities)} activities fall "
            f"outside the host window [{lo}, {hi}]. First: {stray[0]!r}. "
            "The source's timestamp() is probably not in the same domain as its "
            "record timestamps -- see ROCM CONTRACT #1."
        )
