# Contract — rocprofiler-sdk activity shim

Implemented in task 04. Read this before writing code.

The selection and attribution logic is **already written and CPU-verified** in
`src/solexbench_rocm/activity/`. This shim's only job is to emit records.

## Surface

A pybind11 or torch C++ extension named `_rocprof_shim` exposing four symbols:

```
start()      -> None    configure + start a buffered tracing session
stop()       -> None    stop and flush
drain()      -> list    activity tuples (below)
timestamp()  -> int     host timestamp, SAME clock domain as record stamps
```

`drain()` returns tuples of:

```
(kind: str, name: str, start_ns: int, end_ns: int,
 correlation_id: int, copy_kind: int, nbytes: int, value: int)
```

`kind` ∈ `{"KERNEL", "MEMCPY", "MEMSET"}`.

`RocprofActivitySource` in `activity_sources.py` already adapts this to
`GpuActivity`; if you match the tuple shape, nothing downstream changes.

---

## The five traps

### #1 — Clock domain. This is the expensive one.

CUPTI's `get_timestamp()` and its activity start/end stamps share a normalized
domain. **rocprofiler-sdk timestamps come from the HSA clock.**

`timestamp()` must call rocprofiler's own timestamp entry point — *not*
`clock_gettime(CLOCK_MONOTONIC)`, even though the two look interchangeable and
both produce plausible nanosecond values.

Symptom of getting it wrong: window bisection selects zero activities (every
iteration raises `ActivitySequenceNotFound`) or all of them (every iteration
reports the same implausible span). **It does not raise at the point of the
mistake.** In the worst case the offset is small enough that some iterations
still resolve, and you get a plausible-looking distribution that is wrong.

Guard, already written — call it during bring-up:

```python
from activity_sources import verify_clock_domain
verify_clock_domain(activities, host_windows)
```

It asserts every record's `[start, end]` falls inside the union of host windows,
turning a silent failure into an immediate loud one.

### #2 — Name resolution

Kernel-dispatch records carry a `kernel_id`, not a string. Resolve ids to symbol
names via the code-object tracing callback and cache the mapping — **ids are
only valid for the loaded code object**, so a cache that outlives a module
reload will hand back wrong names.

Names arrive Itanium-ABI mangled, exactly as on NVIDIA. Reuse `demangle()` from
`gpu_activity.py`; it is already shared for this reason.

### #3 — Buffer flush ordering

Records must be flushed before `drain()` returns, and `drain()` must be callable
after `stop()`.

Do not assume callback ordering. Activities may arrive out of timestamp order —
which is why the pure layer calls `sort_activities()` defensively, and why
there is a test asserting shuffled input produces identical results.

### #4 — Dispatch-level, not API-level

Trace the **kernel-dispatch** and **memory-copy** buffered categories, not the
HIP API trace.

API-level records include host launch overhead. Excluding that overhead is the
entire reason this methodology exists in preference to event timing; tracing the
API trace reintroduces it while looking like it worked.

### #5 — Memset representation

The benchmark's own LLC flush is a 512 MB memset, and it must stay
distinguishable so the identity filter can drop it.

If rocprofiler reports fills as a memory-copy variant rather than a distinct set
operation, map it to `MEMSET` and carry the fill value through, so
`identity()` — which is
`f"{name}_{copy_kind}_{bytes}_{value}_{kind}"` — stays discriminating.

---

## Validation order

Do these in order; each one makes the next debuggable.

1. `verify_clock_domain()` passes on a real capture.
2. Shim-sourced records pass the existing suite: `pytest tests/ -q`.
3. Freeze a real capture as a `ReplayActivitySource` fixture and check it into
   `tests/fixtures/`. A real trace that once broke selection should never be
   able to regress.
4. Cross-validate against `hip_events` on the full L1 set. **Gate: ≤2% median
   divergence**, with μs-scale kernels reported separately rather than folded
   into the median — they are expected to differ, because event timing includes
   launch slop that dispatch timing excludes. That expected difference is the
   whole point of building this.

## Do not

- Rewrite the selection logic. Seven distinct mutations of it are already caught
  by the test suite; a "fix" that leaves the suite green is probably not fixing
  anything. If you believe it is wrong, **write a failing test first.**
- Switch the default methodology mid-project without re-running. Anything
  measured before the switch is not comparable to anything after it.
