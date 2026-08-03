# CPU-testable GPU timing attribution

Pre-node work for the SOL-ExecBench → ROCm port. **Nothing here needs a GPU**,
a vendor runtime, or torch. `pytest -q test_gpu_activity.py` → 19 passed.

## What problem this solves

The benchmark's default timing methodology is not CUDA events — it is CUPTI
activity tracing. Per timed iteration it records GPU activities, then answers a
surprisingly subtle question: *which of these activities are the user's kernels
for this iteration?* Setup copies, the 512 MB cache-flush memset, allocator
traffic, and leftover warmup dispatches are all in the buffer too.

That selection logic is ~70 lines of pure string-and-integer reasoning. But it
lives in `cupti_utils.py`, which does `from cupti import cupti` at module
scope — so today it cannot even be *imported* without CUDA, let alone tested.
On the AMD port that means the trickiest code on the timing path would be
debugged interactively on the scarcest hardware.

## Layout

| File | Role |
|---|---|
| `gpu_activity.py` | Vendor-neutral `GpuActivity` record + the full selection/attribution logic, extracted behaviour-preserving from upstream. Zero GPU imports. |
| `activity_sources.py` | `ActivitySource` protocol. CUPTI adapter (NVIDIA), `RocprofActivitySource` (AMD, stub + **written contract**), `ReplayActivitySource` (CPU). |
| `trace_fixtures.py` | Deterministic synthetic trace builder modelling a real benchmark iteration, with jitter, noise, interleaving, and out-of-order buffer arrival. |
| `test_gpu_activity.py` | 19 CPU-only tests. |

The port's job reduces to: **supply a record source.** Everything downstream is
shared and already tested.

## The rocprofiler shim contract

`RocprofActivitySource`'s docstring specifies exactly what the C++ shim must
provide — four symbols, one tuple format — and documents five traps, of which
#1 is the expensive one:

> CUPTI's `get_timestamp()` shares a normalized domain with its activity
> stamps. rocprofiler-sdk timestamps come from the HSA clock. If `timestamp()`
> uses `clock_gettime(CLOCK_MONOTONIC)` instead of rocprofiler's own entry
> point, window bisection selects the wrong activities and **every measurement
> is wrong without raising.**

`verify_clock_domain()` turns that silent failure into an immediate loud one.
Run it once during bring-up.

## On test credibility

Passing tests prove nothing unless they can fail. The suite was mutation-tested;
each mutation below is caught:

| Mutation | Caught by |
|---|---|
| drop defensive sort of activity buffer | buffer-arrival-order test |
| drop identity filter (stop excluding setup/flush) | span test |
| neuter the `(LCS, -span)` tiebreak | stage-3 order tests |
| span → sum of durations (lose the gaps) | 3 tests |
| disable exact-match fast path | duplicate-sequence test |
| flip span tiebreak sign | LCS-tie test |
| drop multiset guard | reorder test |

The first pass had **two survivors** (the tiebreak and the fast path both
neutered, suite still green) — the stage-3 tests were added specifically to
close that gap.

## Two properties worth knowing

Asserted as tests so they are documented behaviour, not surprises:

- Work interleaved *between* measured kernels lands in the span even if its
  kernel name is unrecognized. Renaming a kernel does not hide its cost.
- Work starting *after* the final measured kernel escapes the span. Upstream
  mitigates via a post-synchronize host end-stamp plus thread/stream checks;
  the AMD port inherits both the mitigation and the residual gap.

## Next, still no GPU required

- Mock AMD device backend, so the full CLI → packager → subprocess → trace →
  score path integration-tests on CPU.
- Static op-coverage scan of all 235 references against known ROCm gaps
  (needs the HF dataset; it was unreachable from the sandbox this was built in).
- float64 CPU golden references — vendor-neutral ground truth that separates
  "AMD differs from NVIDIA" from "AMD differs from correct math" during
  tolerance calibration.
