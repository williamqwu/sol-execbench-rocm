# Task 04 — rocprofiler-sdk shim

**Goal:** kernel-level timing that excludes host launch overhead, reaching
parity with upstream's CUPTI methodology.

Upstream's default is **not** CUDA events — it is CUPTI activity tracing, which
attributes each iteration's runtime from device-side kernel activity rather than
host-side event pairs. Task 02 shipped on HIP events, which is correct but
includes launch/sync slop on very short kernels (μs-scale problems read slightly
slow, so their SOL scores read slightly low).

## Preconditions

- Task 02 done.

## What already exists

**The hard part is done and CPU-verified.** All selection and attribution logic
lives in `src/solexbench_rocm/activity/`, is behaviour-preserving against
upstream, and is covered by 19 mutation-tested tests. Your job is *only* to
supply a record source.

Read `reference/contracts/rocprof_shim.md` in full before writing code. It
specifies four symbols, one tuple format, and five traps.

## Steps

1. **Implement the shim** — pybind11 or a small torch extension over
   rocprofiler-sdk (the library behind `rocprofv3`). Configure buffered tracing
   for **kernel dispatch** and **memory copy** categories.

2. **Trap #1 first, before anything else works.** rocprofiler-sdk timestamps
   come from the HSA clock, not `CLOCK_MONOTONIC`. `timestamp()` must call
   rocprofiler's own entry point. Mixing domains makes every measurement wrong
   *without raising* — bisection just selects the wrong activities.

   Wire in `verify_clock_domain()` from `activity_sources.py` during bring-up.
   It converts that silent failure into an immediate loud one.

3. **Trap #2: name resolution.** Dispatch records carry a `kernel_id`, not a
   string. Resolve via the code-object callback and cache; ids are only valid
   for the loaded code object. Names arrive Itanium-mangled exactly as on
   NVIDIA — reuse `demangle()`.

4. **Trap #4: dispatch-level, not API-level.** Tracing the HIP API trace instead
   of the dispatch category reintroduces the host launch overhead this whole
   methodology exists to exclude.

5. **Validate against replay.** Feed the shim's output through
   `ReplayActivitySource` and the existing tests before trusting it on real
   problems.

6. **Cross-validate the two methodologies** on the full L1 set:
   ```bash
   python scripts/shard_sweep.py --task methodology-compare \
       --category L1 --gpus 1-7 --out artifacts/04/compare/
   # L1 only is DELIBERATE here: this compares two timing methodologies against
   # each other, not the problem set. L1's 94 single-op kernels span the
   # duration range that matters. Not a scope reduction.
   ```

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 04
```

Passes when: `verify_clock_domain()` passes on real captures; existing tests
pass against shim-sourced records; **median divergence vs `hip_events` ≤ 2%** on
L1, with μs-scale kernels reported separately rather than folded into the
median; every trace records which methodology produced it.

## Guard rails

- **Do not rewrite the selection logic.** If you think it is wrong, write a
  failing test against it first. Seven distinct mutations of that code are
  already caught by the suite; a "fix" that keeps the suite green is probably
  not fixing anything.
- Do not silently switch the default methodology mid-project. Anything measured
  before the switch is not comparable to anything after. If you switch, re-run.
- A large events-vs-rocprof divergence is a **finding**, not an inconvenience.
  Record it before adjusting anything.

## Outputs

- The shim + build integration
- `artifacts/04/compare/` — per-problem methodology divergence
- `artifacts/04/clock-domain-verification.log`
