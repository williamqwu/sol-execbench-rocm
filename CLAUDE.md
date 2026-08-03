# CLAUDE.md — read this first

You are continuing a port of NVIDIA's **SOL-ExecBench** GPU-kernel benchmark to
**AMD Instinct MI355X / ROCm**. Substantial preparatory work is already done and
verified; your job is the part that requires real silicon.

You are running on a **single 8×MI355X node**. That node is the scarce resource.
Everything that could be done without it already has been.

---

## 0. Scope — all 235 problems

The benchmark is **235 problems**: L1 (94), L2 (82), Quant (33),
FlashInfer-Bench (26). **All of them are in scope.** This is a full-parity port,
not a subset.

Note the naming, because it is easy to conflate: the ten items in `tasks/` are
*engineering work items* for the port (calibrate clocks, port the harness,
derive bounds). They are not benchmark problems. Every one of those ten tasks
operates over the whole 235-problem set.

The realistic way scope shrinks is not a decision — it is an omission: a
`--category` flag missing an entry, a sweep that died partway and got marked
done, a crashed worker never retried. Each looks like success.

Guard against it: **`python scripts/check_coverage.py --artifacts <dir>` after
every sweep.** It names every problem not accounted for and exits non-zero. A
gap recorded in `artifacts/deferred.json` with a reason is a decision; a gap
without one is a bug.

The one sanctioned reduction is task 07's MXFP4 contingency (ship 220, defer 15
to v1.1) — and it is only sanctioned because it is explicit, reasoned, and
propagated to every artifact that states a count.

## 1. What "done" means

A working `SOL-ExecBench-AMD`: **all 235 problems** evaluate
end-to-end on MI355X, scored against AMD-derived Speed-of-Light bounds, with a
published scoring manifest. Acceptance is defined per task in `tasks/`, and
overall in `tasks/09-release.md`.

## 2. Prime directives

These are not style preferences. Violating any one of them silently destroys the
benchmark's validity, and the damage is not visible in the output.

1. **Never invent a measurement.** If a run fails, times out, or produces a
   suspicious number, record the failure in `STATE.md` and move on. A plausible
   number that was not actually measured is worse than a missing one, because
   nothing downstream can detect it.
2. **Never copy an NVIDIA constant into an AMD artifact.** Not B200 tolerances,
   not B200 SOL times, not the 1500 MHz clock, not `sm_100a` flags. Every one of
   these must be re-derived on AMD. This is the most tempting shortcut in the
   whole project and the most destructive.
3. **F_LOCK is measured, never guessed.** Every T_SOL and every T_b depends on
   it. `tasks/01-clock-calibration.md` is a hard blocker for tasks 03, 05, 06.
4. **A task is complete only when its acceptance check passes.** Each task file
   names a command. Run it. Paste the real output into `STATE.md`.
5. **Stamp provenance on every artifact.** ROCm version, driver version, torch
   version, F_LOCK, git SHA, UTC timestamp. `scripts/provenance.py` does this;
   use it. An artifact without provenance is not usable for scoring.
6. **Do not "fix" a pinned dependency to make something work.** If the pinned
   ROCm/torch combination is incompatible with something, record the
   incompatibility in `STATE.md` and raise it. Silently upgrading torch changes
   every measured baseline.
7. **When blocked, do not improvise a methodology change.** Write the blocker to
   `STATE.md`, then switch to a task that is not blocked. Changing how something
   is measured, to get past an obstacle, invalidates comparisons with everything
   measured before it.
8. **Surface uncertainty.** If a result looks wrong, say so in `STATE.md`, even
   if you cannot explain it. Especially then.

## 3. How to work

```
1. Read STATE.md            -> current progress, blockers, what is next
2. Read tasks/NN-*.md       -> the next task not marked done
3. Check its preconditions  -> if unmet, go to the task that satisfies them
4. Execute the steps
5. Run the acceptance check -> paste real output
6. Update STATE.md          -> status, artifacts produced, anything surprising
```

`STATE.md` is the single source of truth for progress and is the handoff between
sessions. Update it as you go, not at the end — a session can be interrupted.

Tasks are ordered by dependency, not by importance. `tasks/DEPENDENCIES.md` has
the graph and shows what can run in parallel.

## 4. This node

- **8× MI355X** (CDNA4, gfx950), 288 GB HBM3E each, 8 TB/s, 1400 W per GPU.
- All eight GPUs are yours. Use them — the long sweeps (tasks 05, 06) shard
  8-way via `scripts/shard_sweep.py` and are the difference between a 3-day and
  a 12-hour turnaround.

### GPU discipline

**Timing runs and exploration must not share a GPU.** Beyond that, whether they
can share a *node* is an open question this hardware will answer:

> At 8×1400 W there may be power or thermal coupling between GPUs, meaning a
> loaded GPU 3 perturbs a timing run on GPU 0. **`tasks/01` measures this
> explicitly.** Until it has, treat concurrent load as unsafe for authoritative
> timing.

Working rule until task 01 reports otherwise:

| GPU | Use |
|---|---|
| 0 | Authoritative timing only. Idle otherwise. |
| 1–7 | Compilation, correctness sweeps, exploration, sharded calibration |

Pin with `HIP_VISIBLE_DEVICES`. Record which GPU produced every timing artifact.

## 5. What is already done (do not redo)

| Asset | Location | Status |
|---|---|---|
| Full engineering plan | `PLAN.md` | Reference. Six phases, risks, methodology. |
| Upstream audit | `reference/upstream-audit.md` | Every NVIDIA-specific call site, located. |
| Vendor-neutral timing attribution | `src/solexbench_rocm/activity/` | **CPU-verified, 19 tests, mutation-tested.** Do not rewrite. |
| rocprofiler shim contract | `reference/contracts/rocprof_shim.md` | Spec for the C++ shim. Task 04 implements it. |
| SOLAR arch config generator | `src/solexbench_rocm/solar/gen_arch_yaml.py` | Reproduces AMD spec sheet at peak clock. Needs F_LOCK. |

The activity package is the one piece you should be most reluctant to touch. Its
selection logic is subtle, it is behaviour-preserving against upstream, and its
test suite was mutation-tested — seven distinct mutations are each caught. If you
believe it is wrong, write a failing test first.

## 5b. What is NOT built — read `TODO.md`

`TODO.md` lists every known gap: scripts written but never executed on hardware
(expect small first-contact fixes), the four sweep runners, the port itself, and
five assumptions carried from research that no hardware has confirmed. Nothing
in this repo silently pretends to be finished; if something is missing it is
listed there.

If the node is not ready yet, `TODO.md` also lists what is still doable without
a GPU — the T_b candidate variants being the largest remaining prep win.

## 6. Key facts worth not rediscovering

- **SOL bounds need no GPU.** SOLAR runs on `device="meta"`. Its arch config is
  13 numbers. And because `MAC_per_cycle` is architectural while bandwidth is
  fixed in bytes/second, **T_SOL in cycles is invariant to F_LOCK** — compute it
  once, convert to milliseconds by one division when F_LOCK lands.
- **SOLAR needs no code changes for MXFP4.** It resolves precision by literal
  string lookup (`f"MAC_per_cycle_{precision}_tc"`) and maps
  `float4_e2m1fn_x2 → "nvfp4"`. The generated AMD config emits an `nvfp4` alias
  pointing at the MXFP4 rate.
- **The clock-domain trap.** rocprofiler-sdk timestamps come from the HSA clock,
  not `CLOCK_MONOTONIC`. Getting this wrong makes every measurement wrong
  *without raising*. `verify_clock_domain()` exists to catch it. See
  `reference/contracts/rocprof_shim.md`.
- **CDNA4 uses OCP FP8** (`e4m3fn`/`e5m2`) — same as B200, so the 18 FP8 Quant
  problems port directly. It is CDNA3/MI300X that has the `fnuz` mismatch.
- **NVFP4 ≠ MXFP4.** Block 16 vs 32, FP8-E4M3 scales vs E8M0. The 15 NVFP4
  problems need re-specification, not translation. See `tasks/07`.

## 7. Conventions

- Artifacts → `artifacts/<task-id>/`, always with a provenance header.
- Never commit measurements to `src/`. Code and data stay separate.
- Long sweeps: `nohup` + log to `artifacts/<task>/logs/`, and make them
  **resumable** — assume the session dies mid-sweep, because it will.
- Python: repo code targets 3.12 to match upstream's `requires-python`.
- Run `pytest tests/` before and after touching anything in `src/`.

## 8. If the dataset is missing

`data/` is not in the repo. Fetch with
`huggingface-cli download nvidia/SOL-ExecBench --repo-type dataset --local-dir data/`.
This was never verified from the build environment (network-restricted), so if
it is gated or the layout differs from `reference/upstream-audit.md`, record what
you actually find in `STATE.md` before adapting anything to it.
