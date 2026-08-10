# CLAUDE.md — read this first

You are continuing a port of NVIDIA's **SOL-ExecBench** GPU-kernel benchmark to
**AMD Instinct CDNA4 / ROCm**.

**All ten tasks are `done` and manifest v1 is frozen** — 220 of 235 problems
scoreable, 3717 workloads, measured on **8× MI350X** (`gbt350-odcdh1-a08-1`),
F_LOCK 1300 MHz. Read `STATE.md` for the ledger and `TODO.md` for what is
actually open. The remaining work is v1.1 corrections, the MI355X part, and a
full-benchmark agent baseline — not the initial port.

The node is still the scarce resource. **Start GPU work first and do CPU work in
its shadow**; a session that spends its time on documentation while eight cards
sit at 0% has wasted the only thing that cannot be recovered.

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

A working `SOL-ExecBench-ROCm`: **all 235 problems** evaluate
end-to-end on the target part, scored against AMD-derived Speed-of-Light bounds,
with a published scoring manifest. Acceptance is defined per task in `tasks/`,
and overall in `tasks/09-release.md`.

Reached for MI350X at 220/235, with 15 NVFP4 deferred under the sanctioned
contingency below. **Not** reached for MI355X: nothing there has been measured.

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
2. Read TODO.md             -> the open gaps, and which are v1.1 blockers
3. Execute the work
4. Run the acceptance check -> paste real output
5. Update STATE.md          -> status, artifacts produced, anything surprising
```

All ten `tasks/NN-*.md` are `done`; they remain the specification of what each
acceptance check means, and `verify_artifacts.py --task NN` still runs. Exactly
**one check fails today** and it is known: task 03's `check D: no measurement
beats its T_SOL`. It reads manifest **v1**, the frozen release artifact, which
is meant to go on reporting what v1 shipped — it is not a live signal about the
board, which serves v1.2. Every other task is 0 failed.
**A second failure is a regression** — find out what you broke before doing
anything else.

Task 01 passes (11 checks, 0 failed) and has since commit 2cdb7b0. If you find
a document claiming otherwise, that document is stale; the gate is the
authority. Full matrix, re-run 2026-08-06:

| Task | Result |
|---|---|
| 00 | 13 checks, 0 failed, 1 judgement (dataset census) |
| 01 | 11 checks, 0 failed, 0 judgement (1 WARN: per-GPU floor spread) |
| 02 | 12 checks, 0 failed, 0 judgement |
| 03 | 13 checks, **1 failed**, 2 judgement |
| 04 | 5 checks, 0 failed, 1 judgement (divergence tails) |
| 05 | 10 checks, 0 failed, 1 judgement (>2× tolerances) |
| 06 | 10 checks, 0 failed, 1 judgement (1 WARN: D15 re-time band) |
| 07 | 4 checks, 0 failed, 0 judgement (1 WARN: no FP8 write-up) |
| 08 | 4 checks, 0 failed, 0 judgement |
| 09 | 9 checks, 0 failed, 0 judgement |

`STATE.md` is the single source of truth for progress and is the handoff between
sessions. Update it as you go, not at the end — a session can be interrupted.

Tasks are ordered by dependency, not by importance. `tasks/DEPENDENCIES.md` has
the graph and shows what can run in parallel.

## 4. This node

- **8× MI350X** (CDNA4, gfx950), 288 GB HBM3E each, 8 TB/s, **1000 W** per GPU,
  air-cooled. Host `gbt350-odcdh1-a08-1`. **F_LOCK = 1300 MHz** at
  `--setperfdeterminism 1600`.
- All eight GPUs are yours. Use them — the long sweeps (tasks 05, 06) shard
  8-way via `scripts/shard_sweep.py` and are the difference between a 3-day and
  a 12-hour turnaround.

Session 1 ran on an 8× MI355X node (1400 W, liquid). None of its measurements
transfer; see `docs/HANDOFF.md`.

### GPU discipline

**Timing runs and exploration must not share a GPU.** This is not negotiable and
task 01 did not soften it.

What task 01 *did* settle is whether they may share a **node**. Measured
sibling-GPU interference on this part: **−0.11%**, well inside run-to-run noise.
So sweeps on GPUs 1–7 may run concurrently with authoritative timing on GPU 0.

| GPU | Use |
|---|---|
| 0 | Authoritative timing only. Idle otherwise. |
| 1–7 | Compilation, correctness sweeps, exploration, sharded calibration |

The leaderboard's submission worker holds GPU 0 under a lock file and runs one
job at a time, for the same reason: two jobs on GPU 0 invalidates both timings
*and* every published number they would be compared against.

Pin with `HIP_VISIBLE_DEVICES`. Record which GPU produced every timing artifact.

## 5. What is already done (do not redo)

| Asset | Location | Status |
|---|---|---|
| Full engineering plan | `PLAN.md` | Reference. Six phases, risks, methodology. |
| Upstream audit | `reference/upstream-audit.md` | Every NVIDIA-specific call site, located. |
| Vendor-neutral timing attribution | `src/solexbench_rocm/activity/` | **CPU-verified, 19 tests, mutation-tested.** Do not rewrite. |
| rocprofiler shim | `src/solexbench_rocm/shim/`, `reference/contracts/rocprof_shim.md` | **Built and validated.** Median divergence −0.61% over 1430 pairs; clock domain verified. |
| SOLAR arch config generator | `src/solexbench_rocm/solar/gen_arch_yaml.py` | Reproduces AMD spec sheet at peak clock. |
| The harness port | `src/sol_execbench/`, deltas marked `# AMD:` | **Done.** 3717/3717 non-deferred workloads pass under AMD tolerances. |
| Sweep runners | `scripts/runners/` | All four written and run. |
| T_b candidate variants | `reference/tb-candidates/variants.py` | Source-to-source transforms; one set covers all 235. |
| Anti-reward-hack corpus | `reference/exploits/` | 28/28 replay cases pass, 0 false positives on 235 references. |
| Frozen scoring manifest | `artifacts/09/manifest-v1.json` | 220/235 problems, 3717 workloads. Do not edit; regenerate. |
| Manifest the board serves | `artifacts/09/manifest-v1.2.json` | v1.1 (D18 paged traffic + D35 per-datapath clock) then v1.2 (D37 grouped conv). Both re-derive on `device="meta"`; no measurement was repeated. |
| Leaderboard and submission service | `leaderboard/` | Board, typed `/api/v1`, submit queue, GPU-0 worker. |

The activity package is the one piece you should be most reluctant to touch. Its
selection logic is subtle, it is behaviour-preserving against upstream, and its
test suite was mutation-tested — seven distinct mutations are each caught. If you
believe it is wrong, write a failing test first.

## 5b. What is NOT done — read `TODO.md`

`TODO.md` lists every known gap. The ones that will bite you first:

* **Five T_SOL bounds are known wrong under v1.2**, all diagnosed (D42) and
  **none corrected**. Do not treat scores on those problems as results, and do
  not "fix" a score by adjusting a bound without re-deriving it. Three of the
  five are one defect — the declared-traffic tier prices every declared input at
  its full allocation regardless of what the kernel reads — which is D18, fixed
  in v1.1 for two paged problems rather than at the tier. **328 workloads across
  38 problems still rest on that tier.** Fixing the tier, not another problem,
  is the v1.3 item.
* **The bound check is one-sided and 827 workloads exploit that** (D39). Nothing
  may beat a bound; nothing checks that a bound is tight. 22.3% of workloads sit
  above 100× headroom, where `S` is a PyTorch comparison with no roofline
  content. They are marked (`bound_quality`) and not fixed.
* **`rocprofv3 --pmc` hangs in this container** (D43), so the counter path for
  an independent traffic measurement is closed. The shim is *not* implicated —
  it uses the dispatch-callback timestamp path and every measurement in the repo
  runs on it. The counter-free route is a minimal independent kernel, timed.
* **Some artifacts stamp `f_lock_mhz: null`** even though F_LOCK was measured.
  This is *not* a missing clock preset — `CLOCK_LOCK_PRESETS` has had an
  MI350X entry (`gpu_clk_mhz=1600`, `achieved_gpu_clk_mhz=1300`) since 2cdb7b0
  and task 01 passes. Two unrelated causes, neither of which invalidates a
  measurement; see TODO.md.
* **Nothing is measured on MI355X.** The port needs no work; every number does.
* **D20 is unexplained** — 0.13% of matmul iterations cost 3.9–4.5×. The clock
  hypothesis was tested and falsified. Two upstream tests are skipped behind it.

Nothing in this repo silently pretends to be finished; if something is missing
it is listed there.

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
- **`--setperfdeterminism X` does not give you X.** On this part it yields
  roughly `0.83·X`: setting 1600 achieves **1300 MHz**, and only the achieved
  value is F_LOCK. Whether that ratio holds on the 1400 W part was never asked.
- **A self-consistent bound and anchor cannot detect a shared error.** `T_b`
  comes from a PyTorch reference that over-reads exactly where the
  declared-traffic bound over-counts, so the `T_SOL <= T_b` gate passes while
  both are wrong. Only an independent kernel separates them — which is how all
  three known-bad bounds were found, by agents, after the manifest froze.
- **A run stopped by its budget is a cost measurement, not a score
  measurement.** `pilot8` is off the board for this reason: no session chose
  when to stop, so its mean is survivorship over whatever happened to finish.

## 7. Conventions

- Artifacts → `artifacts/<task-id>/`, always with a provenance header.
- Never commit measurements to `src/`. Code and data stay separate.
- Long sweeps: `nohup` + log to `artifacts/<task>/logs/`, and make them
  **resumable** — assume the session dies mid-sweep, because it will.
- Python: repo code targets 3.12 to match upstream's `requires-python`.
- Run `pytest tests/` before and after touching anything in `src/`. Expect
  **519 passed, 75 skipped** in the container, re-run 2026-08-10: 43 in
  `sol_execbench` (CUPTI-only tests, the two D20 variance tests), 12 in
  `examples` (NVIDIA-only solution languages), and one collection-time skip per
  leaderboard module — each needs fastapi, so that suite runs in
  `leaderboard/.venv` instead: `leaderboard/.venv/bin/python -m pytest
  tests/leaderboard`, all passing, **153 passed / 1 skipped** as of
  2026-08-10. Do not read either
  skip count as a gate: one leaderboard module added moves both by one, which
  is how the old figure of 56 here went stale. Only a drop in *passed*, or a
  skip whose reason is not one of the four above, is a regression.
- The leaderboard is a *view*. Never edit `leaderboard/solbench.db`; change the
  artifact and re-ingest. Runs that live outside the repo go in
  `leaderboard/sources.json` (untracked; see `sources.json.example`), which
  `ingest.py` reads by default — `--agent-runs` **overrides** that list rather
  than adding to it, so passing it is now the way to get the omission, not the
  way to avoid it. Either way the ingest refuses to publish a board that has
  lost a submission the current one has, unless `--allow-drop` says so.

## 8. If the dataset is missing

`data/` is gitignored and does not travel with the repo. The Hub ships parquet,
not the per-problem layout, so fetching is two steps — see the README's *Running
it*. `scripts/materialize_dataset.py` is the exact inverse of the dataset's own
converter and round-trip-verifies all 235 problems.

The census is confirmed against the files, not taken from the paper:
**L1 94, L2 82, Quant 33, FlashInfer-Bench 26**.

Then `scripts/fetch_flashinfer_traces.py`. Without those blobs, 9 of the 26
FlashInfer problems fail at run time as ordinary runtime errors — which looks
like a port defect and is not one.
