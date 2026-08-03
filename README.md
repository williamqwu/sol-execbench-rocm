# SOL-ExecBench-AMD

Port of NVIDIA's [SOL-ExecBench](https://github.com/nvidia/sol-execbench) —
which scores GPU kernels by proximity to an analytically derived hardware
Speed-of-Light bound rather than by speedup over a software baseline — to
**AMD Instinct MI355X / ROCm**.

**Status: preparatory work complete, on-node work pending.** Everything that
could be built and verified without MI355X hardware has been. What remains
needs silicon.

## If you are a Claude Code session on the node

Read [`CLAUDE.md`](CLAUDE.md) first, then [`STATE.md`](STATE.md), then the
lowest-numbered task in [`tasks/`](tasks/) that is not done.

```bash
pytest tests/ -q                              # should be green before you start
bash scripts/node_acceptance.sh               # task 00
python scripts/verify_artifacts.py --task 00  # acceptance gate
```

## Layout

```
CLAUDE.md                  agent contract: prime directives, workflow, node discipline
STATE.md                   progress ledger — the handoff between sessions
PLAN.md                    full engineering plan: 6 phases, risks, methodology
tasks/                     10 ordered tasks, each with acceptance criteria
  DEPENDENCIES.md          the graph, and how to schedule around the long sweeps
scripts/                   node acceptance, clock calibration, sharding, verification
src/solexbench_rocm/
  activity/                CPU-verified timing attribution (19 tests, mutation-tested)
  solar/                   SOLAR arch-config generator for MI355X
reference/
  upstream-audit.md        every NVIDIA-specific call site, located
  contracts/               specs for what remains to be built
tests/                     CPU-only; no GPU required
artifacts/                 measurements land here, always with provenance
```

## What is already done

**Vendor-neutral timing attribution** (`src/solexbench_rocm/activity/`). Upstream's
default timing methodology is CUPTI activity tracing, not CUDA events — per
iteration it must decide which recorded GPU activities are the user's kernels,
as against setup copies, the 512 MB cache-flush memset, and leftover warmup
dispatches. That logic is ~70 lines of pure reasoning trapped in a module that
cannot be imported without CUDA. It is now extracted, behaviour-preserving, with
19 tests. The suite was mutation-tested: seven distinct mutations are each
caught, and the first pass had two survivors that prompted three more tests.

**SOLAR arch-config generator** (`src/solexbench_rocm/solar/`). Reproduces AMD's
published MI355X figures exactly at peak clock (2.517 PFLOPS BF16 vs 2.5
published, 10.066 MXFP4 vs 10.1). A generator rather than a static YAML because
`MAC_per_cycle` is architectural while bandwidth terms must be rescaled when the
locked clock changes — hand-editing `freq_GHz` silently corrupts the roofline
balance point.

**rocprofiler shim contract** (`reference/contracts/`). Four symbols, one tuple
format, five documented traps.

**Upstream audit** (`reference/`). Every NVIDIA-specific call site.

## Three things worth knowing before starting

**SOL bounds need no GPU.** SOLAR runs on `device="meta"` and its arch config is
13 numbers. Because `MAC_per_cycle` is frequency-independent and bandwidth is
fixed in bytes/second, **T_SOL in cycles is invariant to the locked clock** —
compute it once, convert to milliseconds by one division when F_LOCK lands.

**The clock-domain trap.** rocprofiler-sdk timestamps come from the HSA clock,
not `CLOCK_MONOTONIC`. Getting this wrong makes every measurement wrong *without
raising*. `verify_clock_domain()` exists to catch it.

**Scores are within-platform.** An AMD SOL score and an NVIDIA SOL score each
measure the fraction of hardware headroom reclaimed on their own platform. They
are comparable in spirit but are **not** a cross-vendor performance comparison,
because analytic peaks are reachable to different degrees on different
microarchitectures. Measured ceilings are published alongside the analytic peaks
so readers can see the difference for themselves.

## Target

**MI355X** (CDNA4, gfx950) — the correct B200 peer: identical 8 TB/s bandwidth,
comparable dense-matrix throughput, and native OCP FP8 + MXFP4, which keeps the
Quant category portable rather than droppable. MI300X would have needed `fnuz`
FP8 remapping and has no FP4 datapath.

## Licence

Apache-2.0, matching upstream `sol-execbench` and `NVlabs/SOLAR`.
