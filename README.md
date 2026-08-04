# SOL-ExecBench-AMD

Port of NVIDIA's [SOL-ExecBench](https://github.com/nvidia/sol-execbench) — which
scores GPU kernels by proximity to an analytically derived hardware
Speed-of-Light bound rather than by speedup over a software baseline — to
**AMD Instinct CDNA4 / ROCm**.

Supports **MI350X** and **MI355X**. They are the same gfx950 die in different
chassis (air-cooled 1000 W at 2.2 GHz vs liquid-cooled 1400 W at 2.4 GHz), so
they share the build path and the architectural constants and share nothing
that was measured. Every measured quantity is keyed by part.

**Status:** fully measured on 8× MI350X — every number below came off this
hardware. MI355X has a placeholder clock lock and no measurements; see
[MI355X](#running-on-mi355x). See [`STATE.md`](STATE.md) for the live ledger
and [`docs/methodology.md`](docs/methodology.md) for how each number was
derived.

## The one thing to get right before comparing anything

> An AMD SOL score and an NVIDIA SOL score are each **within-platform** measures
> of the fraction of hardware headroom reclaimed. They are comparable in spirit
> but are **not** a cross-vendor performance comparison, because analytic peaks
> are reachable to different degrees on different microarchitectures.

On this hardware, BF16 GEMM reaches 50.6% of its analytic peak
(1168 of 2307 TFLOPS) and HBM copy 56.7% of 8 TB/s (4.53 TB/s). Two kernels at
`S = 0.8` on different vendors have each reclaimed 80% of the distance from
their own platform's anchor to their own platform's bound. They have not been
shown to be equally fast. Measured ceilings are published beside the analytic
peaks (`artifacts/00/roofline-gpu0.json`) so the difference is visible rather
than inferred.

## Scoring

```
S(T_k) = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))
```

`S = 1` at the Speed-of-Light bound; `S = 0.5` at the optimized-PyTorch anchor.
All three inputs — `T_SOL`, `T_b`, and the correctness tolerances — are
re-derived on AMD. None is copied from B200: a copied tolerance either fails
correct kernels or rewards wrong ones, and a copied bound rescales every score
by a constant nobody can see.

Scores are valid only *within* a manifest version
(`artifacts/09/manifest-v1.json`). Any stack change that moves `T_b` requires a
new version.

## What is in the manifest

| | count |
|---|---|
| problems in the dataset | 235 |
| **scoreable** (T_SOL + T_b + AMD tolerance) | **220** |
| deferred, with reason | 15 (all NVFP4 — see [Deferrals](#deferrals)) |
| scoreable workload instances | 3717 |

Every scoreable workload carries a `T_SOL` from one of two derivations, and
**says which**:

| `t_sol_source` | what it is |
|---|---|
| `solar_fused` | SOLAR's roofline over the traced graph — accounts for the arithmetic |
| `declared_traffic` | every declared input read once and output written once, over DRAM bandwidth — accounts for all the traffic and no arithmetic |
| `max_of_both` | both were valid; the larger lower bound wins |

Neither derivation dominates, which is why both are kept and neither is
presented anonymously. Any candidate bound that lands **above** the measured
`T_b` is rejected outright — a lower bound above a measured time is not loose,
it is wrong, and it would push scores past 1. 63 SOLAR bounds and 23 traffic
bounds were rejected that way; no workload lost both.

## Is the scale real?

`scripts/verify_anchor.py` re-times, on hardware, over a 20-problem sample
(`artifacts/06/anchor-verification.json`):

| property | result |
|---|---|
| no measured time below its own `T_SOL` | **349/349** |
| the plain reference never scores above the anchor | **349/349** |
| re-timing `T_b`'s own implementation scores 0.5 ± 0.03 | **336/349** |

The 13 that miss are 12 workloads of one problem
(`FlashInfer-Bench/018_mla_paged_decode`, which re-times a median 1.16× slower
than its recorded `T_b`) and one workload at the tolerance edge. That problem's
anchor is optimistic by ~16%, which depresses its scores rather than inflating
them; it is written up in [`STATE.md`](STATE.md) D15 rather than smoothed away.

**No agent baseline was run** (`artifacts/09/agent-baseline.json` records the
decision). Upstream's median SOL of 0.732 and correlation of r = 0.981 are
results about *agents*, and the four PyTorch formulations here cluster around
`T_b` by construction — `T_b` is defined as the fastest of them. What the
variant set does establish is that the scale is well-formed: every score finite
and in (0, 1], `S = 0.5` at `T_b` to machine precision, and a within-workload
correlation between `S` and headroom reclaimed of **r = 1.000** (median over
2518 workloads).

## Running it

```bash
# 1. Build the pinned measurement container (ROCm 7.2 / torch 2.9.1 / SOLAR)
env/solb bash -lc 'python -c "import torch; print(torch.__version__)"'

# 2. Materialize the dataset. The Hub ships parquet, not the per-problem
#    layout; this is the exact inverse of the dataset's own converter and
#    round-trip-verifies all 235 problems.
env/solb bash -lc '
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(\"nvidia/SOL-ExecBench\", repo_type=\"dataset\",
                  local_dir=\"/var/tmp/solbench/data-hf\")"
  python scripts/materialize_dataset.py \
      --parquet-dir /var/tmp/solbench/data-hf/data \
      --out data/SOL-ExecBench/benchmark'

# 3. Fetch the external FlashInfer blobs. WITHOUT these, 9 of the 26
#    FlashInfer-Bench problems fail at run time as ordinary runtime errors.
env/solb bash -lc 'python scripts/fetch_flashinfer_traces.py'

# 4. Sanity
env/solb bash -lc 'python -m pytest tests/ -q'          # 473 passed, 75 skipped
env/solb bash -lc 'bash scripts/node_acceptance.sh'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'
```

Evaluating one solution:

```bash
env/solb sol-execbench data/SOL-ExecBench/benchmark/L1/<problem> \
    --solution my_solution.json
```

**Correctness runs against the AMD-derived tolerances**, which live in
`artifacts/05/workloads/` rather than in the dataset:

```bash
SOLEXBENCH_WORKLOADS_ROOT=$PWD/artifacts/05/workloads env/solb ...
```

This is opt-in rather than defaulted, so that running against the dataset's
B200 tolerances is a thing someone chose. It matters: the same references fail
8 workloads of `L2/033` under upstream's tolerances and 0 under these.

`env/solb` runs unprivileged as the invoking user. `env/solb-root` is
privileged and exists *only* to apply the clock lock, because `/sys` is
read-only in a stock container.

## Reproducing the measurements

Ordered; each is resumable and each writes failures as artifacts rather than
losing them.

```bash
bash scripts/node_acceptance.sh                       # 00  node census, roofline
python scripts/clock_calibrate.py determinism-sweep   # 01  F_LOCK  (BLOCKS 03/05/06)
python scripts/sol_bounds.py --part MI350X --freq-mhz 1300 --jobs 24 \
       --out artifacts/03/t_sol.json                  # 03  analytic bound, CPU only
python scripts/shard_sweep.py --task tolerances --gpus 0-7 \
       --out artifacts/05/                            # 05  tolerances
python scripts/apply_tolerances.py                    # 05  -> artifacts/05/workloads
python scripts/shard_sweep.py --task tb-candidates --gpus 0-7 \
       --out artifacts/06/candidates/                 # 06  T_b selection, 8 GPUs
python scripts/authoritative_tb.py --gpu 0            # 06  T_b measurement, ONE GPU
python scripts/sol_traffic_floor.py                   # 03  second bound tier
python scripts/sol_cross_checks.py --t-b artifacts/06/authoritative
python scripts/build_manifest.py                      # 09  freeze the manifest
python scripts/check_coverage.py --artifacts artifacts/05
python scripts/verify_artifacts.py --task 09 --full
```

Selection shards across all eight GPUs; **measurement does not**. The eight
GPUs hold clocks spanning 1242–1307 MHz at the same determinism setting, which
is wider than most of the differences this benchmark exists to measure, so
every `T_b` in the manifest is re-timed on GPU 0 alone.

## Running on MI355X

Nothing here needs porting for MI355X — it is the same die — but everything
measured needs re-measuring. In order:

1. `tasks/01` — **re-measure F_LOCK.** `CLOCK_LOCK_PRESETS` carries an MI355X
   entry at 1650 MHz from an earlier session on a different node; it is kept
   and labelled, not trusted. On MI350X `rocm-smi --setperfdeterminism X`
   yields roughly `0.83·X`, and whether that holds on the 1400 W part was never
   asked.
2. Regenerate the arch config at the new F_LOCK
   (`gen_arch_yaml.py --part MI355X`). **T_SOL in cycles does not change** —
   it is clock-invariant by construction — so the millisecond column is one
   scalar division, not a re-run.
3. Re-run tasks 05 and 06. Tolerances and `T_b` are measurements and do not
   transfer.

`src/solexbench_rocm/parts.py` separates ARCHITECTURAL (shared by both parts),
PART (never shared) and MEASURED (never shared, never guessed) so that this
stays hard to get wrong.

## Layout

```
CLAUDE.md                  agent contract: prime directives, node discipline
STATE.md                   progress ledger — the handoff between sessions
PLAN.md                    full engineering plan
docs/methodology.md        how every AMD number was derived
tasks/                     10 ordered tasks, each with acceptance criteria
scripts/
  runners/                 per-problem sweep runners (references, tolerances,
                           T_b candidates, methodology comparison)
  sol_bounds.py            SOLAR bridge — T_SOL for every problem/workload
  sol_traffic_floor.py     the second bound tier, gated against measurement
  sol_cross_checks.py      is T_SOL a bound, and is it the right one?
  authoritative_tb.py      re-time the selected T_b variants on one GPU
  build_manifest.py        freezes the scoring manifest
  verify_anchor.py         the check that proves the score scale is real
  verify_artifacts.py      per-task acceptance checks
src/sol_execbench/         vendored upstream fork; AMD deltas marked "# AMD:"
src/solexbench_rocm/
  parts.py                 dual-SKU constants — one source of truth
  activity/                vendor-neutral timing attribution (mutation-tested)
  shim/                    rocprofiler-sdk activity source (C++)
  solar/                   SOLAR arch-config generator
reference/
  tb-candidates/           T_b variant set
  exploits/                anti-reward-hacking replay corpus
  upstream-audit.md        every NVIDIA-specific call site
artifacts/                 measurements, always with provenance
```

## What differs from upstream

| | upstream (B200) | here (CDNA4) |
|---|---|---|
| Clock lock | `nvidia-smi -lgc 1500` — requested == achieved | `--setperfdeterminism 1600` → **1300 MHz achieved**; the two differ by ~17% and only the achieved value is F_LOCK |
| Timing | CUPTI activity tracing | `hip_events` by default, `rocprof` via a rocprofiler-sdk shim; **recorded per trace**, and the two agree to −0.61% at the median |
| Analytic bound | SOLAR over the traced graph | that, **plus** a declared-traffic tier where the traced graph is partial or absent, with the source recorded per workload |
| Cache flush | `2 × L2_cache_size` | explicit `LLC_BYTES` table — ROCm reports the 4 MiB per-XCD L2, so the upstream sizing would be 64× too small against a 256 MiB Infinity Cache |
| L2 persistence | `cudaCtxResetPersistingL2Cache` | vendor no-op; CDNA has no such API |
| Build | `-gencode arch=…,code=sm_100a` | `--offload-arch=gfx950`; `.hip` accepted as a first-class source |
| Languages | cuda_cpp, cutlass, cudnn, cublas | plus hip_cpp, ck, ck_tile, hipblaslt, miopen, aiter |
| Quant | 18 FP8 + 15 NVFP4 | 18 FP8 port directly (CDNA4 is OCP FP8); **15 NVFP4 have no ROCm path** — see deferrals |

The NVIDIA path is kept working throughout. It is the regression reference:
when an AMD result looks wrong, being able to run the same code on NVIDIA
distinguishes "the refactor broke it" from "AMD genuinely differs". Tests that
assert NVIDIA behaviour are not skipped — they pin the vendor instead.

## Deferrals

Counted honestly and identically in every document; see
[`artifacts/deferred.json`](artifacts/deferred.json), which is the one file
every count quotes.

**15 NVFP4 Quant problems.** Their references fail on ROCm at
`torch._scaled_mm`: block-16 FP8-E4M3-scaled GEMM is CUDA-only. MXFP4 (block
32, E8M0 scales) is the OCP format CDNA4 implements and **does** work — a
Triton `dot_scaled` kernel was compiled, launched and numerically verified. But
the two formats have different block granularity and different scale formats,
so they have different quantization error: an MXFP4 twin is a
re-specification, not a translation, and must never be presented as the NVFP4
problem.

**Not run: the agent baseline.** Upstream reports a median SOL of 0.732 over a
kernel-optimizing agent's submissions, and a headroom correlation of r = 0.981.
No agent was run here. `artifacts/09/score-distribution.json` scores the T_b
variant set against the manifest instead, which validates the scale but is not
that experiment and is labelled as not being it.

## Licence

Apache-2.0, matching upstream `sol-execbench` and `NVlabs/SOLAR`.
