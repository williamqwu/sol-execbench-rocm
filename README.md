# SOL-ExecBench-ROCm

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

Scores are valid only *within* a manifest version. Any stack change that moves
`T_b` requires a new version, and so does a change to `T_SOL`.

There are three. **v1** (`artifacts/09/manifest-v1.json`) is frozen: it is what
the release shipped and every score published against it stays valid against
it. **v1.1** repeats no measurement — every cycle count is v1's — and corrects
two things about the conversion to milliseconds:

* `T_SOL_ms` divides cycles by the measured clock of the datapath the workload
  runs on, not by a single F<sub>LOCK</sub>. Every matrix path holds 1300 MHz;
  the fp32 vector path sustains 1441. See `STATE.md` D35.
* A paged KV cache is priced at the pages the workload gathers, not at its
  allocation. See D18 and D36.

**v1.2** (`artifacts/09/manifest-v1.2.json`) is what the board serves. It is the
first version that re-derives rather than re-divides: SOLAR read a convolution's
`groups` only from `nn.Module` arguments, and every convolution in this
benchmark is functional, so a depthwise convolution was priced as a dense one —
over by exactly the group count, 768× on `L1__006`'s arithmetic term. Seven
problems, 81 workloads. The pipeline was re-run on `device="meta"`, so still no
GPU and no measurement repeated. See D37 and `scripts/rebuild_manifest_v12.py`.

Bounds a real kernel beats: **13 under v1, 6 under v1.1, 3 under v1.2** — and
one of the four that came off in between was not a bound problem at all. The
harness was not seeing work a submission put on its own stream, so `L1__054`'s
*time* was 32% short against a bound that was correct all along (D38). Two of
the published scores were re-measured; the rest of the run was not, because
re-timing 220 problems to correct two moves 218 numbers that were fine.

`scored.json` records which manifest produced it, because a score read against
the wrong one is not off by a little.

### Two means, and they are not interchangeable

`summary` in a `scored.json` carries both, and mixing them is easy and costly —
it produced a wrong headline in this repo on 2026-08-09:

| field | what it is |
|---|---|
| `mean_score` | **the published figure.** Excludes workloads whose bound a kernel beat, because a score against a bound known to be wrong is not a score. |
| `mean_score_including_invalid_bounds` | diagnostic only. Includes them, so it moves a lot whenever a bound is corrected — and a bound correction is exactly when someone reaches for it. |

The two answer different questions and a correction moves them by very
different amounts. On v1 → v1.1, `glm-sweep-2` goes 0.6083 → 0.6111 on the
published field and 0.6288 → 0.6158 on the diagnostic one — different sizes,
and opposite signs.

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

**The variant set is not an agent baseline.** The four PyTorch formulations
cluster around `T_b` by construction — `T_b` is defined as the fastest of them —
so they cannot replicate upstream's median SOL of 0.732, which is a result
*about agents*. What they do establish is that the scale is well-formed: every
score finite and in (0, 1], `S = 0.5` at `T_b` to machine precision, and a
within-workload correlation between `S` and headroom reclaimed of **r = 1.000**
(median over 2518 workloads).

Three agent runs have since happened, none of them a full-benchmark submission;
see [Deferrals](#deferrals).

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
env/solb bash -lc 'python -m pytest tests/ -q'          # 503 passed, 56 skipped
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

## Seeing the results

```bash
leaderboard/run.sh            # http://127.0.0.1:8088
```

A local leaderboard over the frozen manifest, with four kinds of page:

* **rankings**, and a searchable problem index;
* a **problem** page — scoring bounds, tolerance and its derivation, every
  submission's result on every workload;
* a **run** page (`/submissions/<slug>/problems/<key>`) — one submission on one
  problem: per-workload `T_SOL` / `T_b` / `T_k`, the kernel it proposed beside
  the `T_b` formulation it had to beat, the trajectory of every harness eval it
  ran, and what the problem cost in dollars and turns;
* a **submission API** (`POST /api/v1/submit`) with a queue and a GPU-0 worker.

The database is a rebuildable SQLite view of `artifacts/` — never a source of
truth — and scores are computed by importing the repo's own `sol_score`, so the
board cannot drift from the harness. Every `/api/v1` route declares a response
schema. See [`leaderboard/README.md`](leaderboard/README.md).

It ranks on a benchmark score that sums per-workload scores across the *whole*
benchmark and counts anything not passed as zero — so it cannot be raised by
attempting less. Both other denominators are shown beside it and neither is
ranked on: mean over *attempts* (a failed attempt scores zero) and mean over
*passes*. The gap between them is the point. `torch.compile` reads 0.4907 over
passes, the best of any variant, and 0.4002 over attempts, below plain eager —
the difference is the 585 workloads it raises on, which leave the first
denominator and not the second.

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
  agent_baseline.py        run kernel-optimizing agents in sandboxes, GPUs 1-7
  agent_eval.py            the agent's feedback loop — the real harness, no bounds shown
  agent_score.py           re-time the agents' kernels on an idle GPU 0 and score
  agent_cost_report.py     dollars, wall time, GPU occupancy, and what a full run costs
leaderboard/               local leaderboard and submission service
  ingest.py                rebuilds the SQLite view from artifacts/, atomically
  app.py, models.py        pages + a typed JSON API under /api/v1
  submit.py, worker.py     submission queue, and the GPU-0 scoring worker
tests/
  leaderboard/             service tests; skipped in the container, which
                           deliberately has no fastapi
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

**Partial: the agent baseline.** Upstream reports a median SOL of 0.732 over a
kernel-optimizing agent's submissions, and a headroom correlation of r = 0.981.
Four runs have happened here, and `glm-sweep-2` **is** a full-benchmark
submission — 220 of 220. It still does not replicate upstream's number, for a
reason that no amount of coverage fixes: that median is a different model on a
different part, scored against NVIDIA-derived bounds. The board shows each
run's coverage rather than hiding it:

| run | problems | coverage | benchmark score | on the board |
|---|---|---|---|---|
| `glm-sweep-2` — GLM-5.2, codex-cli, 1 h/problem | **220** | **100%** | 0.5921 | yes |
| `glm-run1` — GLM-5.2, amdpilot fleet | 24 submitted, **23 measured** | 10.3% | 0.0672 | yes |
| `opus5-budget100` — Claude-Opus-5, $100/problem, $250 total | 4 | 1.6% | 0.0111 | yes |
| `pilot8` — Claude-Opus-5, $8/problem | 8 | 2.7% | — | **no** |

The 24th `glm-run1` kernel is real and unmeasured: `FlashInfer-Bench__014`'s
authoritative re-time hit `TimeoutExpired` after 1200 s, so it produces no
result rows at all. It contributes zero to the score, exactly like a problem
nobody attempted — which it is not. `STATE.md` D23.

`pilot8` is excluded, with the reason on `/methodology`: all eight sessions hit
the spend cap mid-work, so none chose when to stop, and three submitted a kernel
that does not pass. Its mean of 0.776 is survivorship over the five problems
where anything passed at all. The artifacts are kept; the row is not.

`artifacts/09/score-distribution.json` still carries the T_b variant set, which
validates the scale and is labelled as not being an agent result.

**Eight problems have a T_SOL that is known or suspected wrong.** Three were
caught directly — a correct kernel measured *faster than the speed-of-light
bound*, which is possible only if the bound is wrong:

* `FlashInfer-Bench__019_mla_paged_prefill` (25 of 38 workloads, `pilot8`). The
  declared-traffic tier prices a paged KV cache at its full allocation while the
  kernel gathers 34 pages of 989,669. The same mechanism exposes **six** paged
  FlashInfer problems and **249 scoreable workloads**, badly at the median for
  the three prefill variants — `STATE.md` D18.
* `L1__005_conv_gated_projection_with_causal_conv`, beaten by 1.09–1.15×, and
  `L1__035_flux_ada_layer_norm_zero_modulation_extraction`, by 1.003–1.013×
  (both `glm-run1`). Neither is paged attention, so D18 does not explain them,
  and they are probably not the same defect as each other — `STATE.md` D21.

Scores on all of these are not usable in v1 and are marked wherever they appear.
The v1.1 fixes are in `STATE.md`; the `T_SOL <= T_b` gate cannot catch any of
them, because a bound that over-counts traffic is under-cut by the reference in
exactly the same way.

## Licence

Apache-2.0, matching upstream `sol-execbench` and `NVlabs/SOLAR`.
