# SOL-ExecBench-AMD

Port of NVIDIA's [SOL-ExecBench](https://github.com/nvidia/sol-execbench) — which
scores GPU kernels by proximity to an analytically derived hardware
Speed-of-Light bound rather than by speedup over a software baseline — to
**AMD Instinct CDNA4 / ROCm**.

Supports **MI350X** and **MI355X**. They are the same gfx950 die in different
chassis (air-cooled 1000 W at 2.2 GHz vs liquid-cooled 1400 W at 2.4 GHz), so
they share the build path and the architectural constants and share nothing
that was measured. Every measured quantity is keyed by part.

**Status:** measured on 8× MI350X. See [`STATE.md`](STATE.md) for the live
ledger and [`docs/methodology.md`](docs/methodology.md) for how each number was
derived.

## The one thing to get right before comparing anything

> An AMD SOL score and an NVIDIA SOL score are each **within-platform** measures
> of the fraction of hardware headroom reclaimed. They are comparable in spirit
> but are **not** a cross-vendor performance comparison, because analytic peaks
> are reachable to different degrees on different microarchitectures.

On this hardware, BF16 GEMM reaches 50.6% of its analytic peak and HBM copy
56.7% of 8 TB/s. Two kernels at `S = 0.8` on different vendors have each
reclaimed 80% of the distance from their own platform's anchor to their own
platform's bound. They have not been shown to be equally fast. Measured
ceilings are published beside the analytic peaks so the difference is visible
rather than inferred.

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

## Running it

Two ways into the pinned environment. `env/solb` runs everything inside
`solbench:rocm7.2-torch2.9.1` and is the documented path. `env/solb-native`
satisfies the same environment contract on a node with no docker daemon, and
**asserts** the stack matches the pin rather than assuming it — it refuses to run
when torch or ROCm has drifted, because every baseline here is relative to one
stack and a drifted one produces numbers that look equally authoritative. Swap
`env/solb` for `env/solb-native` throughout if docker is unavailable.

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

## Scoring coding agents against it

Task 10 runs coding agents over the problem set and reports what they achieved
and what it cost.

```bash
# sweep: one agent per problem per harness, each on its own leased GPU
env/solb-native python scripts/run_agents.py --run-id pilot-01 --limit-per-category 6

# score: re-evaluate every harvested solution on the authoritative GPU
env/solb-native python scripts/score_solutions.py --run-id pilot-01

# publish: artifacts/10/scoreboard.json + a self-contained dashboard.html
env/solb-native python scripts/build_scoreboard.py --all-runs
```

Two things about this deserve saying before any number from it is quoted.

**A harness is not a model.** Claude Code and Codex differ in how many tool calls
they make, how they recover from a compile error, and how long they persist. A
difference in score is a difference between *agents*, not between the underlying
models in isolation.

**The score's basis is part of the score.** `S` needs `T_SOL` *and* `T_b`; with
either missing the scoreboard falls back to a weaker basis and labels every
record with which one — `correctness_only`, `speedup_vs_reference`,
`sol_headroom`, or `sol_score_v1`. Records of different bases are never averaged
together, because such a mean moves when the *bounds* land rather than when the
kernels improve. See [`tasks/10-agent-scoreboard.md`](tasks/10-agent-scoreboard.md).

`env/solb` runs unprivileged as the invoking user. `env/solb-root` is
privileged and exists *only* to apply the clock lock, because `/sys` is
read-only in a stock container.

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
  build_manifest.py        freezes the scoring manifest
  verify_anchor.py         the check that proves the score scale is real
  run_agents.py            task 10 — sweep coding agents over the problem set
  agent_verify.py          the on-GPU feedback channel an agent calls
  score_solutions.py       re-evaluate harvested solutions, authoritatively
  build_scoreboard.py      scoreboard.json + self-contained dashboard.html
src/sol_execbench/         vendored upstream fork; AMD deltas marked "# AMD:"
src/solexbench_rocm/
  parts.py                 dual-SKU constants — one source of truth
  activity/                vendor-neutral timing attribution (mutation-tested)
  shim/                    rocprofiler-sdk activity source (C++)
  solar/                   SOLAR arch-config generator
src/solexbench_agents/
  task_packet.py           the sandboxed per-problem directory an agent works in
  harnesses.py             claude-code / codex adapters, with cost accounting
  gpu_pool.py              GPU leasing — agents never share a device
  scoring.py               S, headroom, reference-copy and integrity checks
  aggregate.py             the scoreboard's numbers
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
| Timing | CUPTI activity tracing | `hip_events` by default, `rocprof` via a rocprofiler-sdk shim; **recorded per trace** |
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
[`artifacts/deferred.json`](artifacts/deferred.json).

**15 NVFP4 Quant problems.** Their references fail on ROCm at
`torch._scaled_mm`: block-16 FP8-E4M3-scaled GEMM is CUDA-only. MXFP4 (block
32, E8M0 scales) is the OCP format CDNA4 implements and **does** work — a
Triton `dot_scaled` kernel was compiled, launched and numerically verified. But
the two formats have different block granularity and different scale formats,
so they have different quantization error: an MXFP4 twin is a
re-specification, not a translation, and must never be presented as the NVFP4
problem.

## Licence

Apache-2.0, matching upstream `sol-execbench` and `NVlabs/SOLAR`.
