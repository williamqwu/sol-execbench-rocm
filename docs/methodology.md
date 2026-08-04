# SOL-ExecBench-AMD — methodology

How the AMD numbers were derived, what differs from upstream's B200
methodology, and what is deferred. Every figure here was measured on the
hardware named beside it; nothing was carried over from NVIDIA.

Companion documents: [`README.md`](../README.md) for what this is and how to
run it, [`STATE.md`](../STATE.md) for the progress ledger and the full list of
surprises, [`tasks/`](../tasks/) for the per-task acceptance criteria.

---

## 1. The score, and what it is a score *of*

```
S(T_k) = 1 / (1 + (T_k − T_SOL) / (T_b − T_SOL))
```

`S = 1` at the Speed-of-Light bound, `S = 0.5` at the optimized-PyTorch anchor.
Three inputs, and all three are platform-specific:

| term | what it is | how AMD gets it |
|---|---|---|
| `T_SOL` | analytic roofline bound | SOLAR over the problem's own graph, against a generated MI350X arch config (task 03) |
| `T_b` | optimized-PyTorch anchor | measured, per workload, best of a fixed variant set (task 06) |
| tolerances | what counts as correct | derived from reference-vs-reference variance on MI350X (task 05) |

**None of the three may be copied from B200.** That is not a stylistic
preference. A copied tolerance is either too tight (correct kernels fail) or
too loose (wrong kernels score well); a copied `T_SOL` or `T_b` rescales every
score by a constant nobody can see. All three failure modes produce output
that looks entirely reasonable.

## 2. The cross-vendor caveat — read this before comparing anything

> An AMD SOL score and an NVIDIA SOL score are each **within-platform**
> measures of the fraction of hardware headroom reclaimed. They are comparable
> in spirit but are **not** a cross-vendor performance comparison, because
> analytic peaks are reachable to different degrees on different
> microarchitectures.

Concretely, on this hardware: BF16 GEMM reaches **1168 TFLOPS of a 2307 TFLOPS
analytic peak — 50.6%**. HBM copy reaches **4.53 TB/s of 8.0 TB/s — 56.7%**. A
kernel at `S = 0.8` on MI350X and one at `S = 0.8` on B200 have each reclaimed
80% of the distance from their platform's anchor to their platform's bound.
They have not been shown to be equally fast, equally good, or equally close to
what the silicon can do.

The measured ceilings are published beside the analytic peaks
(`artifacts/00/roofline-gpu0.json`) precisely so that this difference is
visible rather than inferred.

## 3. F_LOCK — and why the definition had to change on MI350X

Every `T_SOL` and every `T_b` is expressed at a locked clock, so that a kernel
measured on Monday is comparable to one measured on Friday. Upstream locks
B200 to 1500 MHz.

**MI350X: determinism setting 1600 MHz, F_LOCK = 1300 MHz achieved.**

Those are two different numbers, and that is the finding. On NVIDIA,
`nvidia-smi -lgc 1500` pins the clock at 1500 and the distinction does not
exist. On MI350X, `rocm-smi --setperfdeterminism X` yields roughly `0.83·X`,
rock-steadily:

| requested | achieved (median) | min | power |
|---|---|---|---|
| 1100 | 934 | 932 | 666 W |
| 1250 | 1049 | 1048 | 729 W |
| 1350 | 1116 | 1114 | 770 W |
| 1500 | 1220 | 1194 | 836 W |
| **1600** | **1303** | **1296** | **885 W** |
| 1700 | 1380 | 1376 | 947 W |
| 1900 | 1403 | 1397 | **1000 W** |
| 2200 | 1402 | 1397 | **1000 W** |

Two things follow, and both are load-bearing:

1. **F_LOCK is the achieved number.** Recording the requested one would
   overstate the clock by ~23% and scale every analytic bound with it.
2. **Above ~1900 the setting stops mattering.** The part pins to its 1000 W cap
   and lands on the same ~1400 MHz whether asked for 1900 or 2200. In that
   regime the clock is set by ambient conditions rather than by us, which is
   what a lock exists to prevent.

1600 was chosen over 1700 despite 1700 giving a higher clock: at 1700 the part
draws 947 W of a 1000 W cap on two of the three GPUs sampled, and a lock one
warm afternoon away from becoming power-bound is not a lock. At 1600 the whole
node draws 868–933 W, so the *setting* binds.

**Unlocked sustained floors**, for reference — 15-minute saturating BF16 GEMM,
p5 of the final 5 minutes: 1390 / 1367 / 1335 MHz on GPUs 0 / 1 / 2. The part
is power-limited, not thermally limited (1000 W cap, junction ≤79 °C against a
100 °C slowdown point).

### The eight GPUs do not share one clock

At the same determinism setting, achieved clocks span **1242–1307 MHz**, a 5%
spread:

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| MHz | 1303 | 1295 | 1264 | 1307 | 1279 | 1296 | 1285 | 1242 |

Each GPU is individually stable (min within ~20 MHz of its own median) but they
differ from each other by more than most of the optimization differences this
benchmark exists to measure. Consequence, applied throughout:

* **authoritative timing runs on GPU 0**, and every timing artifact records its
  GPU;
* sharded sweeps across all eight GPUs are used for correctness and for
  *selecting* a T_b variant, never for the final anchor;
* the winning variant is re-timed on GPU 0 — and so is the runner-up, plus
  anything within 25% of the fastest, because selection noise can only
  mis-order variants that are close and an anchor 3% too slow inflates every
  score measured against it, permanently and invisibly.

Reproducibility at F_LOCK: **CV = 0.0034** over 30 trials in separate processes
(gate 0.02). Sibling interference: **−0.11%** with seven siblings drawing
~6.2 kW between them, so sweeps and authoritative timing may share the node.

**Different GPUs, not the same GPU.** Sibling interference across the node is
negligible; two timing runs *on one GPU* are not. The shard runner originally
assigned `gpus[i % len(gpus)]` at submission time while running one worker per
GPU, which lets two tasks whose indices are congruent mod `len(gpus)` run
together on one device while another idles — observed live, with `L2/060` and
`L2/068` both on GPU 7. Nothing in the output showed it: the artifact records
the device it was told to use, which is identical either way. Workers now take
a GPU from a queue and return it. 176 of the selection-pass artifacts predate
the fix, which is a reason the authoritative pass re-times a band of variants
rather than only the fastest.

### MI355X

`CLOCK_LOCK_PRESETS` carries an MI355X entry at 1650 MHz, measured on a
different node in an earlier session. It is kept and labelled, not reused:
MI355X is the liquid-cooled 1400 W part with a 2400 MHz ceiling, and its
sustained floors were 1725–1757 MHz. Same die, ~350 MHz apart. Anyone running
this on MI355X should re-run `tasks/01` and confirm whether the
requested-vs-achieved split behaves the same way there; it was not observed on
that node, but the node was also never asked the question this way.

## 4. Architectural constants: what may be shared between parts

`src/solexbench_rocm/parts.py` is the single source of truth, and it separates
three kinds of quantity explicitly:

* **ARCHITECTURAL** — a property of the die, shared by MI350X and MI355X. The
  MAC/cycle table earns that status by reproducing *both* parts' published
  peaks from one set of numbers: `524288 × 2 × 2.4 GHz = 2.52 PFLOPS` against
  MI355X's published 2.5, and `× 2.2 GHz = 2.31` against MI350X's 2.3. Likewise
  FP8 (5.03 / 4.61 vs 5.0 / 4.6) and MXFP4 (10.1 / 9.2 vs 10.1 / 9.2).
* **PART** — peak clock, power cap, cooling. Never shared.
* **MEASURED** — F_LOCK above all. Never shared, never guessed.

A constant that derives both parts' published figures is architectural. One
that does not is a measurement in disguise.

## 5. T_SOL

SOLAR (NVlabs, Apache-2.0, pinned at `d3524c4`) runs its five-stage pipeline
over each problem's **own reference**, wrapped as the `nn.Module` +
`get_inputs()` pair it expects. The arch config is *generated*
(`gen_arch_yaml.py`), not hand-written, because `MAC_per_cycle` is
architectural while `*_byte_per_cycle` must be rescaled as
`bytes_per_sec / F_LOCK` — editing `freq_GHz` by hand silently moves the
roofline balance point.

**T_SOL is emitted in cycles as well as milliseconds.** The cycle count is
invariant to F_LOCK, so a future re-lock is one scalar division rather than a
re-run, and the cycle column is directly comparable between MI350X and MI355X,
which differ only in clock.

**The `fused` model is the bound.** SOLAR emits three (`unfused`, `fused`,
`fused_prefetched`). `unfused` assumes every intermediate round-trips to DRAM,
which is above what a competent fused kernel achieves — a "lower bound" that
real measurements would beat. Both are recorded so that a `T_SOL ≤ measured`
violation can be diagnosed rather than merely observed.

### Two tiers of graph extraction, recorded per workload

Some references are data-dependent — they call `.item()` or `nonzero()` — and
cannot be traced on meta tensors at all. Those trace with the problem's **own
input generator on CPU**, and which tier ran is recorded per workload.

Real inputs, deliberately, not zeros: a zero-filled `cu_seqlens` traces an
attention over nothing and yields a confident, wrong, much smaller bound. A
missing bound is recoverable; a wrong one is not.

### Two derivations of the bound, never merged silently

SOLAR's graph is not always the whole kernel, and two separate cross-checks say
so. Fifty problems produce `einsum_graph has no layers` — SOLAR models einsum
layers and a kernel of pure elementwise arithmetic and indexing has none.
Forty-eight more produce a bound *below the traffic the problem's own
definition declares*, which means the traced graph is missing tensors the
problem states it reads.

So there is a second derivation, and it is the simplest bound in the roofline:

```
T ≥ (declared input bytes + declared output bytes) / DRAM bandwidth
```

computed against the same arch config at the same locked clock
(`scripts/sol_traffic_floor.py`). It accounts for *all* the traffic and *none*
of the arithmetic, where SOLAR's accounts for the arithmetic over a graph that
may be partial. Neither dominates, both are valid lower bounds, and the larger
of two valid lower bounds is the better one — so the manifest takes the `max`
and **records which derivation won and what the other one said**
(`t_sol_source` ∈ `solar_fused`, `declared_traffic`, `max_of_both`). They are
never presented as one anonymous column.

One exception is not optional. Where a problem declares a tensor it *indexes*
rather than streams — `L1/018` declares a 131072-position KV cache and touches
one sequence's worth — the declared total exceeds any real kernel's traffic,
and the derived "bound" lands above the measured time. A lower bound above a
measured time is not loose, it is wrong, and it would push scores past 1. Every
traffic bound is therefore gated against the measured `T_b` and falls back to
SOLAR's value when it fails the gate, with the fallback counted.

### Rounding, and the eight bounds that were zero

SOLAR emits `total_cycles` already wrapped in `int()`. At 1.3 GHz a cycle is
0.77 ns and 12 KB at 8 TB/s is two cycles, so sub-cycle bounds are the normal
case at the small end rather than an edge case — eight workloads truncated to
**T_SOL = 0 cycles**, a bound no kernel can approach and a `(T_b − 0)` in the
denominator of the score. A further 204 implied a DRAM bandwidth *above* the
config's own peak, which a roofline cannot do and which is what led here.

The bridge now recomputes SOLAR's own formula —
`max(MACs / MAC_per_cycle, bytes / DRAM_byte_per_cycle)` — from the figures
SOLAR reports beside the result, and ceils, keeping the exact value in
`t_sol_cycles_exact`.

### V1 / V2 / V3, the three flagged unknowns

* **V1 — TF32 on CDNA4.** Deliberately absent from the MAC/cycle table rather
  than guessed. A missing key raises; a wrong key computes a plausible bound.
  No problem in the dataset resolved to `tf32`, so nothing depended on it.
* **V2 — Infinity Cache bandwidth.** The 17 TB/s figure in the generator is a
  placeholder and is **wrong**: measured cache-resident bandwidth peaks at
  5.2 TB/s (64 MiB working set) against 4.5 TB/s from DRAM, a ratio of 1.15×,
  not 3.8×. It is also **inert**: `SRAM_byte_per_cycle` and `SRAM_capacity` are
  referenced nowhere in SOLAR's perf model, which applies DRAM bandwidth to all
  three memory models. Recorded as both wrong and harmless rather than quietly
  left alone.
* **V3 — MXFP4 dense vs sparsity.** The table uses AMD's **dense** MXFP4/MXFP6
  row. AMD separately quotes 10.1 PFLOPS for FP8 *with sparsity*; they are
  different rows and are not conflated.

## 6. Tolerances

Upstream's procedure, mirrored so the numbers mean the same thing: run the
reference repeatedly under perturbation, record the empirical error
distribution between runs, take the maximum × 1.25.

Two details that had to be got right, both found by inspecting the first
output rather than trusting it:

**Compare two executions on the same inputs.** The seed loop varies input
*data*, so the error distribution is sampled across the input space; within a
seed the inputs are identical and only the execution differs. Comparing
outputs across different seeds compares answers to different questions — it
reported a 9.8 absolute error on a problem that is bit-exact run to run.

**Floor the tolerance at one ulp *at the output's scale*.** A reference that is
bit-exact yields zero measured variance, and shipping a zero tolerance would
reject any correct submission that reassociates one accumulation. But `eps` is
a *relative* quantity — bf16's 0.0078 means one ulp at magnitude 1 — so using
it as an absolute floor is a units error that runs 781× looser than upstream on
problems whose measured AMD variance was exactly zero. The floor is therefore
`eps × RMS(|output|)`; RMS rather than max, because max is set by a single
outlier and would grant every small element the slack of the largest.

**Golden references.** Run-to-run agreement shows a kernel is *stable*, not
that it is *correct* — a deterministically wrong kernel is perfectly stable.
`scripts/gen_golden.py` computes float64 CPU references, and task 05 compares
against them where they exist. Two tiers here as well, recorded per workload:
`float64` is arithmetic ground truth (a disagreement is a bug), `native_cpu` is
the same dtypes on CPU kernels (an independent implementation with a different
accumulation order, so still evidence, but a disagreement can also be ordinary
low-precision noise).

**Coverage, and what the tolerances are worth.** 3717 of 3957 workloads carry
an AMD-derived tolerance; the missing 240 are exactly the 15 deferred NVFP4
problems × 16 workloads. A workload without one keeps the dataset's value and
says so in its own record (`_provenance: "NOT AMD-DERIVED"`) — dropping it
would shrink the benchmark, and inheriting B200's silently is the thing this
task exists to prevent.

The check that gives them meaning: re-running every reference against these
tolerances, **3717 of 3717 non-deferred workloads pass**. Against the dataset's
B200 tolerances the same references fail 8 workloads of `L2/033`, where
upstream's `atol = 0.08` is applied to a tensor of magnitude 10¹¹. Both sweeps
are kept (`artifacts/02/references/` and `artifacts/02/references-amd/`).

Two ROCm-specific obstacles had to be cleared to get the last 27 workloads,
and one of them is a platform bug worth knowing about:

* **`masked_select` above 2³² elements.** `t[torch.isfinite(t)]` computes a
  garbage allocation size on ROCm 7.2 / torch 2.9.1 once the tensor exceeds
  2³² elements: it asks for **16781313 GiB** (2⁵⁴ + 2⁴² + 2³⁰ bytes) and raises
  OOM with 200 GiB free. Reproduced in isolation on a flat
  `(2³² + 1000)`-element tensor; promoting the same tensor to float64 and
  reducing it is fine, so it is the mask path, not the size. What made it worth
  chasing rather than filing as an OOM: the same absurd figure appeared *to the
  byte* on three problems sharing no operator.
* **Comparison width.** Promoting a whole 18 GiB output to float64 and
  materializing a difference peaks near 4× its size. Comparisons are chunked at
  64 Mi elements, which changes no result — a maximum over chunks is the
  maximum.

## 7. Timing methodology

Upstream's default is **CUPTI activity tracing**, not CUDA events: it
attributes each iteration from device-side kernel activity rather than from a
host-side event pair, which excludes launch overhead.

CUPTI has no ROCm build. This port therefore ships two methodologies and
**records which one produced every trace** (`Environment.methodology`):

* **`hip_events`** — the default. Correct, and includes host launch overhead,
  so short kernels read slightly slow and their SOL scores read slightly low.
* **`rocprof`** — dispatch-level attribution via a rocprofiler-sdk shim
  (`src/solexbench_rocm/shim/`), reaching parity with upstream's methodology.

All selection and attribution logic is shared: the shim only supplies records,
and `src/solexbench_rocm/activity/` — vendor-neutral, mutation-tested on CPU —
decides which activities belong to which iteration. No selection logic was
written twice.

Two traps were live and are worth restating:

* **Clock domain.** rocprofiler-sdk stamps records with the HSA clock. The
  host bracket must come from rocprofiler's own timestamp entry point, not
  `CLOCK_MONOTONIC`. Mixing domains does not raise — it silently bisects the
  wrong activities into each window. Verified on a real capture
  (`artifacts/04/clock-domain-verification.log`): 8/8 records fall inside their
  host window, and the negative control — the same records shifted by a full
  span — is rejected, so the guard discriminates.

  **On this driver the two clocks coincide: `CLOCK_MONOTONIC − HSA = −730 ns`.**
  That is a measurement, not a reprieve. It means a wrong implementation would
  have passed here by luck, and the trap is a property of the driver rather
  than of the code, so the guard stays.
* **Registration order.** rocprofiler locks its configuration once a ROCm
  runtime initializes, and a session configured after that point produces zero
  records rather than an error. The eval driver therefore registers the shim
  *before* `import torch`, conditional on the methodology.

### How far apart the two methodologies actually are

Measured over **1430 workload pairs**, both timing the same solution on the
same inputs back to back in one process
(`artifacts/04/methodology-comparison.md`):

| group | n | median | p10 | p90 |
|---|---|---|---|---|
| kernels ≥ 100 µs | 1044 | −0.61% | −44.8% | +1.4% |
| kernels < 100 µs | 386 | −4.71% | −43.5% | +9.6% |

Positive means `hip_events` read slower, which was the prediction. The median
came out on the *other* side of zero, and at well under a percent that is
inside the node's own reproducibility (CV 0.0034) — so the claim the
measurement supports is the narrow one: **the two agree to under 1% at the
median**, not that the expected asymmetry was observed.

The tails do not agree. 330 of 1430 pairs differ by more than 20%, in both
directions, concentrated in 22 problems. `hip_events` reads up to 90% slower
where one iteration is many tiny kernels — the event pair contains the host
work between them and the activity sum does not, which is understood.
`rocprof` reads up to 3× slower on some multi-dispatch iterations, which is
not: summing per-dispatch durations exceeds wall time whenever dispatches
overlap. That is written down as a hypothesis, unconfirmed against a dispatch
timeline, and nothing in this port depends on it.

A trace taken under `hip_events` and one under `rocprof` are not
interchangeable, which is exactly why the field exists.

## 8. Anti-reward-hacking

Upstream reports 14.5% of agent submissions flagged on NVIDIA, so this layer is
load-bearing: a benchmark whose detectors silently stopped working looks
healthy until its leaderboard is meaningless.

Every upstream mechanism is torch-level and carries over. Added for AMD:

* **smi lockout.** A submission that raises the clock cap mid-run defeats the
  entire locked-clock calibration and leaves no trace in the output. Blocked at
  two layers: a static source screen before anything is compiled (the layer
  that catches compiled HIP, which runs underneath every Python guard), and
  wrappers on the process-spawning entry points installed before user import.
* **Stream policy** (`check_default_stream`) — an interim guard while timing is
  event-based, since an event pair on the default stream cannot see a kernel on
  another stream. Weaker than the activity-count assertion it stands in for,
  which is why the methodology is recorded per trace.
* **Compute-partition mode** recorded on every trace. SPX vs CPX changes how
  many CUs a kernel can reach, so traces taken under different modes are not
  comparable.

Verification: **28/28 replay cases pass** (`artifacts/08/`). A pass means the
exploit was *detected* or *neutralized* — the corpus states which per case, and
five of those verdicts were corrected after the first run showed the defense
that actually fires was not the one assumed. Two residual gaps are asserted as
passing tests so they stay visible: a side stream that is politely restored,
and threads created during warmup.

## 9. Deferrals

State the count honestly and identically everywhere. See
[`artifacts/deferred.json`](../artifacts/deferred.json) for the machine-readable
list and `artifacts/09/manifest-v1.json` for what the manifest actually
contains.

**15 NVFP4 Quant problems.** These fail at the reference itself on ROCm:

```
scaled_gemm with `torch.float8_e4m3fn` scales of 1x16 blocks
is only supported for CUDA 12.8 and above
```

NVFP4 (block 16, FP8-E4M3 scales) is an NVIDIA format with no ROCm kernel path.
MXFP4 (block 32, E8M0 scales) is the OCP standard CDNA4 implements. **They are
different formats, not two spellings of one** — different block granularity and
a different scale format mean different quantization error, so re-scaling NVFP4
data into MXFP4 does not produce an equivalent problem. Any MXFP4 twin is a
*re-specification*, filed with provenance metadata linking to the NVFP4
original, and must never be presented as the same problem.

Feasibility was established rather than assumed (`artifacts/07/spike.json`):

| path | result |
|---|---|
| `torch._scaled_mm` MXFP4 (block 32, E8M0) | unsupported — ROCm lists float4 only at 1×16 |
| `torch._scaled_mm` NVFP4 (block 16, E4M3) | unsupported — CUDA-gated |
| Triton `dot_scaled`, e2m1 operands + E8M0 scales | **works** — compiled, launched, numerically verified |

So MXFP4 has a real kernel path on gfx950 and NVFP4 has none.
