# SOL-ExecBench-ROCm — methodology

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

#### That question, asked on a second MI355X node

It behaves worse there, and not in the way the paragraph above anticipates: the
split is not between cards, it is between *asking* and *not asking*.
`scripts/gpu_parity_check.py` runs one fixed 8192³ BF16 GEMM on all eight GPUs and
measures throughput by wall clock, so the result does not depend on the clock
telemetry being right. `scripts/unlocked_clock_probe.py` then varies the workload,
the duration and the neighbour load. Both use nothing from the benchmark harness.

**Unlocked, the node is uniform.** All eight cards on `perf_level=auto`, saturating,
45 s: 1447–1498 TFLOPS (**3.4% spread**), 1724–1837 MHz, 1377–1399 W each, 56–63 °C.
Every socket delivers its full 1400 W at once. No weak card, no cooling imbalance.

**Applying `--setperfdeterminism` is what breaks that.** Same run with 1660 requested
on all eight: throughput spread widens to **21.2%**. Six cards sit ~330 MHz below the
frequency they acknowledged, at ~980 W against a 1400 W cap, *cooler* than the one
card that holds, and `amd-smi` reports no violation of any kind on them — not PPT,
thermal, VR, HBM or PROCHOT. Not contention either: GPU 2 alone, all others idle,
still sits at 1325 MHz.

**And on two of the eight the setpoint does nothing at all.** Sweeping the request,
one card at a time under load:

| requested | GPU 1 achieved | GPU 2 achieved |
|---|---|---|
| 1200 MHz | **1657** (1.38×) | 1015 (0.85×) |
| 1400 | **1656** (1.18×) | 1155 (0.83×) |
| 1500 | **1655** (1.10×) | 1214 (0.81×) |
| 1660 | 1656 (1.00×) | 1322 (0.80×) |

GPU 1 runs 1655–1657 MHz whatever it is asked for. It looks like the one healthy card
only because 1656 happens to coincide with the 1660 being requested — a coincidence
that survives exactly one setpoint, which is why the sweep matters and a single-point
check does not. GPUs 2–7 are the ones where determinism functions as a control, with
a 0.80–0.85 scale error. Neither group is correct and the two failure modes are
opposite: one cannot be slowed, the other cannot reach speed. All eight accept the
request and read back as `perf_determinism`.

**The telemetry is honest**, which is worth stating because it was the first
suspicion. Throughput divided by reported clock is **813–856 TFLOPS/GHz on every
card in every condition above** — locked, unlocked, alone, contended, fast and slow.
A card reporting 1326 MHz delivers exactly the work 1326 MHz predicts. The cards
really do run slow; nothing here is a sampling artifact.

Firmware on that node, since this is a firmware-level finding and not a claim about
the part: VBIOS `113-M355-01-1K1-000C`, SMC `04.86.10.05`, MEC 38, RLC 43, SOS
`0x00450028`, identical across all eight; ROCm-SMI-LIB 7.8.0, amdgpu 6.16.6.

Consequence for anyone in this position: `F_LOCK` cannot be *chosen* on such a node,
only measured and verified afterwards. Timing there ran on the one card that holds a
frequency invariantly — 1655–1657 MHz at ~1295 W whether idle-adjacent or with all
eight saturated — while recording that the invariance comes from firmware pinning and
not from the lock, so a firmware change would show up as a changed measurement rather
than as silence.

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

### Both terms of the roofline, not only their max

The two terms of that `max` scale oppositely with the clock, and the arch YAML says
which is which in its own annotations: `MAC_per_cycle` is architectural and
frequency-independent, so the compute term is a fixed number of **cycles** and its
time goes as 1/F; `DRAM_byte_per_cycle` is derived as `bytes_per_sec / freq`, so the
memory term is a fixed **time** and its cycle count scales with F.

Storing only the max discards that. A card boosting from 1650 to 2394 MHz on a
memory-bound kernel has not moved that kernel's bound at all — HBM does not run off
the core clock — while the same boost tightens a compute-bound bound by 31%. From the
max alone neither term is recoverable, and **the bottleneck can flip as F moves**, so
scaling whichever won at the reference clock is wrong in both directions.

`sol_bounds.py` therefore emits `compute_cycles`, `memory_cycles_at_f_ref`,
`mac_per_cycle` and `dram_byte_per_sec` alongside the bound, and
`src/solexbench_rocm/t_sol_at.py` re-maxes them at any clock. Verified on real data
to reproduce the recorded `t_sol_ms` exactly at the reference clock, and to leave a
memory-bound bound flat from 1650 to 2394 MHz. Records written before the split are
refused rather than inferred from `bottleneck`, which would happen to work only while
the evaluated clock stayed above the reference one.

This costs four numbers per workload and makes the bound model explicit enough to
test. It is also what a node that cannot pin its clock needs in order to score at
all — see *That question, asked on a second MI355X node* in §3.

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
  "Slightly" was optimistic; see *How short is the timed window?* below for the
  measured size of it.
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

### How short is the timed window?

`BenchmarkConfig` defaults to `warmup_runs=10, iterations=50`, so `time_runnable`
times 60 back-to-back executions. For the sub-millisecond kernels that make up most
of this corpus that is a **1–13 ms window** — shorter than any telemetry sampler can
observe, which means "what clock was this measured at?" had never actually been
answered for any artifact, only assumed.

It can be answered without telemetry. At a fixed shape, per-iteration time is
proportional to 1/clock, so timing the same kernel at increasing burst lengths turns
the question "was the clock the same?" into "did the work take the same time per
unit?", at whatever timescale is convenient. `scripts/burst_clock_probe.py` does
that. Per-iteration time relative to a fully sustained loop:

Per-iteration time relative to a fully sustained loop, unlocked / locked:

| burst | GEMM 4096³ (compute-bound) | GEMM 1024³ | elementwise (memory-bound) |
|---|---|---|---|
| **60 iters** (the shipped default) | **1.217 / 1.241×** | **2.040 / 2.090×** | 1.042 / 1.044× |
| 400 | 1.079 / 1.112 | 2.516 / 2.543 | 1.009 / 1.010 |
| 2 000 | 1.031 / 1.018 | 1.708 / 1.702 | 1.003 / 1.004 |
| 10 000 | 1.017 / 1.001 | 1.001 / 0.999 | 1.000 / 1.001 |
| 50 000 | 1.000 | 1.000 | 1.000 |

A compute-bound GEMM measures **~22% slower** at the shipped burst length than at
steady state, and a small one **more than twice as slow** — non-monotonically
(2.04, 2.52, 1.71), because at that size the per-iteration cost dominates the whole
window. Convergence needs ~10 000 iterations.

**Slightly worse locked than unlocked** in all three shapes, so it is not an artifact
of a node that cannot pin its clock. It is a property of this timing window that every
`T_b` and every score already carries.

**It is not the clock, and D20 is why we can say that cheaply.** A clock still ramping
out of idle was the obvious reading and it is wrong, on two independent grounds. D20
sampled the active DPM level at ~860 Hz *inside* the timing loop and found the clock
steady to 1.00× during measurement. And the absolute cost per iteration here is not
distributed the way a clock effect would be:

| kernel | Δ per iteration, unlocked | locked |
|---|---|---|
| GEMM 4096³ | 21.2 µs | 25.5 µs |
| GEMM 1024³ | 13.5 µs | 14.2 µs |
| elementwise `a + b` | **0.6 µs** | **0.7 µs** |

A depressed clock slows every kernel roughly in proportion. The elementwise kernel is
effectively immune while both GEMMs pay 13–26 µs — a **33–36× spread** — so whatever
this is lives on the **GEMM path**, not in the clock and not in launch overhead
generally. Note also that the cost is nearly the same in absolute terms for a 4096³
GEMM and a 1024³ one whose sustained iteration is 7.5× shorter, which is why it shows
up as 1.2× on one and 2.0× on the other.

That makes it the same suspect D20 ends on — "kernel selection inside hipBLASLt is the
remaining suspect and has not been tested" — approached from the other side. D20 was
chasing rare 3.9–21× stalls at 0.13% of iterations; this is a systematic cost on every
iteration of a short window. They may or may not be one mechanism. What this
contributes is a control D20 did not have: a non-GEMM kernel measured the same way,
which is immune, and which therefore narrows the search to the GEMM path without
needing any clock instrumentation at all.

What it does and does not invalidate. Baseline and candidate are measured the same
way, so **speedup ratios are unaffected** — the bias divides out. **SOL efficiency is
systematically understated**, since `T_measured` carries this overhead and `T_SOL`
does not model, roughly uniformly, so rankings hold. That is why this is documented
rather than treated as a defect.

Deliberately not fixed here. Raising `iterations` until the window reaches steady
state would remove the bias and make the clock samplable, but 50 is upstream's
methodology; changing it makes these numbers incomparable with upstream's and
requires re-timing everything. That is a decision about what the benchmark is for,
not a bug fix.

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

Verified against the largest corpus of known-good submissions available — the
reference sweep itself: **0 of 235 problems flagged**. A detector that fires on
the problem's own reference would fail every honest submission.

## 8b. Does the scale hold up?

Two checks, and they are different in kind. The first is arithmetic: score the
variant set against the frozen manifest (`artifacts/09/score-distribution.json`,
12248 workload–variant pairs). `S = 0.5` lands on the variant that became `T_b`
with `max |S − 0.5| = 0`, every score is finite and in (0, 1], and the
within-workload correlation between `S` and headroom reclaimed is **r = 1.000**
(median over 2518 workloads).

The second re-measures. `verify_anchor.py` re-times both arms on GPU 0, in
fresh processes, over a 20-problem sample:

| property | result |
|---|---|
| no measured time below its own `T_SOL` | 349/349 |
| the reference never scores above the anchor | 349/349 |
| re-timed `T_b` implementation scores 0.5 ± 0.03 | 336/349 |

The gap between the two is the interesting part. The arithmetic check is exact
by construction; the re-measurement is not, and it found one problem
(`FlashInfer-Bench/018_mla_paged_decode`) whose latency does not reproduce to
3% — 12 of its workloads re-time a median 1.16× slower than the `T_b` recorded
for them, stably across two independent runs. A manifest that only ever checked
itself would have called that problem perfect.

#### When is that third check even answerable?

The `0.5 ± 0.03` check demands a timing precision, and how much precision depends on
something it never looked at. `S` rises from 0.5 at `T_b` to 1.0 at `T_SOL`, so with
headroom `h = (T_b − T_SOL) / T_b` and relative timing error `eps`,

    |dS| = 0.5 * eps / h

A workload whose `T_b` already sits within 3% of the speed of light therefore needs
`T_k` reproduced to about **0.18%** to hold `S` inside ±3% — below any precision
available here, and an order below the 22%-to-2× short-window bias of §7. Such a
workload
cannot pass however sound its bound and its `T_b` are.

Measured on the MI355X run: of 219 workloads, **all 171 with headroom ≥ 25% passed**,
and all 13 failures had headroom ≤ 16% (median 3.2%) while the two groups' timing
reproduction error was indistinguishable (0.51% vs 0.75%). The failures were a
property of the scale, not of the measurement.

| headroom | workloads | failing |
|---|---|---|
| < 5% | 16 | 11 (69%) |
| 5–10% | 9 | 1 (11%) |
| 10–25% | 23 | 1 (4%) |
| ≥ 25% | 171 | **0** |

`verify_anchor.py` now separates **failing** from **undecidable**, with the threshold
derived rather than chosen: `h_min = 0.5 × median(retime_error over workloads with
≥25% headroom) / tolerance`. Estimating `eps` from the well-conditioned workloads
only, rather than per workload, is what stops it being circular — otherwise a broken
measurement would excuse itself by being noisy.

The direction of that estimator matters more than it looks. A *pessimistic* precision
estimate makes `h_min` larger and therefore exempts **more**, which is the unsafe
direction. Using the p90 was tried and gave `eps = 4.0%`, `h_min = 67%`, exempting 89
of 219 including workloads at 60% headroom that were passing perfectly well — an
exemption wide enough to hide anything. The median gives `h_min = 10.5%` and exempts
25 of 219, all in the 2.6–9.6% band.

It does not rubber-stamp the run: one workload still fails, at 16% headroom with a
2.04% reproduction error, so the publication gate stays shut. Undecidable workloads
are excluded from the gate and reported, never counted as passing.

Whether master's own 336/349 residue is the same phenomenon is untested here — a
1.16× reproduction error is 16%, which no achievable headroom would excuse, so under
this criterion `018_mla_paged_decode` would still fail. The criterion says when the
question is answerable; it does not answer it.

**What was not run: the agent baseline.** Upstream's median SOL of 0.732 and
its r = 0.981 headroom correlation are results about a kernel-optimizing
agent's submissions. The four PyTorch formulations here cluster around `T_b` by
construction, because `T_b` is defined as the fastest of them, so the pooled
correlation over them (0.175) is computed over a sample that barely varies and
is not a failed replication of upstream's number — it is not that experiment.
`artifacts/09/agent-baseline.json` records the decision and what a replication
would need.

**What is still not replicated: the agent baseline.** Upstream's median SOL of
0.732 and its r = 0.981 headroom correlation are results about a
kernel-optimizing agent's submissions over the whole benchmark. The four
PyTorch formulations here cluster around `T_b` by construction, because `T_b`
is defined as the fastest of them, so the pooled correlation over them (0.175)
is computed over a sample that barely varies and is not a failed replication of
upstream's number — it is not that experiment.

Four agent runs have since been made. Three of them — `pilot8` (8 problems),
`glm-run1` (24) and `opus5-budget100` (4) — are not that experiment, because a
median over problems chosen for a pilot is not comparable to a median over 220.

`glm-sweep-2` covers all 220, at mean S 0.5921 over 3,690 scored workloads, and
it is the closest thing here to upstream's experiment. It is still not a
replication, and coverage was never what stood in the way: upstream's 0.732 is
a different model, on a different part, against bounds derived for that part.
Both numbers are medians of the same formula over score curves whose T_SOL and
T_b were derived independently, so their difference is not a measurement of the
two models. See [`agent-baseline.md`](agent-baseline.md) for what the runs did
establish, which is coverage, cost, GPU occupancy, and now twelve bounds that a
real kernel beat.

## 9. Deferrals

State the count honestly and identically everywhere. See
[`artifacts/deferred.json`](../artifacts/deferred.json) for the machine-readable
list and `artifacts/09/manifest-v1.json` for what the manifest actually
contains.

**235 in the dataset, 15 deferred, 220 scoreable, 3717 workload instances.**
Those numbers appear in this document, the README and the manifest, and all
three read them from the same file. `build_manifest.py` prints every problem
that is neither scoreable nor deferred and it currently prints none.

Two further limitations that are *not* deferrals but should be read alongside
them: five backends (`ck`, `ck_tile`, `hipblaslt`, `miopen`, `aiter`) are
accepted by the schema and have never been built through — see
[`backend-coverage.md`](backend-coverage.md), which also lists the three
defects the one `hip_cpp` seed found — and the one full-benchmark agent run
(`glm-sweep-2`, 220 problems) carries **no cost figure**, because its gateway
reports GLM-5.2 at $0.00 and 420M tokens were not free. The runs that are
priced cover 4 to 24 problems.

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

## 10. Bounds that are known to be wrong

Not a deferral — these problems *are* in the manifest and *do* produce scores.
The scores are not usable, and saying so here is the only thing that stops them
being read as ordinary results.

A `T_SOL` is a lower bound: nothing can beat it. Three have been beaten, by
correct kernels, on real hardware. That is not a kernel being exceptional; it
is proof the bound is wrong.

| problem | beaten by | found by |
|---|---|---|
| `FlashInfer-Bench__019_mla_paged_prefill` | 25 of 38 workloads | `pilot8` |
| `L1__005_conv_gated_projection_with_causal_conv` | 1.09–1.15×, 4 of 16 | `glm-run1` |
| `L1__035_flux_ada_layer_norm_zero_modulation_extraction` | 1.003–1.013×, 2 of 16 | `glm-run1` |

**The first has a known mechanism and a known blast radius.** The
declared-traffic tier prices a paged KV cache at its full allocation, while a
kernel that honours the page table gathers 34 pages out of 989,669. Six paged
FlashInfer problems and **249 scoreable workloads** are exposed by that
mechanism, whether or not a kernel has yet demonstrated it on each. `STATE.md`
D18 has the derivation; the v1.1 fix derives paged traffic from the page table,
not from `num_pages`.

**The other two are not paged attention and are not explained.** `L1__005` is a
compute-bound SOLAR roofline that is roughly 15% too slow — a rate or a missed
fusion, and a real defect. `L1__035` is beaten by 0.3–1.3% on a problem whose
total headroom is `T_b / T_SOL = 1.008`: there is almost no scoring range there
at all, so a 1% timing difference flips it either way. Whether that is a wrong
bound or a bound too tight to be measurable against is undecided, and the two
need separating before v1.1. `STATE.md` D21.

**Why the `T_SOL <= T_b` gate did not catch any of them.** `T_b` comes from a
PyTorch reference that over-reads in exactly the same way the bound over-counts,
so both numbers move together and their ordering stays valid. It takes a kernel
that *avoids* the traffic to separate them. That is a general property, not a
one-off: a self-consistent bound and anchor cannot detect a shared error, and
only an independent implementation can. It is the strongest argument in this
repo for running an agent against a benchmark before publishing it.
