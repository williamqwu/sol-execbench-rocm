# TODO — MI350X: what is left to fix

**Scope: MI350X only** (`gbt350-odcdh1-a08-1`, F_LOCK 1300 MHz). This is the
single list of owed work for the part the board serves. For the other part see
**`docs/TODO-MI355X.md`**, which since 2026-08-15 is a runbook *and* that part's
own owed-work list — it is measured, at manifest v4, and its numbers are not
these. Conflating the two is the trap its own §2 and §4 warn about, and PR #2's
MI355X clock finding is the exact opposite of MI350X's (see N6). The one item
that crosses both parts is **B2b**.

Everything here is a known gap, deliberately left rather than forgotten. Nothing
in this repo silently pretends to be finished.

**This file absorbs the former `docs/TODO-MI350X.md` (written 2026-08-12), which
is deleted.** Nothing from it was dropped; nothing from the older topic-ordered
`TODO.md` (rewritten 2026-08-06) was dropped either. The ordering principle is
the newer file's: **by how wrong a published number is, not by effort**, with
bugs and decisions kept apart.

Every count below is either computed on 2026-08-12 with the command stated, or
quoted from `STATE.md` with its D-number. Nothing here is restated from memory.
Where a claim has been retracted, the retraction travels with the item, so
nobody rediscovers it. D-numbers are stable identifiers into `STATE.md` /
`docs/findings.md`, not an organising principle — three code comments
(`scripts/guard_authoritative_gpu.py:8`, `:226`, `scripts/gpu_exclusive.py:7`)
cite **`TODO.md` D29** by filename, which is item 4.15 here.

Item ids (`B1`, `D-a`, `4.7`, `C2`, `U3`, `N5`) are stable; cite those.

---

## 1. Where MI350X actually stands

220 of 235 problems are scoreable (15 NVFP4 Quant problems deferred under the
sanctioned task-07 contingency; `artifacts/deferred.json` states
`dataset_total 235, deferred_total 15, shipped_total 220`), 3717 scoreable
workloads, all ten task gates pass except task 03's `check D`, which reads the
frozen v1 manifest by design (N2). The board serves
`artifacts/09/manifest-v1.2.json`.

Re-run 2026-08-15: **03 → 14 checks, 1 failed** (`144 of 7840 measured workloads
are faster than T_SOL, worst 0.27x, across 15 problems`; 25 of 7840 across 5
problems against `manifest-v1.2.json`), **06 → 11 checks, 0 failed**, **09 → 9
checks, 0 failed**. Those three are the standing regression check for any change
that touches shared code. The older "31 of 519, worst 0.29x" figure in `STATE.md`
was taken over a smaller score population and is superseded, not contradicted.

The one caveat that outranks everything else in this file: **the divisor of every
score — `T_b`, the anchor — is not known to reproduce.** 626 of the 3717
published anchors are labelled as a compile variant and carry an eager latency
(D50), the re-sweep that measured the size of that defect showed ~1025 compile
passes that never happened (STATE.md D59), and the one attempt to check whether a
published anchor reproduces at all was **retracted** because the card was not
exclusive (STATE.md D61). Until that is settled on a verified-exclusive GPU 0,
every `S` on the board is arithmetic performed on a number nobody has re-derived.

---

## 2. Blocking, in order of how wrong the number is

### B1. The anchor — `T_b` is mislabelled, and may not reproduce at all

**Statement.** The published anchor is wrong on 626 rows by label and unverified
everywhere. Two facts, one measured and one unknown, ranked first together
because `S = 1/(1 + (T_k − T_SOL)/(T_b − T_SOL))` divides by this.

**Measured (D50).** `torch._dynamo.config.recompile_limit` defaulted to 8 while
`reference/tb-candidates/variants.py:63` built **one** module-level compiled
callable per problem and reused it across every workload, so past the eighth
distinct shape dynamo logged the limit and **silently ran the frame eagerly**
(`fail_on_recompile_limit_hit = False`). 225 of 235 problems have ≥9 distinct
shapes, so **2061 of 3957 workloads were never compiled**.

* The signature is unmissable once you look for it — `L2__009`, indices 0–7
  `FAIL`, indices 8–15 `pass`. Index 8 re-run alone in a fresh process fails
  outright (`mr = 0.428`). Pre-cliff fail rate 522/1776 = **29.4%**; post-cliff
  1/1941 = **0.05%**.
* **523 is a floor, not a count.** Every one of the five problems examined fails
  *all* its workloads when actually compiled.
* **626 of 3717 anchors in `artifacts/06/authoritative` are labelled
  `v2_compile`/`v3_compile_max_autotune` and carry an eager latency.** On the 538
  post-cliff workloads with a recorded compile time, `T_b / compile` is p50
  **1.000**, p90 1.008, max 1.163 — eager times wearing a compile label. The true
  compiled time on those workloads has never been measured, in either direction.
* In the served database **1642 of 3717 scoreable workloads name a compile
  variant as their anchor** (`823` v3, `819` v2); 626 of those sit past the
  cliff, so the board serves a formulation that did not run on that row.

**Scope the board claim precisely, because the obvious check misleads.**
`t_b_variant` renders in `problem.html:251` **and nowhere else in the templates**
— one column, one page type — and it is one of six `c-deriv` derivation columns
hidden by a switch that is **off by default** (`problem.js:80`; `on =
localStorage "cols.deriv" === "1"`, so `no-deriv` applies and `style.css:641`
sets `display:none`). The markup is served on every problem page regardless, and
`/api/v1/problems/{key}` hides nothing — `models.py:191,305` declare it on both
the workload and run models. So *the board asserts something untrue on 626 rows*
holds; *a reader sees it by default* does not. *Open the page, fail to find the
column, and conclude this entry is stale* is the trap. The fix belongs where the
anchor is **stamped**, in the manifest — not in the template. Hiding or blanking
the column would be the board covering for the manifest, which is the one thing
this board is built not to do.

**Status.** Code fixed: `reference/tb-candidates/variants.py:74-76` now sets
`recompile_limit = 256`, `accumulated_recompile_limit = 4096`,
`fail_on_recompile_limit_hit = True` (committed at `4e25a419`). The 235-problem
re-sweep under the fix is in `artifacts/12/tb-recompile-fix/`; STATE.md D59
records `v2_compile 3132 → 2633 (−499)` and `v3_compile_max_autotune 3000 → 2474
(−526)`, 67 problems losing v2 passes and none gaining. **No `T_b` has been
re-selected. Nothing on the board has moved.**

**Unknown (D59/D60, retracted by D61).** STATE.md D59/D60 reported `L2__057`
`v1_eager` re-timing at 2.021× its published anchor on "solo GPU 0". **D61
retracts that.** Two foreign tenants (`sglang::scheduler` pid 2617421, 194 GB
resident; `ray::MegatronTrainRayActor` pid 2638981) started at 18:39 and 18:40 on
2026-08-12 and both re-times (19:32, 20:00) postdate them; `gpu_exclusive.py
--gpu 0` reported `NOT exclusive — 2 foreign process(es)`. **Do not quote 2.021×
as a reproduction gap.** What survives D61, in its own words: the old-SHA/new-SHA
agreement (their p50 ratios against the published anchor, 2.021 vs 2.021; raw
times 88.3389 and 88.0202 ms differ by 0.36%), so the harness change between
`ea94b186` and `dd88de94` (D38 included) is exonerated; and the pre-tenant 17:55
7-wide reading of 76.83 against a published 31.35, which is **2.45× and still
unexplained**, though taken under this session's own sweep load.

**Related, and the same question one level down: the task-06 sweep does not fully
reproduce (D28).** Re-running the 8 problems whose candidate sweep left a variant
short, **10 of 32 variant×problem cells changed verdict** — 6 went from failing
to passing every workload at identical coverage, 1 went from 9 passes to a
timeout. `Quant__011`'s recorded `passed=0 over 3 workloads` was a driver that
died after three, not a variant that failed. Those 8 were *selected* for being
incomplete, so this is not a rate for the whole sweep. New artifacts in
`artifacts/06/candidates-gapfill/`, kept out of `candidates/` because a new
timing there can move `T_b` and every score under it.
*Self-reversal, 2026-08-11:* this used to say the 523 `v2` and 571 `v3`
`INCORRECT_NUMERICAL` failures "have not been shown to be stable". They
reproduce: five problems re-run per-workload in fresh processes fail **every**
workload, including the ones the sweep recorded as passing — because the recorded
passes were never compiled at all (D50 above). The direction of the doubt was
backwards. 523 is a floor, the mechanism is real, and it is safe to describe it
as torch.compile disagreeing with eager — while noting, per D-a, that on the
cases adjudicated against a float64 golden it is *eager* that is the less
accurate of the two.

**Next step, in this order and no other.**
1. Re-time `L2__057` `v1_eager` solo on a *verified-exclusive* GPU 0 and settle
   whether the gap is real.
2. Re-time `L1__085_geglu_activation`, the 0.097× extreme that moves the other
   way and is the better discriminator (STATE.md D60).
3. Only then run `scripts/authoritative_tb.py` to re-select `T_b`. Running (3)
   first launders an unexplained number into the manifest.

The D50 half is a bug, not a methodology question, and it is the one item here
that can be fixed without touching how anything is measured: re-run the candidate
sweep 8-way under the fixed limit and the authoritative re-time on GPU 0. **Do it
first and alone** — it moves the failure count *up*, the opposite direction from
a tolerance fix, and bundling the two makes both illegible afterwards. **~5.5 h
on GPU 0.**

**Do not treat as blocking:** pass/fail counts from D56/D59. Correctness does not
depend on what else is on the card, and D61 says so explicitly.

### B2. The bound — the declared-traffic tier, and the single-clock divisor

**Statement.** Two diagnosed defects, neither corrected, both wrong in the
generous direction, and they overlap.

**Traffic tier (D18/D42).** The tier prices every declared input at its full
allocation regardless of what the kernel reads; the paged-attention case prices a
KV cache at full allocation where the kernel gathers 34 pages of 989,669.
Computed 2026-08-12 from `artifacts/11/bad-bounds-v12.json`
(`traffic_tier_blast_radius`): **328 of 3717 scoreable workloads (8.82%) across
38 problems**, ratio traffic/SOLAR p50 **1.50**, p90 **5.005**, max **128.9**
(`L1__087`). v1.1 fixed it for two paged problems rather than at the tier, which
is why the count keeps coming back. The original D18 exposure table was 6
problems / **249 scoreable workloads**.

**Single-clock divisor (D35).** `T_SOL_ms = cycles / F_LOCK`, and 1300 MHz is the
clock the card holds under a dense bf16 matrix-core load and not under
everything. Light fp32 work clocks **1439–1586 MHz**, so those bounds are 10–22%
too large. **All 759 compute-bound fp32 workloads are scored against a bound
10–22% too generous.** The controls are the argument: three kernels that have
*never* been beaten clock just as high, one higher than any violation on the
board — so the beaten list is not the set of inflated bounds, it is the visible
part of it. Corroborated independently by this session's telemetry (STATE.md
D55: `L2__036` 1584–1586 MHz, `L1__002` 1438, `L1__074` 1444–1446).

**The five known-beaten bounds under v1.2, all diagnosed (D42), none corrected:**

| problem | min T_k/T_SOL | cause |
|---|---|---|
| `L1__018` | 0.312 | declared-traffic tier: the whole 262,144-slot k+v cache priced read+write |
| `L2__045` | 0.357 | SOLAR prices 21.5× of arithmetic the reference discards, partly masked by fp32-priced-at-bf16 |
| `L1__042` | 0.839 | declared-traffic tier: two inputs `run()` never reads — exactly 65/32 |
| `L1__057` | 0.871 | declared-traffic tier: the 157,184-row embedding table |
| `L2__073` | 0.992 | not a modelling error; the D35 clock residue |

**Three of the five are one defect, and it is D18.** D42 establishes the
mechanism exactly, by hand computation landing on the manifest number. It does
not establish the replacement, and a bound must not be "fixed" by adjusting it
until the violation disappears.

**Read the list itself as a warning.** "A kernel beat its T_SOL" is one symptom
with at least three causes — an over-counted traffic term, a wrong clock divisor,
and a measurement that missed work — and the symptom does not say which. Nor does
it catch the same defects when they are too small to break the invariant:
`L1__055` was undercounted 25% by D38 and appeared on no list, because a kernel
scoring 0.57 can be inflated 10% without passing 1.

**How the count moved (history; the board serves v1.2, the frozen v1 still
reports 13).**

| step | correction | where | left |
|---|---|---|---|
| v1.1 | paged-cache traffic (D18) + per-datapath clock divisor (D35) | D36 | 6 |
| — | **not a bound at all**: the timer did not see work a kernel put on its own stream, so `L1__054`'s *time* was 32% short against a correct bound | D38 | 5 |
| v1.2 | SOLAR priced a grouped convolution as a dense one | D37 | 3 |

Then coverage put two back: `gpt-5.6-sol` finishing all 220 problems reached
`L1__018` and `L1__042`, which nothing had ever run hard enough to expose — the
count is 5. The historical thirteen, with the run that found each:

| what | scope | where |
|---|---|---|
| **Paged-attention T_SOL over-counts traffic** (declared-traffic tier). | 6 problems, **249 scoreable workloads** | D18 |
| **`L1__005` bound beaten by 1.09–1.15×.** Compute-bound SOLAR roofline ~15% too slow; not paged. Cleared by v1.2/D37. | 4 of 16 workloads | D21 |
| **`L1__035` bound beaten by 1.003–1.013×.** Total headroom 1.008, so almost no scoring range — may be a bound too tight to measure against rather than a wrong one. Cleared by v1.1/D36. | 2 of 16 workloads | D21 |
| **Nine more bounds beaten** as a real optimizer covered the benchmark: `L1__006`, `L1__057`, `L2__030`, `L2__035`, `L2__045`, `L2__068`, `L2__073`, and — once coverage reached 220 — `FlashInfer-Bench__018` and `L1__054`. | 72 workloads across 11 problems in that run | D31, D31b |
| **`L2__051` bound beaten**, found by `gpt56-40` on 2026-08-09. | 4 workloads | D31c |

**v1 marks three of these thirteen**; the other ten are marked nowhere in the
shipped manifest and are known only from this file, `STATE.md`
D31/D31b/D31c/D36/D37/D38, and the run pages that found them.

The count rose **3 → 10 → 12 → 13** as stronger optimizers covered more of the
benchmark, which is the finding and not an accident of one run: a bound is only
shown wrong by a kernel that beats it, so the number of *known* bad bounds tracks
how hard anything has tried. `L2__051` sharpens it, because it is not more
coverage finding more — it is **less** coverage finding more. `gpt56-40`
attempted 40 problems where `glm-sweep-2` attempted 220 and still turned up a
bound the larger run missed. Any estimate of how many bad bounds remain that is
extrapolated from one model's sweep is extrapolating from one direction.

**Next step.** *For the tier:* derive the replacement traffic model — what the
kernel reads, not what it declares — and apply it **at the tier, not
problem-by-problem**. Note the coupling with B4: pricing a KV cache at the pages
it gathers took `FlashInfer-Bench__018` from 185,274 cycles to **8**, which is
correct and vacuous, so the tier fix and the missing arithmetic term are one
decision.

*For the clock divisor:* the machinery exists and is merged.
`scripts/sol_bounds.py:414-417` emits `compute_cycles`, `mac_per_cycle` and
`dram_byte_per_sec` alongside the max, and `src/solexbench_rocm/t_sol_at.py`
re-evaluates a bound at an arbitrary clock (both landed with PR #2 at
`697749f0`). It does **not** apply retroactively: `artifacts/03/t_sol.json` holds
**3739 workloads, of which 2998 carry `memory_bytes` only and 741 carry none of
`t_sol_at.REQUIRED_FIELDS` (`compute_cycles, memory_bytes, dram_byte_per_sec`);
zero carry `compute_cycles`**, so every existing record raises
`MissingBoundTerms`. `sol_bounds.py` must be re-run, which costs no GPU (SOLAR
runs on `device="meta"`). And note what re-clocking cannot reach: the same count
gives `bottleneck` = memory 1163 / compute 1835 / absent 741, and a memory term
is invariant in milliseconds — re-pricing at an observed clock moves the
compute-bound half only.

### B2b. What MI355X fixed on 2026-08-15 that MI350X still carries

**Statement.** Three bound defects were corrected in shared code and shipped in
`artifacts/09-MI355X/manifest-v4.json`. **The MI350X release artifacts were
deliberately not regenerated** — that is a version cut, not an auditor's edit —
so MI350X still publishes the pre-correction numbers. The code is fixed for both
parts; only the artifacts differ. Verified this session: `git status
artifacts/03 artifacts/09` is empty and both MI350X manifests are untouched.

**(a) The causal-mask stream over-count — D64.** The declared-traffic tier
charges a full read of `q` on `FlashInfer-Bench__014` and `__015`, where the
reference's own empty-window skip leaves **1–25 query rows live out of
10,447–16,384**. The corrected price is ~1.96x lower. On MI355X this deleted 4 of
5 check-D falsifications. **MI350X exposure: the same 2 problems, 68 workloads**
— `check A-published`'s floor and the published bound on those rows are both
computed from the over-count today. This is the same class as D18 one step
further out: a masked **stream**, not a gathered **allocation**.

*Blocked on:* a v1.3 version cut. Do not regenerate `artifacts/03/` or
`artifacts/09/` outside one.

*Caution that travels with it:* on these workloads `required_matched_ratio =
0.99` against 0.010–0.158% live rows, so **lowering the bound makes a degenerate
`(0, -inf)`-fill kernel score better**. Fix the correctness gate in the same cut
or the bound correction is a net loss. Raised three times; unaddressed.

**(b) The gathered SOLAR memory term — D66.** MI350X v1.1 *already* ships this
rule (`rebuild_manifest_v11.py`), so nothing is owed on the artifact. What is
owed is the *record*: the mechanism this repository attributed it to was wrong.
SOLAR is gather-aware — measured, `table[idx]` prices 8 rows — and the over-count
on `FlashInfer-Bench__018` is the reference's own full-tensor `.to(float32)`
executed **before** the index. The durable fix is a slice pushdown in SOLAR's
`graph_analyzer`; it is **recommended, not enacted**, and it would apply to both
parts. Note the size of the trade: `__018` under v1.2 is 97.5x looser than the
best kernel ever written for it, so the correction moved a detectable error to an
undetectable one on purpose.

**(c) The direction rule everything in this file was reasoned with — D69.**
`dS/dT_SOL = (T_b − T_k)/(T_b + T_k − 2·T_SOL)²`. A bound moving **down deflates**
`S` for any kernel faster than `T_b`, which is 74.5% of the measured MI355X
corpus, and inflates it only for one slower. Several items in this file and in
`docs/findings.md` were written under the unqualified "too small ⇒ inflated"
form. **The ranking of this file does not change** — the too-small direction is
still the dangerous one, because nothing can detect it — but any sentence that
justifies an item by "and it inflates every score" is wrong and should be
re-derived before it is quoted at a maintainer.

**Acceptance for (a).** A v1.3 MI350X manifest in which
`FlashInfer-Bench__014/__015` publish at the live-row price, `check D` is re-run
and reported, and `artifacts/deferred.json` plus every count that states 220/3717
is re-checked against it. Nothing may be regenerated piecemeal.

### B3. What passes — three tolerance defects, fixed in code, zero artifacts moved

**Statement.** Each is task 05 failing to do what it already says it does, so
none is a methodology change. All three were fixed in code on 2026-08-12 (commit
`2e653fd9`) and **none has reached a published number**, because closing them
needs a GPU sweep of `calibrate_tolerance.py`, then `apply_tolerances.py`, then
task 06 for the affected problems, then a manifest rebuild.

**D52 — a zero tolerance leaks from an integer output onto float outputs.**
`_dtype_floor`'s `except TypeError` path returned `{"atol":0.0,"rtol":0.0}` for
int/bool, and that zero was applied to the problem's *float* outputs too.
`L2__049` and `Quant__011` are **unpassable by construction**: `L2__049`'s
`topk_idx` is bit-identical (no routing flip) and `topk_weight` is off by exactly
one fp32 ULP on 4488/16384 elements, `mr = 0.726`, and a search to `k = 2^20`
finds no multiplier that passes. **76 workloads, ≥11 failures.** Of the 76, **32
are the defect** (`L2__049` 16, `Quant__011` 16 — the only two that mix an int64
output with a float32 one); the other 44 (`L1__058` 16, `L1__028` 12, `L2__006`
16) are all-integer problems where a zero band *is* exact equality and is right.
*In-place correction of an earlier reading:* those three pass `v1_eager`,
`v2_compile`, `v3_compile_max_autotune` and `v4_contiguous` on every workload and
**fail `v5_compile_contiguous` on every workload** — 16/16, 12/12, 16/16, all
`RUNTIME_ERROR` from a torch-dynamo `_wrap_fx_proxy` traceback, so those failures
are compile failures, not the zero band rejecting a correct answer (recomputed
2026-08-12 from `artifacts/06/candidates/`; v5 also fails 16/16 on `L2__049` and
`Quant__011`). Every `INCORRECT_NUMERICAL` failure among the five is on the two
mixed-dtype problems: `L2__049` 8 of 16 under `v2_compile` and the *same* 8 under
`v3`, `Quant__011` 3 of the 3 workloads each variant reached — 11 distinct
workloads, which is the "≥11" above.
The data model cannot hold a per-output tolerance — `Workload.tolerance` is one
`ToleranceSpec` (`core/data/workload.py:117`) and `eval_driver` applies it to
every output — so the fix lives in two places instead: `_dtype_floor` derives the
band from the floating-point outputs only, and `compute_error_stats` compares
integer and boolean outputs for exact equality whatever the spec says.
*Still open:* recounted from disk 2026-08-12 over
`artifacts/05/workloads/*/*/workload.jsonl`, **76 workloads still carry
`max_atol == 0.0`**, split `L1__058` 16, `Quant__011` 16, `L2__049` 16, `L2__006`
16, `L1__028` 12 — `artifacts/05/workloads/` still ships the old numbers and
nothing on the board has moved. Re-running `calibrate_tolerance.py` for the five
is a GPU sweep (10 seeds × 2 executions × 76 workloads), then
`apply_tolerances.py`, then task 06 for the two mixed-dtype problems, then a
manifest rebuild.

**D52b — the same leak between two *float* dtypes.** `_dtype_floor` took its
epsilon from `tensors[0].dtype` after dropping the integer outputs, and summed
the RMS scale across every float output. **17 of the 235 problems declare more
than one float output dtype** (16 scoreable, **396 workloads**; `Quant__033` is
deferred — counted 2026-08-12 from the 235
`data/SOL-ExecBench/benchmark/*/*/definition.json` files), and in all 17 a
bfloat16 output comes before the float32 ones — so their fp32 outputs were
floored at bf16's epsilon, **0.0078125 against 1.1920929e-07, 65536× looser than
the dtype justifies**, and with a bit-exact reference the floor *is* the whole
shipped band (`FlashInfer-Bench__012` atol 0.004542 / rtol 0.0078125, `L1__013`
atol 0.010156, `L1__051` atol 0.304172, `L2__044` atol 0.074598, each with
`run_to_run.max_abs: 0.0`). Now each float dtype gets its own epsilon and its own
RMS.
**Be precise about what the fix does and does not buy.** Every one of the 17 is
exactly `{bfloat16, float32}` and bf16 has the larger epsilon, so the single
`ToleranceSpec` still forces one band and the **widest** per-dtype floor is
applied — the min would hold a bf16 output to fp32's epsilon, which is D52's
unpassable-by-construction failure again. **The 65536× over-grant on their fp32
outputs survives.** What the fix removes is the cross-dtype RMS in the atol term
and the silence: `_dtype_floors` and `_floor_over_grant` are written into the
tolerance and carried through `apply_tolerances.py` into
`artifacts/05/workloads/`, so the over-grant is a recorded **bound-quality**
figure of B4's kind instead of an invisible one. Closing it for real is a schema
change — see **D-e**.
*Still open:* the 396 workloads on disk carry the old cross-dtype atol, and the
fix changes no artifact until `calibrate_tolerance.py` re-runs for those 16
scoreable problems (10 seeds × 2 executions × 396 workloads), then
`apply_tolerances.py` and a manifest rebuild. The atol may move in either
direction — same epsilon, differently-grouped RMS — so no claim is made about
which scores change.

**D53 — the float64 goldens compare two different problems.** `gen_golden.py`
seeded and generated inputs on **CPU** while `calibrate_tolerance.py` did it at
`device="cuda:0"`, so the same seed drew different numbers (`L1__067`: 7.096e+00
apart). **2302 of 2331 recorded goldens (98.8%) exceed their own derived atol**
and mean nothing — re-derived independently from `artifacts/05/*.json` at
98.756% over 164 problems, so it holds exactly. The check that would have caught
D-a is computed, written to `entry["vs_golden"]`, and never read.
*Code fixed 2026-08-12; the goldens on disk are still invalid.* The device is now
named once as `_common.INPUT_DEVICE`; `gen_golden.py` draws there and moves the
inputs to the CPU before computing in float64;
`tests/scripts/test_golden_input_contract.py` pins the contract (**27 tests** as
of the 2026-08-12 review pass; all of `tests/scripts` is 61 passed, CPU-only).
`calibrate_tolerance.py` now *records* `vs_golden.comparable` and
`golden_comparable` — recorded, not acted on: no tolerance derivation changed,
and it applies the *same* predicate the generator does
(`_common.golden_stamp_matches`: contract version, device and seed), so a stale
golden cannot be stamped comparable by the reader after the writer has decided to
redraw it.
*Still open:* all **165** `.pt` files (2331 workload goldens, `du -sh
artifacts/golden` = **143 GB**) were drawn on the CPU and remain meaningless until
regenerated. `ls artifacts/golden/*.meta.json | wc -l` = **0**, so none is
stamped and the resume path will regenerate rather than reuse. Regeneration now
needs a GPU for the *draw*; the wall-clock cost of the original run was never
recorded, only that it is 235 problems × float64-on-CPU. A separate owed
measurement rides on it — see **4.14**. This is the only independent check that
could adjudicate D-a, so it is a hard prerequisite for that decision, not a
parallel task.

**Also: run-to-run spread is measured in-process only.** Two executions back to
back hold hipBLASLt/MIOpen algorithm selection roughly constant; across processes
it does not. `L2__051` (see D-a) is the ready-made reproducer.

**Next step.** Regenerate the goldens first (needs a GPU for the draw), then
re-derive tolerances for the D52/D52b problems, then re-run task 06 for
`L2__049` and `Quant__011`, then rebuild the manifest.

### B4. What the score means — 827 bounds with no roofline content (D39)

**Statement.** Every entry above is a bound too **large**. The one automatic
check on a bound is that nothing may beat it, and that check is **one-sided**: a
bound that is too *small* is a perfectly valid lower bound, breaks no rule, and
is reported by nothing.

**Scope.** Recomputed 2026-08-12 from `artifacts/11/bound-headroom.json`
(manifest v1.2, 3717 workloads):

```
headroom T_b/T_SOL over 3,717 scoreable workloads   (p50 15.6x)
  under 2x        504  13.6%     variance is a material share of the score
  2x - 10x       1086  29.2%
  10x - 100x     1300  35.0%
  100x - 1000x    397  10.7%  }  827 (22.3%) where S collapses toward
  over 1000x      430  11.6%  }  T_b/(T_b+T_k) and has no roofline content
```

Five visible bad bounds against 827 quiet ones. Worst by median headroom:
`L2__006` at **115,005×**, then six FlashInfer paged problems at 19,000–41,000×,
`L1__016` at 19,474×. **Two honest caveats.** The paged problems are near the top
*because v1.1 corrected them* — `FlashInfer-Bench__018` went 185,274 cycles → 8,
which is correct and vacuous. That is not an argument for undoing D18; it is an
argument that the arithmetic term for those problems is essentially unmodelled.
And the other tail is real: 504 workloads (13.6%) under 2× is where run-to-run
variance is a material share of the score.

**Status.** Marking is **done**: `bound_quality` (narrow / ok / loose / vacuous)
is derived at ingest and shown beside the headroom figure on every problem page.
It changes no score. **Not done:** deriving an arithmetic term for the worst
cases (work), and deciding whether a problem whose bound cannot be modelled
belongs in the scored set (scope — see **D-c**). And `bound_quality` lives **in
the leaderboard, not the manifest** — see **4.12**.

**A second, independent smell test exists and is nearly free.**
`reference/nvidia-b200/published.json` holds NVIDIA's own per-workload SOL for
all 235 problems, fetched from their public site
(`scripts/fetch_nvidia_b200_reference.py`) and matched to 3,915 of our 3,957
workloads by axes. It is **not** a bound and may not become one — different part,
different arch constants, prime directive 2 — but the *ratio* is a tightness
signal of the kind this section says nothing has:

```
B200 SOL / AMD T_SOL, 3,675 scoreable matched workloads
  under 0.5x     1147     AMD bound the larger    (mostly ok/narrow)
  0.5x - 2x      2139  }  58% agree within 2x, which is what two parts of
                       }  this class should look like
  2x - 10x        243     mostly loose
  10x - 100x      134     70 vacuous, 62 loose, 2 marked ok
  over 100x        12     all 12 already marked vacuous
```

The tail lands almost exactly on what `bound_quality` already marks, which is the
reassuring result. **The exception, adjacent and undiagnosed:** two workloads of
`L2__007_multimodal_rotary_embedding_attention` (`batch_size` 32 and 64 at
`seq_len` 256) are marked `ok` while our `T_SOL` is **11.7× and 13.8× below**
NVIDIA's for the same shape. Both come from the `declared_traffic` tier, i.e. B2.
Not diagnosed, not corrected, and not a reason to change a bound on its own — a
cross-part ratio is evidence to go and look, never a derivation.

**Next step.** Ranked below B1–B3 because these numbers are arithmetically
correct and semantically empty. Take D-c, then derive arithmetic terms for the
worst cases, then 4.12.

### B5. `L2__036` — the grouped-conv fix that did not land (D37)

**Statement.** Six of seven grouped-conv problems were corrected in v1.2;
`L2__036` is a backward problem and SOLAR routes backward graphs through
`graph/backward_processor.py`, which never reaches the conv handler the fix
wraps.

**Scope.** One problem. It has never violated a bound, **and that is not
reassuring**: it is the kernel measured at 1584–1586 MHz (STATE.md D55,
`artifacts/12/clock-ab`), the highest on the node, which is what a kernel not
saturating the datapath its bound was priced against looks like from outside.
The v1.2 correction ratios for the six that did land are on record in D37
(`L1__006` ×768.000 exactly, `L2__035` ×6.70–7.07, `L1__029` ×4.999, …), and
`groups` is recovered from shapes, not from argument parsing.

**Retraction inside D37, which must travel with this item:** the original
7-problem scope came from an AST scan of the whole file; `L2__036` was never in
that scope, and D37's own later text making `L2__036` "the witness that the list
is blind" **is wrong and is disowned by the entry itself**. What survives is the
mechanism above.

**Status.** Uncorrected. **Do not quote any `S` on `L2__036`.**

**Next step.** Reach the conv handler from the backward processor, or price
grouped conv before the graph splits; then re-derive on `device="meta"` and fold
into v1.3.

---

## 3. Decisions a human must make

Nobody should settle these unilaterally. Each is a methodology choice under prime
directive 7, and improvising one invalidates comparison with everything measured
before it.

### D-a. What a tolerance is (D51)

**Statement.** `calibrate_tolerance.py` derives `atol`/`rtol` from the spread of
the same reference run twice **in one process**, floored at dtype epsilon. That
spread is exactly `0.000e+00` on **3581 of 3717 workloads (96.3%)**, so **3502 of
3717 tolerances are pure floor** — roughly one ULP — and the gate then demands
99% of *elements* inside it. That is a bit-identity-with-eager test, and the
reference itself cannot always pass it:

> `L2__051`: all 16 workloads measured run-to-run `0.000e+00` and got
> `atol ≈ 1.198e-07`. **`v1_eager` — the unmodified reference — then fails 10 of
> those 16** in a fresh process.

The re-sweep saw the mirror image, 6 passes before and 16 after
(`artifacts/12/tb-recompile-fix/comparison.json` — `L2__051` is the only
`v1_eager` mover in 235 problems).

Four Inductor behaviours produce the divergence, each proved causally: reduction
re-association, FMA contraction, elided intermediate dtype rounding
(`emulate_precision_casts=True` restores **bit-identity** on `L1__062`, `L2__058`
and `Quant__004`), and — max-autotune only — a hipBLASLt→Triton GEMM template
swap.

**The awkward part, which should decide what happens next:** against a float64
golden, eager is **not** the more accurate implementation on any case
adjudicated. `L1__062` compiled is *bit-identical to the correctly-rounded bf16
golden* and eager is not. `L2__058` compiled RMS 2.11e-03 vs eager 2.78e-03.
`Quant__004` compiled 0.923×. `L2__009` and `L1__067` tie. One control went the
other way (`L1__074`, by 0.009%). And the reference misses its own gate against
truth where checked: `L1__067` eager `mr = 0.0899`, `L2__058` eager 0.9869.
**Coverage is thin and the claim is bounded accordingly:** 6 problems, ~10
workloads of 523. A direction, not a distribution. Nobody should widen a
tolerance on the strength of it without the re-measurement below.

**Scoring exposure.** 70 problems lost both compile anchors, so **1115 of 3717
workloads (30.0%)** are anchored to eager-class PyTorch. On the 39 workloads
genuinely compiled *and* passing under a problem-wide-disqualified variant, the
published anchor is **p50 2.02× and up to 6.29× slower** than the compiled time
that was discarded; re-scoring the real submissions moves mean `S` by **−0.026 to
−0.058**. The other 1076 are extrapolation — compile never both compiled and
passed there, and `time_tb_candidates.py:121-123` records a latency only for a
workload that PASSED, so no rejected workload has a time.

**Options.**
1. *Leave it.* Cost: the exposure above stands. Implies the benchmark's baseline
   is systematically slower than PyTorch's own best and the scores are
   correspondingly generous.
2. *Re-derive tolerances against the four formulations rather than against
   reseeding.* Cost: a GPU sweep, plus this is precisely the "loosen a tolerance
   so a kernel passes" move task 05 forbids. Only defensible with a **ceiling
   rule** — e.g. never exceed some fraction of the reference's own measured
   distance from a float64 golden — **written down before the sweep starts**, not
   chosen after seeing the numbers (prime directive 7). Implies the goldens must
   be valid first (B3/D53), so this cannot be executed today even if taken today.
3. *Measure cross-process spread and derive from that instead.* Cost: doubles the
   derivation sweep. Implies admitting in-process spread was never the right
   estimator, which reopens every tolerance in `artifacts/05`.

Full argument and option set in `artifacts/11/compile-diag/REPORT.md` §7. This is
the maintainer's call and no recommendation is made here.

### D-b. Whether to re-select `T_b` and cut v1.3

`scripts/authoritative_tb.py` is the only thing that can re-select `T_b`, and
every fix above terminates in a manifest rebuild.

**Options.**
1. *Re-select now.* Cost: ~5.5 h on GPU 0 for the compile re-time plus the
   re-derivation sweeps. Implies: if B1's reproduction question is still open, an
   unexplained number is laundered into the manifest and becomes the new baseline
   — STATE.md D59 says do not.
2. *Settle B1 first, then re-select, then cut v1.3 carrying the tier fix, the
   tolerance re-derivation, `bound_quality` in the manifest, and `L2__036`.*
   Cost: the board serves known-wrong anchors for longer. Implies one coherent
   version boundary instead of three partial ones, and the v1→v1.1→v1.2 pattern
   of fixing per-problem what is really a tier defect does not repeat.
3. *Cut v1.3 with the bound fixes only, leaving `T_b` alone.* Cost: cheapest.
   Implies bounds and anchors move on different schedules, and the `T_SOL ≤ T_b`
   gate is asked to hold across a version boundary where only one side changed.

### D-c. Whether a problem whose bound cannot be modelled belongs in the scored set

The 827 workloads of B4, and specifically `L2__006` at 115,005× median headroom.
Deriving an arithmetic term for the worst cases is work; dropping them is scope.
Either is defensible; neither is a peer's call.

### D-d. The re-sweep's timeout budget

`artifacts/12/tb-recompile-fix/logs/sweep.log` records `done: 233 ok, 2 failed,
239.9 min`, both failures `timeout after 5400s`
(`L1__094_time_decay_exponential_stabilization`,
`FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`). These are a
real cost of the B1 fix, not flakes: before it, `L1__094` was fast precisely
because it stopped compiling after the eighth shape. `check_coverage.py` counts a
recorded timeout as covered, which is correct and is not the same as measured.
Somebody has to choose the budget for the next full sweep. See also U3.

### D-e. Per-output tolerances

Closing D52b's 65536× over-grant for real needs a per-output tolerance in the
data model — `Workload.tolerance` is one `ToleranceSpec`
(`src/sol_execbench/core/data/workload.py:117`) and `eval_driver` applies it to
every output. That is a schema change and a methodology decision, not a patch. Do
not improvise it.

### D-f. Two accepted, documented positions

Listed so nobody reopens them as bugs.
* **The submission service has no sandbox.** `env/solb` is a reproducibility
  boundary, not a security one: submitted kernels run as the invoking user with
  the repo bind-mounted read-write. Authenticated internal users only; do not
  expose the port. `leaderboard/submit.py` states the boundary in full.
* **Agent evaluation times the reference on every call.** `agent_eval.py` sets
  `benchmark_reference=True` so the agent can see its speedup. Measuring the
  reference once per sandbox would roughly **halve every evaluation**, which is
  what limits agentic optimization on the expensive FlashInfer problems. Changing
  it mid-programme would make runs incomparable, so it is a change for a clean
  batch.

---

## 4. Bugs anyone can fix

Each has a file:line and an acceptance check.

**4.1 `scripts/runners/time_tb_candidates.py` does not refuse a non-exclusive
card.** Grepping the file for `exclusive` and `foreign_on` returns **0 hits**
(2026-08-12). The repo already has `scripts/gpu_exclusive.py`,
`guard_authoritative_gpu.py` and `retime_parallel.py`'s `foreign_on()`; invoking
this runner directly routes around all three, which is exactly how D61 happened.
This is the one item here that prevents a future invalid number rather than
correcting a past one — **pull it forward.**
*Acceptance:* running it with a foreign process on the target GPU exits non-zero
without producing a latency.

**4.2 33 committed artifacts carry no provenance.** `artifacts/12` holds 269 JSON
files, **36 without a `_provenance` block**, of which **33** are the D55
clock-A/B evidence (`clock-ab` 10, `clock-ab-satur` 17, `clock-ab-soak` 6); the
other 3 are the two timeout artifacts and `comparison.json`. Convention 7 and
prime directive 5: no ROCm/driver/torch version, no F_LOCK, no git SHA, no UTC
stamp. D55 may well be right; these files are not usable as evidence by anyone
who was not in the session.
*Acceptance:* every JSON under `artifacts/12/clock-ab*` has a `_provenance` block
written by `scripts/provenance.py`.

**4.3 `f_lock_mhz: null` in the roll-up artifacts.** **This is not a missing
clock preset and not a task-01 gate failure.** Earlier versions of this file,
`CLAUDE.md` and `DESIGN-v2.md` all said it was. `CLOCK_LOCK_PRESETS` has carried
`"AMD Instinct MI350X": ClockPreset(gpu_clk_mhz=1600, dram_clk_mhz=None,
achieved_gpu_clk_mhz=1300)` since commit **2cdb7b0** (2026-08-03 20:36 UTC) —
still the literal entry today at
`src/sol_execbench/core/bench/config/device_config.py:165-167` — and
`verify_artifacts.py --task 01` reports *11 checks, 0 failed*. A later field,
`requested_is_achieved` (`:67`, default `True`), governs the *fallback* when
`achieved_gpu_clk_mhz` is absent (`:91`); MI350X has a measured achieved clock so
it never fires, and it is set `False` only on the MI355X entry, which is what
makes that part's `f_lock_mhz` `None` (see N6).

Two real causes, neither of which loses a measurement:
* **(a) Artifacts written before 2cdb7b0 landed.** Everything in `artifacts/00/`
  and `artifacts/01/` was written between 18:53 and 20:30 UTC on 2026-08-03 —
  before 20:36. Provenance shows `torch.available: true` and eight
  `AMD Instinct MI350X` devices, so `get_clock_preset()` was called and returned
  `None` because the table genuinely had no entry yet. **20 files.** They are
  history and are correct as history.
* **(b) Artifacts written by a host-python process, after the preset existed.**
  **8 files:** `artifacts/10/{pilot8,glm-run1,submitted-apitest}/scored.json`,
  `artifacts/10/pilot8/{run,cost-report}.json`, and
  `artifacts/02/timing-{variance-amd,stall-probe,stall-clock}.json`. Provenance
  says `python: 3.11.7`, `torch: {"available": false}`, `rocm.version: 7.15.0` —
  the *host* interpreter, not the pinned container (`python 3.12.3`, `torch
  2.9.1+rocm7.2.0`, `rocm 7.2.0`). `python3 -c "import torch"` on this host
  raises `ModuleNotFoundError`, so `provenance.f_lock_mhz()` falls through its
  `except Exception` and returns `None`. Verified: with
  `SOLEXBENCH_F_LOCK_MHZ=1300` exported, the same host call returns 1300.

Running on the host is **deliberate**. `agent_score.py`, `agent_cost_report.py`
and `agent_baseline.py` orchestrate: they shell each kernel into the container
through `env/solb` and never touch a GPU themselves (`agent_score.py`'s own
comment says so — it loads `sol_score.py` by file path precisely because host
python has no pydantic). The measurements those runs contain are stamped
correctly: every `artifacts/10/*/retimed/*.json`, written *inside* the container
by `agent_eval.py`, carries `f_lock_mhz: 1300`, `python: 3.12.3`,
`visible_devices: "0"` — **all 654 of them, with no other value**. Only the outer
roll-up, which aggregates rather than measures, is unstamped. The leaderboard
header is unaffected: `meta.f_lock_mhz` is `'1300'`, read from the manifest.

**The 28 above is incomplete, not wrong, and the count depends on where you
look — so state the definition with it.** Counting **only a top-level
`_provenance` dict whose `f_lock_mhz` is `None`** — the shape
`build_manifest.py:162` actually reads — gives **37** (artifacts/01 18,
artifacts/10 7, artifacts/11 7, artifacts/02 3, artifacts/00 2); that is the
definition this item uses. Grepping for the string anywhere in any JSON under
`artifacts/` (`grep -rl '"f_lock_mhz": null' artifacts/ | wc -l`) gives **45**;
the extra 8 carry the null in a nested rather than top-level provenance
(`artifacts/01/f_lock_by_datapath.json`, `artifacts/09/manifest-v1.1.json`,
`artifacts/09/manifest-v1.2.json`, and five under `artifacts/11/`). All 45 are
git-tracked. Both halves of the older enumeration are still exactly right —
`artifacts/00` 2 + `artifacts/01` 18 = 20, plus the eight named host-python
files — and **nine further top-level nulls have appeared since**
(`artifacts/10/glm-sweep-2/scored.json`, `artifacts/10/gpt56-220/scored.json`,
and seven under `artifacts/11/`).

**Adjacent, and a different gap:** `artifacts/10/glm-run1/run.json` and
`artifacts/10/submitted-apitest/run.json` have **no `_provenance` block at all**.
They are hand-assembled / worker-assembled run descriptors rather than
`stamp()`ed artifacts, so the question above does not even arise for them. Only
`pilot8/run.json` went through `agent_baseline.py`.

The fix, in order of preference:
1. Have the three host-side scripts export `SOLEXBENCH_F_LOCK_MHZ` (or pass it
   through to `stamp()`) from the F_LOCK the container reported in the
   per-workload artifacts they just collected. That keeps the value measured
   rather than asserted.
2. Failing that, `provenance.f_lock_mhz()` could read the device name from
   `amd-smi` instead of `torch.cuda.get_device_name(0)`, which works in any
   process that can see the card. Note this resolves only the name → preset
   lookup; it does not observe the running clock.

**What would be wrong is defaulting the field to 1300 in `stamp()`.** A roll-up
written on a different part would then claim a clock it was never measured at,
and nothing downstream could detect it — and note `ClockPreset.f_lock_mhz`
deliberately no longer defaults either (N6).
*Acceptance:* the top-level count of 37 drops to the pre-`2cdb7b0` history files
only, and this item's enumeration matches it.

**4.4 `scripts/clock_ab.py:64,70` address GPUs by raw torch index.** `smi()` runs
`["sudo", "-n", SMI, *args, "-d", str(gpu)]` and `perf_level()` runs `rocm-smi
--showperflevel -d str(gpu)`, both with a torch index. The telemetry side is
correct — `scripts/bounds/clock_ab_probe.py:79` resolves through
`gpu_map.torch_to_amdsmi()` — but the *policy* was applied and read back through
the untranslated path. `scripts/gpu_map.py:113` already provides
`torch_to_rocm_smi()`. This matters because D55's finding rests on this driver:
if torch 0 is not rocm-smi 0 on this node, every condition change landed on a
neighbour and the measured card stayed at the standing setpoint for all three
arms, which would produce D55's result as an artefact. D55 has an independent
argument (a setpoint above the achieved clock is inert either way), but the
measurement does not currently establish it.
*Acceptance:* `gpu_map.torch_to_rocm_smi()` printed on this node, the script
routed through it, and D55's three-condition A/B re-run on an exclusive card.

**4.5 `src/sol_execbench/core/bench/device/amd.py:120-122` and `:144` do the same
thing in the harness.** `lock_clocks` shells `rocm-smi --setperfdeterminism <mhz>
-d <gpu>` and `unlock_clocks` mirrors it, both with a torch index. This is the
path that applies the lock during an actual run. PR #2 fixed exactly this trap in
`scripts/clock_calibrate.py` and left it here.
*Acceptance:* both call sites go through `gpu_map.torch_to_rocm_smi()`; a unit
test pins the translation.

**4.6 `scripts/rebuild_manifest_v11.py:284` hides a second F_LOCK literal.** The
module constant is `F_LOCK_MHZ = 1300.0` at `:68`, but `:284` separately opens
with a bare `f = 1300.0` before the per-datapath override and **`:296`**
(`w["t_sol_ms"] = w["t_sol_cycles"] / (f * 1e3)`) divides by it. Editing the
constant alone leaves the literal in place. `scripts/rebuild_manifest_v12.py:72`
has the same constant but **not** the same defect — its equivalent line `:186`
opens with `f = F_LOCK_MHZ` and `:192` divides by that. Both clamp with
`max(F_LOCK_MHZ, measured)` (`v11:207`, `v12:108`), i.e. 1300 is a hard floor on
the per-datapath clock. Neither script asserts a part, so pointing either at a
non-MI350X manifest produces a plausible wrong answer.
*Acceptance:* no bare clock literal remains in `rebuild_manifest_v11.py`, and
both files refuse a manifest whose `_provenance` part is not MI350X.

**4.7 `scripts/verify_artifacts.py:246` globs `floor-gpu*.json` with no part
filter** and feeds `F_LOCK ≤ min(p5)` from whatever it finds. PR #2 has already
put MI355X files (`artifacts/01/unlocked-clock.json`, `burst-clock*.json`, host
`mia1-p02-g10`) into `artifacts/01` beside MI350X ones. The first MI355X
`floor-gpu*.json` dropped there silently gates the MI350X F_LOCK against MI355X
floors.
*Acceptance:* the glob filters on part or host, and a synthetic MI355X floor file
in `artifacts/01` does not change the MI350X task-01 result.

**4.8 `scripts/verify_artifacts.py` has no test coverage.** It is the acceptance
gate for all ten tasks. A bug in it does not fail loudly; it passes quietly.
Highest leverage-per-hour item in this section after 4.1.
*Acceptance:* a test suite that fails when a check is inverted.

**4.9 Two re-sweep variant failures are not results.**
`artifacts/12/tb-recompile-fix/candidates/Quant__023_fp8_mamba2_ssm_discretization.json`
records `v4_contiguous ok:false` with `HIP out of memory. Tried to allocate 32.00
GiB. GPU 0 has a total capacity of 251.98 GiB of which 5.00 GiB is free`, mtime
18:40 — the minute the 194 GB tenant landed. A card with 5 GiB free of 252 is not
this benchmark's card; do not read this as a cost of the B1 fix. The "GPU 0"
*inside* the HIP message is the renumbered visible device, not the physical card:
the artifact itself records `"gpu": "1"` and `_provenance.visible_devices: "1"`,
so this ran on **physical GPU 1** under `HIP_VISIBLE_DEVICES=1`, where it is
device 0 of one. It is not evidence of anything on GPU 0. Separately,
`L2__012_moe_expert_batched_execution_with_capacity_factor.json` — also `"gpu":
"1"` — records `v3_compile_max_autotune ok:false` with `Memory access fault by
GPU node-3 ... rc=-6` at mtime 17:06, which **predates** the tenants and is a real
undiagnosed fault. **Neither appears in STATE.md D59.**
*Acceptance:* `Quant__023` re-run on an exclusive card; `L2__012`'s fault
diagnosed or recorded as unexplained in STATE.md.

**4.10 D59's latency population is partly tenant-contaminated and has not been
recut.** Over `artifacts/12/tb-recompile-fix/candidates/*.json` mtimes: 235
files, sweep spanning **16:14:25–20:13:19**, **90 written at or after 17:55** and
**41 at or after the 18:39 tenant start**. D59's `T_b` ratio population (p05
0.346 / p50 0.995 / p95 1.062) and its `v1_eager` drift table must be recut on
the uncontaminated prefix before either is quoted. Every pass/fail number is
unaffected.
*Acceptance:* the recut population is in STATE.md with its cut time stated.

**4.11 89 problems mix sweep-GPU timings with authoritative ones** (D28 residue).
The D28 ingest bug itself is fixed (2026-08-07): `ingest_variants()` reads the
per-workload `failures` list instead of painting a problem with its `all_passed`
flag, so **1,239 passing baseline workloads** stopped being under-reported —
torch.compile went 0.3414 → 0.4190 on the whole-benchmark scope and max-autotune
0.3174 → 0.4034, and all four variants now show the 220 problems they actually
attempted where two read as 218 and 213. **No measurement changed.** What is
still open is the mixture: 89 problems have passing workloads whose only timing
came off a sweep GPU rather than GPU 0, labelled per row in `note`. **≈2¼ h on
GPU 0** at the measured 1.5 min/problem.
*Acceptance:* `check_coverage.py` plus a board rebuild in which no scoreable row
carries a sweep-GPU timing.

**4.12 `bound_quality` is not in the manifest** (B4). It is derived at ingest
because a manifest rebuild was unsafe at the time (a scorer was reading the
file), so a consumer of `manifest-v1.2.json` alone cannot see it. A v1.3 item.
*Acceptance:* `manifest-v1.3.json` carries the field and `ingest.py` reads it
instead of deriving it.

**4.13 `ingest.py`'s default is lossy** (D24). Without the right flag, every run
kept outside the repo is silently dropped from the board — this was introduced
three separate times and patched at three call sites, once deleting the $250 Opus
run. Current semantics **invert** the old trap: `--agent-runs` *overrides*
`sources.json` rather than adding to it, so passing it is now the way to get the
omission. Partially mitigated by the refusal to publish a board that lost a
submission unless `--allow-drop`. The durable fix is one config the ingest reads
by default, so "rebuild" cannot mean two different things.
*Acceptance:* `--agent-runs` merges with `sources.json`, with a test.

**4.14 D53's owed measurement.** `gen_golden.py --jobs` defaults to 32, its
pre-fix value. The D53 fix briefly dropped it to 1 on the argument that "32
workers each opening a HIP context on one card" is too many — an assertion, never
measured, for a hazard (`GPU 0 is the authoritative timing card`) that
`HIP_VISIBLE_DEVICES` pinning already handles, and one that would have serialised
the float64 *CPU* execution that is the expensive half. Default restored; what is
owed is a measured per-worker HIP context footprint. One minute on **one of GPUs
1–7**, not GPU 0: N workers each calling `torch.randn(1, device="cuda:<that
gpu>")`, read `rocm-smi --showmeminfo vram`. This is a memory-footprint probe,
not a timing run, so CLAUDE.md §4 places it on the exploration cards — but it
must still be the *only* thing measuring on that card while it runs, because a
co-tenant's VRAM lands in the same readout. Check with `scripts/gpu_exclusive.py
--gpu <n>` first; 4.1 exists precisely because an unqualified "one GPU" produced
D61. Until then, lower `--jobs` by hand if the draw OOMs.
*Acceptance:* the footprint in STATE.md, with the GPU index and the exclusivity
check recorded, and a `--jobs` default justified by it.

**4.15 D29 — the external fleet's GPU-0 hold does not hold.** `dash-overlay`'s J2
sweep placed 34 jobs on GPU 0 despite taking a scheduler reservation on it —
placement 34·42·36·35·35·35·36·34 over GPUs 0–7. Nothing published is affected —
no authoritative timing overlapped them — but the property is unenforced, so the
next overlap will be silent. Same class as 4.1. *(Cited from code as `TODO.md`
D29: `scripts/guard_authoritative_gpu.py:8,:226`, `scripts/gpu_exclusive.py:7`,
`scripts/agent_score.py:183`.)*

**4.16 Nothing publishes an external run to the board.** The `dash-overlay` fleet
writes the leaderboard's address into every job's payload, which reads as a
pipeline and is not one: `sbt collect` → `scripts/agent_score.py` → `ingest.py`
are three manual steps, and until all three are run the kernels exist only in
`~/.jobd/jobs/<id>/kernel.py` — **281 of them on disk as of 2026-08-07, none on
the board.**

---

## 5. Cosmetic / low impact

**C1. The leftover `leaderboard/solbench.db`** (D54). **The live board is
correct — this is a leftover file, not a published defect.** *In-place
correction:* the first version of this entry claimed the board was showing
inverted statuses; that was wrong, and the error was querying
`leaderboard/solbench.db` and taking it for what the site reads. That file, built
`2026-08-06T23:25Z` (before the D28 fix landed 08-07), really is inverted — 585
not-`PASSED` rows for `baseline-v2-compile` against artifacts' 523, **intersection
zero**, and all 585 carrying a latency that today's `ingest.py` cannot produce,
so it was built by code that no longer exists. But `part_databases()`
(`leaderboard/app.py:674`) reads `db/solbench-<PART>.db` first and falls back to
the single-file layout only for a part the per-part layout has not produced.
`db/solbench-MI350X.db` exists: `3694` v2 rows, `523` not-`PASSED` matching
`artifacts/06` exactly, `0` of them carrying a latency. Verified against the
running board:

```
GET /api/v1/submissions/baseline-v2-compile/problems/L2__009_...?part=MI350X
  -> {'INCORRECT_NUMERICAL': 8, 'PASSED': 8},  0 failed rows with a latency
```

No re-ingest is needed, and `leaderboard/solbench.db*` is gitignored so the file
has never reached a deploy and cannot. What is left is a **local-only trap with a
tripwire**: wipe `db/` and the fallback serves it, and `run.sh`'s rebuild guard
counts it as "a database" so it would not rebuild — but the freshness check
reports `stale: True, reasons: ['2 input files added since the last build']`, so
a reader gets the banner. Deleting the leftover removes the trap. **Not deleted:
it is untracked and unreproducible, so it is the maintainer's call** — not a
peer's, and not one to take on a peer's say-so in either direction.

**C2. `run.html`'s `.solbar` step edges** (D58). Three places carry a dated
figure and they do not carry the same one: the two tooltips
(`leaderboard/templates/run.html:305` and `:310`) cite only "all 12,883 scored
cells on this board as of 2026-08-06"; the "3,744 cells are exactly 0.5" figure
appears in **one** place, the comment at `:219`. Both are stale — against
`leaderboard/db/solbench-MI350X.db` today the same queries give **21,040** scored
cells and **3,758** exactly 0.5. (The 12,883/3,744 pair also appears in STATE.md's
grid-ramp section; b1's 29.3% share at exactly 0.5 is structural, because a T_b
variant scores exactly 0.5.) Honestly dated, now stale; re-cutting the scale is a
design decision, not a bug fix.

**C3. The stream-join defence charges honest kernels for itself** (D38 residue).
D38's fix fences and joins a submission's own streams around the timed region.
The fence sits outside the bracket; the join cannot, so a submission that creates
streams pays about **6–7 µs per tracked stream per iteration** inside its own
measurement — measured by running the same single-stream computation with 0 and
with 4 streams live (`artifacts/11/side-stream-timing-hole-fixed.json`: probe C,
1 stream tracked, 0.68547 ms; probe C2, same kernel with 4 tracked, 0.70381 ms;
`(0.70381 − 0.68547) / (4 − 1) = 0.00611 ms`). On a 0.7 ms kernel that is ~1% and
invisible; on a 20 µs kernel it is not, and several FlashInfer problems are that
small. It is in the safe direction — over-measuring cannot inflate a score — but
it penalises exactly the submissions that overlap honestly. **Two ways out,
neither free:**
1. *Move to the rocprofiler methodology*, which stamps each window's end after a
   full synchronize and closes this by construction, with no per-stream event
   ops. Task 04 is built and validated. It is not the default because changing
   the methodology of record moves every measurement ever taken, so it needs a
   clean batch, not a patch.
2. *Join only streams with pending work.* There is no API for that; the nearest
   is a per-stream `query()` before the join, which is a host round trip and may
   cost more than it saves. Unmeasured.

*Also unresolved:* the probe could not get a rocprofiler reading at all (`No GPU
activities recorded during discovery iteration`, all six probes, both runs). The
independent confirmation of D38 is the serialized host wall clock, which is
sufficient — it cannot miss a stream — but the shim not working in that script is
its own small unknown.

**C4. GPU-set defaults are hard-coded, each to a different set.**
`scripts/probe_timing_stall.py:103` defaults `--gpus` to `[1, 2, 3, 4]` — four
cards, not this node's card count and not any other script's set;
`retime_parallel.py:121` defaults `0,1,2,3,4,5,6,7`, `shard_sweep.py:139` `1-7`,
`gpu_broker.py:271` `1,2,3,4,5,6,7`, and `verify_artifacts.py:172` hard-requires
`len(gpus) == 8`. All overridable, both target nodes are 8×, no number is
affected.

---

## 6. Unexplained, and owed rather than broken

**U1. D20 — matmul timing is bimodal on MI350X and the cause is unknown.** 0.13%
of iterations cost 3.9–4.5×. The clock hypothesis was tested and **falsified**
(in-call spread 1.04× against a required 3.9×; the clock is steady at ~1450 MHz
through the measurement). hipBLASLt kernel selection is the untested suspect. Two
upstream tests are skipped behind it because their PASS thresholds were measured
on RTX 4090 / B200 and no defensible AMD constant could be derived —
re-specifying them needs the cause.

**U2. `mm[2048]`'s ~21× outlier did not reproduce** — 3 events in 3,600
iterations, then 0 in 12,000.

**U3. D23 — `FlashInfer-Bench__014`'s authoritative re-time timed out at 1200 s**
for `glm-run1`: 24 kernels on disk, 23 results, and a `TimeoutExpired` that reads
as *never attempted*. The board now carries `retime_ok`/`retime_error`. Whether
the budget is simply too small for a paged-prefill problem of that size, or
whether it is another instance of D18's trouble on the same family, was not
investigated. **New corroboration:** the same problem timed out again at 5400 s
in the 2026-08-12 re-sweep (D-d), which points at budget rather than at D18.

**U4. D43 — `rocprofv3 --pmc` hangs in this container**, so the counter path to
an independent traffic measurement is closed: `--pmc FETCH_SIZE WRITE_SIZE`
produces no output and never exits, even on a 3-kernel `a + 1.0`. Nothing was
changed to work around it. **The shim is *not* implicated** — it uses the
dispatch-callback timestamp path, validated at −0.61% median divergence over 1430
pairs, and every measurement in the repo runs on it. The counter-free route is a
minimal independent kernel, timed. This blocks independent verification of B2 —
see §8. *(STATE.md still carries this as `### D43 — BLOCKED`.)*

**U5. D55's residue.** On MI350X the 1600 setpoint never binds: across twelve
loads spanning 305–900 W and 1303–1586 MHz, locked / unlocked / cap-raised-to-2200
give geomean 1.0000 / 1.0012 / 0.9957, with 1.0005 on the saturated round and
1.0024 on the 60 s soak, against a 0.3–1.3% noise floor, and the achieved clock
is identical in all three conditions. But `artifacts/01` has an unlocked GPU 0 at
1390 MHz / 1001 W and nothing in that session exceeded 900 W; duration was tested
and ruled out. So "the cap never binds" is true of everything measured here and
**not proven in general**. See also 4.4 — the driver's SMI calls are not
index-translated.

---

## 7. What is NOT owed

Listed so nobody re-opens them.

**N1. The 15 NVFP4 Quant problems.** Deferred with evidence under the sanctioned
task-07 contingency; verified from `artifacts/deferred.json` (`dataset_total 235,
deferred_total 15, shipped_total 220`, 15 problem entries). Not a gap in the
port: NVFP4 has no ROCm kernel path and an MXFP4 twin is a re-specification, not
a translation. `tasks/07`.

**N2. Task 03's `check D` failure.** It reads the frozen v1 manifest, which is
meant to go on reporting what v1 shipped. The board serves v1.2. A *second* gate
failure is a regression.

**N3. Five backends the schema accepts and nothing has been built through** —
`ck`, `ck_tile`, `hipblaslt`, `miopen`, `aiter`. ~~A coverage gap, not a wrong
number.~~ **Closed 2026-08-14.** One seed per language, each run end to end
through the real packager + eval_driver on GPU 0 of `mia1-p02-g46` and passing
every workload of its problem: `reference/seeds/{aiter,ck,ck_tile,hipblaslt,
miopen}__*.json`, results in `artifacts/backends/`. Two more packaging defects
of the `-lcuda` family fell out of it (`--use_fast_math` reaching clang++, and
the `-lcuda` fix's emptiness test skipping any submission that set a flag) --
both fixed in `ProblemPackager` with a regression test. What is established is
"the path builds and runs", per language, on one well-conditioned problem;
shape coverage and aiter's gluon path are not. `docs/backend-coverage.md`.

**N4. Goldens capped by tensor size.** 165 `.pt` files; the rest of the 235
problems have workloads recorded as `skipped: N elements > cap` in
`artifacts/golden/_report.json`. The report covers all 235 — recorded, not
missing — though a size-capped golden set cannot check the largest workloads,
which are the ones most likely to expose a layout bug, and it compounds B3/D53
because those workloads have neither a valid golden nor any golden.

**N5. A full-benchmark agent baseline.** ~~No full-benchmark agent baseline.~~
**Closed 2026-08-08.** `agent-glm-sweep-2` covers **220 of 220** problems and all
3,717 scoreable workloads: 3,690 scored, **mean S = 0.6083**, 218 problems swept
clean, 0 flagged. It leads the board on the shared denominator (0.5921 against
eager's 0.4536). Upstream's median SOL of 0.732 on B200 is **not** the comparison
— these are AMD-derived bounds and no cross-vendor number comparison is
defensible. Two caveats travel with it: 168 of 220 sessions were stopped by the
harness's 1 h cap rather than choosing to stop, and 3 submitted a kernel that was
mid-edit when the kill landed while a passing snapshot went unused — so the
figure is a floor by about three problems. (A second full run exists:
`gpt-5.6-sol` over all 220, 3,701 scored, mean S 0.6381, benchmark 0.6332 at
99.2% coverage — STATE.md D41.)

**N6. Anything about MI355X.** Nothing is measured there; the port needs nothing
and every measurement needs redoing, `tasks/01` first, since it blocks 03, 05 and
06 exactly as it did here. **`docs/TODO-MI355X.md` is the runbook.** PR #2's clock
finding — that `--setperfdeterminism` costs ~20% of the clock on that part — is
the exact opposite of U5's MI350X result. Neither transfers; the preset table's
requested-vs-achieved split has to be measured per part and never inferred. Also:
`origin/feat/agent-scoreboard` carries **24 commits of MI355X work and is not
merged**, and its data is deliberately absent from the leaderboard because its
`T_b` is not anchor-verified, so it has no `S` to publish.

**The code now enforces the per-part split, and MI350X is untouched by it.**
`src/sol_execbench/core/bench/config/device_config.py` no longer lets
`ClockPreset.f_lock_mhz` fall back to the *requested* clock. `requested_is_achieved`
defaults to `True`, so every NVIDIA entry is unchanged, and is set `False` on the
MI355X entry (`:135-137`) — so `get_clock_preset("AMD Instinct MI355X").f_lock_mhz`
is now **`None`** rather than 1650. MI350X carries `achieved_gpu_clk_mhz=1300`
(`:165-167`) and still returns **1300**. `CLOCK_LOCK_PRESETS` has **five** entries
and `get_clock_preset` was executed on every one, reported as `(f_lock_mhz,
requested_is_achieved)`: MI355X `(None, False)`, MI350X `(1300, True)`, B200
`(1500, True)`, H100 `(1410, True)`, A100 `(1065, True)`;
`tests/sol_execbench/core/bench/config/test_clock_preset_f_lock.py` pins those
cases plus "an unmeasured AMD part cannot be added by accident".

State the downstream consequence carefully, because it inverts easily.
`scripts/build_manifest.py` has **two** F_LOCK guards and they behave
**differently**. The per-artifact comparison at `:162` — `if f_lock_mhz is not
None and measured_at not in (None, f_lock_mhz)` — **admits** a `None` stamp,
because a null there means "this artifact predates F_LOCK stamping", which is
4.3's population and a different problem from being measured at the wrong clock.
The top-level check at `:244` **exits** when `provenance.f_lock_mhz()` itself
resolves to `None`. So on MI355X today the manifest build refuses outright. That
is the intended behaviour, and it is *not* "the guard admits everything".

**N7. `src/solexbench_rocm/activity/`, `parts.py`, the rocprof shim, and the
exploit corpus.** Built, tested, and not implicated in anything above.

---

## 8. The caveat that outranks the list

`CLAUDE.md` §6 states it and U4 keeps it open: **a self-consistent bound and
anchor cannot detect a shared error.** `T_b` comes from a PyTorch reference that
over-reads exactly where the declared-traffic bound over-counts, so the
`T_SOL ≤ T_b` gate passes while both are wrong. Every known-bad bound in §2 was
found by a kernel that beat it — the count went 3 → 10 → 12 → 13 as optimizers
got stronger, and `gpt56-40` found one on 40 problems that 220 problems of
GLM-5.2 missed. So this file is a **lower bound on the defect set, not the defect
set**. Until an independent kernel or a working counter path exists, it stays
one.
