# STATE.md — progress ledger

**Single source of truth for progress.** Update as you go, not at the end. A
session can be interrupted at any point; whatever is written here is what the
next session inherits. Record real output, not summaries of intent; if something
failed, say so and say how; never mark a task `done` without pasting its
acceptance-check output. **This file is short on purpose** — where the port
stands, what node it runs on, where everything else lives. No narrative history.

## D-numbers are stable anchors, and they moved house

D-numbers (`D1`…`D61`, plus `D31b`, `D31c`, `D52b`) are **permanent
identifiers**, cited from 168 places in the code, but they are no longer the
organising principle of this file. **Every entry now lives in
[`docs/findings.md`](docs/findings.md), grouped by topic, each carrying its
D-number as an anchor.** Read a code comment saying *"STATE.md D35"* as
*"finding D35"* — `docs/findings.md#d35`, or the index below. Nothing was
deleted; entries moved, and retractions stayed adjacent to what they retract.
The convention, once: **the ledger records what was believed when; corrections
attach, they do not rewrite.** `D3` is cited by `scripts/clock_ab.py` but was
never defined; the question it names (is the clock lock costing us anything on
THIS node?) is answered by **D55**. `D4` was never used.

> Session 1 ran on `mia1-p02-g10` (8× **MI355X**); session 2 onward on
> `gbt350-odcdh1-a08-1` (8× **MI350X**). **Everything session 1 measured was
> re-measured here** — F_LOCK came back **1300 MHz**, not 1650, a 21%
> difference that would have corrupted every T_SOL and every T_b. Session-1
> numbers survive in `docs/findings.md` and git history, always labelled.

## Where this stands

**The benchmark is measured and the manifest is frozen.** `manifest-v1` scores
**220 of 235 problems / 3717 workload instances** on MI350X at F_LOCK =
1300 MHz. The 15 that are not scoreable are the NVFP4 Quant problems, whose
*references* fail on ROCm; they are in `artifacts/deferred.json` with the error
text quoted from the calibration artifact, and every count everywhere quotes it.

**The board does not serve v1.** It serves `artifacts/09/manifest-v1.2.json` —
v1.1 (D18 paged traffic, D35 per-datapath clock) then v1.2 (D37 grouped conv).
Both re-derive on `device="meta"`; no measurement was repeated.

What a consumer needs to know before using it:

* **Correctness runs against `artifacts/05/workloads/`**, not the dataset's own
  tolerances. Opt in with `SOLEXBENCH_WORKLOADS_ROOT`. Under upstream's B200
  tolerances the same references fail 8 workloads of `L2/033`.
* **`T_SOL` comes from one of two derivations and every workload says which** —
  SOLAR's roofline over the traced graph, or the traffic the definition declares
  over DRAM bandwidth; the manifest takes the max of the two that survive being
  checked against the measurement. **Five bounds are known wrong under v1.2 and
  none is corrected** (D42), and the check is one-sided: nothing verifies that a
  bound is *tight* (D39).
* **`T_b` was intended to be re-timed on GPU 0 alone; two known exceptions.**
  89 problems have passing workloads whose only timing came off a sweep GPU
  (D28), and 626 of 3717 anchors are labelled a compile variant while carrying
  an eager latency (D50). The eight GPUs span 1242–1307 MHz at one setting.
* **An agent baseline exists.** `glm-sweep-2` covers 220/220, mean S 0.6083
  (D31b); `gpt56-220` scored 3,701 workloads, mean S 0.6381 (D41). `pilot8` is
  off the board — a budget-stopped run is a cost measurement, not a score
  measurement. See `docs/agent-baseline.md`.

## Environment (current node)

| Field | Value |
|---|---|
| Node | `gbt350-odcdh1-a08-1.png-odc.dcgpu` |
| GPUs | 8× AMD Instinct **MI350X**, `gfx950:sramecc+:xnack-`, 252 GiB, 256 CUs each |
| Power cap | **1000 W** per GPU (MI355X node: 1400 W) |
| Max GFX clock | **2200 MHz** (MI355X: 2400 MHz) |
| Cooling | air (MI355X: liquid) |
| ROCm version | 7.2.0 (container) / driver 7.1.1.31500000 |
| torch version + build | `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a` |
| Clock-lock setting | `--setperfdeterminism` **1600** MHz (node-wide) |
| **F_LOCK (achieved)** | **1300 MHz** — measured, GPU 0 under sustained load |
| Sibling-GPU interference | **−0.11%** on a 0.146 ms kernel — sweeps and authoritative timing may share the node. Do **not** extend this to long memory-heavy kernels (D61) |
| Dataset present | yes — 235 problems, L1=94 L2=82 Quant=33 FlashInfer-Bench=26 |
| FlashInfer blobs | yes — 304 external safetensors blobs |
| Measurement container | `solbench:rocm7.2-torch2.9.1`, from `env/Dockerfile` (also carries SOLAR + patched torchview) |
| Node exclusivity | **NOT guaranteed.** Exclusive when audited 2026-08-03 (D2); two foreign tenants (`sglang::scheduler`, `ray::MegatronTrainRayActor`) landed 18:39/18:40 on 2026-08-12 and held GPUs 0–3 at 100% (D61). **Run `scripts/gpu_exclusive.py --gpu 0` before every authoritative timing.** |

Rooflines at **default** clocks, reference points only — per task 00's guard
rails **not** scoring ceilings, and not to be cited downstream: HBM copy
4.53 TB/s (56.7% of 8.0 spec), BF16 GEMM 1168 TFLOPS (50.6% of the MI350X spec
peak of 2307 @2.2 GHz — not MI355X's 2500 @2.4 GHz).

---

## Task status

| ID | Task | Status | Artifacts | Notes |
|---|---|---|---|---|
| 00 | Node acceptance | `done` | `artifacts/00/` | 13 checks, 0 failed |
| 01 | Clock calibration (F_LOCK) | `done` | `artifacts/01/` | **F_LOCK 1300 MHz** at setting 1600; unblocks 03, 05, 06 |
| 02 | Harness port validation | `done` | `artifacts/02/` | 3717/3717 non-deferred workloads passed in the task-02 reference sweep. Later sweeps disagree (D28, D51, D59) — this is a dated result, not a standing guarantee |
| 03 | SOL bounds (T_SOL) | `done` | `artifacts/03/` | 235/235 problems bounded, two derivations, source recorded |
| 04 | rocprofiler shim | `done` | `artifacts/04/` | median divergence −0.61% over 1430 pairs; clock domain verified |
| 05 | Tolerance calibration | `done` | `artifacts/05/` | 3717/3957 AMD-derived; the 240 missing are the deferred NVFP4 |
| 06 | Baselines (T_b) | `done` | `artifacts/06/` | 220 problems anchored; see the two re-timing exceptions above |
| 07 | Quant / MXFP4 | `done` | `artifacts/07/`, `artifacts/deferred.json` | 15 NVFP4 deferred with evidence; 220 ship |
| 08 | Red team | `done` | `reference/exploits/`, `artifacts/08/` | 28/28 replay cases pass, 0 false positives on 235 references |
| 09 | Release | `done` | `artifacts/09/` | manifest v1: **220/235 problems, 3717 workloads scoreable** |

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

**All ten gates re-run 2026-08-06** (`verify_artifacts.py --task NN`, in
`env/solb`). Recorded because three documents were claiming a task-01 failure
that has not existed since 2cdb7b0.

| Task | checks | failed | judgement | WARNs |
|---|---|---|---|---|
| 00 | 13 | 0 | 1 — dataset census | — |
| 01 | 11 | **0** | 0 | per-GPU floor spread 1335–1400 |
| 02 | 12 | 0 | 0 | — |
| 03 | 13 | **1** | 2 | — |
| 04 | 5 | 0 | 1 — divergence tails | — |
| 05 | 10 | 0 | 1 — >2× tolerances | — |
| 06 | 10 | 0 | 1 — node conditions | D15 re-time band 336/349 |
| 07 | 4 | 0 | 0 | no FP8 write-up |
| 08 | 4 | 0 | 0 | — |
| 09 | 9 | 0 | 0 | — |

The single failure is task 03's `check D: no measurement beats its T_SOL` —
31 of 519 measured workloads faster than T_SOL, worst 0.29×, across
`FlashInfer-Bench__019`, `L1__005` and `L1__035`. **That check reads frozen
manifest v1 and is meant to go on reporting what v1 shipped**; it is not a live
signal about the board, which serves v1.2, where the beaten set is a different
five problems (D42). **A second failure anywhere is a regression** — find out
what you broke before doing anything else.

Full task-00/01 acceptance output, the determinism sweep, the per-GPU floor
tables and the interference run: `docs/findings.md` (D8, D55), `artifacts/00/`,
`artifacts/01/`.

## Blockers and open work

**The full owed-work list is [`TODO.md`](TODO.md)** — ordered by how wrong the
number is, every item with an acceptance check. Plan from there, not from here.

Open right now, in the order they would mislead a reader:

1. **The anchor question outranks everything.** Whether a published `T_b`
   reproduces on a verified-exclusive GPU 0 is *not settled*: D59/D60 said it
   did not, **D61 retracted that** — the re-measurements were on a contaminated
   card. **Never quote 2.021× as a reproduction gap.** Owed: a clean re-time of
   `L2__057` `v1_eager`.
2. **`rocprofv3 --pmc` hangs in this container (D43)** — the counter path to an
   independent traffic measurement is closed. The shim is not implicated; the
   route left open is a minimal independent kernel, timed.
3. **Five T_SOL bounds are wrong under v1.2 and none is corrected (D42).** Three
   are one defect, the declared-traffic tier (D18), fixed per-problem in v1.1
   rather than at the tier; **328 workloads across 38 problems still rest on
   it.** Fixing the tier is the v1.3 item.
4. **827 workloads (22.3%) sit above 100× headroom (D39)** — marked via
   `bound_quality`, not fixed, and `bound_quality` is not in the manifest.
5. **Tolerances and goldens: code fixed, artifacts stale** — D52 (76
   artifacts), D52b (396 workloads; closing it is a schema change), D53 (all 165
   `.pt` goldens, 143 GB, invalid until regenerated). D51 says what the
   tolerance actually tests.
6. **626 of 3717 published anchors carry an eager latency under a compile label
   (D50).** The code is fixed (D56); the manifest is not.
7. **Foreign tenants can appear without warning (D61).** Guard every
   authoritative timing; `time_tb_candidates.py` does not yet refuse a
   non-exclusive card.
8. **Nothing has been measured on MI355X** — the port needs no work, every
   number does. See [`docs/TODO-MI355X.md`](docs/TODO-MI355X.md).

---

## Where everything else lives

| Document | What it is for |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | The contract: prime directives, GPU discipline, read order. Read first. |
| [`TODO.md`](TODO.md) | **The** owed-work list for MI350X. Every known gap, with acceptance checks. Absorbs the former `docs/TODO-MI350X.md`. |
| [`docs/findings.md`](docs/findings.md) | Every settled finding, by topic, D-anchored. The former *Surprises and deviations*. |
| [`PLAN.md`](PLAN.md) | Ordering of the bound work. **Last reviewed 2026-08-10 — predates D50–D61.** Where it disagrees with TODO/STATE on a fact, TODO/STATE win. |
| [`docs/methodology.md`](docs/methodology.md) | How every published number was derived, per term, with the B200 comparison. |
| [`docs/TODO-MI355X.md`](docs/TODO-MI355X.md) | Bring-up runbook for the other part. Different clock policy; do not merge it with this one. |
| [`docs/agent-baseline.md`](docs/agent-baseline.md) | How the agent runs were produced, cost modelling, why `pilot8` is off the board. |
| [`docs/backend-coverage.md`](docs/backend-coverage.md) | Schema-accepts vs actually-built-through, per solution language. |
| [`docs/plan-2026-07-31.md`](docs/plan-2026-07-31.md) | Archived pre-work plan. The record of *why*, not of what is true. |
| [`README.md`](README.md) | For an outside consumer: what the benchmark is, the score, how to run it. |
| [`leaderboard/README.md`](leaderboard/README.md) | Board deployment, the git-vs-host storage rule, the API, the trust boundary. |
| [`tasks/NN-*.md`](tasks/) | Per-task spec and acceptance criteria — still the definition of what each gate means. `reference/upstream-audit.md` locates every NVIDIA-specific call site upstream. |

## Findings index

One line each; full entry, evidence and numbers at the anchor shown.
**L** = live defect (also in `TODO.md`) · **S** = settled · **R** = carries a
retraction that must be read with it.

### Bounds and T_SOL

| # | | One line |
|---|---|---|
| [D12](docs/findings.md#d12) | S | T_SOL truncated to whole cycles; eight workloads truncated to 0 and 204 implied DRAM bandwidth above arch peak. |
| [D14](docs/findings.md#d14) | S | The bound was priced at the vector-FP32 rate on 160 of 235 problems — 16× below bf16 matrix; 437 violations became 63. |
| [D18](docs/findings.md#d18) | L | The declared-traffic tier prices the whole allocation (989,669 pages where the kernel gathers 34). Fixed per-problem in v1.1; the tier is still live. |
| [D21](docs/findings.md#d21) | R | Two further beaten bounds (`L1__005`, `L1__035`) — both since cleared, by v1.2/D37 and v1.1/D36. Plus an ingest bug that deleted a problem from /methodology. |
| [D31c](docs/findings.md#d31c) | S | A second model found a bound 220 problems of the first did not: less coverage, more findings. The count cannot be extrapolated from one sweep. |
| [D35](docs/findings.md#d35) | L | F_LOCK is a floor, not a lock: bf16 clocks 1303, fp32 1441, `L2__036` 1586. 759 compute-bound fp32 workloads were scored against a divisor 10–22% too generous. |
| [D36](docs/findings.md#d36) | S | Manifest v1.1 release note: 1048 `T_SOL_ms` changed, 0 re-gated, bad bounds 13 → 6. D18 had a second half. |
| [D37](docs/findings.md#d37) | R/L | SOLAR priced a grouped convolution as a dense one; fixed in v1.2 for six problems. Carries an in-place retraction of its own second half. `L2__036` is still uncorrected. |
| [D38](docs/findings.md#d38) | S/L | The timer never saw a submission's own stream (1743× on the probe). `L1__054` was never a wrong bound — it was a wrong measurement. Residue: 6–7 µs per tracked stream per iteration. |
| [D39](docs/findings.md#d39) | L | Nothing checks that a bound is tight. 827 of 3717 workloads (22.3%) sit above 100× headroom; 13.6% sit under 2×. |
| [D42](docs/findings.md#d42) | L | The five surviving bad bounds are two causes, and one is D18 again. 328 workloads across 38 problems rest on the tier. No bound corrected. |
| [D43](docs/findings.md#d43) | L | **BLOCKED**: `rocprofv3 --pmc` hangs in this container, even on a 3-kernel `a + 1.0`. The shim is not implicated. |

### Clocks, timing and the anchor

| # | | One line |
|---|---|---|
| [D2](docs/findings.md#d2) | R | The node was audited exclusive on 2026-08-03. **Falsified in fact by D61** — do not treat exclusivity as a property of this node. |
| [D8](docs/findings.md#d8) | S | `--setperfdeterminism X` yields ~0.83·X on MI350X and stops responding above ~1900, where the part pins to the 1000 W cap. |
| [D11](docs/findings.md#d11) | S | The shard runner could put two timing runs on one GPU (`gpus[i % n]` is not a pool). Fixed; 176 of 235 selection artifacts predate the fix. |
| [D15](docs/findings.md#d15) | L | `FlashInfer-Bench/018_mla_paged_decode`'s T_b does not reproduce to 3%: re-time median 1.16× the recorded value, twice, cause not established. |
| [D20](docs/findings.md#d20) | L | Matmul timing is bimodal: 0.13% of iterations cost 3.9–4.5×. The clock hypothesis was tested and **falsified**. Two upstream tests are skipped behind it. |
| [D29](docs/findings.md#d29) | L | The external fleet ran 34 jobs on GPU 0 despite a scheduler hold. Nothing published was affected; the property is unenforced. Cited from code as `TODO.md D29`. |
| [D55](docs/findings.md#d55) | S | Locking costs nothing measurable on MI350X — the cap never binds for the kernels the benchmark runs. The MI355X 21% figure does not reproduce here. Answers the orphaned D3. |
| [D59](docs/findings.md#d59) | R | The recompile-fix re-sweep. Its correctness half stands (−499/−526); its T_b half is retracted by D61, and its latency population must be recut on the pre-17:55 prefix. |
| [D60](docs/findings.md#d60) | R | What the T_b non-reproduction is *not*. Conclusion retracted by D61. |
| [D61](docs/findings.md#d61) | L | **CORRECTION**: two foreign tenants held GPUs 0–3 from 18:39 on 2026-08-12; the "solo GPU 0" re-times were contaminated. What survives: the old-SHA/new-SHA exoneration, and the pre-tenant 76.83 vs 31.35 = 2.45×, unexplained. |

### Tolerances, goldens and torch.compile

| # | | One line |
|---|---|---|
| [D9](docs/findings.md#d9) | S | The tolerance runner's memory profile, and two unrelated OOM causes — retention of every seed's outputs (234 GiB of 252) and fp64 comparison peaking at 4× output. |
| [D13](docs/findings.md#d13) | S | `masked_select` asks for 16781313 GiB above 2³² elements — a ROCm 7.2 / torch 2.9.1 boolean-indexing bug. Standing platform hazard; workaround is chunked `torch.where`. |
| [D50](docs/findings.md#d50) | L | `recompile_limit = 8` meant dynamo silently ran eagerly after the 8th shape: **2061 of 3957 workloads never compiled**, and 626 published anchors carry an eager latency under a compile label. |
| [D51](docs/findings.md#d51) | L | The tolerance is a bit-identity-with-eager test: 3502 of 3717 are pure floor. Against a float64 golden, eager is not the more accurate side on any adjudicated case. |
| [D52](docs/findings.md#d52) | L/R | `_dtype_floor`'s int/bool zero was applied to a problem's *float* outputs — 32 of 76 workloads are the defect. Code fixed 2026-08-12; artifacts stale. Carries a v5 correction. |
| [D52b](docs/findings.md#d52b) | L | The same leak between two *float* dtypes: fp32 outputs floored at bf16 epsilon, 65536×. The fix records the over-grant but does not close it — closing it is a schema change. 396 workloads stale. |
| [D53](docs/findings.md#d53) | L | Goldens were drawn on `cpu`, tolerances calibrated on `cuda:0` — same seed, different RNG. **2302 of 2331 goldens exceed their own atol.** All 165 `.pt` files invalid until regenerated. |
| [D56](docs/findings.md#d56) | S | The recompile cliff is fixed (`fail_on_recompile_limit_hit`), and it makes the failure count go **up**: `L2__009` `v2_compile` 8/16 → 0/16. |

### Harness, dataset and tooling

| # | | One line |
|---|---|---|
| [D1](docs/findings.md#d1) | S | The dataset ships as parquet, not per-problem directories; `materialize_dataset.py` is the exact inverse and round-trips all 235. |
| [D5](docs/findings.md#d5) | S | 9 FlashInfer problems need a second dataset — 304 blobs, `FLASHINFER_TRACE_DIR`. Without them they fail as ordinary runtime errors. |
| [D6](docs/findings.md#d6) | S | The vendored data-model package was never committed: an unanchored `data/` in `.gitignore` silently dropped 14 files. Only a fresh clone could have caught it. |
| [D7](docs/findings.md#d7) | S | One of SOLAR's torchview patches is malformed and unnecessary; the Dockerfile now asserts rather than assumes. |
| [D10](docs/findings.md#d10) | S | Stale artifacts read as fresh findings — scratch is keyed by problem, not by (problem, code version). Triage from the end-of-sweep artifact. |
| [D17](docs/findings.md#d17) | S | The scorer wrote into the container and scored every kernel zero; stderr was discarded and `0/0 passed` was indistinguishable from a real null result. |
| [D19](docs/findings.md#d19) | S | 38 dead tests behind a skip that read like a scheduling choice; the skip message's own suggested command was dead too. |
| [D33](docs/findings.md#d33) | S | `agent_score.py --timeout` never reached the evaluation — the inner cap stayed at 1200 s. Forwarded, `FI__014` re-times 30/30. |
| [D57](docs/findings.md#d57) | S | `shard_sweep.py` counted 235 crashes as "235 ok, 0 failed" because it counted exit status. Now reads the artifact it just wrote. |

### Agent runs and scoring

| # | | One line |
|---|---|---|
| [D16](docs/findings.md#d16) | S | The agent pilot billed the wrong gateway key — `~/.claude.json`'s `env` block overrides the process env. Surfaced only by a falsification test. |
| [D31](docs/findings.md#d31) | S | The first near-full-benchmark agent run (192/220, mean 0.5975). **A capped session is not a failed one**: capped 0.6072 vs self-stopped 0.5579. |
| [D31b](docs/findings.md#d31b) | S | The run is complete: 220/220, 3,690 scored, mean 0.6083, 0 flagged. Also: `rocm-smi --showuse` is not a usable signal on this node. |
| [D40](docs/findings.md#d40) | S | The first reward hacks, already published for 1–2 days: 48 of 19,310 results. Two of three used `set_float32_matmul_precision("high")`, not `allow_tf32`. |
| [D41](docs/findings.md#d41) | S | `gpt-5.6-sol` over all 220: 3,701 scored, mean 0.6381. 8-wide re-time measured 1.0139× slower and was reverted. `n_flagged` was structurally incapable of returning non-zero. |

### The leaderboard

| # | | One line |
|---|---|---|
| [D22](docs/findings.md#d22) | S | A failed workload was carrying a score on `/api/problems/{key}`. No ranking was ever wrong — aggregates filter `PASSED`. |
| [D23](docs/findings.md#d23) | L | A submitted kernel whose re-time timed out was invisible — `TimeoutExpired` read as *never attempted*. The board now carries `retime_ok`/`retime_error`; the budget question is open. |
| [D24](docs/findings.md#d24) | L | The "dropped the external run" bug, introduced three separate times. Semantics have since **inverted**: `--agent-runs` now overrides `sources.json` rather than adding to it. |
| [D25](docs/findings.md#d25) | R/L | `f_lock_mhz: null` was blamed on a `CLOCK_LOCK_PRESETS` entry that has existed since 2cdb7b0. Two real causes, no measurement lost; the artifact enumeration is incomplete, not wrong. |
| [D26](docs/findings.md#d26) | S | Three tables ranked means over different denominators (`AVG(score)` skips NULL). No number was changed; the denominator is now printed on every row. |
| [D27](docs/findings.md#d27) | S | The code pane had no test at all. 36/36 panes now byte-identical to the DB; the JS tokenizer and the `hip` branch are explicitly *not* asserted. |
| [D28](docs/findings.md#d28) | S/L/R | The board scored a whole problem by one flag and under-reported 1,239 passing workloads. Four published numbers moved, no measurement changed. Residue: 89 problems mix sweep-GPU with authoritative timings. |
| [D30](docs/findings.md#d30) | S | Two vertical scrollbars on every code pane — CSS Overflow 3 §3. Found from a screenshot; there is no browser on this node. |
| [D32](docs/findings.md#d32) | S | The section nav was laid out by the header's CSS. Same rule as D30, twice in two days; the tests asserted the served HTML, which was correct. |
| [D34](docs/findings.md#d34) | S | The four baselines' pages said their code was not recorded while `variant_source` held 1,175 rows. Fixed with a `submission.variant` column, not a slug round-trip. |
| [D44](docs/findings.md#d44) | S | `/?category=X` filtered the rows and not their labels — scope notes quoting 3,717/220 above a table of 1,480/94. |
| [D45](docs/findings.md#d45) | S | An unknown `?category=` rendered a full board of 0.0000; now a 400 on HTML and all four JSON routes. |
| [D46](docs/findings.md#d46) | S | Every per-workload `axes_json` was `{}` on all 3,957 rows since the column was written. The manifest carries no axes; `workload.jsonl` does. |
| [D47](docs/findings.md#d47) | S | A workload's parameters were only the axes that *vary* — 3 of 7 on `L1__001`. Const now from `definition.json`, expr by a whitelisted AST walk. |
| [D48](docs/findings.md#d48) | S | Workload identity was an 8-char uuid prefix corresponding to nothing anyone else prints; `dataset_index` matches upstream's ordering across all 235. |
| [D49](docs/findings.md#d49) | S | A board built from a bare clone failed silently and differently in five places. Closed by `reference/dataset-meta.json`. |
| [D54](docs/findings.md#d54) | R/L | A stale pre-split `leaderboard/solbench.db` is genuinely inverted, but the live board reads `db/solbench-MI350X.db` first. **Carries its own correction**; the leftover file is still not deleted. |
| [D58](docs/findings.md#d58) | R | One number system for the board — and three figures inside it stated as measured that were not (including 12,883 scored results, actually 21,040). |
| [board coverage](docs/findings.md#coverage-bar) | S | Coverage is one four-state stacked bar in problems, over all 220; widths unrounded because rounded widths do not sum to 100%. |
| [score scopes](docs/findings.md#score-scopes) | S | Two denominators under adjacent headings became one column plus a scope control. `/api/v1/leaderboard` deliberately unchanged. |
| [grid ramp](docs/findings.md#grid-ramp) | S | Grid distribution and contrast closed. b1's share is structural — the T_b variant scores exactly 0.5. Its scored-result count is superseded by D58. |
| [section nav](docs/findings.md#section-nav) | S | Server-rendered sticky nav on the two long reference pages. |

### Scripts fixed on first contact, and audits of the audits

`F1`–`F11` are session-1 fixes on MI355X; see git history. `F12`–`F24` are in
[`docs/findings.md`](docs/findings.md#f12); the three that matter most are
**F17** (a task-01 check that could not fail), **F22** (`check_06` asserted a
schema never produced) and **F23** (`check D` was a literal unconditional PASS —
the one invariant that would have caught D18). **F18** is retracted in part by
**F24**; read them together.

---

## Decisions taken

**F_LOCK = 1300 MHz**

Achieved, at determinism setting 1600. Full reasoning in `docs/findings.md`
(D8, task-01 results). The two-number form is a real structural difference from
NVIDIA, where `nvidia-smi -lgc` makes them the same; `ClockPreset` carries both
and `f_lock_mhz` returns the achieved one.

The line above is the canonical marker `verify_artifacts.py` parses, and the
only place in this file that form appears. **Do not reformat it.** Prose may
mention other parts' clocks freely — that is why a marker is required rather
than the first number after the first mention of F_LOCK (F17).

**Authoritative timing is pinned to GPU 0.** At the same determinism setting the
eight GPUs hold clocks spanning 1242–1307 MHz (5%), larger than most of the
optimization differences the benchmark exists to measure. Sharding is fine for
correctness and for *selecting* a T_b variant; the winner is re-timed on GPU 0.

**Architectural constants are shared between MI350X and MI355X; measured ones
are not.** `solexbench_rocm/parts.py` separates the kinds explicitly; the shared
MAC/cycle table is justified by reproducing *both* parts' published peak FLOPS
from one set of numbers — 524288 MAC/cycle × 2 × 2.4 GHz = 2.52 PFLOPS (MI355X
spec 2.5) and × 2.2 GHz = 2.31 PFLOPS (MI350X spec 2.3). A constant that derives
both is architectural; one that does not is a measurement in disguise.

**T_SOL uses SOLAR's `fused` model.** `unfused` assumes every intermediate
round-trips to DRAM, which would make the "lower bound" exceed real
measurements. Both are recorded so a violation can be diagnosed, not merely
observed.

**Timing methodology is recorded on every trace.** `Environment.methodology` is
resolved once per run and passed to both timer and trace, so "recorded" and
"used" cannot drift.

**MI355X clock methodology: bracket the window** (`docs/TODO-MI355X.md` §4.3
option 2), chosen by the maintainer. Sample the clock immediately before and
after the timed window, record both, refuse the measurement if they disagree by
more than the threshold. Off unless `SOLEXBENCH_CLOCK_BASIS=unlocked`; the
MI350X corpus takes exactly the path it took before.

* **The window is the `time_runnable` call in the eval driver**, and only that.
  Compilation and `max_autotune` are behind it, driven by the correctness pass.
  Bracketing `evaluate()` was tried and does not work: the kernel is 0.8–55% of
  the span and 85% of measurements refuse.
* **Threshold 0.0078**, the p99 of 6,544 consecutive-sample clock spreads in
  `artifacts/01/unlocked-clock.json` (median 0.111%, p90 0.284%, max 26.4%).
  Overridable by `SOLEXBENCH_CLOCK_BRACKET_THRESHOLD`; the value in force is
  stamped on every measurement. **Caveat:** that artifact is from `g10`, and no
  within-window bracket distribution has been measured on `mia1-p02-g46` at all —
  the g46 figures (36.8% across shapes, 3.9% across cards) are between-shape and
  between-card, the wrong statistic for this. Re-derive from the first sweep's
  own recorded spreads.
* **`T_b` and `T_k` are re-timed back to back in one session on one card**, and
  both brackets are recorded (§4.4).
* **The bracket bounds the clock error and nothing else.** It does not touch the
  short-window bias (`docs/methodology.md` §7), which on g46 is +106.9% for the
  worst shape at the shipped burst length and is measured *not* to be a clock
  effect (per-iteration cost 21.1 / 12.6 / 1.2 µs across shapes, an 18× spread).
  No summary of this work may imply otherwise.

**Uncertainty to carry forward, none of it measured away:**

* The **`amdsmi_init()` SIGSEGV under concurrency** is now serialised behind an
  `flock` in `device/amd.py`. **The fix is unverified** — reproducing the crash
  needs concurrent load on cards 1–7, which were running live sweeps. Do not
  record it as confirmed.
* Measured while wiring this up, on g46 GPU 0 (telemetry read only, no kernel):
  an SMI read costs **0.23–0.55 ms**, so on a 1 ms window the bracket spans about
  twice the region it brackets — recorded per measurement as
  `clock_bracket_lag_ns`. And an **idle** card (~193 MHz, 2 MHz jitter) yields a
  1.04% relative spread and is refused, where the same 2 MHz on a loaded card is
  0.08–0.12%. Scoring windows always follow warmup so this should never bite, but
  the relative-spread rule does have a quantisation floor at low clocks. No
  absolute floor was added: that would be improvising a methodology change to get
  past an obstacle (prime directive 7).

**The other three step-6 decisions** (`docs/TODO-MI355X.md` §5 step 6 requires
four written before any sweep that divides by a clock; the clock methodology
above is the second). All four were put to the maintainer and answered on
2026-08-14.

1. **Where F_LOCK comes from: nowhere, and that is the answer.**
   `get_clock_preset("AMD Instinct MI355X").f_lock_mhz` returns `None` and stays
   that way; `SOLEXBENCH_F_LOCK_MHZ` is **not** exported. There is no single
   achieved clock on this part to name — the measurement below puts the spread
   across kernel shapes at 36.8% — so any one number would be a fabrication
   dressed as a constant, which is what `4f7b06fd` removed. The bound instead
   travels as separately-scalable terms (`compute_cycles`, `memory_bytes`,
   `dram_byte_per_sec`, `mac_per_cycle`) and is re-maxed per measurement at that
   measurement's own bracket clock. `build_manifest.py`'s top-level `:243` guard
   therefore no longer hard-exits under `SOLEXBENCH_CLOCK_BASIS=unlocked`; the
   refusal **moves down a level** into `collect_t_b`, which drops any winner
   without a non-refused bracket and rejects outright any artifact carrying no
   clock evidence at all. An unknown clock is still not a permissive one — the
   guard was relocated, not relaxed.

3. **Task 01's acceptance check becomes "the clock basis is characterised."**
   The three lock-presupposing checks (preset exists, preset agrees with
   STATE.md, every GPU at the setpoint) cannot pass unlocked, and task 01 is a
   hard blocker for 03/05/06, so "not applicable" was not available. The
   unlocked arm requires instead: the per-card clock distribution under
   sustained load is recorded; the eight-card spread is ≤ 7%; and the bracket
   refusal rate is ≤ 2%. The locked arm is untouched, so MI350X takes the same
   path byte for byte. **The 2% is the weakest number in this file** — no
   refusal rate had been observed anywhere when it was set, and it is derived
   from a calibration distribution rather than measured. It is flagged as such
   in `verify_artifacts.py` and must be re-derived from the first completed
   sweep.

4. **The low-headroom anchor exemption stays, and the gate is taught about it.**
   A workload below `h_min` genuinely cannot hold `S` inside ±3% — that is what
   `h_min` means — so counting it as a failure would report a measurement-
   precision limit as an anchor defect. But an exemption that only ever raises
   the pass rate is not a gate, so `verify_artifacts` now bounds the exemption
   *count* separately in both check_06 and check_09: over-exempting is itself a
   failure. See the corrected `h_min` below, which shrinks the exempt set by 47%
   on its own.

**T_b and T_k are re-timed back to back, same card, same session** (§4.4). This
is the answer to the two-clock problem: unlocked, `T_b` was measured at whatever
clock its kernel pulled and `T_k` at whatever the candidate's kernel pulls, and
nothing in the repo normalized them, so a candidate that turns a compute-bound
kernel into a memory-bound one would be rewarded twice — once for the real
speedup and once for boosting. Both brackets are recorded on every score.

A consequence worth stating because it changes the schedule and could look like
a shortcut: **the authoritative pass is no longer pinned to one card.** MI350X
pinned it to GPU 0 because eight cards spanned 1242–1307 MHz and a T_b from one
card was not comparable to a T_b from another. Under same-card back-to-back
re-timing the requirement is weaker and different — `T_b` and its `T_k` must
share a card, not all `T_b` must share one card. So the authoritative pass runs
8-way with per-problem card pinning (`plan[i::8]` over the sorted plan, a pure
function of the problem name, so every replicate of a problem lands on the same
card) and the card identity is recorded in the artifact and enforced at scoring
time. This is a consequence of decision §4.4, not an independent relaxation, and
it is what turns an 11.4-hour serial pass into roughly 1.5 hours.

**`h_min` was wrong on trunk, and the correction is exact.** With
`T_k = T_b(1+d)` and `h = (T_b − T_SOL)/T_b`, `S = 1/(2 + d/h)` exactly. The
anchor property holds while `d/h ≤ 2 − 1/(0.5+tol)` on the fast arm (6/53 at
tol = 0.03) and `1/(0.5−tol) − 2` on the slow arm (6/47). The fast arm binds, so
`h_min = eps/(2 − 1/(0.5+tol)) = 8.833·eps`, against trunk's `0.5·eps/tol =
16.667·eps` — larger by exactly `1/(0.5+tol)`. Trunk never produced a false
failure; it silently excused workloads the gate could have adjudicated. Against
`artifacts/06/anchor-verification.json` the threshold moves 7.210% → 3.821%,
four workloads move from exempt to checked and **all four pass** (worst
|S−0.5| = 0.0043).

## What this node measured about its own clock

`artifacts/00/gpu-parity-mia1-p02-g46.json`, `artifacts/01/unlocked-clock-*.json`,
`artifacts/01/burst-clock-*.json`. These are the numbers the decisions above
rest on, and they are **wider than the g10 figures `docs/TODO-MI355X.md` quotes**:

| workload, GPU 1 alone | clock | power |
|---|---|---|
| `gemm_dense` | **1800 MHz** | 1383 W |
| `gemm_small` | 2392 MHz | 673 W |
| `memory_bound` | 2392 MHz | 1143 W |
| `reduction` | 2391 MHz | 870 W |

**36.8% spread across kernel shapes**, against the 27.9% quoted from g10. The
cards are **power-capped at 1400 W, not clock-capped**, which is why the heavy
shape is the slow-clocked one — and it is the direct refutation of §4.3's
option 5, assuming one clock for everything. Eight cards loaded together span
1739–1855 MHz and 3.9% in throughput.

`burst_clock_probe` is the one that constrains what bracketing can claim: at
`time_runnable`'s own burst length the worst shape reads **+106.9%** against a
50,000-iteration sustained loop, and the per-iteration attribution is 21.1 µs /
12.6 µs / 1.2 µs across three shapes — an **18× spread**. A depressed clock
would slow all three alike. **So the short-window bias is largely not a clock
effect, and bracketing bounds the clock error without touching it.** Do not read
a passing bracket as evidence that the window is unbiased.

### The first bracketed sweep refused 100% of its workloads — two bugs, both fixed

**Root cause.** `_clock_info` resolved its device with
`int(getattr(device, "index", device))`. Correct for `None`, an `int` and a
`torch.device`; **silently wrong for the string `"cuda:0"`**, because `str` has
an `.index` *method* — `getattr` finds it, the default is never reached, and
`int(<built-in method>)` raises `TypeError` inside a bare
`except Exception: return None`. `eval_driver.py:351` sets
`_device = "cuda:0"`, so every sample returned `None` and every measurement was
refused for absent clock evidence, at exit status 0. Verified on three cards:
`sample_clock_mhz(0)` → 159/204/235 MHz, `sample_clock_mhz("cuda:0")` → None on
all three. **Not** the PCI-ordering trap — the three distinct readings prove the
translation addresses distinct physical cards correctly under a restricted
`HIP_VISIBLE_DEVICES`.

Fixed by `device/amd.py:torch_index_of()`, which parses every spelling
explicitly and **raises** on anything else rather than defaulting to 0 — a
default of 0 would read some card and return a plausible number, which is §8.1
again and undetectable rather than merely total. Device resolution now sits
*outside* the `try`, so a malformed request raises while genuine telemetry
failure still returns None.

**Second bug, independent.** The artifact's `clock_bracket_summary` was built
from the **winners**, and a refused measurement never becomes a winner — so it
read `n_bracketed: 0, n_refused: 0, refused_by_reason: {}` three lines below a
correct `n_workloads_refused_on_clock: 16`. The field a reader checks went blind
exactly when it mattered. Both counters now come from one list built in
`select_winners`.

**Loudness.** `clock_fatalities()` makes zero-brackets, any `sampler_error`, or
a 100% refusal rate a non-zero exit with a `FATAL:` line and a
`clock_bracket_fatal` field, while still writing the full artifact. Verified end
to end: forcing the threshold to 1e-9 exits 1.

### D-new: the 0.0078 threshold is WRONG for g46, and the reason is a clock ramp

First within-window bracket-spread distribution ever measured on this part
(GPU 0, `L1/009`, 4 variants × 16 workloads = 64 brackets):

| q05 | q25 | **median** | q75 | q90 | q95 | max |
|---|---|---|---|---|---|---|
| 0.084% | 0.463% | **0.971%** | 3.58% | 4.54% | 5.41% | 6.50% |

Against the shipped 0.0078 that is a **57.8% refusal rate** (37/64). The
threshold was calibrated on `g10`'s 1-second-gap samples (median 0.111%, p99
0.778%) and the realised median is **9× that**.

**The refusals are a clock ramp, not noise.** 60 of 64 brackets have
`after > before`, and **37 of 37 refusals do** — not one refusal is the clock
falling. Refused windows start at a median **2271 MHz** and end at 2375;
admitted windows start at 2373 and end at 2383. The card is still climbing into
boost when the window opens. Two negative controls: window duration does not
explain it (Pearson r = **−0.06** over the 64), and raising warmup from 10 to
200 made it **worse**, not better (median 0.0078 → 0.0130, 8/16 → 13/16 refused,
n=16, not over-interpreted).

**Recommendation, for the maintainer — I did not change the constant.** Do not
raise the threshold to fit. At 0.05 the rate is still 9.4%, and admitting a 5%
bracket means admitting 5% of uncertainty into a compute-bound bound, against a
3% anchor tolerance. The refusal is working: it is reporting that on this part a
majority of measurements genuinely do not have one clock. The real options are
(a) settle the card immediately before the window so it opens at steady boost,
(b) accept the yield, or (c) re-derive the threshold from this distribution and
state the resulting bound uncertainty. All three are methodology decisions.

**Note for whoever takes that decision:** the bracket currently spans the
`ShiftingMemoryPoolAllocator` construction as well as warmup + iterations,
because that allocation happens inside `time_runnable`. That is CPU work with an
idle GPU at the head of every window, and it is a plausible contributor to the
ramp. Tightening the bracket to the measured loop alone means returning a
bracket *from* `time_runnable`. Untested — do not treat it as diagnosed.

## Session log

Handoff record. What each session *found* is in `docs/findings.md`; what it
*owes* is in `TODO.md`.

| Date | Node | Worked | Left behind |
|---|---|---|---|
| 2026-08-03 | `mia1-p02-g10`, 8× MI355X | Session 1: tasks 00, 01 (F_LOCK 1650), 02 port written | Work moved to an MI350X node; every measurement re-taken there |
| 2026-08-03 | MI350X | Session 2: environment rebuild, tasks 00 and 01 (F_LOCK 1300), runners, SOLAR bridge, exploit corpus, T_b variants | `artifacts/00/`, `artifacts/01/`, `parts.py`, `scripts/runners/`, `reference/exploits/`; restored the never-committed `core/data` (D6) |
| 2026-08-06/07 | MI350X | Session 3: leaderboard only — no measurement taken or changed | D25, D26, D27; `tests/leaderboard/` from 0 to 91 passing; the MI350X↔MI355X part switch |
| 2026-08-11 | MI350X | Session 4: leaderboard front end only; the only new API behaviour is a 400 | D44–D49, D58 |
| 2026-08-11 | MI350X | Why torch.compile fails 71 problems | D50–D54 |
| 2026-08-12 | MI350X | The lock measured; D50 acted on; D52/D52b/D53 fixed in code | D55–D58; the re-sweep and its retraction, D59–D61 |

**Next session:** start with `TODO.md` item 1 (the anchor), and run
`scripts/gpu_exclusive.py --gpu 0` before you time anything.
