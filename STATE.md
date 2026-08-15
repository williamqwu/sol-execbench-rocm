# STATE.md — progress ledger

**Single source of truth for progress.** Update as you go, not at the end. A
session can be interrupted at any point; whatever is written here is what the
next session inherits. Record real output, not summaries of intent; if something
failed, say so and say how; never mark a task `done` without pasting its
acceptance-check output. **This file is short on purpose** — where the port
stands, what node it runs on, where everything else lives. No narrative history.

## D-numbers are stable anchors, and they moved house

D-numbers (`D1`…`D72`, plus `D31b`, `D31c`, `D52b`; `D4` and `D62` are gaps) are **permanent
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

### MI355X: measured, and the manifest is v4 (2026-08-15)

**The line above — "nothing has been measured on MI355X" — is no longer true.**
`artifacts/09-MI355X/manifest-v4.json` scores **220 of 235 problems / 3717
workload instances** on 8× MI355X, `clock_basis: unlocked`, anchors from
`artifacts/06-MI355X/authoritative-merged`, coverage 235/235. The same 15 NVFP4
Quant problems are deferred, for the same reason, with the same evidence.

**This part is never clock-locked.** There is no F_LOCK for it and there must not
be one (`docs/TODO-MI355X.md` §5 step 6, decision 1). Every command carries
`SOLEXBENCH_CLOCK_BASIS=unlocked`; every measurement carries its own clock
bracket; every bound is re-derived at that measurement's own bracket from
clock-free terms. **Do not compare an MI355X millisecond with an MI350X one.**

What moved on 2026-08-15, and it is 238 of 3717 published bounds across 16
problems — exactly additive across three disjoint causes, which is also the proof
that nothing else moved:

| cause | n | problems | ratio new/old | direction |
|---|---:|---|---|---|
| causal-mask live-row pricing (**D64**) | 64 | `FI__014`, `FI__015` | 0.5039 … 0.5103 … 0.8081 | all DOWN |
| SOLAR memory term dropped on gathered (**D66**) | 47 | `FI__018` | 3.87e-05 … 0.0079 … 0.0781 | all DOWN |
| tier compared at the measurement's clock (**D65**) | 127 | 13 | 4.576 … 39.47 … 249.07 | all UP |

The third is the largest correction of the session and ran in the *undetectable*
direction until it was found: a SOLAR tier stored at 1.8 GHz was judged
impossible against a `T_b` measured near 2.4 GHz, dropped, and the bound fell to
the declared-traffic floor 4.58x–249x too small. `solar_rejected_above_t_b` went
127 → 0.

**What still fails, and it is one thing:** `L2__050_vae_decoder_mid_block_attention_resnet`
at 0.69x. SOLAR's compute count is exactly right and is priced at the fp32 SIMD
rate while the kernel runs autocast fp16 on the matrix cores, which
`max_rtol = 0.5583` admits. It is a precision-pricing decision, not an arithmetic
error, and **it needs a maintainer's signature** — the two candidate resolutions
are mutually exclusive and both are methodology changes. `docs/TODO-MI355X.md`
§13 M1.

**Where MI355X was measured.** Anchors in `authoritative-merged` come off **24
distinct cards across 3 nodes** — `mia1-p02-g46` (91 problems), `mia1-p02-g45`
(65), `mia1-p02-g05` (61), 4 unidentified — all 8× MI355X, 1400 W liquid,
2400 MHz ceiling, ROCm 7.2.0 / driver 6.16.6 / torch
`2.9.1+rocm7.2.0.git7e1940d4`. **Cross-node clock comparability has never been
measured on this part**, which is why `CARD_KEYS` includes the hostname and why
nobody may drop it without measuring that first.

**The board lost coverage, and that is the honest outcome (D70).** `full-01`
`sol_score_v1` went **1619 → 594** and `quant-fill` **230 → 70**, because 1025
already-published records had their `T_b` and `T_k` measured on different physical
cards, which §4.4 forbids with no bypass. **None of it is the manifest** — a
control backfill with HEAD code and manifest-v3 collapses identically. Read it as
*"these records cannot support the claim they make"*, not as *"undoing
inflation"*: the workaround that used to hide it yields a *lower* `S` on 543 of
831 records.

**And it was recovered, on-card, the same day — to more than it started with.**
Re-scoring each affected problem on the card
`artifacts/06-MI355X/card-assignment.json` assigns needs no new anchor. Run
across g46, g45 and g05 and **finished** (0 workers left, census stable over a
re-check):

| `full-01` | before | after the card fix | after re-scoring |
|---|---:|---:|---:|
| `sol_score_v1` | 1619 | 594 | **1750** |
| `sol_headroom` | 229 | 1254 | **100** |
| `correctness_only` | 139 | 139 | 137 |
| mean `S` | 0.6565 | 0.6495 | **0.6584** |

`quant-fill` went 230 → 70 → **230**, mean `S` 0.3785 → 0.4397 → **0.3787**.

Status census over the finished tree: **PASSED 1850, INCORRECT_NUMERICAL 96,
RUNTIME_ERROR 41**. The correctness verdicts moved a little across the session
because the re-scoring genuinely re-ran those kernels on their assigned cards —
this was not an arithmetic recompute, and the earlier audit figure of
PASSED 1960 / 98 / 41 was over the pre-re-score population. **Compare the
verdict mix, not the raw totals, and re-derive both rather than quoting either.**

**Read the end state, not the middle, and read what changed about it.** The board
now has *more* scored records than before the session (1750 against 1619) at
essentially the same mean, and — the point of the exercise — **every one of them
has its `T_b` and its `T_k` measured on the same physical card.** The 594 was
never a result; it was the honest floor while the anchors were being put back on
the right cards. 100 records remain `sol_headroom` and stay there.

Re-derive it yourself before quoting it — one command over the tree, and
`score_basis` is the field that carries the answer:

```bash
env/solb python - <<'EOF'
import json, glob, collections
for run in ('full-01', 'quant-fill'):
    c, ss = collections.Counter(), []
    for f in glob.glob(f'artifacts/10/scores/{run}/*/*.json'):
        if f.endswith('summary.json'): continue
        for r in json.load(open(f)).get('records') or []:
            c[r.get('score_basis')] += 1
            if r.get('sol_score') is not None: ss.append(r['sol_score'])
    print(run, dict(c), 'mean S', round(sum(ss)/len(ss), 4) if ss else None)
EOF
```

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

The single failure is task 03's `check D: no measurement beats its T_SOL`.
**That check reads frozen manifest v1 and is meant to go on reporting what v1
shipped**; it is not a live signal about the board, which serves v1.2, where the
beaten set is a different five problems (D42). **A second failure anywhere is a
regression** — find out what you broke before doing anything else.

**Both parts' gates, re-run 2026-08-15 10:2x UTC** in `/var/tmp/solbench/m2`,
after the close-out rebuild of the tier, the manifest and the report.
This table supersedes the counts above wherever they differ.

| Task | MI350X (default manifest) | MI355X (`--manifest manifest-v4.json`) |
|---|---|---|
| 00 | — | 13 checks, 0 failed, 1 judgement |
| 02 | — | 12 checks, 0 failed |
| 03 | 16 checks, **1 failed**, 3 judgement | 18 checks, **1 failed**, 2 judgement, 0 WARN |
| 05 | — | 10 checks, 0 failed, 1 judgement |
| 06 | 11 checks, 0 failed, 1 judgement | 12 checks, 0 failed, 1 judgement |
| 07 | — | 4 checks, 0 failed |
| 08 | — | 4 checks, 0 failed |
| 09 | 9 checks, 0 failed | 9 checks, 0 failed |

**Read the pass/fail column, not the check count.** Task 03 went 13 → 14 → 18
checks in one day as `check A-published`, its manifest-binding guard, the
report-inputs guard, `check D-terms` and the legacy-column WARN were added. A
check count that grows is coverage; only the failed column is the rule.

* **MI350X task 03** — `144 of 7840 measured workloads are faster than T_SOL
  (worst 0.27x) across 15 problems`, first-named `FlashInfer-Bench__018`. Against
  `--manifest manifest-v1.2.json`: 25 of 7840, worst 0.31x, 5 problems.
  Unchanged by the 2026-08-15 session and verified so. **The older figure of
  "31 of 519, worst 0.29x, 3 problems" in this file was from a smaller score
  population and is superseded.**
* **MI355X task 03** — `1 of N measured workloads … 0.69x … L2__050`, every one
  checked against a bound re-derived at that measurement's own clock bracket. It
  was 2 failures before this session (the other was a regex artefact, D68) and 12
  phantom check-D rows before that (D63). **`N` was 2078 at the start of the day
  and was moving (2048, 2035, …) as the on-card re-scoring rewrote
  `artifacts/10`; it settled at 2080. Re-run the gate before quoting it; the
  count of failures is the stable claim, not the denominator.**
* **The legacy-column WARN is gone, and measurement closed it, not wording.**
  Mid-session it read `70 of 2065 measurements beat the manifest's plain
  t_sol_ms` against 1 beating the bound check D re-derives — the one thing the
  session had made measurably worse, because correcting the tier comparison
  (D65) pushed the legacy column away from the published bound. Re-deriving
  `t_sol.json` at a single 2400 MHz reference and rebuilding the tier and the
  manifest on top of it collapses that: the check now reads **`identical to
  check D's own count (1 of 2080); 0 of 3957 records carry no f_ref_mhz`**. The
  whole manifest is legible through `t_sol_at.bound_ms` and the two columns no
  longer disagree about which measurements beat their bound.
  `bound_headroom.published_bound_ms` is the single place the choice of column
  is made, and `leaderboard/ingest.py`, `leaderboard/app.py`,
  `scripts/score_distribution.py`, `scripts/bound_headroom.py` **and
  `scripts/agent_score.py` — the submission path — now all read through it.**

**MI350X must stay at 03 → 1 failed, 06 → 0, 09 → 0.** Movement there is a
regression regardless of what it does to MI355X.

**Nothing from this session is committed, and several release artifacts are
untracked.** `git status` at handoff shows `artifacts/09-MI355X/manifest-v4.json`,
`artifacts/03-MI355X/llc/` and `scripts/llc_bandwidth_probe.py` as **untracked**,
alongside 14 new test files. CLAUDE.md §7 says git carries what is needed to READ
and AUDIT a number, and manifests are tracked — so **the MI355X release manifest
is not yet in git.** Whoever commits should read §7's table first; `artifacts/10`
score trees and `reference/dataset-meta.json` are tracked, `transcripts/`,
`data/` and `artifacts/golden/` are not.

**D71 is closed as a state and open as a diagnosis.** `artifacts/03-MI355X/t_sol.json`
was re-derived near the end of the session (single clock, 2400 MHz, 2998 of 2998
records stamped), which left the shipped `manifest-v4.json` no longer a function
of the artifacts it names. The close-out rebuilt the chain once, in order —
traffic tier gated against `authoritative-merged`, then `manifest-v4.json`, then
`cross-checks.md` — so the manifest is a function of its own declared sources
again and task 03's input checks are PASS. **What that rebuild moved is
unexplained and is still owed a diagnosis**: 15 published bounds on
`L1__021_vision_cu_seqlens_variable_length_attention`, ×0.9501–×1.0789, 8 up and
7 down, off a `compute_cycles` that differs between two SOLAR derivations at the
same clock because the `cpu_real` tier draws `cu_seqlens` unseeded. Every one of
the 15 still sits ≥4.65× below its own T_b, so no score is near a bound, but the
bound is **not reproducible** and no gate can see that. `docs/TODO-MI355X.md`
§13 M8.

**One operational rule came out of this and will bite you.** Task 03 now binds
`check A-published` to the manifest under audit by **sha256, not path**, because
a manifest rebuilt in place keeps its filename and is a different manifest. So
**regenerate `artifacts/03-MI355X/cross-checks.md` after any manifest rebuild**,
or the gate goes red — the failure message names the command. The companion check
on the report's other inputs (`t_sol.json`, `t_sol_traffic.json`, the arch YAML)
**is now a FAIL too**, and A-published's own count REFUSES when either binding
breaks. It was a WARN only while `t_sol.json` was known-broken and awaiting
re-derivation, so that landing a fix would not read as a regression; that landed,
nothing on the tree trips it, and a gate is only honestly hardened at the moment
it costs nothing. Practical consequence: **rebuild the tier and the manifest and
regenerate the report as one sequence, in that order.**

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
   *2026-08-15: the tier IS fixed on MI355X* — `scripts/sol_gathered_traffic.py`
   derives the gather pairing from each problem's own reference,
   `sol_traffic_floor.py` uses it, and MI355X manifest v2 check D goes **102
   violations across 13 problems → 28 across 11**, worst ratio 0.02 → 0.53. It
   fires on 7 of 235 problems and moves 265 workloads; nothing else changes by
   a byte. Not applied to the MI350X artifacts.
   *Update, same day:* the SOLAR side of `FlashInfer-Bench__018` is closed too,
   but **not** by the mechanism D18 predicted — SOLAR is gather-aware and the
   over-count was the reference's own full-tensor cast (**D66**). And the tier
   defect has a second guise on the **query** axis, corrected on MI355X only
   (**D64**). MI350X still carries both, deliberately, pending a version cut.
   See D18, D64, D66.
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
8. **MI355X is measured and at manifest v4** — see the section above and
   [`docs/TODO-MI355X.md`](docs/TODO-MI355X.md) for what that part still owes.
   Its own five headline items, ordered by how wrong the number is:
   `L2__050`'s precision pricing (the last real falsification, and a maintainer
   decision — D64/§Issue 2); the declared-traffic tier's per-input allocation
   pricing, which is the same v1.3 item as MI350X and governs **1181 of 3717
   scoreable MI355X workloads across 82 problems** (MI350X's "328 across 38" is
   a different, MI350X-only count — do not quote it here);
   127 newly-published bounds that are all narrow and carry no `bound_quality` in
   the shipped manifest (D65 — the build is fixed, the artifact is not); no
   independent *measurement* of any corrected byte count anywhere (D43 closed the
   counter route; the counter-free probe is **unrun**); and the board arithmetic
   that follows the on-card re-scoring — **re-ingest and re-run task 03; neither
   is done** (D70).

---

## Where everything else lives

| Document | What it is for |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | The contract: prime directives, GPU discipline, read order. Read first. |
| [`TODO.md`](TODO.md) | **The** owed-work list for MI350X. Every known gap, with acceptance checks. Absorbs the former `docs/TODO-MI350X.md`. |
| [`docs/findings.md`](docs/findings.md) | Every settled finding, by topic, D-anchored. The former *Surprises and deviations*. |
| [`PLAN.md`](PLAN.md) | Ordering of the bound work. **Last reviewed 2026-08-10 — predates D50–D61.** Where it disagrees with TODO/STATE on a fact, TODO/STATE win. |
| [`docs/methodology.md`](docs/methodology.md) | How every published number was derived, per term, with the B200 comparison. |
| [`docs/TODO-MI355X.md`](docs/TODO-MI355X.md) | Bring-up runbook for the other part, and now its owed-work list too. Different clock policy; do not merge it with this one. |
| [`docs/issues/mi355x-bound-quality.md`](docs/issues/mi355x-bound-quality.md) | The seven MI355X bound-quality issues, as a **resolved-issue record**: what each turned out to be, where the original brief was wrong, and what is still open. |
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
| [D63](docs/findings.md#d63) | S | The two T_SOL tiers were compared at different clocks — 1.8 GHz against 2.4. Analysis half; the enforcement half is D65. |
| [D64](docs/findings.md#d64) | L | D18's second guise: a masked **stream**, not a gathered allocation. 64 MI355X bounds down ~1.96x, 4 of 5 falsifications deleted. MI350X carries the same defect, uncorrected. |
| [D65](docs/findings.md#d65) | L | A SOLAR tier stored at 1.8 GHz was judged impossible against a 2.4 GHz `T_b` and dropped: **127 bounds were 4.58x–249x too small**. Not 127 findings — one forced band. `bound_quality` on 0 of them. |
| [D66](docs/findings.md#d66) | R/L | `FlashInfer-Bench__018`: SOLAR is **gather-aware**; the over-count is the reference's own full-tensor cast before the gather. Refutes the standing hypothesis. Ships MI350X v1.1 parity, not the durable fix. |
| [D67](docs/findings.md#d67) | R | The achievable-bandwidth curve, measured: 7.24–7.31 TB/s at ~258 MiB, no knee at 256 MiB, >8 TB/s only under ~64 MiB. **The Infinity-Cache hypothesis is refuted.** Three reusable measurement traps. |
| [D68](docs/findings.md#d68) | R/L | The "120 published bounds below their floor" never existed — a regex that ran past its section. A-published's real verdict was 0. Two of its three residual defects were closed the same day; the gate is now bound to its manifest by sha256. |
| [D69](docs/findings.md#d69) | S | `dS/dT_SOL = (T_b−T_k)/(T_b+T_k−2T_SOL)²`. A bound moving **down deflates** `S` for the 74.5% of the corpus faster than `T_b`. The repo said it both ways. |
| [D70](docs/findings.md#d70) | S/L | The board lost 63% of its scored records to honest card refusals — then recovered on-card to **1750**, above the pre-session 1619, every one now card-matched. Two backfill columns were wrong. `bound_violation 102 → 1` was 84% population removal, not correction. |
| [D71](docs/findings.md#d71) | L | The shipped MI355X manifest was built from a `t_sol.json` that no longer existed on disk. **Rebuilt at close-out**, which moved 15 bounds on `L1__021`, 8 up / 7 down — and the cause is now known: that problem's `cu_seqlens` are drawn **unseeded**, so its bound is not reproducible and no gate can see it. |
| [D72](docs/findings.md#d72) | L | The declared-traffic tier's own rejection gate ran against `authoritative`, not the `authoritative-merged` the manifest declares: 237 records shipped ungated, one at 1.68× its anchor. No published bound was wrong — the manifest re-gates. Now refused in code, and the tier rebuilt: `traffic_rejected_above_t_b` 8 → 7, 0 bounds move. |

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

**MI355X: T_SOL is an interval, published at the minimum clock of the bracket.**
Approved by the maintainer 2026-08-14; gated on `SOLEXBENCH_CLOCK_BASIS=unlocked`,
the locked MI350X path unchanged. Three parts:

1. **Published at f_min** — the largest T_SOL, hence the tightest bound. Chosen for
   the direction of its error: too strict is *detectable* (a measurement beats its
   own bound and the existing check fires), too loose is not, and CLAUDE.md §6 names
   an undetectable bound as the failure this repo has already had three times. The
   argument is in `src/solexbench_rocm/t_sol_at.py`'s docstring, not only here.
2. **Both ends recorded** — `t_sol_ms_at_clock_min` / `_max`, the clock at each end,
   and `sol_score_at_clock_min` / `_max` beside the published `S`.
3. **Width is a field**, per workload (`t_sol_interval_halfwidth_rel`) and per
   problem (`t_sol_interval_halfwidth_max`), so it sorts without reprocessing.

**Refusal is demoted from a gate to a label.** A bracket refused for *spread* no
longer discards the measurement: it publishes with a stated width.
`clock_bracket_refused` is still set, still counted, and `summarize_brackets` reports
the same `n_refused` / `refusal_rate` / `refused_by_reason` it always did, plus an
added interval split. A bracket with *no* samples — `no_clock_evidence`,
`sampler_error` — is still refused, and `clock_fatalities()` still exits 1 on those.
The demotion is applied at manifest-build time
(`build_manifest._recover_interval_anchors`) rather than in the sweep runner,
deliberately: changing selection mid-flight would split the corpus into problems
chosen under two rules with nothing recording which.

**Measured**, rebuilding the manifest off `artifacts/06-MI355X/authoritative` to a
scratch path (the sweep was still live, so these move): 212/235 problems, 3558
scoreable workloads, **every one of them carrying an interval**. Halfwidth **median
0.19%**, **max 9.92%** (`L1__071_kv_cache_update_with_rope`); 2 problems above 5%;
**2** bottleneck flips; **368 workloads admitted by the demoted label**, which under
the gate were missing T_b. Per problem: L2__004 max 3.53%, L2__005 max 1.78%,
L1__013 max 0.015%, L2__012 max 8.5e-6% (memory-bound at both ends — the width is
zero for the right reason).

**The ±33–43% figures quoted when this was requested are the clock span across a
problem's workloads and variants, not within one timed window.** Within-window
brackets on the same problems are much tighter (L2__004's widest single window is
1811–1949 MHz), so the published intervals are correspondingly narrow. Do not quote
the larger figures as interval widths.

**One defect found doing it, and fixed:** the two T_SOL tiers emit DRAM bandwidth as
`7999919999999.999` and `7999920000000.0` — the same 7.99992e12 reached by two
routes. `_reclock_terms` compared them for exact equality and so refused to merge
*every* two-tier MI355X record. **`artifacts/09-MI355X/manifest-v1.json` was built
before the fix and carries `reclock_terms_conflicting_bandwidth: 2977`, with only
576 of 3415 scoreable workloads holding an interval — it must be rebuilt.** After the
fix that count is 0 conflicts and 3558/3558. Now compared at 1e-9 relative; a real
disagreement is still a conflict.

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

### The allocator hypothesis was WRONG, and tightening the bracket would not help

I proposed that `ShiftingMemoryPoolAllocator` construction — CPU work with an
idle GPU, inside `time_runnable` and therefore inside the bracket — was holding
the card down at the "before" sample. **Measured, and it is not.** Instrumented
phase breakdown on GPU 0, same problem, four workloads:

| workload | allocator ctor | clock before ctor | clock after ctor |
|---|---|---|---|
| 65d4e462 | 0.164 ms | 1597 MHz | 1597 MHz |
| 8d38a764 | 0.083 ms | 2020 MHz | 2020 MHz |
| 3f768220 | 0.081 ms | 2257 MHz | 2257 MHz |
| 6a068dec | 0.118 ms | 2323 MHz | 2323 MHz |

Construction costs 0.08–0.16 ms and moves the clock by **exactly 0 MHz** in all
four. It is not the gap.

**And tightening the bracket would not have fixed it**, computed from the same
run rather than by building it first: bracketing only the measured loop instead
of all of `time_runnable` changes the spread 0.2186 → 0.1954, 0.1046 → 0.0938,
0.0421 → 0.0385, 0.0200 → **0.0205** (one gets *worse*). Every one is still far
above 0.0078. Per the maintainer's instruction I stopped here and did **not**
move the sampling inward.

**What it actually is: a DVFS boost ramp slower than the window.** The card
climbs throughout the timed region and is still climbing when it ends — the
highest reading of every run is the one taken after the loop. Ramp
characterisation after 3 s idle: **117 ms to come within 1% of peak**
(2364 → 2398 MHz). From deep idle it is far larger: the first workload of a
process opens its window at 1597 MHz against a ~2400 MHz ceiling. The effect is
positional — median spread by position within a variant run is 0.029 / 0.028 /
0.026 for the first three workloads (start clock 2243–2302 MHz) and ~0.008
thereafter (start clock 2356–2384 MHz).

This is a property of the hardware and of the harness's duty cycle, not a
bracketing defect. It is **not** the same class as the `evaluate()` mistake.

### D-new: keep 0.0078. It is right on score-error grounds; the yield is a separate defect

Re-derived from the score rather than from the refusal rate. Only compute-bound
bounds are clock-sensitive — memory-bound bounds are clock-invariant — which is
**47.1%** of scoreable workloads, using MI350X v1.2 as a proxy for the mix since
no MI355X manifest exists yet. A relative clock uncertainty `u` produces a score
error of at most `u·(1−h)/h`, reached as `T_k → T_SOL`; it is exactly 0 at the
anchor, so the anchor tolerance is a ceiling on the error, not a description of
it.

| u | median h (0.764) | q10 h (0.263) | q01 h (0.122) |
|---|---|---|---|
| **0.0078** | 0.24% | **2.2%** | 5.6% |
| 0.0128 | 0.40% | 3.6% | 9.2% |
| 0.0200 | 0.62% | 5.6% | 14.4% |
| 0.0500 | 1.55% | 14.0% | 36.0% |

Holding the q10-headroom compute-bound workload inside the ±3% the anchor gate
already uses implies a ceiling of **u = 0.0107**. The shipped 0.0078 sits below
it with margin, so **the calibration holds** — it was derived from the wrong
distribution (g10's 1-second-gap steady-load samples) and lands in the right
place for the right reason. That is luck, and it is worth saying so.

**And that settles the yield question.** Moving to the largest defensible value,
0.0107, lifts the admit rate only from **42.2% to 56.2%**. No threshold that
keeps scores honest fixes this. The 58% refusal rate is therefore not a
calibration problem and must not be treated as one — it is the boost ramp. The
options are to settle the card with the real kernel before the window
(~120–200 ms), to re-measure refused workloads once the card is warm, or to
accept the yield. All three change what a measurement means and are the
maintainer's call.

### Allocator is NOT in the timed region — checked, because it would have been serious

Were per-iteration allocator work inside the event pair, every latency this
benchmark has produced on any part would be inflated and no artifact would say
so. It is not. Causal test: injecting a **50 ms** sleep into `setup()` moved the
reported median by **0.055 ms** (0.949 → 1.004 ms). `setup()`, the L2 flush and
`_fence_streams()` all precede `start_events[i].record()`; only `fn(args)` and
`_join_streams()` lie between the two events. Both the causal and the structural
form are now pinned by
`tests/sol_execbench/core/bench/test_setup_outside_timed_region.py`.

### Pre-window settle: implemented, measured, and it works

`docs/TODO-MI355X.md` §4.3 option (a), chosen by the maintainer. Before the
window opens, run **the real kernel** until the clock stops climbing, then
sample. Off unless `SOLEXBENCH_CLOCK_BASIS=unlocked`; `SOLEXBENCH_CLOCK_SETTLE=0`
disables it so the effect can be measured against its own absence in one session.

**Why this is not methodology drift, and say so when asked.** The measured
quantity does not move: the timed region is still upstream's `warmup_runs=10,
iterations=50` around the same callable, and the settle runs entirely before the
first clock sample and the first timing event. That is the distinction from
§4.3 option 1 — lengthening the window to ~10,000 iterations *would* change what
is measured and break comparability with upstream, which is why it was declined.
Settling changes only the state the card is in when measurement starts.

**Measured A/B, GPU 0, `L1/009`, both arms back to back in the same session**
(so neighbour load is held constant; GPUs 1–7 were running a reference sweep
throughout). The no-settle arm reproduced at 59.4% against the 57.8% measured
when the fleet was idle, which incidentally bounds the neighbour effect at ~1.6
points:

| | refusal | median spread | q90 | max | anchors |
|---|---|---|---|---|---|
| no settle | 59.4% | 0.00948 | 0.04678 | 0.05838 | 11/16 |
| **settle** | **4.7%** | **0.00294** | **0.00631** | **0.00926** | **16/16** |

**The positional effect is flattened**, which was the sharpest test:
first-three-workloads refusal **9/12 → 1/12** (median 0.0297 → 0.0048), and
position ≥3 **29/52 → 2/52** (median 0.0089 → 0.0029).

**And the steady-state tail I could not explain is gone with it.** I flagged 56%
refusal at position ≥3 as possibly a second mechanism; it fell to 3.8% under the
same fix, so it was the same ramp seen at a different phase, not a second one.
One problem, one card — not proof, but the open question is narrower now.

**Threshold unchanged at 0.0078**, as instructed. It was not retuned and did not
need to be: the realised distribution moved under it rather than the other way
round.

**Cost.** +11.7 s on an 86.4 s problem (**+13.5%**), settle median 186 ms, max
778 ms, 64/64 settled without hitting a cap. Extrapolated over the full
candidate sweep — 235 problems, 3957 workloads, 5 variants = 19,785 timed calls
at ~146 ms added each: **≈48 min serial, ≈6 min at 8-way shard**.

**Two implementation traps, both hit and both worth keeping written down.**

1. *Stability must be judged over the window's own duration.* The first version
   exited when three consecutive ~10 ms samples agreed within the band. That is a
   test on the slope, and it passes early on a slow ramp: 2340→2348→2355 spans
   0.64% over 30 ms and reads as settled, while the same ramp across the ~100 ms
   window is 6% and is refused. It exited after a median of **12 ms / 6
   iterations** and made things **worse — 53.1% → 78.1% refusal**. The horizon is
   now `window_iters × measured per-iteration cost`, recorded per measurement as
   `settle_stability_horizon_ms`.
2. *A self-contradictory coverage check made the criterion dead code.* Having
   filtered samples to those within the trailing horizon, it then asked whether
   the oldest of *those* was at least a horizon old — satisfiable only by exact
   equality. Every settle silently ran to the 1000 ms cap. It still worked
   (1.6% refusal) but cost 5.6× more and stamped `settled: false` on every good
   measurement. Fixed to check coverage against the oldest sample overall;
   `settled` is now true 64/64 and the cost fell from +66 s to +11.7 s.

Both are pinned by tests, including `test_a_slow_ramp_does_not_read_as_settled`.

**Still unverified:** one problem on one card. The caps (1000 ms / 20000 iters)
have never been hit in a real run, so the capped path is covered only by unit
tests. And the settle's own GPU work is not free of side effects in principle —
it warms the card for the *next* workload too, which is a direction the A/B
cannot separate.

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
| 2026-08-14 | `mia1-p02-g46`, MI355X, GPU 0 only | Solution-language coverage: one real seed each for `aiter`, `ck`, `ck_tile`, `hipblaslt`, `miopen`, run end to end through packager + eval_driver, all workloads passing | `reference/seeds/*.json`, `artifacts/backends/`, TODO N3 closed, `docs/backend-coverage.md` rewritten; two packager defects fixed (`--use_fast_math`, `-lcuda` substitution) with a regression test; `agent_eval.py --solution` added. **No timing artifact produced — these runs had no locked clock (`f_lock_mhz: null`) and none of their latencies is usable for scoring.** |

| 2026-08-15 | `mia1-p02-g46` (+g45, g05 anchors), MI355X | MI355X bound quality: seven issues investigated, six adversarially reviewed, three corrections shipped as **manifest v4**. Three hypotheses refuted by measurement (D66, D67, D68). Gates MI355X 03: 2 failures → 1. Tests 921 → **1149** passed / 168 skipped. Closed out by rebuilding the chain once — tier (gated against `authoritative-merged`), `manifest-v4.json`, `cross-checks.md` — determinism proved first; 220/235, 3717, coverage 235/235. | `artifacts/09-MI355X/manifest-v4.json`, `artifacts/03-MI355X/t_sol_traffic.json`, `artifacts/03-MI355X/llc/llc-bandwidth-gpu0.json`, `artifacts/06-MI355X/card-assignment.json` (**nothing consumes it yet**), 419 rewritten `artifacts/10` score files (the stale-stamp defect is fixed: they now carry a separate `_backfill_provenance` beside the untouched measurement block, D70). D63–D72. **`tests/leaderboard` was not run** — no `leaderboard/.venv` in this worktree and fastapi is not importable on this node; 4 new worker tests have never executed. |

**Next session, on MI350X:** start with `TODO.md` item 1 (the anchor), and run
`scripts/gpu_exclusive.py --gpu 0` before you time anything.

**Next session, on MI355X:** `docs/TODO-MI355X.md` §13. Two of the items there
need a maintainer's signature before any code moves (`L2__050`'s precision
pricing, and the `merge_authoritative_tb` tiebreak, which is **measured
inflating**). The cheapest real work left is arithmetic, not measurement:
**re-ingest the board** against the re-scored tree and **re-run task 03 and paste
it**, neither of which was done after the on-card re-scoring finished.
