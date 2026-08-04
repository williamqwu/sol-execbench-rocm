# STATE.md — progress ledger

**Single source of truth for progress.** Update as you go, not at the end.
A session can be interrupted at any point; whatever is written here is what the
next session inherits.

Rules: record real output, not summaries of intent. If something failed, say so
and say how. Never mark a task `done` without pasting its acceptance-check
output.

> Session 1 ran on `mia1-p02-g10` (8× **MI355X**). Session 2 ran on
> `gbt350-odcdh1-a08-1` (8× **MI350X**). **Session 3 is back on `mia1-p02-g10`,
> the MI355X node.** `HANDOFF.md` says which session-1 results transfer. Numbers
> from a part other than the current node's are kept where they are useful as a
> second data point, and are always labelled.

> **Read this before reading any number below.** The artifacts tree currently
> mixes two parts, deliberately and visibly:
>
> | artifacts | part | why |
> |---|---|---|
> | `00/`, `01/` | **MI355X** (this node) | regenerated in session 3; `00-MI350X/` and `01-MI350X/` hold the session-2 record |
> | `02/`, `03/`, `04/`, `05/`, `07/`, `08/` | **MI350X** | measured in session 2 and not yet re-derived here |
> | `10/` | **MI355X** | agent scoreboard, session 3 |
>
> Every artifact carries its own part in its provenance block, so no reader has
> to infer this. What it means in practice: **`artifacts/03/t_sol.json` is
> MI350X at F_LOCK 1300 and must not be used to score an MI355X measurement.**
> `scripts/score_solutions.py` enforces that by comparing the artifact's `part`
> against the live node and refusing rather than rescaling.

---

## Where this stands

**On MI350X the benchmark is measured and the manifest is frozen.**
`manifest-v1` scores **220 of 235 problems / 3717 workload instances** on MI350X
at F_LOCK = 1300 MHz. The 15 that are not scoreable are the NVFP4 Quant
problems, whose *references* fail on ROCm; they are in `artifacts/deferred.json`
with the error text quoted from the calibration artifact, and every count in
every document quotes that file.

**On MI355X — this node, session 3 — it is not.** `manifest-v1` is an MI350X
artifact and cannot score a measurement taken here: F_LOCK is 1640 rather than
1300, so every `T_b` in it is a wall-clock time at the wrong clock.
`scripts/score_solutions.py` compares an artifact's `part` against the live node
and refuses rather than rescaling. What session 3 has re-derived is listed in the
task table below; `T_b` for this part is measured but **not yet anchor-verified**
(blocker B2).

What a consumer needs to know before using either:

* **Correctness runs against `artifacts/05/workloads/`**, not the dataset's own
  tolerances. Opt in with `SOLEXBENCH_WORKLOADS_ROOT`. Under upstream's B200
  tolerances the same references fail 8 workloads of `L2/033`.
* **`T_SOL` comes from one of two derivations and every workload says which.**
  SOLAR's roofline over the traced graph, or the traffic the definition itself
  declares over DRAM bandwidth. Neither dominates; the manifest takes the max
  of the two that survive being checked against the measurement.
* **Every `T_b` was re-timed on one GPU alone.** On MI350X the eight GPUs span
  1242–1307 MHz at the same determinism setting; on MI355X they span
  **1318–1644**, which is a 25% spread and a much stronger reason (D16).
* **An agent baseline now exists, for MI355X only** — task 10,
  `artifacts/10/scoreboard.json`. It is not comparable to upstream's median of
  0.732: that figure is a `sol_score_v1` number and this one currently rests on a
  weaker basis, because B2 holds. `artifacts/09/score-distribution.json` remains
  what it says it is — evidence the MI350X scale is well formed, not an agent
  result.

## Environment (current node — session 3)

| Field | Value |
|---|---|
| Node | `mia1-p02-g10` |
| GPUs | 8× AMD Instinct **MI355X**, `gfx950:sramecc+:xnack-`, 288 GiB, 256 CUs each |
| Power cap | **1400 W** per GPU (MI350X node: 1000 W) |
| Max GFX clock | **2400 MHz** (MI350X: 2200 MHz) |
| Cooling | liquid (MI350X: air) |
| ROCm version | 7.2.0 / HIP `7.2.26015-fc0010cf6a` |
| torch version + build | `2.9.1+rocm7.2.0.git7e1940d4` — the pinned version, natively |
| triton | `3.6.0+git42270451` from `/sgl-workspace/triton-custom` — **a dev checkout, not a release** (D14) |
| Clock-lock setting | see task 01 below |
| **F_LOCK (achieved)** | see task 01 below |
| Sibling-GPU interference | see task 01 below |
| Dataset present | yes — 235 problems, L1=94 L2=82 Quant=33 FlashInfer-Bench=26, round-trip verified |
| FlashInfer blobs | yes — 304 external safetensors blobs |
| Measurement environment | **`env/solb-native`** — no docker on this node (D15) |
| Node exclusivity | **exclusive at session start** — no other KFD processes, all GPUs at `auto` |

### GPU index mapping — confirmed live on this node

```
torch -> amdsmi    {0: 3, 1: 0, 2: 2, 3: 1, 4: 7, 5: 4, 6: 6, 7: 5}
torch -> rocm-smi  {0: 3, 1: 0, 2: 2, 3: 1, 4: 7, 5: 4, 6: 6, 7: 5}
```

Identical to the permutation in `scripts/gpu_map.py`'s docstring, which was
written for this node. Worth re-confirming rather than trusting, because
indexing `amdsmi` handles by a torch index samples a *different physical GPU* —
in task 01 that means reading an idle GPU's clock while the load runs elsewhere,
producing a low, stable, entirely plausible "floor" that is fiction.

Concretely observed while the first floor ran: `--gpu 0` put the load on
`rocm-smi` device **3**, exactly as the mapping predicts.

---

## Task status

| ID | Task | MI350X | MI355X (this node) | Artifacts | Notes |
|---|---|---|---|---|---|
| 00 | Node acceptance | `done` | `done` | `artifacts/00/`, `00-MI350X/` | 13 checks, 0 failed on both; MI355X rooflines reproduce session 1 to within 1% |
| 01 | Clock calibration (F_LOCK) | `done` 1300 | `done` **1640** | `artifacts/01/`, `01-MI350X/` | achieved, not requested. D16: on MI355X only two of eight GPUs obey the request |
| 02 | Harness port validation | `done` | `done` | `artifacts/02/` | 3717/3717 non-deferred workloads pass under AMD tolerances. Port is part-independent |
| 03 | SOL bounds (T_SOL) | `done` | `in-progress` | `artifacts/03/` | MI350X: 235/235, two derivations, source recorded. MI355X: **re-derived after the fp32-rate fix** — 166/235 problems from SOLAR, and all 90 bound violations went to 0 (D25). The traffic tier still owes the other 69 |
| 04 | rocprofiler shim | `done` | `n/a` | `artifacts/04/` | median divergence −0.61% over 1430 pairs; clock domain verified. Not part-specific |
| 05 | Tolerance calibration | `done` | `inherited` | `artifacts/05/` | 3717/3957 AMD-derived. Numerics of the same gfx950 ISA; not re-derived here |
| 06 | Baselines (T_b) | `done` | `blocked` | `artifacts/06/`, `06-MI350X/` | MI350X: 220 anchored, 336/349 anchor checks pass. MI355X: selection 223/235 and 132 re-timed, but **not anchor-verified** — blocker B2 |
| 07 | Quant / MXFP4 | `done` | `blocked` | `artifacts/07/`, `artifacts/deferred.json` | 15 NVFP4 deferred with evidence; 220 ship. On MI355X the other 18 are blocked by B1 (`artifacts/blocked.json`) |
| 08 | Red team | `done` | `done` | `reference/exploits/`, `artifacts/08/` | 28/28 replay cases, 0 false positives on 235 references. Extended in session 3: the screen now screens filenames, after D21 |
| 09 | Release | `done` | `in-progress` | `artifacts/09/` | MI350X: **manifest v1, 220/235 problems, 3717 workloads scoreable**. MI355X: `manifest-MI355X-v1.json` builds and reports its own gaps; 0 scoreable while B2 holds |
| 10 | Agent scoreboard | `not-run` | `in-progress` | `artifacts/10/` | **pilot done and scored**, 24 problems × 2 harnesses, acceptance 24 checks / 0 failed. Full 404-session sweep paused for this merge |

Read the two status columns as two independent ports of the same benchmark. A
task `done` on MI350X is not done here whenever its result is a measured
quantity: F_LOCK, T_SOL in milliseconds and T_b all differ by part, and
`scripts/score_solutions.py` refuses an artifact whose `part` is not the live
node's rather than rescaling it.

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

**Task 06 correction.** Session 2's entry said "selection sweep running on 8
GPUs; authoritative pass after" and left the status `in-progress`. A fresh clone
has no `artifacts/06/` at all, so whatever ran was never committed. Corrected to
`not-started` rather than left as inherited optimism — this is the same class of
loss as D6, where a green test suite hid a third of the port never reaching git.
It is also the binding constraint on the scoreboard: `S` is not computable for
any problem without `T_b`.

### Task 00 acceptance output (2026-08-04, **MI355X**, session 3)

```
Acceptance check — task 00

  [PASS              ] node-report.json exists
  [PASS              ] node report has provenance
  [PASS              ] 8 GPUs present                         found 8
  [PASS              ] all GPUs are gfx950
  [PASS              ] power caps probed on every GPU          8/8
  [PASS              ] power caps uniform (±5%)                all 1400.0
  [PASS              ] max GFX clocks probed on every GPU      8/8
  [PASS              ] max GFX clocks uniform (±5%)            all 2400
  [PASS              ] idle temperatures probed on every GPU   8/8
  [PASS              ] idle temperatures uniform (±25%)        all 38
  [PASS              ] HBM roofline measured
  [PASS              ] BF16 GEMM roofline measured
  [REQUIRES-JUDGEMENT] dataset layout matches audit            confirm categories L1=94 L2=82 Quant=33 FlashInfer=26

  13 checks, 0 failed, 1 require human judgement
```

Judgement item resolved: `find -L data -name definition.json` returns
**235**, split 94 / 82 / 33 / 26, against real files after materializing.

The 1400 W caps and 2400 MHz ceilings are the positive identification that this
is the MI355X part and not the MI350X one, independent of the device name string.

**Rooflines at DEFAULT clocks** (reference points only — per task 00's guard
rails these are NOT scoring ceilings and must not be cited downstream):

| | MI355X (session 3, this node) | MI355X (session 1, same node) | MI350X (session 2) |
|---|---|---|---|
| HBM copy | **4.91 TB/s** (61.4% of 8.0 spec) | 4.87 TB/s (61%) | 4.53 TB/s (56.7%) |
| BF16 GEMM | **1437.6 TFLOPS** (57.1% of 2517 @2.4 GHz) | 1433 TFLOPS (57%) | 1168 TFLOPS (50.6% of 2307 @2.2 GHz) |

Session 3 reproduces session 1 to within 0.8% on bandwidth and 0.3% on GEMM,
measured months apart by a different code path (`env/solb-native` rather than the
container). That is the strongest evidence available that the native environment
of D15 is equivalent to the container it replaces — the alternative would have
been to trust the version assertion alone.

### Task 00 acceptance output (2026-08-03, MI350X — archived, session 2)

```
Acceptance check — task 00

  [PASS              ] node-report.json exists
  [PASS              ] node report has provenance
  [PASS              ] 8 GPUs present                         found 8
  [PASS              ] all GPUs are gfx950
  [PASS              ] power caps probed on every GPU         8/8
  [PASS              ] power caps uniform (±5%)               all 1000.0
  [PASS              ] max GFX clocks probed on every GPU     8/8
  [PASS              ] max GFX clocks uniform (±5%)           all 2200
  [PASS              ] idle temperatures probed on every GPU  8/8
  [PASS              ] idle temperatures uniform (±25%)       all 60
  [PASS              ] HBM roofline measured
  [PASS              ] BF16 GEMM roofline measured
  [REQUIRES-JUDGEMENT] dataset layout matches audit           confirm categories L1=94 L2=82 Quant=33 FlashInfer=26

  13 checks, 0 failed, 1 require human judgement
```

Judgement item resolved: 94/82/33/26 = 235 verified against real files after
materializing the dataset (deviation D1).

**Rooflines at DEFAULT clocks** (reference points only — per the task's guard
rails these are NOT scoring ceilings and must not be cited downstream):

| | MI350X (this node) | MI355X (session 1) |
|---|---|---|
| HBM copy | 4.53 TB/s (56.7% of 8.0 spec) | 4.87 TB/s (61%) |
| BF16 GEMM | 1168 TFLOPS (50.6% of 2307 spec @2.2 GHz) | 1433 TFLOPS (57% of 2500 @2.4 GHz) |

The MI350X spec peak is 2307 TFLOPS, not 2500: same die, lower clock. Comparing
an MI350X measurement against the MI355X peak would have understated the
achieved fraction by 9%. `scripts/roofline_probe.py` now looks the peak up per
part (`solexbench_rocm/parts.py`) instead of hardcoding one.

### Task 01 results (2026-08-04, **MI355X**, session 3) — F_LOCK = 1640 MHz on GPU 0

**Step 1 — sustained clock floors, UNLOCKED** (15 min saturating BF16 GEMM, p5
of the final 5 minutes, one GPU at a time so per-GPU variation is not confounded
with cross-GPU power coupling):

| Run | p5 | p50 | min | max | session 1 |
|---|---|---|---|---|---|
| GPU 0, siblings idle | **1725** | 1729 | 1723 | 1753 | 1725 |
| GPU 1, siblings idle | 1734 | — | — | — | 1734 |
| GPU 2, siblings idle | 1751 | — | — | — | 1757 |
| GPU 0, all 7 siblings loaded | **1724** | — | — | — | 1728 |

Session 1's floors are reproduced to within 6 MHz on every GPU. Spread across
GPUs is **26 MHz**, under the task's 50 MHz threshold (MI350X was 55 MHz).

The loaded case is the striking one: seven siblings at ~1.1–1.4 kW each moved
GPU 0's floor by **1 MHz**. At 8×1400 W this node does not couple.

**Step 2 — the determinism sweep, and it does not do what MI350X did.**

`--setperfdeterminism X` on GPU 0 (`determinism-sweep.json`):

| requested | achieved (median) | min | ratio | power |
|---|---|---|---|---|
| 1100 | 942 | 941 | 0.86 | 712 W |
| 1250 | 1050 | 1048 | 0.84 | 762 W |
| 1350 | 1121 | 1121 | 0.83 | 793 W |
| 1500 | 1214 | 1213 | 0.81 | 871 W |
| **1600** | **1593** | 1591 | **1.00** | 1219 W |
| **1700** | **1696** | 1693 | **1.00** | 1337 W |
| 1800 | 1728 | 1725 | — | **1400 W** |
| 1900 | 1729 | 1727 | — | **1400 W** |
| 2000 | 1730 | 1727 | — | **1400 W** |
| 2200 | 1730 | 1727 | — | **1400 W** |
| 2400 | 1728 | 1724 | — | **1400 W** |

Three regimes, not one:

1. **Below 1500 the MI350X rule holds** — achieved ≈ 0.81–0.86 · requested.
2. **At 1600–1700 the request is obeyed** to within 0.4%. The jump is not
   smooth: 100 MHz more *request* between 1500 and 1600 buys **379 MHz** more
   *achieved* clock. That is a DPM state boundary, not a curve.
3. **At 1800 and above the part pins to the 1400 W cap** and lands on the same
   ~1729 MHz whether asked for 1800 or 2400. In that regime ambient conditions
   set the clock, which is precisely what a lock exists to prevent.

**Step 3 — F_LOCK = 1640 achieved, at setting 1650.**

1650 is the round number ≥50 MHz below the lowest observed floor (1724, busy).
It sits inside the obedient window, and it draws 1269 W of 1400 — so the setting
binds and not the power limit. 1700 was rejected for margin: it draws 1337 W,
63 W from the cap, and a lock one warm afternoon from becoming power-bound is
not a lock.

Verified under load: `expected 1650 MHz, observed median 1644.0 MHz (drift 6.0)
PASS`. F_LOCK is recorded as **1640** — a round number 0.24% below the measured
median and below the measured minimum of 1642, which makes every T_SOL
marginally conservative rather than marginally optimistic.

**Step 4 — stability at F_LOCK:** `CV = 0.0041` over 30 trials in separate
processes (gate 0.02). Noise is ~5× below the gate. (MI350X: 0.0034; MI355X
session 1: 0.0015.)

**Step 5 — sibling interference:**
```
baseline: timing GPU 0, siblings idle
loaded: siblings [1, 2, 3, 4, 5, 6, 7] under sustained load
  sibling power now: [1400, 1088, 1101, 1100, 1115, 1100, 1070] W

quiet 0.1094 ms -> busy 0.1094 ms  (+0.02%)
verdict: negligible
Sweeps and authoritative timing can share the node.
```

Seven siblings drawing ~7.9 kW between them moved GPU 0's timing by +0.02%.
**Scheduling consequence, acted on:** the task 10 agent sweep runs on GPUs 1–7
while scoring runs serially on GPU 0, concurrently.

### Task 01 acceptance output (2026-08-04, MI355X, session 3)

```
Acceptance check — task 01

  [PASS              ] F_LOCK recorded in STATE.md                       1640 MHz
  [PASS              ] F_LOCK present in CLOCK_LOCK_PRESETS              1640 MHz for MI355X
  [PASS              ] STATE.md and CLOCK_LOCK_PRESETS agree on F_LOCK   both 1640 MHz
  [PASS              ] clock floor sampled on >=3 GPUs                   4 GPUs
  [PASS              ] F_LOCK at or below lowest observed floor          F_LOCK 1640 <= min p5 1724
  [PASS              ] stability measured
  [PASS              ] timing CV < 2%                                    CV=0.0041
  [PASS              ] sibling interference measured
  [PASS              ] interference has a stated scheduling consequence  negligible

  9 checks, 0 failed, 0 require human judgement
```

The first two checks and the agreement check are **new**, added because the
original ones could not fail. See F17.

### Task 01 results (2026-08-03, MI350X — archived, session 2) — F_LOCK = 1300 MHz

**Step 1 — sustained clock floors, UNLOCKED** (15 min saturating BF16 GEMM, p5
of the final 5 minutes, one GPU at a time so per-GPU variation is not
confounded with cross-GPU power coupling):

| Run | p5 | p50 | min | peak power | peak junction |
|---|---|---|---|---|---|
| GPU 0, siblings idle | **1390** | 1396 | 1369 | 1001 W | 72 °C |
| GPU 1, siblings idle | 1367 | 1377 | 1286 | 1001 W | 79 °C |
| GPU 2, siblings idle | **1335** | 1338 | 1265 | 1001 W | 79 °C |
| GPU 0, all 7 siblings loaded | 1400 | 1407 | 1317 | 1002 W | 65 °C |

Every run sits at the 1000 W cap with junction ≤79 °C — well below the 100 °C
slowdown point. **MI350X is power-limited, not thermally limited**, same as
MI355X but at a 400 W lower budget, which is the whole reason its floor is
~350 MHz lower (1335–1390 vs 1725–1757).

Per-GPU spread is **55 MHz**, over the 50 MHz threshold in the task, so F_LOCK
had to be chosen for the worst GPU rather than the best.

**Step 2/3 — and the finding that changed how F_LOCK is defined here.**

Applying the MI355X procedure directly produced a **failed verification**:

```
locking GPU 0 -> 1250 MHz
  8/8 card(s) at perf_determinism
locked

expected 1250 MHz, observed median 1049.0 MHz (drift 201.0)
FAIL: drift exceeds 50 MHz
```

On MI350X, `rocm-smi --setperfdeterminism X` does **not** yield X. It yields
about 0.81–0.85·X, rock-steadily. That is not a documented behaviour we could
look up, so it was measured
(`clock_calibrate.py determinism-sweep`, artifacts `01/determinism-sweep*.json`):

| requested | achieved (median) | min | power |
|---|---|---|---|
| 1100 | 934 | 932 | 666 W |
| 1250 | 1049 | 1048 | 729 W |
| 1350 | 1116 | 1114 | 770 W |
| 1500 | 1220 | 1194 | 836 W |
| 1600 | **1303** | 1296 | 885 W |
| 1700 | 1380 | 1376 | 947 W |
| 1900 | 1403 | 1397 | **1000 W** |
| 2200 | 1402 | 1397 | **1000 W** |

Two things fall out of that table:

1. **Requested ≠ achieved**, so F_LOCK must be the *achieved* number. Recording
   the requested one would overstate the clock by ~23% and make every T_SOL and
   every T_b wrong by that factor, plausibly and undetectably.
2. **Above ~1900 the part stops obeying the setting and pins to the 1000 W
   cap**, landing on the same ~1400 MHz whether you ask for 1900 or 2200. In
   that regime the clock is set by ambient conditions, not by us, which is
   precisely what a lock is supposed to prevent.

**Setting = 1600 MHz, F_LOCK = 1300 MHz.** 1600 was chosen over 1700 for power
margin: 1700 draws 947 W of a 1000 W cap on two of the three GPUs sampled, and
a lock that is one warm afternoon from becoming power-bound is not a lock. At
1600 the part draws 868–933 W, so the *setting* binds, not the power limit.

Achieved clock at setting 1600, all eight GPUs measured under sustained load:

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| median MHz | 1303 | 1295 | 1264 | 1307 | 1279 | 1296 | 1285 | **1242** |
| min MHz | 1296 | 1293 | 1250 | 1295 | 1278 | 1280 | 1283 | 1217 |

Each GPU is individually stable (min within ~20 MHz of its own median) but they
differ from each other by up to 65 MHz (5%). **Determinism mode gives each GPU
its own steady clock, not a node-wide one.** Consequence, which is now a
standing rule: authoritative timing is pinned to **GPU 0** and every timing
artifact records its GPU. Task 06's candidate sweep may shard across GPUs 1–7
because it only *selects* a variant; the winner is re-timed on GPU 0.

F_LOCK is 1300 rather than 1303 — a round number 0.2% below GPU 0's measured
median, which makes T_SOL marginally conservative. Cross-check 4
(T_SOL ≤ best measured) will catch it empirically if that is ever the wrong
call.

**Step 4 — stability at F_LOCK:** `CV = 0.0034` over 30 trials in separate
processes (gate 0.02). Timing noise is ~6× below the gate. (MI355X: 0.0015.)

**Step 5 — sibling interference:**
```
baseline: timing GPU 0, siblings idle
loaded: siblings [1, 2, 3, 4, 5, 6, 7] under sustained load
  sibling power now: [871, 888, 934, 875, 882, 883, 895] W

quiet 0.1464 ms -> busy 0.1462 ms  (-0.11%)
verdict: negligible
Sweeps and authoritative timing can share the node.
```
Seven siblings drawing ~6.2 kW between them moved GPU 0's timing by −0.11%.
This also confirms the 1600 setting keeps binding under full-node load: if
power had started to bind, the timing would have moved.

### Task 01 acceptance output (2026-08-03, MI350X — archived, session 2)

```
Acceptance check — task 01

  [PASS              ] F_LOCK recorded in STATE.md                       1300 MHz
  [PASS              ] clock floor sampled on >=3 GPUs                   4 GPUs
  [PASS              ] F_LOCK at or below lowest observed floor          F_LOCK 1300 <= min p5 1335
  [WARN              ] per-GPU floor spread >50MHz                       1335-1400 MHz; F_LOCK must suit the worst
  [PASS              ] stability measured
  [PASS              ] timing CV < 2%                                    CV=0.0034
  [PASS              ] sibling interference measured
  [PASS              ] interference has a stated scheduling consequence  negligible

  8 checks, 0 failed, 0 require human judgement
```

The WARN is expected and is acted on, not waived: the 65 MHz spread is exactly
why authoritative timing is pinned to GPU 0 (see *Decisions taken*).

### Task 10 pilot results (2026-08-04, MI355X, session 3)

24 problems — 6 sampled **evenly** per category, not the first 6 — against both
harnesses, K=5 verify attempts, 30-minute wallclock cap. 48 sessions in
**157 minutes** across 7 GPUs. Scored serially on GPU 0 at F_LOCK 1640.

| | claude-code (`claude-opus-5`) | codex (`gpt-5.5`) |
|---|---|---|
| Problems solved | 15/24 | 18/24 |
| **Excluding Quant** (blocker B1) | **15/18 (83%)** | **18/18 (100%)** |
| Workloads passed | 277/309 (89.6%) | 325/325 (100%) |
| Median speedup vs reference | 1.83× | 1.56× |
| Median headroom reclaimed | 0.457 | 0.318 |
| Mean wallclock per problem | 28.1 min | 12.2 min |
| **Mean verify attempts used** | **0.9 of 5** | **4.2 of 5** |
| Input tokens | 2.2 M | 45.8 M |
| Cost reported | $17.85 (4 of 24 sessions priced) | none reported |
| Languages chosen | triton 17, pytorch 1 | triton 14, pytorch 11 |
| Reference copies | 0 | 0 |

**The headline number is the verify-attempt row, not the solve rate.** Claude Code
averaged 0.9 verifications of its 5 and hit the 30-minute cap on most problems;
Codex averaged 4.2 and finished in 12 minutes. The two are not being measured at
the same budget in any meaningful sense — the cap binds hard on one and not the
other. Where both produced a working kernel the *quality* is comparable
(headroom 0.46 vs 0.32, and the two medians are over different problem subsets).

So the defensible claim from this pilot is: **at a 30-minute cap on this node,
Codex converges and Claude Code frequently does not get to iterate.** It is not
evidence about the underlying models' kernel-writing ability, and the task file's
guard rail applies — the budget is part of the measurement.

Every problem unsolved by *both* harnesses is one of the 6 Quant problems, i.e.
all of them are blocker B1 and none is a kernel nobody could write.

**Why Claude Code runs out of time, from its own transcripts.** It is not
thinking slowly or failing to understand the task. It is choosing a different
strategy: on `L1__003_lm_head_projection`, of 21 shell calls it wrote and ran
`tune.py`, `tune2.py`, `tune3.py`, `scan_m.py`, `probe2.py`, `probe_aiter.py`
and `sweep_all.py` — its own autotuning harness — and read AMD's tuned GEMM
tables out of `/sgl-workspace/aiter/hsa/gfx950/bf16gemm/*.csv`. The sweeps it
launched carried internal budgets of `timeout 3600` and `timeout 5400`, i.e. 60
and 90 minutes, **inside a 30-minute session cap**.

So it invests the budget in offline tile search and expects to verify once at the
end; the cap arrives first. Codex instead writes a plausible kernel early and
converges against `./verify`, which is why it averages 4.2 verifications and
finishes in 12 minutes.

That reframes the headline: the gap is a **strategy/budget interaction**, not a
difference in kernel-writing ability. Neither strategy is wrong — offline
autotuning is what a human performance engineer does — but one of them cannot pay
off inside 30 minutes. The obvious follow-up experiment is the same pilot at a
90-minute cap, which would test that explanation directly. It has not been run;
changing the cap mid-flight would make the full sweep incomparable to the pilot.

**Cost accounting is incomplete and the reason is structural.** A session killed
at the wallclock cap never emits its result object, which is where Claude reports
cost — so 20 of 24 claude sessions are unpriced. Tokens are recovered from the
CLI's own transcript (D18) so effort is still comparable; cost is left null rather
than inferred from a guessed rate. Codex reports no cost at all.

**`speedup_vs_reference` must not be read as a performance figure.** The largest
in this run is **7055×** (`FlashInfer-Bench__014`, `t_ref` = 506 ms against
`t_k` = 0.07 ms). The FlashInfer references are executable specifications, not
optimized baselines, so a fused kernel beats them by three orders of magnitude
and the number says almost nothing about hardware utilization. This is precisely
why upstream scores against `T_b` and `T_SOL` instead, and it is the strongest
argument in this repo for finishing task 06.

### Pilot-derived budget for the full run

| | measured in pilot | extrapolated to 220 problems × 2 harnesses |
|---|---|---|
| Sessions | 48 | 440 |
| Wallclock (7 GPUs) | 2.6 h | ~24 h |
| Agent GPU-hours | 16.1 h | ~148 h |
| Scoring (serial, GPU 0) | 2.4 h | ~22 h |
| Claude cost, priced sessions only | $17.85 / 4 sessions ≈ $4.5 each | ~$1000 for claude alone |
| Input tokens | 48 M | ~440 M |

220 rather than 235 because the 15 NVFP4 problems are deferred; and Quant is
unscoreable until B1 is fixed, which would reduce it to 187 in practice.

---

## Blockers

### B2 — T_b was measured under CPU contention and fails the anchor property; S is not published

T_b was carried further than it has ever been in this project: candidate
selection completed for **223 of 235 problems** (`artifacts/06/candidates/`,
223 ok / 3 failed / 107 min sharded across GPUs 1–7), and the authoritative
re-time on GPU 0 covered the first 12 before this was found.

`scripts/verify_anchor.py` then rejected the scale outright:

```
anchor property   13/204        (rule: submitting T_b's own implementation scores 0.5 +- 0.03)
reference < 0.5   203/204
T_SOL violations  0
```

The mechanism is visible in a single check:

```
variant v1_eager    t_b_ms 0.01372   (recorded anchor)
                    t_k_ms 0.06876   (the same code, re-measured)
score_of_anchor 0.144            (should be 0.5)
```

Re-running the *identical* variant takes 5× longer than the recorded anchor. So
`T_b` is not reproducible, and `S` built on it would be wrong by that factor
while looking entirely plausible.

**The cause is a scheduling mistake, and it was mine.** Both the authoritative
re-time and the anchor verification were run on GPU 0 *while the 404-session
agent sweep saturated the node's 120 CPUs*. Task 01 measured GPU-to-GPU
interference at +0.02% and the D20 addendum already recorded that the CPUs are
the resource that actually contends — and then this violated that rule anyway.
Triton autotuning and `torch.compile` are CPU-bound, so a compile-heavy timing
run next to seven compile-heavy agents measures the CPU scheduler.

Actions taken rather than deferred:

- `artifacts/06/anchor-verification.md` renamed to
  **`anchor-verification-VOID-contended.md`**, so a future session cannot mistake
  a failing report for a passing one, and cannot mistake either for a clean
  measurement.
- The manifest was rebuilt **without** `T_b`, and `scripts/backfill_scores.py`
  rolled the 98 `sol_score_v1` records back to `sol_headroom`. Each keeps
  `score_basis_history: ["sol_score_v1"]`, so the retraction is on the record
  instead of being invisible.
- `verify_artifacts --task 10` now reads the anchor report's **verdict** rather
  than its existence. The earlier version tested only that the file was there,
  which a failing report satisfies exactly as well as a passing one.

**What it takes to finish:** re-run `authoritative_tb.py` on an otherwise idle
node, then `verify_anchor.py`, then `backfill_scores.py`. No agent work needs
repeating — `T_k`, `T_ref` and every pass/fail verdict were measured on GPU 0 at
F_LOCK before the sweep started and remain valid. That is the whole reason the
backfill exists as a separate step from scoring.

For the record, the S values that were briefly computed had median **0.500**
(claude-code 0.503, codex 0.499, n=49 each) over just three problems —
`FlashInfer-Bench__001`, `005`, `009`, all rmsnorm or GEMM, i.e. exactly the
shapes where `torch.compile` and hipBLASLt are already near-optimal and S≈0.5 is
the expected answer. Even had the anchor held, three problems would not have
supported a headline.

### B1 — this node runs Python 3.10; the project requires 3.12, and all 33 Quant references need ≥3.11

`pyproject.toml` declares `requires-python = ">=3.12"`, matching upstream. Every
interpreter on this node — including `/opt/venv/bin/python3`, which owns the
pinned ROCm torch build — is **Python 3.10.12**. No 3.11 or 3.12 is installed.

The consequence is narrow and total: `StrEnum` arrived in 3.11, and

```
$ grep -rl StrEnum data/SOL-ExecBench/benchmark/*/*/definition.json | wc -l
33
```

is **exactly the 33 Quant problems, and nothing else**. Their references fail to
import before any submission is involved:

```
File ".../_reference.py", line 5, in <module>
    from enum import StrEnum
ImportError: cannot import name 'StrEnum' from 'enum' (/usr/lib/python3.10/enum.py)
RuntimeError: Failed to exec reference code
```

**Why this is recorded rather than fixed.** The obvious repair — define
`class StrEnum(str, Enum)` — is *not* equivalent to CPython's:

```python
>>> str(A.X)      # naive shim
'A.X'
>>> str(A.X)      # real 3.11 StrEnum
'x'
```

For `Quant__001` the enum is used only as dict keys and attribute lookups, so the
shim is behaviourally identical *there* — verified by reading it. That is one
problem of 33. Shipping a shim whose semantics differ from the real class in the
`str()` path, across 33 references nobody has read line by line, is exactly the
kind of quiet methodology change prime directive 7 forbids: any reference that
formats the enum into a value would compute something different, and the
"correct" answer would silently stop being the dataset's.

Fixing it properly means a 3.12 interpreter with the same ROCm torch build, which
is an environment change, not a code change (prime directive 6).

**The evidence that this is the node and not the port** is already in the repo:
`artifacts/deferred.json`'s own provenance block records `"python": "3.12.3"` on
`gbt350-odcdh1-a08-1`, session 2's MI350X node, where all 33 Quant problems
evaluated normally. Nothing about the port changed; the interpreter did.

**Where the 18 are recorded.** In a new `artifacts/blocked.json`, deliberately
*not* in `artifacts/deferred.json`. The latter states what the port does not
ship, and its `shipped_total: 220` is quoted by every document that states a
count; folding an environment limitation into it would restate the port's
coverage as 202 when it is 220. `scripts/check_coverage.py` now reads both, so
15 deferred + 18 blocked accounts for all 33 Quant problems with no overlap and
a sweep that skips Quant can still exit zero.

**Effect on task 10, stated wherever the numbers are:** the pilot's Quant
results are **not comparable for either harness** and are excluded from the
headline. Claude Code's 0/6 is an environment failure, not a model failure.
Codex's 6/6 is real work but was only scoreable because it diagnosed this and
shipped its own shim (D21), which the other harness did not do — so the column
measures environment-workaround initiative, not kernel quality.

Everything else — L1, L2, FlashInfer-Bench, 202 of 235 problems — is unaffected,
which is why the rest of the pilot stands.

---

## Surprises and deviations

### D1 — dataset ships as parquet, not per-problem directories
(Carried from session 1, still true.) The Hub publishes
`data/{L1,L2,Quant,FlashInfer-Bench}.parquet`, one row per problem.
`scripts/materialize_dataset.py` is the exact inverse of the dataset's own
converter and round-trip-verifies all 235.

**New in session 2:** the materializer wrote `reference` only to `reference.py`,
not into `definition.json`. `Definition` declares `reference` as a required
field, so *every* problem failed to load with a pydantic `Field required`
error the first time a runner touched one. The audit described the directory
*contents*; it did not imply a different schema. Now written to both places,
with the round-trip check comparing them so they cannot drift.

### D2 — this node IS exclusively ours
The MI355X node was shared (another user, another container). This one is not:
no other logins, no KFD processes. The node-wide clock lock is therefore safe
to leave in place, and the sibling-power contamination flagging in
`clock_calibrate.py floor` reported no busy siblings during any tail window.

### D5 — 9 FlashInfer-Bench problems need a second, separate dataset
(Carried from session 1.) 304 blobs from `flashinfer-ai/flashinfer-trace`,
fetched by `scripts/fetch_flashinfer_traces.py`, and `FLASHINFER_TRACE_DIR`
must be set. Both confirmed working on this node.

### D6 — the vendored data-model package was never committed [session 1 loss]
`.gitignore` contained `data/` **unanchored**, which matches
`src/sol_execbench/core/data/` and `tests/sol_execbench/core/data/` as readily
as it matches the dataset directory it was written for. Session 1's commit
therefore silently omitted nine source files and five test files that the code
imports — `Definition`, `Workload`, `Solution`, `Trace`, the dtype map, the
whole schema layer. Session 1's tests passed because the files existed in its
working tree; git simply never took them.

Recovered by re-vendoring from upstream at the pinned SHA (`a9fa080`) and
re-applying the AMD delta (`hip_cpp`/`ck`/`ck_tile`/`hipblaslt`/`miopen`/
`aiter` languages, `MI350X`/`MI355X` hardware, `.hip` entry points). Pattern
changed to `/data/`, anchored, with a comment saying why.

Worth stating plainly because the failure mode generalizes: the tests were
green, the working tree was correct, and the artifact that would have been
shipped was missing a third of the port. Nothing in the session-1 workflow
could have caught it — only a fresh clone could.

### D7 — one of SOLAR's own torchview patches is malformed and unnecessary
SOLAR ships two patches for torchview. `torchview-collect-attributes.patch` is
**corrupt** (its first hunk header declares 9 lines and supplies 8), so
`git apply` and GNU `patch` both refuse it, and SOLAR's own `install.sh`
silently skips it on failure.

Investigated rather than skipped, because a silently-dropped patch that
mattered would have left reduction-op attributes (`dim`/`keepdim`) uncaptured
and quietly changed every analysis. It does not matter: **both of its changes
are already present upstream** in torchview at the commit SOLAR pins, written
with `functools.partial` where the patch used a lambda. `env/Dockerfile` now
asserts both changes are present rather than assuming, so a future torchview
bump that drops them fails the build instead of producing subtly wrong bounds.

### D8 — determinism mode does not do what the name suggests
See task 01 above. `--setperfdeterminism X` yields ~0.83·X on MI350X, and above
~1900 stops responding to X at all. Recorded here because it is the single most
likely thing for a future session on other AMD hardware to get wrong: the
MI355X procedure (pick F_LOCK from the unlocked floor, request it, verify) is
*correct in form* and produced a wrong answer *in fact* on this part.

### D9 — the tolerance runner's memory profile, and one absurd allocation

Twenty-seven workloads across five problems failed calibration with HIP OOM,
and the two causes are unrelated:

*Retention.* Relative error can only be measured once `atol` is known, and
`atol` is only known after the last seed, so the first implementation kept
every seed's outputs — `seeds × 2 × output_size` of device memory, 234 GiB of
252 held at the point of failure. `--low-memory` keeps one seed's outputs and
re-runs the seed loop instead. Same derivation, twice the executions.

*Comparison width.* The comparison promoted whole tensors to float64 and
materialized a difference, peaking near 4× the output's size — an 18 GiB
output cannot be compared to itself on a 252 GiB GPU. Now chunked at 64 Mi
elements. This changes no number: a maximum over chunks is the maximum.

Four of the five problems calibrate after those two fixes. What is left is
**not** an OOM and is not yet explained:

```
L1/018 (batch 8, seq 128, cache_len 0)   Tried to allocate 16781313.00 GiB
L1/026 (batch 1, 224, 224)               Tried to allocate 16781313.00 GiB
```

16781313 GiB is 2^54 bytes. It is identical to the byte on two problems that
share no operator, so it is one bug and not two, and the shape it is derived
from is almost certainly garbage rather than large. Other workloads of both
problems calibrate normally. Not investigated yet: it needs a GPU, and every
GPU is currently producing timing numbers that concurrent load would corrupt.
Until it is, those two workloads carry `NOT AMD-DERIVED` in their tolerance
record.

### D10 — stale artifacts read as fresh findings

Half an hour was spent classifying 52 "SOLAR failures" that turned out to be
records written before the dataset was re-materialized and before the container
image was fixed. Re-running one of them by hand produced a *different* error,
which is the only reason the staleness was noticed at all.

The scratch directory is keyed by problem, not by (problem, code version), so a
failure recorded by an older build looks exactly like one recorded by the
current build. `--resume` re-runs failures precisely so they refresh — but
anything that *reads* the scratch directory mid-sweep is reading a mixture.
Rule for the rest of this port: triage failures from the artifact the sweep
wrote at the end, never from scratch state while it is still running.

### D11 — the shard runner could put two timing runs on one GPU

`shard_sweep.py` assigned `gpus[i % len(gpus)]` at *submission* time and ran
`len(gpus)` worker threads. Those are not the same constraint. Two tasks whose
indices are congruent mod `len(gpus)` can be in flight simultaneously, and they
then share a GPU while another sits idle. Caught live by reading
`/proc/<pid>/environ` for every running runner: `L2/060` and `L2/068` were both
on GPU 7 while GPUs 5 and 6 idled.

Two timing runs on one GPU inflate each other, and **nothing in the output says
so** — the artifact records the device it was told to use, which is the same
either way. Now each worker takes a GPU from a queue and returns it when done,
so concurrency is bounded by the pool and not by arithmetic on the task index.

Consequence for the numbers already collected: 176 of the 235 selection-pass
artifacts were produced before the fix, so an unknown subset of them contain
inflated per-variant times. That affects **selection**, never the anchor —
`authoritative_tb.py` re-times on GPU 0 alone, and every T_b in the manifest
comes from that pass. To keep a mis-ordered variant from being dropped before
it gets there, the authoritative pass now re-times the top two variants *plus
anything within 25% of the fastest*, rather than the top two alone.

### D12 — T_SOL was truncated to whole cycles, and eight of them truncated to 0

SOLAR emits `total_cycles` already wrapped in `int()`, and the bridge wrapped it
again. At 1.3 GHz a cycle is 0.77 ns, and the smallest workloads here are
genuinely sub-cycle — 12 KB at 8 TB/s is about two cycles — so the rounding is
not an edge case at the small end, it is the normal case. Eight workloads
ended up with **T_SOL = 0 cycles**: a bound of zero time, which no kernel can
approach and which puts a division by `(T_b − 0)` into the score.

A further 204 workloads implied a DRAM bandwidth *above* the arch config's own
peak, which is impossible for a roofline and was the symptom that led here.

The bridge now recomputes the roofline from the quantities SOLAR reports beside
the result — `max(MACs / MAC_per_cycle, bytes / DRAM_byte_per_cycle)`, its own
formula — and ceils, keeping the exact figure in `t_sol_cycles_exact`. All 185
successful problems were refreshed with `--only-status ok`, which re-runs the
successes without paying again for the failures (a failure means SOLAR ran to
the timeout, so they are the expensive ones).

### D13 — `masked_select` asks for 16781313 GiB above 2³² elements

The eight workloads D9 could not explain were not an OOM. Boolean indexing on
ROCm 7.2 / torch 2.9.1 computes a garbage allocation size once the tensor has
more than 2³² elements. Reproduced in isolation, on a flat tensor with nothing
else on the GPU:

```python
n = (1 << 32) + 1000
t = torch.ones(n, dtype=torch.float16, device="cuda")
t[torch.isfinite(t)]
# OutOfMemoryError: Tried to allocate 16781313.00 GiB
#                   (2**54 + 2**42 + 2**30 bytes), 70 GiB free
```

Promoting the same tensor to float64 and reducing it is fine, so it is the
mask path and not the size. The tolerance floor now accumulates over bounded
chunks with `torch.where`.

What made it worth chasing rather than filing as an OOM: the *same* absurd
number appeared to the byte on three problems that share no operator. An
allocator under pressure does not do that.

Every non-NVFP4 workload now has an AMD-derived tolerance — 3717 of 3957, with
the missing 240 exactly the 15 deferred NVFP4 problems × 16 workloads.

### D14 — the bound was priced at the vector-FP32 rate on 160 of 235 problems

The single largest error found in this session, and it was invisible until
`T_SOL ≤ T_b` could be checked against real measurements: **437 workloads had
a T_SOL above their own measured time**, by up to 13.4×.

`_precision_for()` chose the *widest* dtype among a problem's inputs, on the
reasoning that the widest drives both the compute peak and the byte count.
That is right for bytes and exactly wrong for the rate. In SOLAR's config
`fp32` is `MAC_per_cycle_fp32_sm` — the **vector** rate, 32768 MAC/cycle,
16× below the bf16 matrix rate — so a bf16 kernel with one `float32` epsilon
argument was priced at 0.085 PFLOPS instead of 1.36. 160 of the 235 problems
resolved to `fp32` that way, most of them mixed-precision kernels whose scalar
`eps` decided the answer.

T_SOL is a **lower** bound, so every term in it must be a lower bound: the
fastest rate the arithmetic could plausibly run at, and the least traffic it
could plausibly move. The fix is therefore two changes in the same direction:

* scalars (`shape: null`) are excluded — an `eps` rides in a kernel argument
  and is not a compute precision;
* among the tensor inputs the **narrowest** floating type wins, not the widest.

After the fix: `fp32` 108, `bf16` 104, `fp16` 12, `fp8` 4, `nvfp4` 1, and the
violations fall from 437 to **63**.

The 63 that remain are two kinds, both recorded rather than smoothed over:
depthwise-convolution problems where SOLAR appears to count a grouped
convolution as dense (`L1/006`, `L1/029`, `L2/035`, ratios 2–5.8×), and eight
workloads within 1–6% where the model and the measurement are simply that
close. Neither is shipped: `build_manifest` rejects **any** candidate bound
above the measured time, from either derivation, and falls back to the other —
a bound above a measured time makes `(T_b − T_SOL)` negative and pushes scores
past 1.

What this says about the method, and it is worth saying plainly: cross-check D
is the only one of the four that could have caught this, and it could not run
until task 06 finished. Checks A–C all passed throughout on a bound that was
wrong by 13× on some problems. A roofline that is internally consistent is not
thereby right.

### D15 — one problem's T_b does not reproduce to 3%

The anchor check re-times `T_b`'s own implementation and requires it to score
0.5 ± 0.03. Over a 20-problem sample, **336 of 349 workloads pass**, no
measured time falls below its `T_SOL`, and the reference never scores above the
anchor. Of the 13 that fail, **12 are one problem** —
`FlashInfer-Bench/018_mla_paged_decode` — where the re-timed latency comes back
a median of **1.16×** the recorded `T_b`. Two independent runs of the check
reproduced 13 and 12 failures on it, so the effect is stable and it is the
problem, not the check.

That problem's `T_b` is therefore optimistic by roughly 16%, which makes every
score on it *lower* than it should be — the conservative direction, but wrong.
The cause is not established: MLA paged decode is the most input-layout-
dependent kernel in the set and loads its inputs from safetensors blobs, so
allocation and page-table state are the obvious suspects. Recorded rather than
smoothed away, and it is a reason to be careful about drawing conclusions from
that one problem's scores.

The remaining single failure (`L1/072`) is one workload at the tolerance edge.

### D16 — `triton` here is a development checkout, not a release [session 3]

On this node `import triton` resolves to
`/sgl-workspace/triton-custom/python/triton/__init__.py`, version
`3.6.0+git42270451`. The pinned container installs whatever
`rocm/pytorch:rocm7.2...2.9.1` ships; this is a different build of the same
version number.

**Recorded as a deviation, not fixed.** Removing it would break the node's other
tenants, and pinning a release Triton would change every Triton timing measured
here — prime directive 6 says record the incompatibility rather than silently
adjusting the stack. Two consequences worth stating:

- `triton.experimental.gluon` **imports**, which it may not in a release wheel.
  Gluon is therefore a language an agent can legitimately use on this node and
  might not be able to elsewhere. That is a property of the environment, not of
  the benchmark.
- A Triton kernel's performance is a property of the compiler that built it. Two
  Triton numbers from different Triton builds are not comparable.

So `scripts/provenance.py` now stamps `kernel_stack.triton.path`,
`dist_version`, `is_release_wheel` and `gluon` onto every artifact, alongside
`aiter`'s git SHA and the presence of `ck`/`ck_tile`/`hipblaslt`/`miopen`. Before
this, an artifact recorded `torch` and `rocm` and nothing about the toolchain
that actually produced the kernel.

The probe uses `importlib.util.find_spec` rather than importing: importing
`aiter` loads a compiled extension, and a provenance stamp taken mid-measurement
must not create a HIP context on the device it is describing.

### D25 — 19% of bounded workloads beat their own bound, and the mechanism I proposed was wrong

**Cause: master's D14, in full. Resolved by merging it. Both explanations offered
below when this was first written were incorrect, and that is the useful part.**

Session 3 branched from `ea94b18` and re-derived `T_SOL` for MI355X with the
pre-fix `sol_bounds.py`, in which `_precision_for()` took the widest dtype among
a problem's inputs. A bf16 kernel with one float32 `eps` argument was therefore
priced at the fp32 **vector** rate, 32768 MAC/cycle, 16x below the bf16 matrix
rate. Across the pilot, **90 of 474** correct-and-bounded workloads came in faster
than their own lower bound.

After merging the fix and re-deriving:

```
violations with the PRE-FIX bounds:   90
violations with the POST-FIX bounds:   0  of 380 re-checkable workloads
precision resolution   fp32 157 -> 81     bf16 48 -> 100     (+ fp8 5, nvfp4 14)
```

Two problems make the size of it concrete:

| | pre-fix T_SOL | post-fix | ratio | bottleneck |
|---|---|---|---|---|
| `L2__028_gqa_rotary_attention_core_backward` | 4.152 ms | 0.2595 ms | **0.063x** | compute -> compute |
| `L1__001_attention_softmax_dropout_value_matmul_backward` | 3.197 ms | 0.5180 ms | 0.162x | **compute -> memory** |

`L2__028` moves by almost exactly 1/16, which is the bf16-to-fp32-vector ratio
and is the fingerprint of the defect. `L1__001` additionally *reclassifies* from
compute-bound to memory-bound, because once the arithmetic is priced correctly it
is no longer the binding term.

**What I got wrong, and why it is worth recording rather than quietly deleting.**
The first version of this entry proposed two mechanisms, neither of which was the
cause:

1. *"Paged KV decode: the bound counts memory the kernel never touches."* Offered
   for `FlashInfer-Bench__018`, on the evidence that SOLAR charged 2.28 GB of
   traffic for a single decode step. Plausible, and it fit the numbers -- but it
   was never tested, and `018` is not even in the post-fix set, because SOLAR
   fails on it and the traffic tier bounds it instead.
2. *"The reference does redundant work, so the traced MAC count overstates it."*
   Offered for `L2__028`. Simply wrong: the MAC count was fine, the price per MAC
   was not.

Both were reasoned from a real observation and both were confident. The
observation was worth surfacing; the explanations should have been marked as
hypotheses rather than written as findings, and prime directive 8 asks for exactly
that distinction -- surface the uncertainty, especially when you cannot explain it.

Master had already found and fixed this on MI350X, where it produced 437
violations reduced to 63. Reading the mainline before theorising about a shared
code path would have cost nothing.

**What still stands.** Master's own **D15** records `FlashInfer-Bench__018`
re-timing a median 1.16x slower than its recorded `T_b` on MI350X, reproduced over
two independent runs, with its anchor called optimistic by ~16%. That is a `T_b`
observation rather than a `T_SOL` one, it is independent of the rate bug, and it
is about the same problem my first hypothesis pointed at. So `018` remains the one
problem both ports flag, for reasons neither has yet closed out.

### D24 — the scorer trusted the packet's copy of the problem, and an agent edited it

The most serious defect found in session 3, and it was mine, not an agent's.

While investigating why two Codex Quant submissions passed without the
`sitecustomize.py` shim of D23, a diff of the packet against the dataset showed
the packet's `definition.json` had been **rewritten**:

```diff
  # data/SOL-ExecBench/benchmark/Quant/001_.../definition.json  (authoritative)
- from enum import StrEnum
  # .../pilot-01/codex/Quant__001_.../packet/definition.json     (as scored)
+ try:
+     from enum import StrEnum
+ except ImportError:
+     from enum import Enum
+
+     class StrEnum(str, Enum):
+         pass
```

`definition.json` carries the `reference` field — the PyTorch semantics the
kernel is checked against. `score_solutions.py` loaded it **from the packet**, so
it scored the submission against the submission's own edited definition of
correctness.

For these two cases the edit was a benign workaround for blocker B1, and the
kernels themselves are real: `Quant__001` is a legitimate Triton FP8 GEMM at 5.2×
the reference. That is not the point. **In the limit an agent can replace the
reference with a call to its own kernel and score 100% on every workload**, and
nothing in the pipeline would have noticed. The whole "nothing an agent produced
is trusted to score itself" principle was implemented for the *timing* — clean
re-evaluation on the authoritative GPU — and left the *specification* trusted.

Fixed:

- `score_one()` now loads `definition.json` from `data/SOL-ExecBench/benchmark/`
  and `workload.jsonl` from `artifacts/05/workloads/`, always. Only
  `solution.json` and its sources come from the packet.
- The packet's copies are diffed against the authoritative ones and any
  divergence is recorded per field (`packet_spec_divergence`), because the fields
  are not equally serious: a changed `reference` redefines correctness, a changed
  `description` is cosmetic.
- Divergence does not abort scoring. The authoritative spec is used either way,
  so the honest outcome is a score against the real problem plus a record that
  the packet disagreed.
- Every pilot score was recomputed from scratch after the fix.

Blast radius, measured rather than assumed: **2 of 48 packets**, both Codex, both
Quant, both `definition.reference`. No workload file was touched, and no L1, L2
or FlashInfer-Bench packet diverged at all.

The reason it stayed contained is luck, not design — B1 gave the agents a reason
to touch the reference, and B1 only affects Quant. A submission with a different
motive would have had the same opening on any problem.

### D23 — an agent repaired the harness's own environment, and it counted as a solve

Blocker B1 makes all 33 Quant references unimportable on this node. Claude Code
scored 0/6 on the pilot's Quant sample as a result. Codex scored **6/6**, and the
mechanism is worth the whole entry:

```
artifacts/10/runs/pilot-01/codex/Quant__004_fp8_moe_expert_linear/packet/
    sitecustomize.py:
        import enum
        if not hasattr(enum, "StrEnum"):
            class StrEnum(str, enum.Enum): ...
            enum.StrEnum = StrEnum
```

Four of its six Quant submissions ship one. `sitecustomize.py` is imported by the
interpreter **automatically at startup**, before any harness code runs, so the
reference then imports and the problem becomes scoreable.

This is not cheating in the usual sense and is not treated as such: it does not
touch the numerics, the timing path, or the tolerance. It is a competent
diagnosis of a broken environment — Codex had 4.2 verify attempts on average to
notice, where Claude Code averaged 0.9 and never saw the error.

It is nonetheless **disqualifying as a measurement**, for two reasons that have
nothing to do with intent:

1. It is the difference between 16/16 and not evaluating at all. That is far too
   much leverage to sit outside the record.
2. The shim's semantics differ from real `StrEnum` in the `str()` path (B1). It
   was verified equivalent for one of the six, by reading the reference. The other
   five were not verified, so five of those 16/16 results rest on an unchecked
   assumption about what the reference computes.

Changes made:

- **The static screen now screens filenames, not just contents.**
  `sitecustomize.py`, `usercustomize.py`, `.pth` and `conftest.py` are flagged as
  startup hooks: code the harness never invoked, running outside every timed
  region. A content scan could never have caught this, because the content is
  innocuous — the hazard is entirely in the name.
- Quant is excluded from the pilot headline for both harnesses, with the reason
  stated at every count.

The general lesson is the uncomfortable one: the agent found a real defect in the
benchmark that the benchmark's own authors had not, and the mechanism it used to
route around it would have gone unrecorded. Anti-gaming machinery built by
enumerating known exploits catches known exploits.

### D22 — an agent's autotuner outlived the agent and kept a GPU busy for an hour

Found by noticing a GPU at 100% while the pilot was supposed to be between
waves, and tracing it to:

```
pid 174506  python3 explore10.py 17 32 64
  cwd  .../pilot-01/claude-code/FlashInfer-Bench__009_gemm_n5120_k2048/packet
  HIP_VISIBLE_DEVICES=6
```

`explore10.py` is a tile-size search script **the agent wrote**. Its session had
been killed at the 30-minute wallclock cap **63 minutes earlier**. The script,
launched from a shell tool call under `timeout 5400`, was never in the CLI's own
process tree in a way `subprocess.run(timeout=...)` could reach: that kills the
direct child and nothing else. Seven processes were still alive.

Why it matters more than the wasted watts: an orphan holding a GPU **inflates
every later measurement on that device**, and nothing in the output says so. It
is deviation D11's failure mode arriving through a door D11 did not cover — there
the sweep put two tasks on one GPU, here a dead session did.

Fixed in three layers, because the first two are individually escapable:

1. Agents are launched with `start_new_session=True`, so the whole process group
   can be signalled; the timeout path now sends SIGTERM then SIGKILL to the
   group rather than to the CLI.
2. After **every** session, `_reap_orphans()` kills any surviving process whose
   working directory is inside the packet. The cwd is the reliable marker: the
   agent is launched with the packet as its cwd and everything it spawns inherits
   it, including through a double fork that escapes the group. Matching on cwd
   rather than a command name is what makes it work at all — nothing could have
   predicted the name `explore10.py`.
3. What was killed is recorded on the session (`orphans_killed`), so this shows
   up as a fact in the artifact instead of as an anomaly someone has to notice on
   a power meter.

**The pilot's scores are unaffected, and the reason is structural rather than
lucky.** The orphan was on GPU 6, where agents self-verify; every scored number
is re-measured on GPU 0 by `score_solutions.py`. This is precisely the case the
"nothing an agent produced is trusted to score itself" rule was written for — it
turned a contaminated-measurement incident into a wasted-GPU incident.

**`scripts/shard_sweep.py` has the same defect, and it is not fixed.** After the
T_b selection sweep reported `done: 223 ok, 3 failed`, an `eval_driver.py` with
about 35 Triton autotune children was still running on GPU 4. It did not corrupt
anything — GPU 0 was doing the authoritative work — but it consumed most of the
CPU, and the concurrent scoring pass slowed from ~3 minutes per problem to
~6 minutes, with one problem starved past the 20-minute eval timeout and recorded
as `eval_failed`. Those were re-scored once the orphan was killed.

So: **do not run a shard sweep concurrently with authoritative scoring.** The
GPUs do not interfere (task 01 measured +0.02%), but the CPUs do, and the
symptom is a spurious timeout recorded as a result. The reaping logic in
`solexbench_agents.harnesses` is generic enough to lift into `shard_sweep.py`;
that is left as a known gap rather than done, because it is the repo's own runner
and changing it mid-sweep would have invalidated the T_b pass in flight.

### D19 — the agent left its packet and read the harness [session 3]

The first smoke run of task 10 was given one problem in an isolated packet. Its
transcript shows it doing this:

```
[tool Bash] cd /sgl-workspace/sol-execbench-rocm &&
            sed -n '360,470p' src/sol_execbench/core/bench/timing.py
[tool Bash] cd /sgl-workspace/sol-execbench-rocm &&
            grep -n "def bench_time_with_cuda_events" -A 60 ...
```

It was reading how it would be timed. Taken on its own that is defensible — the
harness is Apache-2.0 code it could have read anywhere, and understanding the
measurement is part of optimizing for it. The problem is that the same reach
exposes three things that are not defensible:

- `artifacts/05/` — the tolerance **derivations**, with per-seed error
  statistics. An agent reading those can target the tolerance rather than the
  semantics.
- `artifacts/03/t_sol.json` — the analytic bounds. The answer key for how fast
  the kernel is meant to get.
- `artifacts/02/` — reference timings, and every other problem in the set.

Fixed by pointing each packet's `./verify` at a **reduced harness tree**
(`task_packet.build_verify_root`) holding only the evaluation code — `src/`,
`scripts/agent_verify.py`, `_common.py`, `provenance.py`, and a symlink to the
FlashInfer blobs. No artifacts, no other problems. The feedback loop is
unchanged; the answer key is not on a path the packet references.

**This is a reduction in exposure, not a security boundary, and the difference
is worth stating rather than blurring.** The agent is a local root process and
can still walk the filesystem. What makes the result defensible is the two
things that do not depend on containment: scoring re-evaluates every solution
from a tree fingerprinted at sweep start, and the static source screen runs on
every submission before it is scored.

### D20 — a gateway 403 mid-session is not a model failure [session 3]

The same smoke run died after 27 productive turns and $1.36 with:

```
Failed to authenticate. API Error: 403 AMD gateway error: ... returned
Forbidden. Reason: Access denied due to Virtual Network/Firewall rules.
```

It had already written a working-looking Triton kernel to `kernel.py`; it had
not yet written `solution.json`, so the harvest collected nothing and the
session recorded as "no solution". Counting that as "the model could not solve
it" would understate the score by however often the gateway happens to wobble
across 470 sessions.

Two changes, and the second matters more than the first:

1. **Transient failures are retried, not recorded as results.** This is the one
   deliberate exception to prime directive 1: the failure is real and is logged,
   but it is not a failure *of the thing being measured*. The signature list is
   in `harnesses.TRANSIENT_SIGNATURES`; a session that produced a solution is
   never treated as transient, so a retry can never discard a real answer.
2. **`TASK.md` now tells the agent to write `solution.json` first**, before
   tuning, because only that file is collected and a session can end without
   warning. A working kernel with no `solution.json` scores identically to no
   kernel at all.

Also fixed while here: the session's raw stdout and stderr are now persisted per
unit. The 403 message existed nowhere except Claude's own transcript, which is
not somewhere a sweep of 470 sessions can be debugged from.

And `is_error: true` arrived alongside `subtype: "success"`, so reporting the
subtype described a clean run that merely produced nothing. The `result` field
carries the real message and is now preferred.

### D21 — two integration defaults that silently removed the speed axis

`BenchmarkConfig.benchmark_reference` defaults to **False**, so
`reference_latency_ms` stays `0.0` and every speedup is undefined. The first
scored smoke run produced `speedup_vs_reference: null` on all seven workloads and
fell back to `correctness_only` — a scoreboard with no speed column at all, for a
benchmark about speed. Set to `True` in both `score_solutions.py` and
`agent_verify.py`; the agent's own `TASK.md` promises "your latency beside the
reference's" and could not deliver it either.

`are_clocks_locked()` reads the `SOL_EXECBENCH_CLOCKS_LOCKED` environment
variable, which upstream's Docker entrypoint sets after locking. It is a
**declaration, not a probe** — exporting it locks nothing. Setting
`lock_clocks=True` without it turned all seven workloads into
`RUNTIME_ERROR: lock_clocks=True but GPU clocks are not locked on this server`,
on a node whose clocks were in fact locked.

The fix is not to export the variable. `scoring.clock_lock_state()` reads
`power_dpm_force_performance_level` from sysfs for the authoritative GPU —
resolved through the PCI-bus mapping, because `card1` is torch 0 and `card57` is
torch 7 here — and `score_solutions.py` refuses to score unless it reads
`perf_determinism`. Only then is the flag set. Exporting it unconditionally would
have made every latency read as authoritative while being taken at a boost clock
that varies 10–30% under load.

### D18 — on MI355X, two GPUs obey the clock request and six do not [session 3]

The single most consequential finding of session 3, and it is not what D8
predicted.

D8 established that on MI350X `--setperfdeterminism X` yields ~0.83·X. The
obvious hypothesis was that MI355X behaves the same way with different
constants. It does not. **At one setting, the eight GPUs of this node hold
clocks spanning 326 MHz:**

| torch GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| achieved @ setting 1650 | **1644** | **1643** | 1318 | 1341 | 1370 | 1357 | 1352 | 1327 |
| power | 1269 W | 1270 W | 973 W | 949 W | 976 W | 995 W | 976 W | 951 W |
| ratio to request | 1.00 | 1.00 | 0.80 | 0.81 | 0.83 | 0.82 | 0.82 | 0.80 |

GPUs 0 and 1 obey the request. The other six land at ~0.80·request — the MI350X
rule — and **it is not power**: they draw 949–995 W of a 1400 W cap, some 400 W
of headroom unused. A power-limited GPU pins at its cap; these do not.

Confirmed by sweeping GPU 2 across the range (`determinism-sweep-gpu2.json`):

| requested | 1600 | 1700 | 1750 | 1800 | 1900 |
|---|---|---|---|---|---|
| GPU 2 achieved | 1288 | 1352 | 1390 | 1415 | 1470 |
| ratio | 0.81 | 0.80 | 0.79 | 0.79 | 0.77 |
| power | 942 W | 997 W | 1015 W | 1048 W | 1088 W |

GPU 2 has no obedient window anywhere in the useful range, while GPU 0 crosses
into one discontinuously between 1500 (0.81) and 1600 (1.00) — 100 MHz more
request buying 379 MHz more clock. So this is a **per-GPU DPM/firmware
difference within one node and one part**, not a property of MI355X.

Three things follow, and the third is the point:

1. **No node-wide setting makes all eight deterministic at the same clock.**
   Targeting the worst GPU's ~1320 MHz would put GPU 0 in its own discontinuity
   gap (1500→1214, 1600→1593; there is no setting that lands GPU 0 near 1320
   reliably). Targeting GPU 2's 1470 needs setting 1900, at which GPU 0 pins to
   the power cap and lets ambient temperature choose the clock.
2. **F_LOCK is therefore a per-GPU quantity, and 1640 is GPU 0's.** It is not
   the node's, and any timing from another GPU is at a different clock.
3. **This is why authoritative timing is pinned to GPU 0** — a rule session 2
   adopted for a 65 MHz spread on MI350X. Here the spread is 326 MHz, or 25%,
   which is larger than almost every optimization difference the benchmark
   exists to measure. `scripts/score_solutions.py` runs every scored evaluation
   on GPU 0, serially, and the agent pool is constructed to exclude it.

The agents' own `./verify` timings therefore come from a ~20% slower clock than
the score does. That is harmless and deliberate: an agent compares its kernel to
the reference *on its own GPU*, so the ratio it optimizes against is right even
though the absolute milliseconds are not, and its numbers were never the score.

Not investigated: whether the two obedient GPUs differ in firmware revision, or
whether the boundary moves with temperature. Both matter for whether F_LOCK is
stable across a long sweep, and the CV of 0.0041 over 30 processes is evidence
that it is stable *now*. Recorded rather than explained, per prime directive 8.

### D17 — no docker on this node, so `env/solb` cannot work [session 3]

`docker` is not installed. `env/solb` is a `docker exec` wrapper, so the entire
documented run path was unavailable.

What made a native path legitimate rather than a shortcut: the interpreter
already carries **exactly** the pinned stack — `torch 2.9.1+rocm7.2.0`, HIP
`7.2.26015`, `hipcc` from `/opt/rocm` — so nothing had to be relaxed to make it
work. `env/solb-native` reproduces `env/solb`'s environment contract
(`FLASHINFER_TRACE_DIR`, `HF_HOME`, `SOLEXBENCH_SCRATCH`, `PYTHONPATH`, the
`sol-execbench` entry point, `git safe.directory`) and then **asserts** the stack
via `env/check_stack.py`, refusing to run when it drifts. A native environment
that merely happened to be right would have been the shortcut; one that fails
loudly when it is not is the same guarantee the container gave.

Missing Python deps were installed against a constraints file pinning
`torch`/`numpy`/`triton`, for the reason `env/Dockerfile` gives: several of
these declare a bare `torch` dependency and pip would happily replace the ROCm
build with a CUDA wheel from PyPI, invalidating every measurement afterwards on
a box where nothing would obviously break until the timings did.

SOLAR and the patched torchview were installed at the Dockerfile's pinned SHAs.
Its assertion that the malformed `torchview-collect-attributes.patch` is
redundant still holds here: `collect-attributes changes verified present
upstream`.

`env/solb` is left untouched and remains the documented path on a node that has
docker.

---

## Fixes to scripts on first contact

F1–F11 are session-1 fixes on MI355X; see git history for their detail. They
all still hold. Session-2 fixes:

**F12 — `scripts/gen_golden.py` assumed a `get_inputs()` that does not exist.**
It was written against the KernelBench convention; SOL-ExecBench problems
declare their inputs in `definition.json` and generate them through
`gen_inputs`/`load_safetensors`/`custom_inputs_entrypoint`. As written it would
have failed on all 235 problems. Rewritten against the real schema, per
workload, keyed by workload uuid — which is what task 05 compares against.

**F13 — `gen_golden.py` fp64 promotion breaks dtype-literal references.**
Promoting only the inputs to float64 raises on any reference that constructs
internal tensors at a literal dtype (`torch.zeros(..., bfloat16)`, weights made
inside `run`). 63 of 1480 L1 workloads. Now falls back to a native-dtype CPU
run and **records which tier produced each golden** (`ok:float64` vs
`ok:native_cpu`), because they are not equally strong evidence: a disagreement
against float64 is a bug, against native CPU it may be ordinary noise.

**F14 — `scripts/roofline_probe.py` hardcoded MI355X spec peaks.** Printed
"spec peak 2500 @ 2.4GHz" on an MI350X, whose peak is 2307 @ 2.2 GHz. Now
resolved per part from `solexbench_rocm/parts.py`, and the achieved fraction is
written into the artifact so no reader has to infer the denominator.

**F15 — `scripts/sol_bounds.py` (new) tripped over scalar inputs.**
`get_input_shapes` returns `None` for a scalar input (e.g. a dropout
probability). Iterating that raised `TypeError: 'NoneType' object is not
iterable` and killed **41 of 94** L1 problems before any bound was computed.
Scalars are now passed as Python numbers, which is also semantically required:
the reference uses them in control flow, where a meta tensor would silently
change the traced graph.

**F17 — `verify_artifacts.py --task 01` had a check that could not fail.**
[session 3]

`f_lock_from_state()` matched `F_LOCK.*?(\d{3,4})` against the whole of STATE.md,
i.e. the first number following the first mention of F_LOCK anywhere. Once the
file documented two parts, that resolved to a sentence about the MI350X bound,
and the acceptance check reported:

```
[PASS] F_LOCK recorded in STATE.md                1300 MHz
[PASS] F_LOCK at or below lowest observed floor   F_LOCK 1300 <= min p5 1724
```

Both green, both meaningless: it validated a stale MI350X number against MI355X
floors, and 1300 clears 1724 so comfortably that no wrong answer could ever have
failed it. The check that exists to protect the most consequential measurement in
the project was inert.

Now: the match requires a canonical `**F_LOCK = <n> MHz**` line, and a new check
compares it against `CLOCK_LOCK_PRESETS` — the value the code actually applies
and stamps. A document and a constant that disagree about the clock every bound
is expressed at is exactly the failure nothing downstream can detect.

**F16 — SOLAR's five-stage pipeline needed per-problem process isolation.**
Stage 1 traces arbitrary reference code, and some references trace
pathologically. In a `ProcessPoolExecutor` a stuck worker cannot be cancelled,
so one bad problem stalls the sweep behind it — the classic way a "finished"
sweep silently covers 200 problems instead of 235. Each problem now runs as a
killable subprocess with a timeout, and a timeout is recorded as a result.

---

## Decisions taken

**F_LOCK = 1640 MHz** achieved on **GPU 0**, at determinism setting 1650, on
MI355X. This is the canonical declaration; `scripts/verify_artifacts.py --task 01`
parses this exact line and fails if it disagrees with `CLOCK_LOCK_PRESETS`.
Full reasoning in the task 01 session-3 section above.

*(Session 2, MI350X, archived: F_LOCK was 1300 MHz achieved at setting 1600.)*

The two-number form is a real structural difference from NVIDIA, where
`nvidia-smi -lgc` makes them the same; `ClockPreset` carries both and
`f_lock_mhz` returns the achieved one.

**Authoritative timing is pinned to GPU 0.** Not a style choice: at the same
determinism setting the eight GPUs hold clocks spanning 1242–1307 MHz (5%),
which is larger than most of the optimization differences the benchmark exists
to measure. Sharding is fine for correctness and for *selecting* a T_b variant;
the winning variant is re-timed on GPU 0.

**Architectural constants are shared between MI350X and MI355X; measured ones
are not.** `solexbench_rocm/parts.py` separates the three kinds explicitly, and
the shared MAC/cycle table is justified by reproducing *both* parts' published
peak FLOPS from one set of numbers — 524288 MAC/cycle × 2 × 2.4 GHz = 2.52
PFLOPS (MI355X spec 2.5) and × 2.2 GHz = 2.31 PFLOPS (MI350X spec 2.3). A
constant that derives both parts' published figures is architectural; one that
does not is a measurement in disguise.

**T_SOL uses SOLAR's `fused` model.** `unfused` assumes every intermediate
round-trips to DRAM, which is above what a competent fused kernel achieves and
would make the "lower bound" exceed real measurements. Both are recorded so a
T_SOL ≤ measured violation can be diagnosed rather than merely observed.

**Timing methodology is recorded on every trace.** Upstream had one default
(CUPTI) and so needed no field; the AMD port ships on `hip_events` until task
04, and the two are not comparable on short kernels. `Environment.methodology`
is resolved once per run and passed to both the timer and the trace, so
"recorded" and "used" cannot drift.

---

## Session log

```
### 2026-08-03 — session 1  (node: mia1-p02-g10, 8x MI355X)
Worked: task 00 (done), 01 (done, F_LOCK 1650), 02 (port written, sweep not run)
Ended because: work moved to an MI350X node.

### 2026-08-04 — session 3  (node: mia1-p02-g10, 8x MI355X)
Worked: environment bring-up without docker (D15), task 00 (done), task 01
        (done, F_LOCK 1640 on GPU 0), task 03 re-derived for MI355X (222/235),
        task 06 selection (223/235, authoritative pass VOID — B2),
        task 10 built end to end and piloted.
Produced: env/solb-native + env/check_stack.py — the pinned stack asserted, not assumed
          artifacts/00/, artifacts/01/ — MI355X node record; MI350X archived beside
          artifacts/03/t_sol-MI355X.json — bounds at the measured F_LOCK
          artifacts/06/candidates/ — T_b selection for 223 problems
          artifacts/09/manifest-MI355X-v1.json — builds, reports its own gaps
          artifacts/10/ — agent harness, 48 scored pilot sessions, dashboard
          src/solexbench_agents/ — packets, harness adapters, GPU pool, scoring
          scripts/{run_agents,agent_verify,score_solutions,build_scoreboard,
                   backfill_scores}.py
          tasks/10-agent-scoreboard.md
          tests/solexbench_agents/ — 93 tests, all CPU-only
Found: six defects in the benchmark, four of them only because an agent walked
       into them — D16 (two of eight GPUs obey the clock lock), D17 (agent read
       the harness), D18 (gateway 403 scored as a model failure), D20 (agent's
       autotuner outlived its session by an hour), D21 (agent patched the
       interpreter via sitecustomize.py), D22 (the scorer trusted the agent's
       copy of the problem, and one had edited the reference), D23 (19% of
       bounded workloads beat their own bound), F17 (an acceptance check that
       could not fail), B1 (Python 3.10 where 3.12 is required), B2 (T_b void).
Left running: the 404-session full sweep, resumable.
Next session: re-run authoritative_tb + verify_anchor on an IDLE node, then
        backfill_scores. No agent work needs repeating.

### 2026-08-03 — session 2  (node: gbt350-odcdh1-a08-1, 8x MI350X)
Worked: environment rebuild, task 00 (done), task 01 (done, F_LOCK 1300)
Produced: restored src/sol_execbench/core/data (D6 — never committed)
          artifacts/00/ — node report, part-aware rooflines, acceptance log
          artifacts/01/ — 4 floors, determinism sweeps, stability, interference
          solexbench_rocm/parts.py — dual-SKU constants, one source of truth
          scripts/runners/{_common,run_reference,time_tb_candidates,
                           calibrate_tolerance}.py
          scripts/sol_bounds.py — SOLAR bridge, works end to end
          reference/exploits/ — replay corpus (concurrency, caching,
                           environment, AMD-specific) + static source screen
          reference/tb-candidates/variants.py — generic T_b variant set
          methodology + compute-partition recorded on every trace
          smi lockout, stream policy, static source screen (task 08 defenses)
```
