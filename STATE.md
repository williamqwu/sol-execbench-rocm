# STATE.md — progress ledger

**Single source of truth for progress.** Update as you go, not at the end.
A session can be interrupted at any point; whatever is written here is what the
next session inherits.

Rules: record real output, not summaries of intent. If something failed, say so
and say how. Never mark a task `done` without pasting its acceptance-check
output.

> Session 1 ran on `mia1-p02-g10` (8× **MI355X**). Session 2 onward runs on
> `gbt350-odcdh1-a08-1` (8× **MI350X**). `HANDOFF.md` says which session-1
> results transfer. The MI355X numbers are kept in this file where they are
> useful as a second data point, and are always labelled.

---

## Where this stands

**The benchmark is measured and the manifest is frozen.** `manifest-v1` scores
**220 of 235 problems / 3717 workload instances** on MI350X at F_LOCK =
1300 MHz. The 15 that are not scoreable are the NVFP4 Quant problems, whose
*references* fail on ROCm; they are in `artifacts/deferred.json` with the error
text quoted from the calibration artifact, and every count in every document
quotes that file.

What a consumer needs to know before using it:

* **Correctness runs against `artifacts/05/workloads/`**, not the dataset's own
  tolerances. Opt in with `SOLEXBENCH_WORKLOADS_ROOT`. Under upstream's B200
  tolerances the same references fail 8 workloads of `L2/033`.
* **`T_SOL` comes from one of two derivations and every workload says which.**
  SOLAR's roofline over the traced graph, or the traffic the definition itself
  declares over DRAM bandwidth. Neither dominates; the manifest takes the max
  of the two that survive being checked against the measurement.
* **Every `T_b` was re-timed on GPU 0 alone.** The eight GPUs span 1242–1307
  MHz at the same determinism setting.
* **No agent baseline was run.** Upstream's median of 0.732 has no counterpart
  here, and `artifacts/09/score-distribution.json` is labelled as not being one.

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
| Sibling-GPU interference | **negligible (−0.11%)** — sweeps and authoritative timing may share the node |
| Dataset present | yes — 235 problems, L1=94 L2=82 Quant=33 FlashInfer-Bench=26 |
| FlashInfer blobs | yes — 304 external safetensors blobs |
| Measurement container | `solbench:rocm7.2-torch2.9.1`, from `env/Dockerfile` (now also carries SOLAR + patched torchview) |
| Node exclusivity | **exclusive** — no other user, no other KFD processes (unlike the MI355X node) |

---

## Task status

| ID | Task | Status | Artifacts | Notes |
|---|---|---|---|---|
| 00 | Node acceptance | `done` | `artifacts/00/` | 13 checks, 0 failed |
| 01 | Clock calibration (F_LOCK) | `done` | `artifacts/01/` | **F_LOCK = 1300 MHz** at setting 1600; unblocks 03, 05, 06 |
| 02 | Harness port validation | `done` | `artifacts/02/` | 3717/3717 non-deferred workloads pass under AMD tolerances |
| 03 | SOL bounds (T_SOL) | `done` | `artifacts/03/` | 235/235 problems bounded, two derivations, source recorded |
| 04 | rocprofiler shim | `done` | `artifacts/04/` | median divergence −0.61% over 1430 pairs; clock domain verified |
| 05 | Tolerance calibration | `done` | `artifacts/05/` | 3717/3957 AMD-derived; the 240 missing are the deferred NVFP4 |
| 06 | Baselines (T_b) | `done` | `artifacts/06/` | 220 problems anchored, all re-timed on GPU 0 |
| 07 | Quant / MXFP4 | `done` | `artifacts/07/`, `artifacts/deferred.json` | 15 NVFP4 deferred with evidence; 220 ship |
| 08 | Red team | `done` | `reference/exploits/`, `artifacts/08/` | 28/28 replay cases pass, 0 false positives on 235 references |
| 09 | Release | `done` | `artifacts/09/` | manifest v1: **220/235 problems, 3717 workloads scoreable** |

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

**All ten gates re-run 2026-08-06** (`verify_artifacts.py --task NN`, in
`env/solb`). Recorded here because three documents were claiming a task-01
failure that has not existed since 2cdb7b0:

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
`FlashInfer-Bench__019`, `L1__005` and `L1__035`. That is D18/D21, known and
carried deliberately. **A second failure anywhere is a regression.**

### Task 00 acceptance output (2026-08-03, MI350X)

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

### Task 01 results (2026-08-03, MI350X) — F_LOCK = 1300 MHz

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

### Task 01 acceptance output (2026-08-03, MI350X)

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

---

## Blockers

None open.

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

### D16 — the agent pilot billed the wrong gateway key, and looked fine doing it

`~/.claude.json` carries an `env` block, and Claude Code applies it **over** the
process environment. On this host it sets

```
ANTHROPIC_CUSTOM_HEADERS = Ocp-Apim-Subscription-Key: 6838c76b…
```

so every header `agent_baseline.py` exported — including the
`fb97d25…` key it was told to use — was discarded. The `pilot8` run
($65.08 over 8 problems) therefore authenticated with the wrong AMD gateway
subscription key.

Nothing about the run looked wrong. It completed, produced kernels, reported
costs. It surfaced only from a falsification test: a session was run with a
deliberately invalid subscription key and **succeeded anyway**, which is
impossible if the exported header were being used. Directly against the
gateway, `fb97d25…` → 200, `6838c76b…` → 200, `0000…` → **401**, so the
gateway does validate the key and the earlier success proves the override.

No personal Anthropic credential was ever involved: `ANTHROPIC_API_KEY` on this
host is the literal string `dummy` and there are no stored OAuth credentials.
Both keys are AMD gateway keys reaching `llm-api.amd.com/Anthropic`.

Fixed by passing `--settings` with an explicit `env` block, which does take
precedence — verified the same way round: an invalid key passed through
`--settings` makes the session fail. The run record now carries
`gateway_key_prefix` so which key paid for a run is an artifact, not a
recollection.

### D17 — the scorer wrote into the container and scored every kernel zero

`agent_score.py` passed the container a **host** path for `--out`. Only two
trees are bind-mounted — the repo at `/work`, and `SOLEXBENCH_SCRATCH` at its
own absolute path — so a run directory anywhere else (`artifacts/…` given as an
absolute path, or a scratch experiment under `$HOME`) resolves inside the
container to a directory the unprivileged user cannot create. The runner died
before writing anything, `retime()` discarded its stderr, and all eight
problems reported `0/0 passed, 0 scored`.

That output is indistinguishable from eight kernels that genuinely failed,
which is what makes it dangerous: a real result of "the agent achieved
nothing" was available for the taking. The artifact is now staged through
scratch and copied out, and a runner that never ran reports `RUNNER FAILED`
with its stderr instead of a zero score.

The earlier one-problem validation run passed only because `--run` was given as
a *relative* path, which happened to resolve against the container's `/work`
working directory.

### D18 — T_SOL is wrong on the paged FlashInfer problems, and an agent proved it

The first agent kernel to beat the reference by a large margin on
`FlashInfer-Bench__019_mla_paged_prefill` came in **faster than T_SOL on 25 of
its 38 workloads**, scoring up to 1.115. Nothing can be faster than the
speed-of-light bound, so this is a defect in the bound, not a result.

The kernel is legitimate — a fused Triton MLA prefill that passes 38/38 against
the AMD-derived tolerances and is not flagged by any anti-reward-hack check.
What it does differently is read only the pages `kv_indices` names.

The declared-traffic bound counts the whole declared tensor:

```
ckv_cache: [num_pages, page_size, head_dim_ckv]   num_pages = 989,669
kpe_cache: [num_pages, page_size, head_dim_kpe]
  full cache = 1.140 GB -> at 8 TB/s = 0.14251 ms   (manifest T_SOL = 0.14266 ms)
  pages actually gathered = 34 of 989,669 -> 39.2 KB, a factor of 29,108 less
```

The bound matches "read the entire cache" to 0.1%. It is not a bound on the
work; it is a bound on the allocation.

**Why no existing check caught it.** The gate is `T_SOL <= T_b`, and T_b is
measured from the PyTorch reference — which *also* reads the whole cache. Both
numbers are wrong in the same direction, so they agree. Only a gather-aware
kernel separates them, and until this run nothing on this benchmark was
gather-aware.

**Exposure — 6 problems, 249 scoreable workloads:**

| problem | scoreable wl | median over-count | worst |
|---|---:|---:|---:|
| `FlashInfer-Bench__014_gqa_paged_prefill…kv4` | 30 | 4,612× | 176,707× |
| `FlashInfer-Bench__015_gqa_paged_prefill…kv8` | 38 | 4,531× | 276,155× |
| `FlashInfer-Bench__019_mla_paged_prefill` | 38 | 7,367× | 197,934× |
| `FlashInfer-Bench__018_mla_paged_decode` | 47 | 136× | 123,709× |
| `FlashInfer-Bench__012_gqa_paged_decode…kv4` | 48 | 1× | 129× |
| `FlashInfer-Bench__013_gqa_paged_decode…kv8` | 48 | 1× | 128× |

Over-count is `num_pages / num_kv_indices` per workload. The two decode
problems are mostly sound at the median and wrong only in the tail; the three
prefill problems and `018` are wrong at the median by three to four orders of
magnitude. This is very likely also what makes `018`'s anchor irreproducible
(D15) — same family, same layout dependence.

**What was done.** `agent_score.py` now enforces the invariant directly: a
workload whose measured time falls below its T_SOL is marked
`bound_violation`, excluded from the headline mean, and reported. On the pilot
that is the difference between a reported mean S of 0.8401 and the honest
0.7757.

**What was not done.** Manifest v1 is frozen and has been published, so the
bound is not being silently recomputed. The fix for v1.1 is to derive paged
traffic from `num_kv_indices × page_size`, the pages the workload actually
names, rather than from `num_pages`. Until then, scores on those six problems
are not usable and are marked as such wherever they appear.

### D19 — 38 dead tests behind a skip that read like a scheduling choice

`pytest tests/` reported `75 skipped`. 63 of those carried the `timing_serial`
marker, which `conftest.py` auto-skipped **unconditionally**. Running them
showed **40 of 63 failing**.

Two separate causes, and neither was visible from the skip line:

* **38 are CUPTI tests.** `timing.py` imports cupti *lazily* — correctly, so the
  module stays importable on ROCm — which means these collect fine here and then
  fail at call time with `ModuleNotFoundError` instead of being skipped.
  `_NVIDIA_ONLY_TEST_FILES` never caught them because the file imports cleanly.
  CUPTI has no ROCm build, so no hardware makes them pass; the AMD path is the
  task-04 rocprofiler shim. Now skipped by class, with that reason.
* **The marker's own instruction did not work.** The skip reason printed 63
  times said to run `pytest tests -m timing_serial -n 0`. `-n 0` is a
  pytest-xdist flag and xdist is not installed in the pinned image, so that
  command errors. The one pointer at the largest block of unrun tests was dead.

The blanket skip was also wrong in intent: `timing_serial` exists because these
measure GPU wall-clock and a co-scheduled worker corrupts them — a reason to
skip *under parallelism*, not always. Gated on actual xdist now, so on an idle
node they run. `pytest tests/` went from **483 passed / 75 skipped** to
**503 passed / 55 skipped**, 0 failed.

### D20 — matmul timing spread on MI350X is bimodal, and the cause is unknown

The remaining 2 non-CUPTI failures were `test_matmul_timing_variance`, whose
thresholds its docstring sources to "measured ranges on **RTX 4090 and B200**" —
NVIDIA constants, forbidden by directive 2. Re-derived on this part with
`scripts/derive_timing_variance.py`: 120 invocations per size across GPUs 1-4,
clock-locked at `perf_determinism`, same statistic the test computes
(`max/min` over one `time_runnable(return_mode="all")` call). Only the constant
was re-derived; the statistic was not changed.
Artifact: `artifacts/02/timing-variance-amd.json`.

| size | median | p95 | % >2× | outliers cluster at | NVIDIA k | k fails |
|---|---|---|---|---|---|---|
| 64 | 1.76× | 5.73× | 14.2% | scattered, ≤9.2× | 1.25 | 100% |
| 512 | 1.49× | 2.68× | 5.8% | ≤3.9× | 1.30 | 93% |
| 2048 | 1.02× | 1.04× | 2.5% | **21.4 / 21.6 / 22.5×** | 1.15 | 2.5% |
| 4096 | 1.01× | 3.35× | 7.5% | **3.3–3.4×, nine times** | 1.15 | 7.5% |

**No constant fixes this test.** At 2048 the normal spread is 1.02× and then it
cliffs straight to 21×, with nothing between: stopping the flake needs k=25,
and a 25× threshold on a 1.02× quantity asserts nothing. So the tests are
deferred with this evidence rather than given an invented number — the same
treatment as the NVFP4 problems, and for the same reason: re-specification, not
translation.

**Narrowed, and the leading hypothesis falsified.**
`scripts/probe_timing_stall.py` → `artifacts/02/timing-stall-probe.json`:

* **Not cold start.** The first call of every size is the *tightest* sample
  taken — the opposite of the obvious hypothesis, checked because it was obvious.
* **Not a fixed structural offset.** Stall indices are scattered (3–9 distinct
  per cell, concentration 0.11–0.5, median position 0.45 through the call), so
  it is not an allocator pool wrap or a page boundary at a reproducible index.
* **A per-iteration hazard, not per-call.** mm[4096] stalls at 0.135% of
  iterations at rep=100 and 0.125% at rep=25 — flat in the iteration count.
* **Even across GPUs** (5/3/4/4 over GPUs 1–4), so not one bad card.
* **mm[2048] did not reproduce**: 0 stalls in 12,000 iterations, against 3 at
  ~21× in 3,600 earlier. The 21× event is real but conditional on something not
  yet identified, and that discrepancy is unresolved.

**The clock is not the mechanism.** A discrete DPM step was the natural
explanation for a fixed multiplier, and this part has exactly one intermediate
step to fall to (`pp_dpm_sclk`: 38 / 500 / 2200 MHz, determinism range
500–1600). `scripts/probe_stall_clock.py` samples the active level at ~860 Hz
while the timing loop runs, sharing `CLOCK_MONOTONIC` across the two processes
so each sample can be attributed to the call it fell inside
(`artifacts/02/timing-stall-clock.json`):

```
calls WITH a stall   n=  6   clock 1401-1452 MHz   in-call spread 1.04x
calls without        n=114   clock 1449-1451 MHz   in-call spread 1.00x
```

A 1.04× clock spread cannot produce a 3.9–4.5× stall. **Hypothesis rejected.**

Two things that alignment step was load-bearing for. Without it the raw
histogram showed the clock sweeping 107→1598 MHz and looked like a confirmation;
that spread is entirely process startup and the idle gaps *between* calls. And
the DRM card had to be resolved by PCI bus — on this node torch 1 is `card9`
while `card1` is torch 0, so sampling `card{gpu}` would have read an idle GPU,
produced a flat trace, and "falsified" the hypothesis for the wrong reason. Same
trap as D11 and the task-01 floor that was fiction.

**Useful side result:** during actual measurement the clock is steady to 1.00×
at ~1450 MHz. That is reassurance for T_b, and it also shows F_LOCK = 1300 is
the *sustained-load* figure — short bursty runs on an idle node sit ~150 MHz
higher, which is why T_b and submissions must be timed the same way.

Still open: what costs a steady-clock kernel 3.9–4.5× at 0.13% of iterations.
Kernel selection inside hipBLASLt is the remaining suspect and has not been
tested.

### D21 — two more bounds a real kernel beat, and neither is D18's mechanism

`glm-run1` (GLM-5.2, 24 problems, re-timed on GPU 0) beat T_SOL on two problems
that are **not** paged attention, so the D18 explanation does not cover them:

| problem | T_SOL source | beaten by | workloads |
|---|---|---|---|
| `L1__005_conv_gated_projection_with_causal_conv` | `solar_fused`, compute | **1.09–1.15×** | 4 of 16 |
| `L1__035_flux_ada_layer_norm_zero_modulation_extraction` | mixed | **1.003–1.013×** | 2 of 16 |

Different in kind, and probably from each other too. L1__005 is a compute-bound
SOLAR roofline that is ~15% too slow — a rate or a missed fusion, and a real
defect. L1__035 is beaten by 0.3–1.3%, on a problem whose headroom is only
T_b/T_SOL = 1.008: there is almost no scoring range there at all, so a 1%
timing difference flips it either way. Whether that is a wrong bound or a bound
too tight to be measurable against is not yet decided, and the two need
separating before v1.1.

**Ingest bug found on the way, and fixed.** A bound a real kernel beat is a fact
about the *bound*, not the run that exposed it, but `ingest.py` wrote the list
with `INSERT OR REPLACE` per run and skipped excluded runs entirely. So each run
overwrote the last, and taking the pilot off the board deleted
`FlashInfer-Bench__019` — the D18 problem — from `/methodology`. Now accumulated
across every run read, excluded or not.

### D22 — a failed workload was carrying a score

Found while building the submission × problem view, which is the first page
that puts a status and its score on the same row.

`ingest.py` scored the four reference variants unconditionally:

```python
rows.append((sub_id, pkey, uuid, "PASSED" if all_passed else "FAILED",
             ms, sol_score(ms, t_b, t_sol), 0, label))   # score even on FAILED
```

A variant that fails the correctness check still has a latency, and `sol_score`
turns it into a perfectly plausible number — the speed at which the *wrong
answer* was produced. `torch.compile` on
`L1__002_vae_conv3x3_groupnorm_silu_residual_fused` displayed `FAILED … S =
0.4956` on thirteen rows, and the page's own summary card read "best workload S
= 0.5832" for a submission that passed nothing on that problem.

**No ranking was ever wrong.** Every aggregate filters `status='PASSED'`, and
the board totals are byte-identical before and after the fix. It was confined
to per-workload *display* — but `/api/problems/{key}` served it too, so anything
consuming the API would have inherited it.

The agent path never had this: `agent_score.py` leaves `score` at `None` unless
the workload passed. The variant path had simply diverged. Fixed at ingest
rather than in the templates, so every consumer gets one answer. The run page
additionally suppresses the T_b speedup on a failed row, for the same reason —
"0.99× vs optimized PyTorch" next to `FAILED` reads as near-parity.

This is the second defect of exactly this shape (D21 was the first): a number
that is real, plausible, and attached to the wrong claim. Both were invisible
until a page forced two facts next to each other.

### D23 — a submitted kernel whose re-time timed out is invisible

`glm-run1` has 24 kernels on disk and results for 23. The missing one,
`FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`, has a
`retimed/*.json` recording `TimeoutExpired` after 1200 s.

It produces no result rows, so every aggregate treats it as *not attempted* —
identical to a problem the agent never opened. The agent wrote 180 lines and
the harness could not measure them; that is a different fact, and it was not
recoverable from the board.

Found only because ingesting kernels-from-disk disagreed with
kernels-from-results, 24 against 23. Nothing else would have surfaced it.

The board now carries `run_kernel.retime_ok` / `retime_error` and shows the
state on both the run page and the submission page. **The score is unchanged
and should be**: no measurement exists, so there is nothing to score. The fix
is to stop the absence from being silent, not to fill it in.

Still open: whether the 1200 s timeout is the right budget for a paged-prefill
problem of that size, or whether this is another instance of D18's
paged-attention trouble on the same FlashInfer family. Not investigated.

### D24 — the same "dropped the external run" bug, twice more

`ingest.py` reads agent runs from `artifacts/10` unless told otherwise, so any
rebuild that omits `--agent-runs` silently deletes every run kept outside the
repo — currently the $250 Opus run.

This has now been introduced three separate times:

1. The staleness banner told the reader to run a bare `ingest.py`. Fixed by
   printing the roots the build actually used.
2. `worker.py` shelled out to a bare `ingest.py` after scoring. **Caught in the
   first end-to-end test**: the job scored correctly, and the Opus run
   disappeared from the board. Fixed by reading `meta.input_extra_roots` back
   out of the database that is about to be replaced.
3. Any manual rebuild, still, if the flag is forgotten.

The shape of the bug is that the *default* is lossy and the loss is silent —
the board still renders, still looks complete, and only the person whose
submission vanished would notice. `worker.py` now diffs the submission set
across a rebuild and reports a drop rather than trusting the exit code.

The durable fix is for the roots to live in a config the ingest reads by
default, so that "rebuild" cannot mean two different things. Not done.

### D25 — `f_lock_mhz: null` was blamed on a preset that exists

Three documents (`CLAUDE.md` §3 and §5b, `TODO.md`, `leaderboard/DESIGN-v2.md`
§6) stated that `CLOCK_LOCK_PRESETS` has no MI350X entry, that this is why some
artifacts stamp `f_lock_mhz: null`, and that it is the one remaining task-01
gate failure. All three claims were false. `HANDOFF.md` §1 says the same thing
and is *not* wrong — it was written before the entry existed and is marked
superseded — but it is the sentence a grep lands on, so it now carries an
inline correction. The entry was added in **2cdb7b0**, 2026-08-03 20:36 UTC,
and `--task 01` reports 11 checks, 0 failed.

The null is real, and it has two causes — neither of which is a missing preset,
and neither of which loses a measurement. 28 artifacts carry it:

**20 predate the preset.** All of `artifacts/00/` (2) and `artifacts/01/` (18),
written 18:53–20:30 UTC on 2026-08-03, i.e. *before* 20:36. Their provenance
records `torch.available: true` and eight MI350X devices, so
`get_clock_preset()` ran and correctly returned `None`: the table had no entry
yet because task 01 was in the middle of producing the number that would go in
it. Correct as history.

**8 were written by the host interpreter, after the preset existed.**
`artifacts/10/{pilot8,glm-run1,submitted-apitest}/scored.json`,
`artifacts/10/pilot8/{run,cost-report}.json`, and
`artifacts/02/timing-{variance-amd,stall-probe,stall-clock}.json`. All eight
stamp `python: 3.11.7`, `torch: {"available": false}`, `rocm.version: 7.15.0`
— the host, not the pinned container (`3.12.3` / `torch 2.9.1+rocm7.2.0` /
`rocm 7.2.0`). `provenance.f_lock_mhz()` resolves the preset through
`torch.cuda.get_device_name(0)`, `import torch` raises `ModuleNotFoundError` on
this host, and the function's `except Exception` returns `None`. Confirmed both
directions: bare host call returns `None`; the same call with
`SOLEXBENCH_F_LOCK_MHZ=1300` returns `1300`.

Host execution is by design. `agent_score.py`, `agent_cost_report.py` and
`agent_baseline.py` orchestrate and never touch a GPU — `agent_score.py` loads
`sol_score.py` by file path specifically because host python has no pydantic,
and each kernel is shelled into the container through `env/solb`. So the
timings are stamped and the roll-up is not: every
`artifacts/10/*/retimed/*.json`, written inside the container by
`agent_eval.py`, carries `f_lock_mhz: 1300`, `python: 3.12.3`,
`visible_devices: "0"`. `meta.f_lock_mhz` in the leaderboard database is
`'1300'`, from the manifest, so nothing published is affected.

Fix is in `TODO.md`. The tempting non-fix — defaulting `stamp()` to 1300 — is
prime directive 2 in miniature: it would make a roll-up written on MI355X claim
a clock it was never measured at, undetectably.

The lesson is not about clocks. A wrong *explanation* propagates faster than a
wrong number, because nothing runs it and nothing checks it. This one survived
into a `TODO.md` whose own header says it was rewritten to remove stale items —
and the gate that refutes it runs in seconds.

### D26 — three tables ranked means with different denominators

`AVG(score)` skips NULL, and a `PASSED` result stores `score IS NULL` when the
kernel beat `T_SOL` — the bound is invalid there, so no score is defensible.
The denominator therefore varies per row, and the leaderboard put such rows
next to each other under one `mean S` heading, in a **sortable** column.

The real instance: `agent-pilot8` on
`FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` passes
38 of 38 workloads, 25 of which beat the bound. Its mean was printed as
**0.9899 over 13 workloads**, directly above `agent-glm-run1`'s **0.9430 over
all 38** — so the run with a third of the evidence sorted to the top. Same
shape in `problem_detail()`, the submission page, and the `peers` query.

All 31 such results repo-wide were traced and they map exactly onto defects
already recorded: **25 to D18** (paged-attention `declared_traffic`
over-counting) and **6 to D21** (`L1__005` 4/16, `L1__035` 2/16). Nothing new
was found and no bound was silently adjusted.

**No number was changed.** Every mean is arithmetically correct and stays as
it was; what was missing is the count it was divided by, which is now printed
under each one (`over 13 of 38 — bound beaten on 25`). Changing the mean was
the tempting alternative and it is unavailable in both directions: scoring a
beaten bound as 0 punishes a kernel for the bound being wrong, and dropping
the workload inflates the run. Publishing the denominator is the only move
that does not invent a number.

The trial switcher had the opposite bug and was fixed the other way: it used
`AVG(score)` where the run card used `score_sum / attempted` with
`COALESCE(score, 0)`, so the two disagreed on exactly these rows. The switcher
now matches the card and says why a clean sweep can still score low.

Guarded by `tests/leaderboard/test_score_denominator.py` (6 tests). The
denominator is printed on **every** row, not only the short ones — a caveat
shown only when it applies is indistinguishable from one nobody checked.

### D27 — the code pane had no test at all

`leaderboard/static/highlight.js` (syntax highlighting, copy button) and the
`<pre>` that feeds it were untested. The pane is the only place a reader sees
the kernel that produced a number, and the copy button hands `pre.textContent`
to the clipboard — so a template that mangles the source does not look wrong,
it *exports* something that was never run. Escaped-once and escaped-twice
render identically for most sources; the check has to be mechanical.

Added `tests/leaderboard/test_code_pane.py` (4 tests): unescape the served
pane and compare byte-for-byte against the database, on a constructed hostile
source and on every real kernel. Verified against the built board — **36/36
panes byte-identical, and all 36 stored `sha256` values match their own
source.** Mutation-checked: removing the protective leading newline after
`<pre>` and adding a `|trim` filter each fail it.

Two things are asserted here and two are not:

* **Not** the tokenizer. This node has no JavaScript runtime (`node`, `deno`,
  `qjs`, `d8` all absent), so `highlight.js` is unexecuted. Its guarantees are
  structural in that file — every emitted chunk passes through `esc()`, sticky
  regexes that cannot match ahead of the offset, a total fallback branch, and
  `data-hl` so a block is never scanned twice — and were read, not run.
* **Not** the `hip` branch of language detection. `run.html` picks `hip` when
  the source starts with `#include` and `python` otherwise. Checked against an
  independent signal (`__global__`, `extern "C"`, `#pragma unroll`,
  `template<` vs `import`/`def`/`class`): **36 of 36 kernels and all 1175
  `variant_source` rows are Python, 0 disagreements.** So the heuristic is
  correct on everything the board holds and the `hip` path has never been
  exercised by real data. The first HIP kernel submitted is the test.

### D28 — the board scores a whole problem by one flag, and under-reports 1,239 passing workloads

**No measurement is wrong. The board reads 1,239 of them as failures that the
artifacts record as passes.**

`ingest_variants()` (`leaderboard/ingest.py`) writes each variant row as
`"PASSED" if all_passed else "FAILED"`, where `all_passed` is a **per-problem**
flag on the variant. A problem where `torch.compile` matches on 13 of 20
workloads therefore lands on the board with all 13 marked `FAILED` and scored
`NULL` — even though `latency_ms_by_workload` holds exactly those 13 timings and
nothing else. The per-workload truth was available the whole time, one key
across: each variant also carries `failures: [{workload_uuid, status, log}]`,
which the ingest never reads.

Board figure against the artifacts, over the 220 scoreable problems:

| variant | board says | artifacts say | delta |
|---|---|---|---|
| `v1_eager` | 3701 / 3717 | 3707 passed of 3717 attempted | 6 |
| `v4_contiguous` | 3687 / 3717 | 3688 of 3717 | 1 |
| `v2_compile` | 2586 / 3717 (69.6%) | **3171 of 3694** (85.8%) | 585 |
| `v3_compile_max_autotune` | 2394 / 3717 (64.4%) | **3030 of 3611** (83.9%) | 636 |

The problem counts (`149/220`, `136/220`) are *not* wrong — they are counts of
problems where the variant passed everything, which is what `all_passed` means.
It is the workload column that is answering a different question than its
heading.

The residue after the fix is a real result and not a gap: `v2_compile` fails
**523 workloads on `INCORRECT_NUMERICAL`** and never produces a timing for 23
more; `v3` fails 581 (571 numerical, 10 `RUNTIME_ERROR`) and misses 106. That
is torch.compile disagreeing with eager beyond tolerance on this part, which is
worth publishing accurately rather than rounding into a coverage number.

`v1_eager`'s one non-clean problem is
`L2__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block`: 6 of 16
workloads pass, 10 `INCORRECT_NUMERICAL` — the *reference formulation* against
its own golden. That is the `219/220`, and it is a finding, not a missing run.

**Costs no GPU time to fix.** It is a re-ingest: read `failures` per workload
instead of painting the problem. What it does need is a decision the ingest
cannot make alone — 89 of the 220 problems have at least one variant with
passing workloads but **no GPU-0 authoritative re-time**, because the
authoritative pass only re-timed variants that were clean (`v2` lacks 69, `v3`
76, `v1` 9, `v4` 8). Those rows would go on the board carrying a sweep timing
from GPUs 1–7, which is a different measurement class and has to be labelled as
one. At the authoritative sweep's measured rate — **217 problems in 323.3 min,
~1.5 min/problem** (`artifacts/06/logs/authoritative.log`) — re-timing all 89 is
**about 2¼ hours on GPU 0**, and a full re-derivation from scratch is ~5.4 h.

Not fixed here. Recorded, with the numbers, so the fix is a re-ingest against a
stated target rather than a rediscovery.

### D29 — the external fleet ran 34 jobs on GPU 0

`dash-overlay`'s J2 backfill sweep places one agent per GPU and takes a
scheduler hold on the authoritative device (`sbt`'s
`reserve_authoritative_gpu`). The hold did not hold: across the recorded J2
jobs the placement is 34 · 42 · 36 · 35 · 35 · 35 · 36 · 34 over GPUs 0–7 —
**GPU 0 took a full share.** Two of them are from the sweep still in flight.

No published number is affected: no authoritative timing was running during
those jobs, and every score on the board was re-timed before the sweep began.
The failure is that the property was not enforced, so the next time the two do
overlap nothing will say so. Whichever side owns it, the check belongs where
the placement happens, not in a comment.

### D30 — two vertical scrollbars on every code pane (fixed)

`pre.code.has-ln` split the axes: `overflow-y:auto` on the `<pre>`,
`overflow-x:auto` on the `<code class="src">` inside it. But CSS Overflow 3 §3
forbids one axis staying `visible` while the other is not, so `.src`'s
`overflow-y` computed to `auto` as well. Under the default
`align-items:stretch` that gave `.src` a definite 608px height and a scrollbar
of its own, while the gutter — same stretch, no overflow — spilled past the
640px cap and gave the `<pre>` a second one. Dragging the outer bar then moved
the line numbers and not the code, which is the blank space it appeared to
reveal: the gutter's overhang.

Now one scroll container, the `<pre>`, both axes; `align-items:flex-start` so
neither child is stretched, and the gutter is `position:sticky; left:0` so a
200-column line still cannot push the numbers off the left edge. The copy
button moved from `right:10px` to `22px` — it is positioned against
`.codewrap`, outside the scrollport, so it sat under the scrollbar of every
pane long enough to have one.

Found from a screenshot, not from a test, and it stays that way: verifying it
needs layout, and this node has no browser. `tests/leaderboard/test_code_pane.py`
asserts what the pane *contains*, which is the part that can be checked here.

### Section nav on the two long reference pages

`/methodology` and `/problems/<key>` are eight and seven sections of reference
material with no way to see the shape of the page or jump within it. Both now
carry a sticky left nav, server-rendered from `TOC_METHODOLOGY` / `TOC_PROBLEM`
in `app.py` — not scraped from the DOM, so it is there with JavaScript off and
a test can check it. The scroll spy that marks the current section is the only
part that needs JS, and its absence costs nothing but the highlight.

`tests/leaderboard/test_sidenav.py` (7 tests) asserts every `href="#x"`
resolves to an `id="x"` in the body, that the nav is in document order (the spy
assumes it), that no `h2` is missing from the nav, and that a page passing no
`toc` still renders single-column. Two files with no compiler between them
would otherwise drift into links that render, look live, and scroll nowhere.

### The grid ramp: distribution and contrast, closed

Both halves of the earlier bunching finding are now measured rather than
eyeballed.

**Distribution**, over 12,883 scored results: u5 9.0 · u4 9.0 · u3 13.3 ·
u2 19.5 · u1 17.1 · b1 29.3 · b2 0.6 · b3 0.6 · b4 0.7 · b5 0.9 %. The prior
89%-in-two-buckets bunching is gone. b1's 29.3% is structural, not a binning
error: the reference variant that wins `T_b` scores exactly S = 0.5 there by
construction.

**Contrast**, recomputed independently from the tokens in `style.css`: every
ratio the file documents reproduces to the digit — dark beat 2.70/3.68/5.03/
6.96/9.47 and under 2.70/3.67/5.06/6.87/9.36 against `--panel #161a21`; light
beat 1.74/2.18/2.76/3.44/4.36 and under 1.74/2.24/2.86/3.68/4.68 against
`#ffffff`. The light floor of 1.74:1 is below WCAG 1.4.11's 3:1 for non-text
objects and is carried by the border on every scored cell instead, measured at
3.44:1 and 3.68:1. Polarity is never on colour: `.g-u*` are circles, `.g-b*`
squares, and a `forced-colors` block restates all of it in system colours.

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

**F16 — SOLAR's five-stage pipeline needed per-problem process isolation.**
Stage 1 traces arbitrary reference code, and some references trace
pathologically. In a `ProcessPoolExecutor` a stuck worker cannot be cancelled,
so one bad problem stalls the sweep behind it — the classic way a "finished"
sweep silently covers 200 problems instead of 235. Each problem now runs as a
killable subprocess with a timeout, and a timeout is recorded as a result.

**F17 — `verify_artifacts.py --task 01` had a check that could not fail.**

`f_lock_from_state()` matched `F_LOCK.*?(\d{3,4})` against the whole of STATE.md,
i.e. the first number following the first mention of F_LOCK anywhere — a prose
sentence, a table cell, a deviation write-up, whichever came first. That is fine
while the file documents one part. On a node whose STATE.md still discussed the
other part's bound it resolved to the wrong number and reported:

```
[PASS] F_LOCK recorded in STATE.md                1300 MHz
[PASS] F_LOCK at or below lowest observed floor   F_LOCK 1300 <= min p5 1724
```

Both green, and neither could have failed: 1300 clears a 1724 floor so
comfortably that no wrong answer would ever trip it. The check protecting the
most consequential measurement in the project was inert.

Now the match requires a canonical `**F_LOCK = <n> MHz**` line, and a new check
compares it against `CLOCK_LOCK_PRESETS` — the value the code actually applies
and stamps. A document and a constant that disagree about the frequency every
bound is expressed at is exactly the failure nothing downstream can detect.

**F18 — `build_manifest.py` built from a T_b directory holding two clocks.**

Merging two ports of this benchmark added 87 T_b artifacts measured at F_LOCK
1300 into a directory of artifacts measured at 1640. No conflict was raised,
because a three-way merge does not conflict on a file present on only one side,
and the manifest then built from the mixture without complaint.

Every one of those files was internally correct and correctly stamped. The
*directory* was wrong — and a directory has no provenance of its own, which is
precisely why every artifact here carries one. T_b is a wall-clock time, so those
87 problems would have carried an anchor from one clock while their kernels were
timed at another, rescaling those problems' scores by the ratio, per problem,
invisibly.

`collect_t_b()` now takes the expected F_LOCK — read from the same
`CLOCK_LOCK_PRESETS` table `lock_clocks()` applies from, so the manifest and the
hardware cannot disagree — and rejects any artifact measured elsewhere with a
loud count. An artifact with *no* recorded clock is still admitted: that is a
missing-provenance defect, which `check_06` already covers separately, and
conflating the two would hide both.

Regression tests in `tests/scripts/test_build_manifest.py`. The lesson generalizes
past clocks: a three-way merge reasons about files, while a measurement's validity
is a property of the set it belongs to, so the durable defence is the consumer
checking provenance rather than the merger being careful.

**This check is necessary and not sufficient, and the first version of this entry
claimed more than that.** It compares the artifact's stamp against the preset
table. Both read from the same place, so it catches an artifact from *another
clock* and is blind to an artifact whose stamp is simply wrong. That is not a
hypothetical gap: an unreset determinism sweep left a node at a 1900 MHz setpoint,
`provenance.f_lock_mhz()` returned the preset's 1640 without reading a device, and
143 artifacts measured at ~1860 MHz were stamped 1640 — then 1640 was checked
against 1640 and passed. Eleven hours of measurement, every value about 12% faster
than the number it claimed.

The original argument here was that reading the expected clock from the same table
`lock_clocks()` applies from meant the two could not disagree. **The table is not
the hardware.** Closing it needs the setpoint read back off the GPUs before
measuring, and the observed clock stamped rather than the requested one — a change
to the timing runners, not to `collect_t_b()`.

**F19 — the static source screen could not see a startup hook.**

`static_source_screen()` scanned file *contents* only. Python imports some names
automatically at interpreter startup — `sitecustomize.py`, `usercustomize.py`,
a `.pth` line beginning with `import`, and `conftest.py` under pytest — before any
runtime guard has installed itself and outside every timed region. A submission
shipping one of those executes code the harness never invoked, which is the same
escape as monkey-patching.

**A content scan cannot catch it, because the content does not have to look
suspicious.** The case that surfaced this shipped a two-line
`sitecustomize.py` defining `enum.StrEnum` to work around an interpreter older
than `requires-python`. It was not malicious, did not touch the numerics, and was
arguably a repair — it was also the difference between a problem scoring 16/16 and
not evaluating at all, which is more leverage than any submission should hold
outside the record.

The screen now checks the filename as well as the contents. False-positive
anchors are in `reference/exploits/test_source_screen.py`: all 235 dataset
references still pass, as do near-misses like `my_conftest.py`, `site_customize.py`
and `path.py`.

**F20–F24 — review of PR #1, and three more checks in the same file.**

PR #1 (`fix/verification-gaps`) contributed F17–F19 above. All three describe
real defects and the fixes are directionally right. Two needed correction
before merge, and auditing them surfaced three further checks in
`verify_artifacts.py` that were not reporting the state of the work.

**F20 — the PR failed its own check.** F17 requires a canonical
`**F_LOCK = <n> MHz**` line, and the PR states master's line already matches.
It does not: master's line reads `**F_LOCK = 1300 MHz achieved, at determinism
setting 1600.**`, and the pattern requires `MHz**` immediately after the digits.
The regex matched nothing on `b1c53dc`, on `654864c`, on the PR head itself, or
on current master, so `--task 01` failed on the PR's own branch. Fixed by
restructuring the *Decisions taken* line to lead with the bare marker.

Worse, the tightening silently removed a working check. The floor comparison is
guarded by `if p5s and fl:`, so a missing marker took `F_LOCK at or below
lowest observed floor` out of the run entirely — task 01 went from 8 checks to
7, losing the one that catches a clock the GPU cannot hold. It now falls back
to the preset, so a documentation defect can no longer delete a physics check.

**F21 — F18's clock guard failed open.** `collect_t_b(dir, None)` admits every
artifact regardless of clock, and `f_lock_mhz()` returns `None` off-GPU with no
override set. Building the manifest in the wrong environment would therefore
restore the exact defect F18 fixes, silently, with normal-looking output.
`build_manifest.py` now refuses to build when F_LOCK cannot be resolved.
Verified the guard does not change the result: rebuilding with
`SOLEXBENCH_F_LOCK_MHZ=1300` reproduces manifest v1 exactly — 235 problems,
220 scoreable, 3717 workloads, identical `bound_sources`.

**F24 — the determinism setpoint, read back off the hardware.** The PR's own
follow-up (`254cdd1`) retracts part of F18: the guard compares an artifact's
stamp against `CLOCK_LOCK_PRESETS`, and `provenance.f_lock_mhz()` answers from
that same table without reading a device, so it is blind to an artifact whose
stamp is simply wrong. On another node an unreset sweep left a 1900 MHz
setpoint while the preset returned 1640; 143 artifacts were measured at ~1860
and stamped 1640, and 1640 checked against 1640 passed.

The retraction is right, but it scopes the fix out as "a change to the timing
runners". It is not. `amd-smi metric -c` reports the setpoint as `MAX_CLK` per
GFX block on an **idle** device, so task 01 now compares the hardware against
the preset's *requested* clock with no load and no timed region:

```
[PASS] every GPU is at the preset's determinism setpoint   all 8 GPUs at 1600 MHz
```

Run against a negative control rather than merely observed passing, since a
check nobody has watched fail is the subject of this whole PR: against a preset
requesting 1640 the same hardware reports `FAIL (8 GPUs at [1600])`.

**This node audited while implementing it.** All 8 GPUs at setpoint 1600, and
GPU 0 under sustained load holds **1295–1303 MHz** against a stamped 1300 — so
the 220 T_b artifacts here are *not* affected by the defect `254cdd1`
describes. Verified rather than assumed.

Not fixed here: stamping the observed clock rather than the requested one in
`provenance.f_lock_mhz()`. That is the real end state and it changes the
provenance of every artifact the project produces, so it is its own change.

**F22 — `check_06` asserted a schema that was never produced.** It required
`artifacts/06/t_b.json` with a `problems` map, and `anchor-verification.md`.
Task 06 writes one file per problem under `authoritative/` keyed by
`winner_by_workload`, and the anchor result as `.json`. So `t_b.json exists`
has failed on every run this repo has ever had, while STATE.md recorded task 06
as done. The mirror image of F17: not a check that could not fail, but one that
could not pass. Rewritten against the real layout, and extended with F18's
one-clock invariant at acceptance time. It now reports 220/220 problems
anchored at a single F_LOCK, and surfaces D15 as a WARN (336/349) rather than
silence.

While rewriting it I keyed the anchor check on `n_failed`, a field that does not
exist in the artifact, which resolved to `None` and printed "every checked
workload within tolerance" over 13 real failures — the audited defect,
reproduced inside the audit. Worth recording precisely because it took thirty
seconds to introduce.

**F23 — `check D` was a literal unconditional PASS.** The line was

```python
c.add(JUDGE if "PENDING" in text else PASS,
      "check D: T_SOL <= best measured", "needs task 06")
```

`cross-checks.md` contains no `PENDING` and no check-D section, so this passed
always and compared nothing. It is also the one invariant that would have
caught D18. It now compares T_SOL against every measurement under
`artifacts/10/*/scored.json`, and reports, correctly:

```
[FAIL] check D: no measurement beats its T_SOL   25 of 115 measured workloads are
       faster than T_SOL (worst 0.29x the bound) across 1 problem(s):
       FlashInfer-Bench__019_mla_paged_prefill — the bound is wrong (D18)
```

With no submissions on disk it reports JUDGE rather than PASS, because the T_b
variants cannot falsify a bound that is too slow — the reference over-reads
exactly the way the bound does.

**Also: `check_07` required a write-up, not evidence.** It asserted
`artifacts/07/fp8-validation.md`, never written, and so failed permanently
while the validation it stands for had been done. It now checks the evidence —
all 18 non-NVFP4 Quant problems pass every workload in the task 02 reference
sweep, 18/18 — and warns separately that no summary document exists.

`.pth` in `_PATH_HAZARDS` flags any such file, where the hazard is specific to
site directories and `.pth` is also the conventional PyTorch checkpoint suffix.
Left as-is deliberately: submissions here carry source text, a `.pth` among
them is never legitimate, and the screen reports rather than raises.

---

## Decisions taken

**F_LOCK = 1300 MHz**

Achieved, at determinism setting 1600. Full reasoning in the task 01 section
above. The two-number form is a real structural difference from NVIDIA, where
`nvidia-smi -lgc` makes them the same; `ClockPreset` now carries both and
`f_lock_mhz` returns the achieved one.

The line above is the canonical marker `verify_artifacts.py` parses, and it is
the only place in this file that form appears. Prose may mention other parts'
clocks freely — that is the whole point of requiring a marker rather than
matching the first number that follows the first mention of F_LOCK (F17).

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

### 2026-08-06/07 — session 3  (node: gbt350-odcdh1-a08-1, 8x MI350X)
Worked: leaderboard only — no GPU work, no measurement taken or changed.
Produced: D25 (f_lock null misattributed), D26 (mean-S denominator),
          D27 (code pane untested), grid ramp distribution + contrast closed
          tests/leaderboard/ — 11 files, 91 passed, was 0 before this session
          MI350X<->MI355X part switch: db/solbench-<PART>.db, TODO-MI355X.md
Verified: container `pytest tests/` 503 passed / 66 skipped (65 -> 66 is
          test_code_pane.py's module-level importorskip, not a regression)
          /api/v1/leaderboard byte-identical to the pre-session baseline on
          every field the baseline records — six runs, six ranks, unchanged
Open:     nothing committed; the HTML explainer stack (a separate repo,
          branch qwu-dev/solrocm-bench) was still being written at session end
```
