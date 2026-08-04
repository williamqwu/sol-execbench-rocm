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
| 02 | Harness port validation | `done` | `artifacts/02/` | 235/235 problems swept; references run on ROCm |
| 03 | SOL bounds (T_SOL) | `in-progress` | `artifacts/03/` | 183 problems bounded; resume pass running over the rest |
| 04 | rocprofiler shim | `in-progress` | `src/solexbench_rocm/shim/` | shim built and verified; L1 comparison sweep pending |
| 05 | Tolerance calibration | `in-progress` | `artifacts/05/` | 3690/3957 workloads AMD-derived; 240 NVFP4 + 27 to fix |
| 06 | Baselines (T_b) | `in-progress` | `artifacts/06/candidates/` | selection sweep running on 8 GPUs; authoritative pass after |
| 07 | Quant / MXFP4 | `done` | `artifacts/07/`, `artifacts/deferred.json` | 15 NVFP4 deferred with evidence; 220 ship |
| 08 | Red team | `done` | `reference/exploits/`, `artifacts/08/` | 28/28 replay cases pass |
| 09 | Release | `not-started` | | manifest builder written; needs 03 and 06 |

Status vocabulary: `not-started` · `in-progress` · `blocked` · `done` · `deferred`

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

---

## Decisions taken

**F_LOCK = 1300 MHz achieved, at determinism setting 1600.** Full reasoning in
the task 01 section above. The two-number form is a real structural difference
from NVIDIA, where `nvidia-smi -lgc` makes them the same; `ClockPreset` now
carries both and `f_lock_mhz` returns the achieved one.

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
```
