# Findings — the settled record

**What this file is.** Every finding this port has established, organised by
what it is *about* rather than by when it was found. It is the archive: things
here are understood. What is still *owed* lives in `TODO.md`; where the port
stands right now lives in `STATE.md`. If an entry here describes something fixed
in code but not yet re-derived in an artifact, it says so and names the owed
work, and `TODO.md` carries the item.

**How to find things.** Read the topic headings — they are written in plain
language and say what was found. Every entry carries its **D-number as a
trailing anchor tag** (`{#d35}`), because **29 D-numbers are cited from code
comments across 168 call sites**, and those comments must keep resolving. The
number is an identifier, not an organising principle: D35 sits next to the other
clock findings, not between D34 and D36. If you arrived here from a comment
saying "see D35", use the **numeric index at the end of this file**, which maps
every D-number to its heading and anchor. Three conventions hold throughout:

* **Corrections attach; they do not rewrite.** Where a later finding retracted
  or narrowed an earlier one, the retraction is stated *inside* the entry, next
  to the claim it invalidates. The ledger is a record of what was believed when.
  The important case is **D61 retracting the headline of D59 and D60** — do not
  quote the 2.021x figure as a reproduction gap.
* **Nothing here is a measurement you can re-derive by reading it.** Numbers are
  quoted from the artifact or session that produced them. Where a figure is
  known stale, the entry says so and gives the newer one.
* **`STATE.md` remains the canonical home of the F_LOCK marker line** that
  `verify_artifacts.py` parses. This file deliberately does not reproduce that
  marker's exact form.

---

# 1. The clock: what F_LOCK is, and what it is not

Every `T_SOL` in milliseconds is a cycle count divided by F_LOCK, and every
`T_b` is a wall-clock time taken at it. Read this section before any other: two
of the largest bound errors in the project are clock errors wearing a modelling
costume.

### Determinism mode does not do what its name suggests {#d8}

`rocm-smi --setperfdeterminism X` does **not** yield X on MI350X. It yields
about **0.81–0.85·X**, rock-steadily, and above ~1900 it stops responding to X
at all and pins to the 1000 W cap. Measured, not looked up
(`clock_calibrate.py determinism-sweep`, `artifacts/01/determinism-sweep*.json`):

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

Applying the MI355X procedure directly produced a failed verification —
requested 1250, observed median 1049.0, drift 201.0 MHz.

Two consequences. **F_LOCK must be the achieved number**: recording the
requested one would overstate the clock by ~23% and make every T_SOL and every
T_b wrong by that factor, plausibly and undetectably. And **above ~1900 the part
stops obeying the setting**, landing on ~1400 MHz whether you ask for 1900 or
2200 — in that regime the clock is set by ambient conditions, which is exactly
what a lock is supposed to prevent.

Setting **1600**, achieved **1300 MHz**. 1600 was chosen over 1700 for power
margin: 1700 draws 947 W of a 1000 W cap on two of the three GPUs sampled, and a
lock that is one warm afternoon from becoming power-bound is not a lock. At 1600
the part draws 868–933 W, so the *setting* binds, not the power limit.

Recorded prominently because it is the single most likely thing for a future
session on other AMD hardware to get wrong: the MI355X procedure (pick F_LOCK
from the unlocked floor, request it, verify) is *correct in form* and produced a
wrong answer *in fact* on this part.

**Per-GPU spread.** At setting 1600, all eight GPUs under sustained load:

| GPU | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| median MHz | 1303 | 1295 | 1264 | 1307 | 1279 | 1296 | 1285 | **1242** |
| min MHz | 1296 | 1293 | 1250 | 1295 | 1278 | 1280 | 1283 | 1217 |

Each GPU is individually stable (min within ~20 MHz of its own median) but they
differ from each other by up to 65 MHz (5%). **Determinism mode gives each GPU
its own steady clock, not a node-wide one** — which is why authoritative timing
is pinned to GPU 0 and every timing artifact records its GPU. F_LOCK is 1300
rather than 1303: a round number 0.2% below GPU 0's median, making T_SOL
marginally conservative.

Unlocked sustained floors (15 min saturating BF16 GEMM, p5 of the final five
minutes, one GPU at a time so per-GPU variation is not confounded with
cross-GPU power coupling): GPU 0 siblings idle **1390**, GPU 1 **1367**, GPU 2
**1335**, GPU 0 with all seven siblings loaded **1400**. Every run sits at the
1000 W cap with junction <= 79 °C, well below the 100 °C slowdown point —
**MI350X is power-limited, not thermally limited**, same as MI355X but at a
400 W lower budget, which is the whole reason its floor is ~350 MHz lower
(1335–1390 against 1725–1757). Per-GPU spread 55 MHz, over the task's 50 MHz
threshold, so F_LOCK had to be chosen for the worst GPU rather than the best.

Stability at F_LOCK: **CV = 0.0034** over 30 trials in separate processes,
against a 0.02 gate — timing noise ~6x below it. (MI355X: 0.0015.) Sibling
interference: seven siblings drawing ~6.2 kW between them moved GPU 0's timing
from 0.1464 ms to 0.1462 ms, **−0.11%**, negligible — which is why sweeps and
authoritative timing may share the node. That figure was measured on a
**0.146 ms** kernel, and [D61](#d61) records that it does not license extending
it to a 31 ms memory-heavy one contending for HBM and fabric.

### F_LOCK is a floor, not a lock, and most "wrong bounds" were that {#d35}

The thirteen problems whose T_SOL a real kernel beat are not thirteen defects,
or ten. They are **three causes**, and the largest one is not a SOLAR error.

`T_SOL_ms = t_sol_cycles / F_LOCK`, and F_LOCK is 1300 MHz — the clock the card
holds under a dense bf16 matrix-core load, which is what task 01 calibrated
against and is not wrong. It is simply not the clock it holds under everything.
Measured with amdsmi sampled every 2 ms on GPU 0, idle node, on the physical
card resolved by PCI bus:

```
bf16 matmul 8192   (the calibration load)   1303 MHz   x1.002
fp32 matmul 8192                            1441 MHz   x1.108
L2__073 kernel     (violating)              1466 MHz   x1.128
L1__074 kernel     (control, not violating) 1446 MHz   x1.112
L1__002 kernel     (control, not violating) 1439 MHz   x1.107
L2__036 kernel     (control, not violating) 1586 MHz   x1.220
```

Lighter arithmetic draws less power and the card clocks up. `--setperfdeterminism
1600` caps the ceiling; it does not pin the clock. So every bound whose workload
clocks above 1300 is too large in milliseconds by exactly that ratio.

**The controls are the argument.** Three of those kernels have never been beaten
by anything, and they clock as high as the violators — one of them higher than
any violation observed anywhere in the benchmark. The bad-bound list is not the
set of problems with inflated bounds. It is the subset where a kernel got good
enough to expose one.

This accounts, on its own, for six of the thirteen: `L2__073` (worst 1.120
against 1.128 of headroom), `L2__068` (1.033), `L2__051` (1.081), `L2__035`
(1.021), `L1__035` (1.015), `L2__030` (1.001). Every one lands under the clock
ratio.

The population says the same thing from the other side. Over every
compute-bound workload of `glm-sweep-2`, `T_SOL / T_measured` by declared dtype:

```
             n    median   p90    >1     within 20% of the bound
float32     759   0.538   0.940   4.6%        25.6%
bfloat16    823   0.222   0.700   1.7%         3.8%
float16     103   0.365   0.686   0.0%         4.9%
fp8_e4m3fn   46   0.184   0.230   0.0%         0.0%
```

fp32 sits two and a half times closer to its bound than bf16 does. The arch
table is not the cause and was checked: strict fp32 matmul measures
**31,834 MAC/cycle** against its own clock versus the table's 32,768 — 97%, an
ordinary GEMM efficiency.

**Blast radius is the part that matters.** This is not about six problems. All
**759 compute-bound fp32 workloads** are scored against a bound 10–22% too
large, so all 759 scores are slightly too generous, and nothing flagged it
because nothing beat most of them. bf16 and fp8 bounds are sound. The divisor
was corrected for the fp32 and fp8 datapaths in v1.1 ([D36](#d36)); the general
blindness argument stands unchanged.

**Undiagnosed residue as of this entry:** five problems the clock cannot buy —
`L2__045` (2.547, bf16), `L1__006` (1.327), `L1__054` (1.269), `L1__005` (1.218,
bf16, [D21](#d21)) and `L1__057` (1.148, bf16, memory-bound). The two bf16 ones
run at ~1300 MHz, so for them the clock is not available as an explanation at
all. Three of the five involve a convolution; that is a hint and not a
diagnosis. **Narrowed since:** `L1__054` was a *wrong measurement, not a wrong
bound* ([D38](#d38)); `L1__006` and `L1__005` were cleared by the
grouped-convolution fix ([D37](#d37)).

**One exposure that had not happened yet.** Setting
`torch.backends.cuda.matmul.allow_tf32 = True` takes fp32 matmul from 31,834 to
**147,954 MAC/cycle — 4.52x** — from one flag, on unchanged fp32 tensors. No
kernel in any run had set it. One that did, and still passed the AMD-derived
tolerances, would beat its fp32 bound by four and a half times while playing
entirely by the rules: nothing in the harness requires the arithmetic to happen
in the declared dtype. The bound prices a datapath, not a problem. **This
stopped being hypothetical the same day** — see [D40](#d40), where three
published submissions did exactly this.

*Corrected 2026-08-10, in place.* This entry's D18 confirmation paragraph first
said the paged bound "moves less than 0.1% across workloads of very different
shape". **It does not**, and the claim was generalised from the violating subset
rather than checked against the problems: `__018` spans 185,274–185,680 cycles
(0.219%, 12 of 47 at the floor) and `__019` spans 185,274–278,144 (**50.1%**,
1 of 38 at the floor). What *is* exact is the floor itself — the time to stream
the whole KV allocation once: 989,669 pages x 576 x 2 B = 1.140 GB, and
1.140 GB / 8 TB/s x 1300 MHz = **185,266** cycles against the manifest's
**185,274**. The allocation term is workload-independent and swamps everything
that is not. The constancy was never measured.

Nothing in this entry changes a manifest number. Full artifact with method and
caveats: `artifacts/11/bound-violations-diagnosis.json`; the measurement scripts
are in `scripts/bounds/`.

### Does the lock cost us anything? On this part, no — it never binds {#d55}

**2026-08-12.** The MI355X report (`html_design/d3`, node `mia1-p02-g10`: 2.70%
spread unlocked -> 21.21% locked, six cards ~320 MHz under a setpoint they
acknowledged) does **not** reproduce here. This is the locked-vs-unlocked parity
check that page 3 lists as never measured.

Method: `scripts/clock_ab.py` + `scripts/bounds/clock_ab_probe.py`, GPU 0 alone,
three conditions visited in **ABBA order** across blocks so drift cancels, five
reps of a 3 s sustained hold per visit, GFX clock and socket power sampled at
2 ms through the PCI-resolved amdsmi handle. The lock is restored in a `finally`
**and** an `atexit`, the perf level is read back and logged, and the node was
left at `setperfdeterminism 1600`, verified.

Six real kernels spanning the measured clock band (`artifacts/12/clock-ab`):

```
condition     geomean vs locked   range          between-block spread
locked1600    1.0000              --             1.33%
unlocked      1.0012              0.996-1.008    0.92%
locked2200    0.9957              0.992-1.000    0.40%
```

Nothing here beats the noise floor, and the reason is in the telemetry rather
than in the ratios: **the achieved clock is the same in all three conditions.**
`L1__002` reads 1438/1438/1438 MHz, `L2__036` 1584/1586/1585, `L1__074`
1444/1446/1445. These kernels draw 305–785 W of a 1000 W cap, so the 1600
setpoint is not the binding constraint for any of them and the policy has
nothing to act on. Unlocked is also not less stable here (0.92% against 1.33%
between blocks, each block a fresh process), which was the other half of the
worry.

Two more rounds went after the regime the first cannot reach, since the cap can
only bind on a load that approaches the power limit.
`artifacts/12/clock-ab-satur` — five loads including the calibration GEMM
itself, five ABBA blocks each:

```
synthetic bf16 GEMM 8192   locked 1305 MHz / 878 W   unlocked 1307 / 879
synthetic fp32 GEMM 8192   locked 1443 / 590         unlocked 1443 / 589
synthetic fp16 GEMM 8192   locked 1306 / 880         unlocked 1306 / 879
L1__003 (bf16 GEMM)        locked 1358 / 900         unlocked 1358 / 900
L1__074                    locked 1446 / 619         unlocked 1446 / 619
aggregate                  unlocked geomean 1.0005, 0.997-1.004, spread 0.6%
```

`artifacts/12/clock-ab-soak` — the same GEMM under **60 s** sustained holds
rather than 3 s, because `artifacts/01`'s unlocked floor run recorded 1390 MHz
at 1001 W and nothing here had reproduced it, so duration was the obvious
suspect. It is not the variable: locked 1303 MHz / 886 W, unlocked 1303 MHz /
886 W, 1.0024x, and the **unlocked arm is the marginally slower one**.

**So the finding is stronger than "locking is safe here". On this node the lock
is not doing anything.** Across twelve loads spanning 305–900 W and 1303–1586
MHz, no condition was found in which the 1600 setpoint binds: the card lands on
the same clock with the cap, without it, and with the cap raised to 2200. The
stability the benchmark relies on (rep CV 0.05–0.5%) is a property of a
power-limited part sitting at a consistent operating point, **not** a product of
`perf_determinism`.

**Unresolved, and recorded rather than smoothed over:** `artifacts/01` has an
unlocked GPU 0 reaching 1390 MHz at 1001 W and this session never got the card
above 900 W on any load. Duration is ruled out. The remaining explanation is
that the earlier probe's loop differs from `torch.mm` in a way that draws the
last 100 W — it was not identified, and until it is, the claim "the cap never
binds" is true of everything measured here and not proven of everything.

### D3 is the question D55 answers {#d3}

`scripts/clock_ab.py:3` opens with the docstring
`"""D3 — is the clock lock costing us performance on THIS node?"""`. **No D3
entry was ever written anywhere in this repo.** The question it names is the one
[D55](#d55) measures and answers: on this node, no, because the 1600 setpoint
never binds. This anchor exists so that the citation resolves rather than
dangling. (D4 is a gap in the numbering — it appears nowhere, in code or prose.)

### `f_lock_mhz: null` was blamed on a preset that exists {#d25}

**This entry is itself a retraction.** Three documents (`CLAUDE.md` §3 and §5b,
`TODO.md`, `leaderboard/DESIGN-v2.md` §6) stated that `CLOCK_LOCK_PRESETS` has
no MI350X entry, that this is why some artifacts stamp `f_lock_mhz: null`, and
that it is the one remaining task-01 gate failure. **All three claims were
false.** The claim originated in the session-1 handoff, which was correct when
written — the entry did not exist yet — and was then copied forward into
documents that outlived it. That is the failure mode: not a wrong statement, a
true one that nobody re-checked. The entry was added in **2cdb7b0**, 2026-08-03
20:36 UTC, and `--task 01` reports 11 checks, 0 failed.

The null is real, it has two causes, and **neither loses a measurement**. 28
artifacts carried it at the time of writing:

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
stamp `python: 3.11.7`, `torch: {"available": false}`, `rocm.version: 7.15.0` —
the host, not the pinned container (`3.12.3` / `torch 2.9.1+rocm7.2.0` /
`rocm 7.2.0`). `provenance.f_lock_mhz()` resolves the preset through
`torch.cuda.get_device_name(0)`, `import torch` raises `ModuleNotFoundError` on
this host, and the function's `except Exception` returns `None`. Confirmed both
directions: a bare host call returns `None`; the same call with
`SOLEXBENCH_F_LOCK_MHZ=1300` returns `1300`.

Host execution is by design. `agent_score.py`, `agent_cost_report.py` and
`agent_baseline.py` orchestrate and never touch a GPU — `agent_score.py` loads
`sol_score.py` by file path specifically because host python has no pydantic,
and each kernel is shelled into the container through `env/solb`. So the timings
are stamped and the roll-up is not: every `artifacts/10/*/retimed/*.json`,
written inside the container by `agent_eval.py`, carries `f_lock_mhz: 1300`,
`python: 3.12.3`, `visible_devices: "0"`. `meta.f_lock_mhz` in the leaderboard
database is `'1300'`, from the manifest, so **nothing published is affected**.

The tempting non-fix — defaulting `stamp()` to 1300 — is prime directive 2 in
miniature: it would make a roll-up written on MI355X claim a clock it was never
measured at, undetectably.

**Narrowed since, and the enumeration above is now incomplete rather than
wrong.** A later recount found **37** top-level and **45** by grep; nine further
nulls have appeared since. `TODO.md` carries the current count, the two ranked
fix options, and the note that `glm-run1`/`submitted-apitest` `run.json` have no
`_provenance` block at all.

The lesson is not about clocks. **A wrong explanation propagates faster than a
wrong number**, because nothing runs it and nothing checks it. This one survived
into a `TODO.md` whose own header said it was rewritten to remove stale items —
and the gate that refutes it runs in seconds.

### A manifest built from a directory holding two clocks {#f18}

Merging two ports of this benchmark added 87 T_b artifacts measured at F_LOCK
1300 into a directory of artifacts measured at 1640. No conflict was raised —
a three-way merge does not conflict on a file present on only one side — and the
manifest then built from the mixture without complaint.

Every one of those files was internally correct and correctly stamped. **The
*directory* was wrong, and a directory has no provenance of its own**, which is
precisely why every artifact here carries one. T_b is a wall-clock time, so
those 87 problems would have carried an anchor from one clock while their
kernels were timed at another, rescaling those problems' scores by the ratio,
per problem, invisibly.

`collect_t_b()` now takes the expected F_LOCK — read from the same
`CLOCK_LOCK_PRESETS` table `lock_clocks()` applies from, so the manifest and the
hardware cannot disagree — and rejects any artifact measured elsewhere with a
loud count. An artifact with *no* recorded clock is still admitted: that is a
missing-provenance defect, which `check_06` already covers separately, and
conflating the two would hide both. Regression tests in
`tests/scripts/test_build_manifest.py`.

**Retracted in part, in place, and again by [F24](#f24).** The first version of
this entry claimed more than the check delivers. It compares the artifact's
stamp against the preset table; both read from the same place, so it catches an
artifact from *another clock* and is **blind to an artifact whose stamp is
simply wrong**. That is not a hypothetical gap: an unreset determinism sweep
left a node at a 1900 MHz setpoint, `provenance.f_lock_mhz()` returned the
preset's 1640 without reading a device, and **143 artifacts measured at
~1860 MHz were stamped 1640** — then 1640 was checked against 1640 and passed.
Eleven hours of measurement, every value about 12% faster than the number it
claimed. The original argument was that reading the expected clock from the same
table `lock_clocks()` applies from meant the two could not disagree. **The table
is not the hardware.**

The lesson generalises past clocks: a three-way merge reasons about files, while
a measurement's validity is a property of the set it belongs to, so the durable
defence is the consumer checking provenance rather than the merger being
careful.

### The clock guard failed open off-GPU {#f21}

`collect_t_b(dir, None)` admits every artifact regardless of clock, and
`f_lock_mhz()` returns `None` off-GPU with no override set. Building the
manifest in the wrong environment would therefore restore the exact defect
[F18](#f18) fixes — silently, with normal-looking output. `build_manifest.py`
now refuses to build when F_LOCK cannot be resolved. Verified the guard does not
change the result: rebuilding with `SOLEXBENCH_F_LOCK_MHZ=1300` reproduces
manifest v1 exactly — 235 problems, 220 scoreable, 3717 workloads, identical
`bound_sources`.

### The determinism setpoint, read back off the hardware {#f24}

F18's own retraction scopes its fix out as "a change to the timing runners". It
is not. `amd-smi metric -c` reports the setpoint as `MAX_CLK` per GFX block on
an **idle** device, so task 01 now compares the hardware against the preset's
*requested* clock with no load and no timed region:

```
[PASS] every GPU is at the preset's determinism setpoint   all 8 GPUs at 1600 MHz
```

Run against a **negative control** rather than merely observed passing, since a
check nobody has watched fail is the subject of the whole review it came from:
against a preset requesting 1640 the same hardware reports
`FAIL (8 GPUs at [1600])`.

**This node audited while implementing it.** All 8 GPUs at setpoint 1600, and
GPU 0 under sustained load holds **1295–1303 MHz** against a stamped 1300 — so
the 220 T_b artifacts here are *not* affected by the defect F18's retraction
describes. Verified rather than assumed.

Not fixed here: stamping the **observed** clock rather than the requested one in
`provenance.f_lock_mhz()`. That is the real end state and it changes the
provenance of every artifact the project produces, so it is its own change.

---

# 2. The bounds: how a speed-of-light number goes wrong

`T_SOL` is a **lower** bound, so every term in it must be a lower bound: the
fastest rate the arithmetic could plausibly run at, and the least traffic it
could plausibly move. Each entry below is a way that failed. The single most
important structural fact is stated in D14 and repeated by every entry after it:
**a roofline that is internally consistent is not thereby right**, and the only
automatic check in the repo looks in one direction.

### The bound was priced at the vector-FP32 rate on 160 of 235 problems {#d14}

The single largest error found in the port, and it was invisible until
`T_SOL <= T_b` could be checked against real measurements: **437 workloads had a
T_SOL above their own measured time**, by up to 13.4x.

`_precision_for()` chose the *widest* dtype among a problem's inputs, on the
reasoning that the widest drives both the compute peak and the byte count. That
is right for bytes and exactly wrong for the rate. In SOLAR's config `fp32` is
`MAC_per_cycle_fp32_sm` — the **vector** rate, 32,768 MAC/cycle, 16x below the
bf16 matrix rate — so a bf16 kernel with one `float32` epsilon argument was
priced at 0.085 PFLOPS instead of 1.36. **160 of the 235 problems** resolved to
`fp32` that way, most of them mixed-precision kernels whose scalar `eps` decided
the answer.

The fix is two changes in the same direction: scalars (`shape: null`) are
excluded — an `eps` rides in a kernel argument and is not a compute precision —
and among the tensor inputs the **narrowest** floating type wins, not the
widest. After the fix: `fp32` 108, `bf16` 104, `fp16` 12, `fp8` 4, `nvfp4` 1,
and the violations fall from 437 to **63**.

The 63 that remain are two kinds, both recorded rather than smoothed over:
depthwise-convolution problems where SOLAR counts a grouped convolution as dense
(`L1/006`, `L1/029`, `L2/035`, ratios 2–5.8x — this is [D37](#d37)), and eight
workloads within 1–6% where the model and the measurement are simply that close.
Neither is shipped: `build_manifest` rejects **any** candidate bound above the
measured time, from either derivation, and falls back to the other — a bound
above a measured time makes `(T_b − T_SOL)` negative and pushes scores past 1.

**What this says about the method, and it is worth saying plainly:** cross-check
D is the only one of the four that could have caught this, and it could not run
until task 06 finished. Checks A–C all passed throughout, on a bound that was
wrong by 13x on some problems.

### T_SOL was truncated to whole cycles, and eight truncated to zero {#d12}

SOLAR emits `total_cycles` already wrapped in `int()`, and the bridge wrapped it
again. At 1.3 GHz a cycle is 0.77 ns and the smallest workloads here are
genuinely sub-cycle — 12 KB at 8 TB/s is about two cycles — so the rounding is
not an edge case at the small end, it is the normal case. **Eight workloads
ended up with `T_SOL = 0` cycles**: a bound of zero time, which no kernel can
approach and which puts a division by `(T_b − 0)` into the score. A further
**204 workloads implied a DRAM bandwidth above the arch config's own peak**,
which is impossible for a roofline and was the symptom that led here.

The bridge now recomputes the roofline from the quantities SOLAR reports beside
the result — `max(MACs / MAC_per_cycle, bytes / DRAM_byte_per_cycle)`, its own
formula — and ceils, keeping the exact figure in `t_sol_cycles_exact`. All 185
successful problems were refreshed with `--only-status ok`, which re-runs the
successes without paying again for the failures (a failure means SOLAR ran to
the timeout, so they are the expensive ones).

### The bound prices the allocation, not the work — paged attention {#d18}

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

The bound matches "read the entire cache" to 0.1%. **It is not a bound on the
work; it is a bound on the allocation.**

**Why no existing check caught it.** The gate is `T_SOL <= T_b`, and T_b is
measured from the PyTorch reference — which *also* reads the whole cache. Both
numbers are wrong in the same direction, so they agree. Only a gather-aware
kernel separates them, and until this run nothing on this benchmark was
gather-aware.

**Exposure — 6 problems, 249 scoreable workloads:**

| problem | scoreable wl | median over-count | worst |
|---|---:|---:|---:|
| `FlashInfer-Bench__014_gqa_paged_prefill…kv4` | 30 | 4,612x | 176,707x |
| `FlashInfer-Bench__015_gqa_paged_prefill…kv8` | 38 | 4,531x | 276,155x |
| `FlashInfer-Bench__019_mla_paged_prefill` | 38 | 7,367x | 197,934x |
| `FlashInfer-Bench__018_mla_paged_decode` | 47 | 136x | 123,709x |
| `FlashInfer-Bench__012_gqa_paged_decode…kv4` | 48 | 1x | 129x |
| `FlashInfer-Bench__013_gqa_paged_decode…kv8` | 48 | 1x | 128x |

Over-count is `num_pages / num_kv_indices` per workload. The two decode problems
are mostly sound at the median and wrong only in the tail; the three prefill
problems and `018` are wrong at the median by three to four orders of magnitude.
This is very likely also what makes `018`'s anchor irreproducible ([D15](#d15))
— same family, same layout dependence.

**What was done at the time.** `agent_score.py` enforces the invariant directly:
a workload whose measured time falls below its T_SOL is marked
`bound_violation`, excluded from the headline mean, and reported. On the pilot
that is the difference between a reported mean S of 0.8401 and the honest
0.7757.

**Status now.** v1.1 corrected the six paged problems ([D36](#d36)), and found
that D18 needed a second half nobody had noticed. **The underlying tier defect
is not fixed** — it was fixed per-problem, not at the tier — and
[D42](#d42) measures what still rests on it: **328 workloads across 38
problems**. That is the v1.3 item in `TODO.md`.

### The five surviving bad bounds are two causes, and one is D18 again {#d42}

**2026-08-10.** `scripts/bounds/diagnose_bad_bounds.py`,
`artifacts/11/bad-bounds-v12.json`. Under manifest v1.2, five problems still
have a `T_SOL` that a measured kernel beat. They are not five modelling errors.

**Cause A — the declared-traffic tier prices every declared input at its full
allocation, whether or not the kernel reads it.** Three of the five, and the
hand computation lands on the manifest number exactly in every case:

| problem | what is priced but not read | hand ÷ tier |
|---|---|---|
| `L1__018` | the whole 262,144-slot k+v cache, read **and** written (2.15 GB), for `seq_len` slots touched | 0.996, 1/1 |
| `L1__042` | `expert_mask` (int64) and `topk_idx` — `run()` reads neither, it uses `topk_idx.shape[0]` | 1.000, 16/16 |
| `L1__057` | the 157,184-row embedding table (1.29 GB), for a gather of `B×S` rows | 1.000, 11/11 |

`L1__042` is exact to the bit: the tier is 4.0625 units of
`batch_seq_len×256×4` against a live 2 units — **65/32 = 2.03125x on all sixteen
workloads**.

**This is D18.** Pricing a KV cache at its allocation rather than at the part
touched is the same defect; v1.1 fixed it for the two FlashInfer problems rather
than at the tier, so it is still live. **328 workloads across 38 problems** are
scored against a bound where `max_of_both` picked traffic over SOLAR — p50
1.50x, p90 5.00x, max **128.9x** (`L1__087`, the embedding table again). Most
produce no violation only because nothing has got close enough.

**Cause B — SOLAR counts arithmetic the reference discards.** `L2__045` prices
the Q-Former and projector over all `ceil(audio_seq_len/15)` windows, but the
reference reads only the first `ceil(num_audio_tokens/40)`: **21.3–22.0% — a
factor of 21.3–22.0x — of the counted MACs never reach the output**. A second,
independent error partly masks it: those fp32 einsums are priced at the **bf16**
MAC rate, exactly 1/16 on all sixteen workloads. Net 1.34x too large against an
fp32 kernel, and a kernel using bf16 where the tolerance permits crosses it by
2.8x.

Note the direction of that second error: pricing fp32 work at the bf16 rate
makes the bound *smaller*, which never breaks the lower-bound invariant and so
is never reported. It is [D39](#d39), not a violation — and here it hid a 21.5x
over-count for long enough that only a kernel doing both could expose either.

**`L2__073` is not a modelling error.** Worst violation 0.9916, inside the band
[D35](#d35) predicts for a compute-bound fp32 workload clocking above the
1300 MHz divisor. Nothing to derive.

The mechanisms are confirmed; **no bound is corrected by this entry.** A hand
check that matches proves what the number *is*, not what it should be, and
re-deriving is a separate step that must not be done by adjusting a value until
the violation disappears.

### SOLAR prices a grouped convolution as a dense one {#d37}

**Fixed 2026-08-10 in manifest v1.2**, six problems. Correction:
`src/solexbench_rocm/solar/conv_groups.py`, applied explicitly by
`sol_bounds.py`. Re-derived bounds: `artifacts/11/d37/`. Manifest:
`scripts/rebuild_manifest_v12.py` -> `artifacts/09/manifest-v1.2.json`, 81
workloads moved, 25 bottleneck flips, 0 re-gated. Bound violations 5 -> **3**
(`L1__005` and `L1__006` cleared).

**SOLAR ignores `groups`**, so a depthwise or grouped convolution is priced as
though every input channel fed every output channel, and the arithmetic term is
over-counted by the group count. The mechanism is exact:

```
L1__006  true MACs   599,040 = B x inner_width x (L+2) x K = 2 x 768 x 130 x 3
         SOLAR   460,062,720
         ratio         768.0  = inner_width, to the digit

L1__029  SOLAR 10,995,116,277,760
         = dense conv 16 x 16384 x 16384 x 512 x 4  (8,796,093,022,208)
         + in_proj    16 x 512 x 32768 x 8192       (2,199,023,255,552)
         exactly. The conv term alone is over-counted 16,384x.
```

`L1__029` is the one to read carefully, because it shows the effect is not the
mechanism: its convolution is a small share of its real work, so a 16,384x error
on that term is about **5x** on T_SOL. The mechanism is exact; what it costs is
per problem.

Measured MAC ratios after the fix: `L1__006` **x768.000** exactly, `L2__035`
x6.70–7.07, `L1__029` x4.999, `L2__058` x4.66–4.75, `L2__051` x3.17–3.26,
`L1__005` x1.999.

`groups` is recovered from the tensor shapes — `groups = in_channels //
weight.shape[1]`, exact for every convolution however it was called — rather
than from a new argument-parsing branch. The argument can be positional, a
keyword, or carried on a module, and a parser that misses one spelling fails
silently in the direction that inflates the bound, which is the failure being
fixed. Shapes cannot be spelled two ways.

The wrapper is **not a patch to SOLAR**. SOLAR is pinned by SHA in
`env/Dockerfile` and deliberately not vendored; a patch file would mean
rebuilding the measurement image to change a bound derivation, and the image is
the thing that must not move under a measurement. The wrapper lives in the port,
is version-controlled with the artifacts it produces, and `apply()` raises if
SOLAR's handler class names move rather than silently deriving bounds that lost
the fix.

#### The in-place retraction inside this entry, kept adjacent to what it invalidates

**`L2__036` was never in scope, and the scope was wrong when this entry was
first written.** Its ratio is x1.000, and the reason is not that the fix missed
it. Re-scanning by `run()`'s function body rather than by the reference file,
**six** problems call a grouped convolution in the traced graph — `L1__005`,
`L1__006`, `L1__029`, `L2__035`, `L2__051`, `L2__058` — and v1.2 corrects all
six. `L2__036`'s `F.conv2d(..., groups=C)` is in `get_inputs`, which builds the
intermediates its backward pass consumes: **not traced by SOLAR, not timed by
the harness.**

The original scope of seven came from an AST scan of the whole file. The
superseded text read:

> Seven problems call a convolution with an explicit non-1 `groups`: `L1__005`,
> `L1__006`, `L1__029`, `L2__035`, `L2__036`, `L2__051`, `L2__058`. (A plain
> grep for `groups=` returns 44 and is wrong — most are GQA's
> `num_key_value_groups`, which is not a convolution.) Four of the seven have
> been on the bad-bounds list and three never have, and the three are not the
> ones with correct bounds. `L2__036` is the case in point: never flagged, and
> it is the kernel that clocked **1586 MHz**, the highest measured anywhere on
> this node — a kernel running far above its datapath's saturating clock is not
> saturating that datapath, which is what a compute bound priced for work the
> kernel never does looks like from outside.

**Everything in that paragraph about `L2__036` being the witness for "the list
is blind to bounds nothing can reach" is wrong, and wrong in the opposite
direction.** Its bound errs *low*, not high, and its 1586 MHz was never evidence
of anything but the fp32 vector datapath doing what [D35](#d35) measured it
doing. The general claim it was being used to support survives on other evidence
and is [D39](#d39).

What is kept from the same passage, because it is independently true: **a bound
is only shown to be wrong by a kernel that beats it**, so the bad-bounds list
finds errors small enough for a kernel to cross and is structurally blind to
errors large enough that nothing gets near. That inverts the reading in
[D31c](#d31c): a longer bad-bounds list is not a worse benchmark and a shorter
one is not a better one. The list length tracks how close anything has got, and
the worst bounds are the ones nothing can reach.

#### The residue this entry left open, and what happened to it

Clocks of all five residue kernels, measured — the earlier attempt exec'd kernel
source from a string and triton's `@jit` refused three of them, so they are
loaded from a real file now (`scripts/bounds/clocks_residue.py`):

```
L1__005  1352 MHz  x1.040  bf16     L1__057  1313 MHz  x1.010  bf16, memory-bound
L1__006  1516 MHz  x1.166  fp32     L2__045  1554 MHz  x1.195  bf16
L1__054  1449 MHz  x1.115  fp32
```

**`L1__005`**: SOLAR is exactly **2.0x** a hand count of its three terms, and
its conv is 0.05% of the work, so `groups` does not explain a factor of two — an
exact 2.0 smells like a MAC/FLOP conflation on part of the graph. **`L1__054`**:
SOLAR's MAC count is **correct** to the digit, so it is not a counting defect.
At this problem's exact shape a plain fp32 GEMM reaches **29,627 MAC/cycle**,
*below* the table's 32,768; transposing B the way the kernel does changes
nothing; the three GEMMs the reference performs take 602.5 µs back to back and
the fused single GEMM 590.5 µs. The kernel is timed at 476.5 µs, and its source
explains the wall clock honestly — for M >= 1024 it runs the `value` GEMM on a
second stream, so two of three overlap, and 602.5 x 2/3 = 402 µs is the right
order. It stays on the default stream and synchronises before returning, so it
is an optimisation and not an exploit. What is left is the rate: 25.77e9 MACs in
476.5 µs at 1449 MHz is ~37,300 MAC/cycle, about **14% above** the table's
32,768, and no arrangement of streams exceeds a device's peak. So either
`MAC_per_cycle_fp32_sm` is too low — an 8192-cube GEMM cannot tell "the peak is
32,768 and one GEMM reaches 97%" apart from "the peak is higher and one GEMM
cannot reach it", and two concurrent GEMMs is exactly the experiment that
separates them — or the harness's reported latency for this workload is not the
wall time of `run()`. **The second answer turned out to be the right one: see
[D38](#d38), which shows `L1__054` was a wrong *measurement*, short by 32%, and
never a wrong bound at all.** **`L2__045`**: 2.13x residue after its 1554 MHz is
accounted for; largest and least understood, and still open under
[D42](#d42) Cause B. **`L1__057`**: memory-bound, so the clock cancels; a
traffic question, and still open under [D42](#d42) Cause A.

Full artifact: `artifacts/11/grouped-conv-bound-defect.json`.

### Manifest v1.1: what the two corrections actually moved {#d36}

v1 is frozen and unchanged. `artifacts/09/manifest-v1.1.json` sits beside it,
built by `scripts/rebuild_manifest_v11.py`, which **repeats no measurement**:
every cycle count is v1's, and only the conversion to milliseconds and the paged
byte count change.

```
clock-corrected workloads : 805   (fp32_sm 759 @ 1441 MHz, fp8_tc 46 @ 1314)
paged-corrected workloads : 249   (6 FlashInfer problems)
total T_SOL_ms changed    : 1048
re-gated (kept v1 value)  : 0
```

**Bounds a real kernel beats: 13 -> 6.** Gone: `FlashInfer-Bench__018`,
`L1__035`, `L2__030`, `L2__035`, `L2__068`. Remaining: `L1__005`, `L1__006`,
`L1__054`, `L1__057`, `L2__045`, `L2__073`.

The per-datapath clock is `max(F_LOCK, measured)`, not the measurement alone.
T_SOL is a lower bound, so where the part can go faster the bound must assume it
does — and the matrix-core measurements (bf16 1296, fp16 1299) sit inside
scatter of 1300, so writing them in would assert a 0.3% correction the
measurement does not support and move 926 bounds for nothing. Only fp32 (1441)
and fp8 (1314) clear F_LOCK; only those move.

**D18 needed a second half nobody had noticed.** Correcting the declared-traffic
tier fixed four of the six paged problems and **not `018`**, because SOLAR has
the same defect independently: its `memory_bytes` for that problem is
**1,140,133,554** — the whole allocation, to the byte — and its bottleneck is
`memory`, so `max(solar, traffic)` kept exactly the number being corrected. A
memory term computed over an allocation is the same error wherever it appears,
so both tiers are recomputed from the gathered bytes and only SOLAR's
*arithmetic* term survives from it. **018's first workload goes 185,274 cycles
-> 8.**

That is worth saying plainly: the corrected bound for a small paged workload is
very loose, and the score there degenerates towards a plain speedup against T_b.
That is a worse bound to read and a correct one to use — too small is loose, too
large is wrong, and only one of the two lets a kernel score above 1. See
[D39](#d39) for what that costs.

**The residue validates the diagnosis rather than surviving it.** `L2__073`
falls from 1.120 to **1.010**: its kernel holds 1466 MHz where the saturating
fp32 GEMM holds 1441, and 1.010 x (1441/1466) = 0.993. The 1% left is exactly
the gap between "the clock this datapath holds flat out" and "the clock this
particular kernel held", and closing it would mean dividing bounds by a clock
the datapath does not sustain under load. `L1__006` 1.327 -> 1.197 and `L1__054`
1.269 -> 1.145 — part clock, part real defect. `L1__005` (bf16), `L2__045`
(bf16) and `L1__057` (memory) do not move at all, which is what the diagnosis
predicted for those two families.

Rescored from stored `retimed/*.json` with `--reuse-retimed`; no GPU time and no
re-measurement. `scored.json` now records `manifest_version`, because a score is
only meaningful inside one.

**What it moved on the board, which is far less than the size of the correction
suggests.** The published figure is `summary.mean_score`, and that already
EXCLUDES bound-violating workloads — exactly the ones a bound correction moves
most. So the correction mostly re-priced workloads that were not being scored:

```
                     mean_score          incl. invalid bounds     violated
glm-sweep-2   0.6083 -> 0.6111 (+.0028)  0.6288 -> 0.6158          72 -> 26
gpt56-40      0.6457 -> 0.6423 (-.0034)  0.6884 -> 0.6480          12 ->  9
```

glm-sweep-2 goes **up**: it had 72 violating workloads and now has 26, so 46
workloads that scored nothing now score something. The two runs move in opposite
directions and by about a third of a percent each. Head to head on the shared
workloads both runs scored, which is the comparison that matters: **+0.0173
under v1 (n=643) and +0.0155 under v1.1 (n=656)**. The lead shrinks by a tenth
and the ordering does not change.

*Corrected 2026-08-10, in place.* An earlier reading of this entry said the
correction removed about two thirds of gpt-5.6-sol's lead. That was computed on
`mean_score_including_invalid_bounds` — a basis that includes the broken bounds
— and `including_invalid` is not what anything publishes. On the published basis
the effect is roughly a tenth. **The error is kept visible because it is the
same shape as the defect being fixed: a number that looked like the headline, on
a denominator nobody had checked.**

`verify_artifacts.py --task 03` still reports 13 and still fails check D. That
check reads **v1**, the frozen release artifact, and is meant to keep reporting
what v1 shipped. It is not a regression and it is not stale.

### Two bounds a real kernel beat, and neither was D18's mechanism {#d21}

`glm-run1` (GLM-5.2, 24 problems, re-timed on GPU 0) beat T_SOL on two problems
that are **not** paged attention:

| problem | T_SOL source | beaten by | workloads |
|---|---|---|---|
| `L1__005_conv_gated_projection_with_causal_conv` | `solar_fused`, compute | **1.09–1.15x** | 4 of 16 |
| `L1__035_flux_ada_layer_norm_zero_modulation_extraction` | mixed | **1.003–1.013x** | 2 of 16 |

Different in kind, and probably from each other too. `L1__005` is a
compute-bound SOLAR roofline ~15% too slow. `L1__035` is beaten by 0.3–1.3%, on
a problem whose headroom is only `T_b/T_SOL = 1.008`: there is almost no scoring
range there at all, so a 1% timing difference flips it either way.

**Both are now resolved, and neither resolution was a bound re-derivation of the
kind this entry expected.** `L1__035` was cleared by v1.1's per-datapath clock
([D36](#d36)); `L1__005` was cleared by the grouped-convolution fix in v1.2
([D37](#d37)). The open question this entry posed — "whether that is a wrong
bound or a bound too tight to be measurable against" — became [D39](#d39)'s
other tail: 13.6% of workloads have under 2x of headroom, and `L1__035` at
1.008x is the visible end of that band rather than an outlier.

**Ingest bug found on the way, and fixed.** A bound a real kernel beat is a fact
about the *bound*, not the run that exposed it, but `ingest.py` wrote the list
with `INSERT OR REPLACE` per run and skipped excluded runs entirely. So each run
overwrote the last, and taking the pilot off the board deleted
`FlashInfer-Bench__019` — the D18 problem — from `/methodology`. Now accumulated
across every run read, excluded or not.

### A second model found a bound that 220 problems of the first did not {#d31c}

`gpt56-40` (codex-cli, `gpt-5.6-sol`, 40 problems, 2026-08-09) scored **670
workloads, mean S = 0.6457, 0 flagged, 40/40 problems clean**. Twelve workloads
were excluded because a real kernel beat T_SOL: `L1__006` and `L2__045` were
already known ([D31](#d31)), and **`L2__051` is new**. Including the twelve
would report 0.6884, which is why the scorer refuses to.

The interesting part is not the count going 12 -> 13. It is the direction. Every
previous increase came from *more coverage*: a stronger optimizer reaching
problems nobody had reached. This one came from **less** — 40 problems against
220 — and it still exposed a bound the larger sweep missed. A second model with
different habits is a second search direction, not more of the first. So "how
many bad bounds are left" cannot be extrapolated from one model's sweep, however
complete, and the 13 remains a lower bound in a way that adding coverage alone
will not close.

**Narrowed by [D37](#d37).** This entry's reading — that the count going up is
the finding — is inverted where it is read as a quality signal. The list length
tracks how close anything has got; the *worst* bounds are the ones nothing can
reach and which therefore never appear on it at all. Both readings are true of
different things: coverage does not bound the count (this entry), and the count
does not bound the defect set ([D39](#d39)).

On the 643 workloads the two runs share, `gpt-5.6-sol` scores **0.6469** against
GLM-5.2's **0.6295**, winning 24 problems to 16 — a real lead, on an eighth of
the benchmark, and not the gap the headline mean-S figures suggest.
Whole-benchmark score is **0.1143**, because 180 problems are untouched and
coverage is part of the score. Both numbers are on the board; the run page
labels which is which.

### The defect class nothing checks: a bound far below anything achievable {#d39}

**Measured 2026-08-10.** `scripts/bound_headroom.py`,
`artifacts/11/bound-headroom.json`. CPU only, no GPU, no measurement.

The board enforces exactly one invariant on a bound: **no measurement may beat
it.** That catches a `T_SOL` too *large*, and it is the only automatic check in
the repo. A `T_SOL` too *small* breaks nothing — it is a perfectly valid lower
bound, just a uselessly weak one — so nothing reports it, ever.

It is not harmless. With

    S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))

as `T_SOL -> 0` the score becomes `T_b / (T_b + T_k)`: a comparison against the
PyTorch anchor with no roofline content left in it. Those problems are scored on
a different question from the rest of the board.

Headroom `T_b / T_SOL` over manifest v1.2, all 3,717 scoreable workloads:

```
p10   1.74      under 2x      504  (13.6%)
p50  15.61      2x - 10x     1086  (29.2%)
p90 1978        10x - 100x   1300  (35.0%)
p99 1.66e5      100x - 1000x  397  (10.7%)
max 1.50e6      over 1000x    430  (11.6%)
```

**827 workloads, 22.3%, have more than 100x of headroom.** Against the three
problems on the known-wrong-bounds list, that is the proportion of this
benchmark whose scores are dominated by `T_b`. Worst by median headroom:
`L2__006` 115,005x, then six FlashInfer paged problems between 19,000x and
41,000x, `L1__016` 19,474x.

**Two things follow that are uncomfortable and both should be said.**

The FlashInfer paged problems are at the top of that list *because of v1.1*.
D18's correction was right — the kernel gathers a few pages and the bound was
pricing the whole allocation — but `FlashInfer-Bench__018` went from
**185,274 cycles to 8**, and a bound of 8 cycles against a `T_b` in milliseconds
is correct and vacuous. Those bounds went from wrong-and-tight to
right-and-empty, and nothing in the pipeline noticed because the only check
looks the other way.

And the opposite tail is real too: **13.6% of workloads have under 2x of
headroom**, a scoring range so narrow that run-to-run variance is a material
share of the score. `L1__035` at 1.008x total headroom ([D21](#d21)) is not an
outlier so much as the visible end of that band.

**No threshold is asserted.** A large headroom is not automatically an error — a
memory-bound problem whose PyTorch reference is simply a poor implementation
shows one honestly. What is wrong is that nobody was looking. The workloads are
marked (`bound_quality`) and **not fixed**; the marking is not in the manifest.
Nothing about D39 itself changed when the board's number formatting did — 827 of
3,717 are still at or above 100x, still one-sided, still unfixed.

### BLOCKED: rocprofv3 counter collection hangs in this container {#d43}

**2026-08-10.** [D42](#d42) establishes by hand that the declared-traffic tier
prices bytes the kernel never moves. Deriving the replacement by hand and
checking it by hand is the mistake `CLAUDE.md` §6 names, so the next step needs
an **independent measurement of the bytes actually moved**.

`scripts/bounds/measure_traffic.py` was written for it — rocprofv3
`--pmc FETCH_SIZE WRITE_SIZE` over the reference, cards 1–7, card 0 refused.
**It does not run.** `rocprofv3 --pmc` produces no output and never exits:

| probe | kernels | waited | result |
|---|---|---|---|
| `L1__042` reference, 5 reps | ~10 | 15 min | killed, no counter rows |
| `a + 1.0` on a 4096² bf16 tensor, 3 reps | 3 | 4 min | killed, no counter rows |

Three kernels is not a workload problem. Checked and ruled out: no stale
rocprofv3 session in the container holding the counter block (only zombie
pythons from 2026-08-07), and the hang reproduces from a cold start.

**This does not implicate the shim.** `src/solexbench_rocm/shim/` uses
rocprofiler-sdk's *dispatch callback* path — timestamps — and that is built,
validated over 1430 pairs at −0.61% median divergence, and used by every
measurement in the repo. The PMC counter path is a different interface and is
the one that is broken. Nothing measured so far is affected.

Not diagnosed further and **nothing was changed to work around it** — a
container rebuild or a capability change to make counters work would alter the
image every baseline was measured in. Recorded and left.

**The alternative does not need counters.** `CLAUDE.md` §6 says only an
independent kernel separates a bound from its anchor, and that is a *timing*
measurement: write a minimal kernel that moves only the traffic the problem
actually requires, time it on GPU 0, and compare against the tier. Slower path,
no counters, and it is how all three known-bad bounds were found in the first
place. That is the next step, not a fix to this one.

---

# 3. The anchor and the timer: what a measured time actually measures

`T_b` is the other half of every score. These entries are about the act of
measuring: what the bracket contains, whose GPU it ran on, and whether the
number comes back the same twice.

### The timer never saw a submission's own stream {#d38}

**Fixed 2026-08-10.** Two probe artifacts, before and after, on the same probes:
`artifacts/11/side-stream-timing-hole.json` and `…-hole-fixed.json`. Script
`scripts/bounds/side_stream_timing.py`. GPU 0, idle, F_LOCK stamped, hip_events
— the methodology every published score on this board was measured with
(`manifest.methodology`).

`bench_time_with_cuda_events` brackets each timed iteration with two events
recorded on the **current** stream, and the loop deliberately does not
synchronize between iterations "to keep the driver's GPU queue full". So the
default stream carries a deep backlog while a submission's own stream is empty:
work launched there executes immediately, against *earlier* iterations, and its
duration never lands inside the bracket that enqueued it.

| probe | before | after | true (serialized wall) |
|---|---|---|---|
| 8192³ fp32 GEMM, side stream, host-synced in-call | 12.059 | 12.049 | 12.04 |
| same, default stream (control) | 12.067 | 12.021 | 12.03 |
| **same, join deferred to the next call** | **0.0069** | **12.058** | 12.04 |
| **`L1__054` as submitted** | **0.4780** | **0.6974** | 0.707 |
| the same computation, one stream (control) | 0.6781 | 0.6855 | 0.699 |
| the same overlap, joined with `wait_stream` | 0.6867 | 0.7107 | 0.725 |

Milliseconds, medians of 50. **1743x** on the deferred-join probe: that is a
bypass, not a shave.

**What the guard that existed did and did not do.** `check_default_stream` has
been in `reward_hack.py` since the port, and its own docstring names this exact
hole — "an event pair recorded on the default stream simply does not observe a
kernel running on another one, so the work is free". It catches a submission
that *leaves* a non-default stream current. `test_side_stream_restored` in the
corpus records, in writing, that a submission which restores it walks past. So
the gap was known and documented for months. What was never done was **measure
it**, and a status assertion cannot see a time. It is 1743x.

**The fix closes the bracket at both ends** (`core/bench/streams.py`). Streams
constructed after `install_stream_tracking()` are tracked; the timed loop fences
them from the current stream before the start event and joins them into it
before the end event. Closing only the end was tried first and only half worked
— a kernel that host-synchronizes inside `run()` has already drained its stream
by the time control returns, so the join waits on nothing, and `L1__054` still
read 1.41x fast. Both halves together are exactly what
`current_stream().wait_stream(s)` does, which is why a kernel already written
that way measures the same either way.

**Neutralize, do not flag**: a kernel that overlaps its work is a legitimate
kernel, and nothing in the source distinguishes the incidental case from the
adversarial one. Two corpus cases, `test_side_stream_host_sync_no_advantage` and
`test_side_stream_join_deferred_to_next_call`, assert **no time advantage**
rather than a status, and are mutation-checked: both fail with the join removed,
both pass with it.

**What it cost, in published numbers.** Five kernels in `agent-glm-sweep-2`
create a stream; they were re-timed on GPU 0 and nothing else was (`--retime`,
new in `agent_score.py`, because re-timing 220 problems to fix five moves 215
numbers that had nothing wrong with them).

```
L1__054   0.14395 -> 0.18948 ms  (1.316x)   mean S 1.0735 -> 0.5487
L1__055   0.05871 -> 0.07320 ms  (1.247x)   mean S 0.5651 -> 0.5064
L2__073   3.02058 -> 3.02197 ms  (1.000x)   mean S 0.9067 -> 0.9027
FIB__008  0.04800 -> 0.04784 ms  (0.997x)   mean S 0.5472 -> 0.5460
FIB__004  no timings either way (its kernel calls cpp_extension.load, refused)
```

Run mean S 0.6111 -> 0.6104. Bound violations 26 workloads / 6 problems -> 21/5.

Two things in that table matter more than the mean.

**`L1__054` was never a wrong bound.** It was carried as one of the thirteen and
was scoring a mean of **1.0735** — above the roofline, which is what a wrong
bound looks like from outside. Its bound was fine. Its *measurement* was short
by 32%. So the thirteen were not two causes plus a residue; they were **three
kinds of defect wearing one symptom**, and "a kernel beat its T_SOL" is evidence
that something is wrong, not evidence of *what*.

**`L1__055` never appeared on any list.** It was undercounted by 25%, its score
inflated by 10%, and nothing anywhere detected it — because a 25% inflation on a
kernel scoring 0.57 does not push it past 1. This is [D35](#d35)'s blindness
argument again with a different subject: the bad-bounds list finds only defects
large enough to break the one invariant we check, and this one was not.

**What it costs a legitimate kernel, and this is live residue.** The fence is
recorded outside the bracket, but the join cannot be, so a submission that
creates streams pays roughly **6–7 µs per tracked stream per iteration** inside
its own measurement. On `L1__054`'s 0.7 ms that is ~1%; on a 20 µs kernel it
would be material. It is in the safe direction — over-measuring cannot inflate a
score — but it is a real penalty on honest overlapping kernels and it is in
`TODO.md`. The durable answer is the rocprofiler methodology, which stamps each
window's end after a full synchronize and closes this by construction; but
switching the methodology of record would move every measurement ever taken,
which is not something to do to get past an obstacle. Arithmetic for the residue:
`(0.70381 − 0.68547) / (4 − 1) = 0.00611 ms`.

**Not re-timed:** `glm-run1`'s `FlashInfer-Bench__007`, the sixth stream-using
kernel found. That run is withdrawn from the board, so its numbers are not
published; recorded here so the omission is a decision.

### The shard runner could put two timing runs on one GPU {#d11}

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

**Consequence for numbers already collected:** 176 of the 235 selection-pass
artifacts were produced before the fix, so an unknown subset contain inflated
per-variant times. That affects **selection, never the anchor** —
`authoritative_tb.py` re-times on GPU 0 alone, and every T_b in the manifest
comes from that pass. To keep a mis-ordered variant from being dropped before it
gets there, the authoritative pass now re-times the top two variants *plus
anything within 25% of the fastest*, rather than the top two alone.

**The same trap in a different disguise, worth knowing before you sample any
device:** on this node torch device 1 is DRM `card9` while `card1` is torch 0,
so addressing a GPU by raw index against a `/sys` or DRM path reads a different
card. This bit the [D20](#d20) clock probe (see there) and `TODO.md` records
three further call sites that still address by raw index.

### Matmul timing spread on MI350X is bimodal, and the cause is unknown {#d20}

Two `test_matmul_timing_variance` tests sourced their thresholds to "measured
ranges on **RTX 4090 and B200**" — NVIDIA constants, forbidden by prime
directive 2. Re-derived on this part with `scripts/derive_timing_variance.py`:
120 invocations per size across GPUs 1–4, clock-locked at `perf_determinism`,
the same statistic the test computes (`max/min` over one
`time_runnable(return_mode="all")` call). Only the constant was re-derived; the
statistic was not changed. Artifact: `artifacts/02/timing-variance-amd.json`.

| size | median | p95 | % >2x | outliers cluster at | NVIDIA k | k fails |
|---|---|---|---|---|---|---|
| 64 | 1.76x | 5.73x | 14.2% | scattered, <= 9.2x | 1.25 | 100% |
| 512 | 1.49x | 2.68x | 5.8% | <= 3.9x | 1.30 | 93% |
| 2048 | 1.02x | 1.04x | 2.5% | **21.4 / 21.6 / 22.5x** | 1.15 | 2.5% |
| 4096 | 1.01x | 3.35x | 7.5% | **3.3–3.4x, nine times** | 1.15 | 7.5% |

**No constant fixes this test.** At 2048 the normal spread is 1.02x and then it
cliffs straight to 21x, with nothing between: stopping the flake needs k=25, and
a 25x threshold on a 1.02x quantity asserts nothing. So the two tests are
deferred with this evidence rather than given an invented number — the same
treatment as the NVFP4 problems, and for the same reason: re-specification, not
translation.

**Narrowed, and the leading hypothesis falsified.**
`scripts/probe_timing_stall.py` -> `artifacts/02/timing-stall-probe.json`:

* **Not cold start.** The first call of every size is the *tightest* sample
  taken — the opposite of the obvious hypothesis, checked because it was obvious.
* **Not a fixed structural offset.** Stall indices are scattered (3–9 distinct
  per cell, concentration 0.11–0.5, median position 0.45 through the call), so it
  is not an allocator pool wrap or a page boundary at a reproducible index.
* **A per-iteration hazard, not per-call.** mm[4096] stalls at 0.135% of
  iterations at rep=100 and 0.125% at rep=25 — flat in the iteration count.
* **Even across GPUs** (5/3/4/4 over GPUs 1–4), so not one bad card.
* **mm[2048] did not reproduce**: 0 stalls in 12,000 iterations, against 3 at
  ~21x in 3,600 earlier. The 21x event is real but conditional on something not
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

A 1.04x clock spread cannot produce a 3.9–4.5x stall. **Hypothesis rejected.**

Two things that alignment step was load-bearing for. Without it the raw
histogram showed the clock sweeping 107 -> 1598 MHz and looked like a
confirmation; that spread is entirely process startup and the idle gaps
*between* calls. And the DRM card had to be resolved by PCI bus — on this node
torch 1 is `card9` while `card1` is torch 0, so sampling `card{gpu}` would have
read an idle GPU, produced a flat trace, and "falsified" the hypothesis for the
wrong reason. Same trap as [D11](#d11) and the task-01 floor that was fiction.

**Useful side result:** during actual measurement the clock is steady to 1.00x
at ~1450 MHz. That is reassurance for T_b, and it also shows F_LOCK = 1300 is
the *sustained-load* figure — short bursty runs on an idle node sit ~150 MHz
higher, which is why T_b and submissions must be timed the same way.

**Still open:** what costs a steady-clock kernel 3.9–4.5x at 0.13% of
iterations. Kernel selection inside hipBLASLt is the remaining suspect and has
not been tested. Two upstream tests remain skipped behind this.

### One problem's T_b does not reproduce to 3% {#d15}

The anchor check re-times `T_b`'s own implementation and requires it to score
0.5 ± 0.03. Over a 20-problem sample, **336 of 349 workloads pass**, no measured
time falls below its `T_SOL`, and the reference never scores above the anchor.
Of the 13 that fail, **12 are one problem** —
`FlashInfer-Bench/018_mla_paged_decode` — where the re-timed latency comes back
a median of **1.16x** the recorded `T_b`. Two independent runs of the check
reproduced 13 and 12 failures on it, so the effect is stable and it is the
problem, not the check.

That problem's `T_b` is therefore optimistic by roughly 16%, which makes every
score on it *lower* than it should be — the conservative direction, but wrong.
The cause is not established: MLA paged decode is the most input-layout-
dependent kernel in the set and loads its inputs from safetensors blobs, so
allocation and page-table state are the obvious suspects. It is very likely the
same family effect as [D18](#d18). It remains a WARN on
`verify_artifacts.py --task 06` (336/349). The remaining single failure
(`L1/072`) is one workload at the tolerance edge.

**Now a special case of a larger question.** [D59](#d59) found that a subset of
published T_b anchors does not reproduce at all, on problems unrelated to paged
attention. D15 is the first instance of that class to be noticed, not an isolate.

### A T_b re-sweep, and what it does and does not establish {#d59}

The 235-problem `tb-candidates` re-sweep under the [D56](#d56) recompile fix is
in `artifacts/12/tb-recompile-fix/candidates`, compared against the shipped
`artifacts/06/candidates` by `scripts/compare_candidates.py`.

**Complete: `done: 233 ok, 2 failed, 239.9 min` on GPUs 1–7.**
`check_coverage.py` reports 235/235 across all four categories — note that it
counts a recorded timeout as covered, which is correct (a recorded failure is a
result) but is not the same as 235 measured.

The two failures are both `timeout after 5400s`, and both are a **real cost of
the fix** rather than flakes: `L1__094_time_decay_exponential_stabilization` and
`FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`. Before the
fix `L1__094` recorded `v2_compile 9/16` and `v3_compile_max_autotune ok=False`
— it was fast precisely because it stopped compiling after the 8th shape.
Compiling all 16 under max-autotune does not fit in 90 minutes. Whatever budget
the next full sweep uses, these two need a bigger one; they are not evidence
against the fix.

**What is established, and is the point: the compile variants lose a sixth of
their passes.**

```
                          before   after            (over all 235 problems)
v1_eager                    3661 -> 3671   (+10)
v2_compile                  3132 -> 2633  (-499)   15.9% of its passes were false
v3_compile_max_autotune     3000 -> 2474  (-526)   17.5%
v4_contiguous               3636 -> 3651   (+15)
v5_compile_contiguous          0 ->    0
```

67 problems lose v2 passes and **none gains any**; 75 lose v3 passes and two
gain. The per-problem signature is the cliff, unmistakably: 13->1, 10->0, 8->0,
8->0, 8->0 … A pass count that only ever falls, and falls to the shape boundary,
is not noise. **523 was a floor, as [D50](#d50) said. The floor is now measured,
and it is ~1025 more failures across the two variants.** Correctness does not
depend on node load, so this comparison is sound even though the sweep ran
7-wide.

**What is NOT established: anything about T_b.** The winner moved on 2025 of
3633 workloads and the T_b ratio came out p05 0.346 / p50 0.995 / p95 1.062,
which would be a large and interesting two-sided result — except that the
baseline does not reproduce. `L2__057` is the check that settled it: its T_b
looked 2.6–3.2x worse after the fix, so every variant was compared, and **all**
of them moved, including `v1_eager`, which the fix cannot touch. The comparison
that matters is against `artifacts/06/`**`authoritative`** — the serial GPU-0
re-time that IS the published anchor — not against `candidates`, a selection
sweep:

```
workload (largest 2)    authoritative   candidates   GPU 0 SOLO, today
ea5ae433                     31.3462      32.6514             88.0202
7b2424f5                     17.4742      18.2696             54.6464
over all 16 workloads:  today / authoritative  p50 2.021, min 1.071, max 3.892
```

Authoritative and candidates agree to 4%, so the sharded sweep was not the
problem.

> **RETRACTED by [D61](#d61).** Both "solo GPU 0" re-times above were taken
> **after** two foreign tenants started on this node. The 2.021x is not a
> reproduction gap and must not be quoted as one. What survives is stated in
> D61: the 17:55 pre-tenant sweep value of **76.83 ms against a published
> 31.35 = 2.45x**, still unexplained, and the internal old-SHA/new-SHA
> comparison. Every pass/fail number above is unaffected — correctness does not
> depend on what else is on the card.

**How far it spreads, measured rather than left as a worry.** `v1_eager`, same
workload uuid, `artifacts/06/candidates` against the re-sweep, 3623 workloads:

```
p01 0.315   p05 0.809   p25 0.986   p50 0.998   p75 1.007   p95 1.072   p99 1.187   max 3.00
slower today: >1.10x  123 (3.39%)   >1.25x 18   >1.50x 11   >2.00x  6
faster today: <0.90x  310 (8.56%)   <0.80x 171  <0.667x 119 <0.50x  86 (2.37%)
```

The bulk reproduces — half the workloads inside 1.4%. The tails do not, and they
are not symmetric: **8.56% are more than 10% faster today and 2.37% are more
than twice as fast**, against 3.39% and 0.17% the other way. The re-sweep ran
7-wide and the baseline largely ran solo, which biases toward *slower* today, so
the faster tail cannot be explained by load and is the direction that should
worry us — an anchor recorded too slow makes every score against it too
generous. Worst movers by problem median: `L1__085` **0.097x** (a tenfold
discrepancy on geglu activation), `L2__059` 0.308x, `L1__079` 0.451x, `L1__050`
0.473x, and `L2__057` 1.737x in the other direction.

**This latency population is partly contaminated and must be recut before it is
quoted.** Per [D61](#d61): 235 files, sweep 16:14:25–20:13:19, **90 written at
or after 17:55** and **41 at or after 18:39**.

**Two re-sweep variant failures that are not results**, recorded here because
they belong with this sweep: `Quant__023` OOM'd at 18:40 — the minute the 194 GB
tenant landed, and on **physical GPU 1**, not GPU 0 (the HIP message's "GPU 0"
is the renumbered visible device); and `L2__012` died with a memory-access fault
`rc=-6` at 17:06, **predating** the tenants and genuinely undiagnosed.

**Also observed, and it is [D51](#d51)'s own example turning up unprompted:**
`v1_eager` on `L2__051` passed **6 of 16** in `artifacts/06` and **16 of 16**
here. D51 reported the mirror image — eager re-run in a fresh process failing
10 of 16 on that exact problem. Eager is not deterministic across processes on
`L2__051`, so a tolerance derived from one in-process pair does not bound it.
That is the tolerance-derivation defect, seen from the eager side.

**Not done:** the authoritative T_b pass is a serial GPU-0 re-time
(`scripts/authoritative_tb.py`) and it is the only thing that can re-select T_b.
Running it before `L2__057` is explained would launder an unexplained number
into the manifest.

### What the T_b non-reproduction is NOT {#d60}

Chased D59's discrepancy far enough to eliminate the obvious answers. Recorded
so the next session does not re-run these.

**The environment is identical.** `_provenance` on the 08-04 authoritative
artifact and on the re-time agree on every field that could matter: torch
`2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, ROCm 7.2.0, driver
7.1.1.31500000, amd_smi 26.2.1, host `gbt350-odcdh1-a08-1`, `visible_devices 0`,
`f_lock_mhz 1300`, and the same `artifacts/05/workloads/.../workload.jsonl`. The
only difference is the repo: `ea94b186` then, `dd88de94` now.

**It is not sibling load.** The re-time was solo on GPU 0 and came out SLOWER
(88.02 ms) than the 7-wide re-sweep (76.83), against 31.35 published.

**It is not the [D38](#d38) side-stream timing fix (`05cb6e3e`, 2026-08-10),
which was the obvious suspect and is the reason to write this down.** Two
independent checks kill it: no reference in the dataset uses a side stream at
all — grepping all 235 `definition.json` references for `Stream` /
`current_stream` / `wait_stream` returns **0** — and `_fence_streams` /
`_join_streams` are documented and written as no-ops when nothing is tracked.
`L2__057` in particular creates no stream.

**It is not the clock.** [D55](#d55) measured that the policy does not move the
clock on this part, and both measurements carry the same F_LOCK.

**The decisive experiment was run, and it exonerates the harness.** `ea94b186`
checked out into a worktree and run under an identical container (same image,
same devices, same scratch, the worktree's own `artifacts/05/workloads`, the
same `data/` bind-mounted read-only), `v1_eager`, on GPU 0:

```
workload (largest 3)        published    OLD sha, today    NEW sha, today
ea5ae433                      31.3462           88.3389           88.0202
7b2424f5                      17.4742           54.8049           54.6464
2a151224                       4.4462            8.9160            9.0110
over all 16:  OLD/published p50 2.021 (1.072-3.674)   NEW/published p50 2.021 (1.071-3.892)
```

The two SHAs agree. Precisely: the **ratios** against the published anchor agree
to three decimals (2.021 and 2.021); the raw times do not and should not be
described that way — 88.0202 ms and 88.3389 ms differ by 0.36%, ordinary
run-to-run spread on a contended card. So nothing committed between 08-04 and
08-12 caused this, D38 included, **and the harness is not implicated at all.**

> **RETRACTED in part by [D61](#d61).** This entry's headline conclusion — that
> a published measurement is not reproducible from everything the artifact
> records about it, and that provenance is therefore insufficient — **is not
> established**, because both re-times were taken on a card two foreign tenants
> were using. What survives is exactly the old-SHA/new-SHA exoneration above:
> both arms were taken 28 minutes apart under the same co-tenant load, so their
> agreement is internally valid. The claim that the environment was identical
> does not survive: something outside `provenance.stamp()` *was* different, and
> it was the tenants.

**One observation for whoever picks this up, recorded because it is suggestive
and NOT because it is tested.** The caches the container carries across runs
live under `SOLEXBENCH_SCRATCH/home` and they are large and old:

```
/var/tmp/solbench/home/.triton        4.1G
/var/tmp/solbench/home/.cache/comgr   3.7G
/var/tmp/solbench/home/.cache/miopen  1.4M    dir mtime 2026-08-04 18:43
/var/tmp/solbench/home/.cache         dir mtime 2026-08-04 18:43
```

Nothing has written to `.cache` since **2026-08-04**, the day the anchors were
taken — including the three re-times of `L2__057`, which would have added
entries had they missed and then tuned. MIOpen is the conv library and `L2__057`
is a coupling-flow block. That is a coincidence of dates and subject matter, not
evidence: nobody checked whether the reference uses convolutions, whether MIOpen
is consulted at all on this path, or what `MIOPEN_USER_DB_PATH` resolves to
inside the container. **Do not repeat the D38 mistake and treat a plausible
mechanism as the answer before testing it.**

The experiments, in order: (1) move the scratch caches aside and re-time
`L2__057` cold, then restore and re-time — if the gap moves, the cause is a
cache no provenance records; (2) re-time `L1__085_geglu_activation`, the 0.097x
extreme, which moves the OTHER way and so is the better discriminator; (3) only
then measure how many of the 3717 anchors move, because a population sweep
before the mechanism is understood costs a day and answers nothing.

### CORRECTION: the re-measurements were not on an exclusive card {#d61}

**Retracting the claim that a published anchor does not reproduce.** It may
still be true. It is not shown, and the reason is a discipline failure.

At **18:39 and 18:40 on 2026-08-12** two foreign tenants started on this node:
`sglang::scheduler` (pid 2617421, 194 GB resident) and
`ray::MegatronTrainRayActor` (pid 2638981). Both were still running at 21:00,
with GPUs 0–3 reading 100% utilisation. Both "solo GPU 0" re-times of `L2__057`
were taken after that:

```
artifacts/06 authoritative      2026-08-04T06:38:30Z      31.3462 ms
re-sweep (7-wide, mine only)    2026-08-12T17:55:03Z      76.8261 ms   before the tenants
"solo" GPU 0, new sha           2026-08-12T19:32:44Z      88.0202 ms   CONTAMINATED
"solo" GPU 0, old sha ea94b186  2026-08-12T20:00:42Z      88.3389 ms   CONTAMINATED
```

`scripts/gpu_exclusive.py --gpu 0` run afterwards reports plainly:
`gpu_id 36538 is NOT exclusive -- 2 foreign process(es)`.

**What survives.** The old-SHA/new-SHA comparison is still internally valid:
both were taken 28 minutes apart under the same co-tenant load and their ratios
against the published anchor agree to three decimals (2.021 against 2.021, over
raw times of 88.3389 and 88.0202 ms that themselves differ by 0.36%), so the
harness change between `ea94b186` and `dd88de94` is still exonerated. And the
**17:55 sweep value predates the tenants**, so the gap against the published
anchor is not purely their doing — **76.83 against 31.35 is still 2.45x**, under
a 7-wide sweep load of our own. Task 01's −0.11% sibling figure was measured on
a 0.146 ms kernel and does not license extending it to a 31 ms memory-heavy one
contending for HBM and fabric.

**What does not survive:** the 2.021x as a reproduction gap, the claim that the
environment was identical, and therefore [D60](#d60)'s conclusion that
provenance is insufficient to reproduce a number. All three need a clean re-run
on an exclusive card.

**Why it happened, because the mechanism matters more than the apology.**
`CLAUDE.md` §4 says timing runs and exploration must not share a GPU, and the
repo already has the guards: `scripts/gpu_exclusive.py`,
`guard_authoritative_gpu.py`, and `retime_parallel.py`'s `foreign_on()` check on
every measurement. `time_tb_candidates.py` was invoked directly, which runs none
of them, and "solo on an idle GPU 0" was asserted from having seen the card idle
earlier rather than from checking at the time. **The guard exists exactly so
that a shared card is a refusal instead of a number, and it was routed around.**

**Unaffected: every pass/fail number in [D56](#d56) and [D59](#d59).**
Correctness does not depend on what else is on the card, and the −499/−526
compile result stands. The owed work is in `TODO.md`.

### `agent_score.py --timeout` never reached the evaluation {#d33}

The flag set the *subprocess* timeout for the `env/solb` call and was never
passed to `agent_eval.py`, whose own `--timeout` defaults to **1200 s**. So
every re-time was capped at twenty minutes regardless, and raising the outer
value — which is what anyone would try — bought nothing at all.

Found through `FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`,
whose re-time recorded `TimeoutExpired ... after 1200 seconds` with 0 workloads.
On the board that is indistinguishable from a kernel that produced nothing. With
the timeout actually forwarded it re-times at **30/30 passed**. The kernel was
never the problem; the scorer was not given enough time and said so in a place
that read like the kernel's fault.

The inner cap is now `outer − 120 s`, deliberately the smaller of the two:
whichever fires first decides what the artifact says, and only the inner one
writes a real result document — the outer can just kill the process and leave
`retime()` synthesising "produced no artifact".

Any problem whose evaluation legitimately needs more than twenty minutes was
unscoreable before this, silently. `FI__014` is the only one found so far, but
it was found by accident. The budget question is still open — see [D23](#d23),
and the two 5400 s timeouts in [D59](#d59)'s re-sweep.

---

# 4. Tolerances, goldens and correctness

The context for this whole section: **all five of these findings came out of one
question — "why does the board show torch.compile failing ~70 problems?"** The
answer is that some of those failures are not the compiler's, some are not
failures, and the instrument used to judge them is narrower than its name
suggests. Five problems were reproduced on GPUs 4–7 (numerics only, no timing;
GPUs 0–3 belonged to other tenants), each root-caused and then attacked by three
adversarial verifiers on a different lens. Artifacts, scripts and the full
report are in `artifacts/11/compile-diag/`. Ground truth throughout is
`artifacts/06/candidates`: **523 `INCORRECT_NUMERICAL` for `v2_compile` over 71
problems** (L1 24, L2 41, Quant 6) and 571 over 80 for v3. **Zero trace errors** —
these are not compile errors. `v5_compile_contiguous` is the one that
`RUNTIME_ERROR`s, and it is already off the board.

### torch.compile silently stopped compiling after the 8th shape {#d50}

`torch._dynamo.config.recompile_limit` is **8**, and `variants.py` builds **one
module-level compiled callable per problem**, reused across every workload.
After the 8th distinct shape dynamo logs the limit and **silently runs the frame
eagerly**. **225 of 235 problems have >= 9 distinct shapes, so 2061 of 3957
workloads were never compiled at all.**

The signature is unmistakable on `L2__009`: indices 0–7 FAIL, 8–15 "pass". Re-run
index 8 alone in a fresh process and it fails outright (mr = 0.428, verified by
hand, not only by an agent).

Two consequences:

* **The 523 is a floor, not a count.** [D59](#d59) later measured the floor:
  −499 on v2 and −526 on v3.
* **626 of 3717 published anchors in `artifacts/06/authoritative` are labelled
  v2/v3 while carrying an eager latency.** On the 538 post-cliff workloads with
  a recorded compile time, `T_b/compile` is p50 **1.000**, max 1.163 — they ARE
  eager times.

**This reaches the board.** 1642 of 3717 scoreable workloads name a compile
variant (823 v3, 819 v2), so **626 rows serve a formulation that did not run on
them.** Scope, because the obvious check misleads: `t_b_variant` renders in
`problem.html:251` and nowhere else in the templates, and it sits in the
`c-deriv` band which is hidden by a switch that is **off by default**
(`problem.js:80`). The markup ships on every problem page anyway and `/api/v1`
hides nothing (`models.py:191,305`) — so the board asserts it, but a reader does
not see it unless they opt in. **Anyone spot-checking the page and not finding
the column should not conclude this is stale.**

**The fix belongs where the anchor is stamped, in the manifest, NOT in the
template**: blanking the column would be the board covering for the manifest.
This is a plain bug, not a methodology question, and fixing it moves the failure
count **up**. Estimated at ~5.5 h on GPU 0, and `TODO.md` records that it must
be done first and alone.

### The recompile cliff is fixed, and the failure count goes up {#d56}

**2026-08-12.** `reference/tb-candidates/variants.py` now sets
`recompile_limit = 256`, `accumulated_recompile_limit = 4096` and — the part
that matters more — `fail_on_recompile_limit_hit = True`. A variant that cannot
compile is a result the runner already records; a variant that quietly **stops**
compiling and keeps reporting times is not detectable downstream by anything,
and that is what the default 8 was doing to 2061 of 3957 workloads.

Verified by hand before trusting any sweep. `L2__009` under `v2_compile`:

```
before   8 passed / 16      (indices 0-7 fail, 8-15 "pass")
after    0 passed / 16      (all 16 INCORRECT_NUMERICAL)
```

The eight "passes" were eager runs wearing a compile label. Over the first 14
problems of the re-sweep, v2 loses 24 passes and v3 loses 28, and every problem
that moved moved **at the 8-shape boundary** (13->1, 8->0, 10->6, 8->0). The
full 235-problem re-sweep result is [D59](#d59). No manifest, bound or tolerance
was touched, and no T_b has been re-selected.

### The tolerance is a bit-identity-with-eager test {#d51}

`calibrate_tolerance.py` derives atol/rtol from the spread of the **same
reference run twice in one process**, floored at dtype epsilon. That spread is
exactly **0.000e+00 on 3581 of 3717 workloads (96.3%)**, so **3502 of 3717
tolerances are pure floor** — about one ULP — and the gate then demands 99% of
*elements* inside it. So torch.compile is rejected for being *different*, not
for being *wrong*.

Measured divergence against that: `L2__009` 5.8e-06 against atol 1.3e-07
(mr 0.415), `L1__067` 2.0e-06 vs 3.4e-08 (mr 0.249), `Quant__004` 2.0e-01 vs
4.9e-03 (mr 0.734), `L1__062` 1.3e-01 vs 8.3e-03 (mr 0.977), `L2__058` 2.3e-02
vs 4.7e-03 (mr 0.985).

Four distinct Inductor behaviours produce it, **each proved causally rather than
argued**: reduction re-association; FMA contraction; elided intermediate dtype
rounding (`emulate_precision_casts=True` restores **bit-identity** on `L1__062`,
`L2__058` and `Quant__004` — the cleanest control in the set); and, under
max-autotune only, a hipBLASLt -> Triton GEMM template swap.

**Surfacing the uncertainty, per prime directive 8: on the cases where anyone
adjudicated it against a float64 golden, eager is NOT the more accurate
implementation.** `L1__062` compiled is bit-identical to the correctly-rounded
bf16 golden and eager is not; `L2__058` compiled RMS 2.11e-03 against eager
2.78e-03; `Quant__004` compiled 0.923x eager. `L2__009` and `L1__067` are ties;
the one control that went the other way, `L1__074`, did so by 0.009%. Worse, the
reference misses its own gate against truth where checked — `L1__067` eager
scores mr = 0.0899, `L2__058` eager 0.9869 — and with no golden involved at all,
`v1_eager` re-run in a fresh process **fails 10 of 16 workloads of `L2__051`**
under the tolerance task 05 derived for `L2__051`.

**On the measured cases the benchmark is grading agreement-with-eager, not
correctness.** Bounded explicitly: this is **6 problems and ~10 workloads of
523**. It is a direction, not a distribution, and no more is claimed. (Note that
the goldens those adjudications used are themselves invalid — see
[D53](#d53) — which is a reason to redo the adjudication, not a reason the
mechanism is wrong.)

**Score exposure, measured.** T_b is the fastest variant passing **every**
workload of a problem, so 70 problems lost both compile anchors and **1115 of
3717 workloads (30.0%)** fall back to eager-class PyTorch. On the **39** workloads
that were genuinely compiled AND passed under a problem-wide-disqualified
variant, the published anchor is **p50 2.02x** and up to 6.29x slower than the
compiled time that was discarded; an agent tying T_b there scores 0.5 where it
should score p50 0.312. Re-scoring the real submissions (moving only T_b) drops
mean S by **0.026–0.058**. For scale, the 8-wide re-time debate turned on ±0.001.
**The other 1076 are an extrapolation, not a measurement**: compile never both
compiled and passed there.

**Not measured, and it matters:** `artifacts/06` records a latency only for
workloads that PASSED (`time_tb_candidates.py:121-123`), so there is no compiled
time for a numerically-rejected workload. Any statement about what those would
have cost is inference.

This is a live item with a maintainer decision attached, in `TODO.md`.

### A zero tolerance leaked from an integer output onto float outputs {#d52}

`_dtype_floor` read `torch.finfo(tensors[0].dtype)` under
`except TypeError: return {"atol": 0.0, "rtol": 0.0}`, and `tensors[0]` is the
problem's **first** output. A problem whose first output is an index tensor
therefore got a **zero floor for its float outputs too**, and against a bit-exact
reference (`max_abs == 0`), `max(max_abs * 1.25, 0.0)` is exactly zero — a
bit-identity-with-eager gate. **The reverse leak is also real:** with a float
output first no `TypeError` fires at all, and the integer output's magnitudes
are then summed into the RMS scale, widening the float atol by whatever an index
happens to be worth.

**Scope, read from `artifacts/05` and NOT re-measured:** 76 workloads carry
`max_atol 0.0, max_rtol 0.0` with `_derivation "... floored at torch.int64
epsilon"`, over five problems. Only **two** of the five mix dtypes, and only
those two are unpassable: `L2__049` and `Quant__011`, both (int64 `topk_idx`,
float32 `topk_weight`), 16 workloads each = **32 are the defect**. The other 44
(`L1__058` 16, `L1__028` 12, `L2__006` 16) are **all-integer** problems where a
zero band is exact equality and is correct.

`L2__049`'s `topk_idx` is bit-identical (no routing flip) and its `topk_weight`
is off by exactly **one fp32 ULP on 4488 of 16384 elements**, mr 0.726, and a
search to k = 2²⁰ finds no multiplier that passes. **Unpassable by
construction.**

**Fixed in code, 2026-08-12, CPU-only.** The data model cannot express a
per-output tolerance — `Workload.tolerance` is one `ToleranceSpec` and
`eval_driver.py` applies it to every output — so the split lives in two places.
`_dtype_floor` now derives the band from the **floating-point outputs only**,
records which outputs are held exact, and reports integer run-to-run difference
separately so a non-deterministic index is still visible and still cannot widen
a float band. `compute_error_stats`
(`src/sol_execbench/core/bench/correctness.py`) now compares integer and boolean
outputs for **exact equality whatever the spec says**, in the output's own dtype
rather than through the float32 cast (2⁵³ and 2⁵³+1 are the same float32), after
`check_tensor_sanity` so the all-zeros anti-hack guard still runs first.
`required_matched_ratio` and `max_error_cap` are unchanged: only the width of
the band moved.

Tests: `tests/scripts/test_calibrate_tolerance.py` (new, 10 cases) and
`tests/sol_execbench/core/bench/test_correctness.py::TestIntegerOutputsAreComparedExactly`
(new, 7 cases). Against the pre-fix tree (`git archive HEAD` into a scratch
dir) 8 of the 17 fail; against the fixed tree all 17 pass.

**Blast radius of the harness half, checked rather than assumed:** seven
problems emit an integer or boolean output. Three are all-integer, two are the
mixed pair, and the remaining two — `Quant__027` and `Quant__033` — are NVFP4,
already in `artifacts/deferred.json`, with `ok: false` and no tolerance at all
(`scaled_gemm ... only supported for CUDA 12.8`). All five non-deferred ones
ship `max_atol 0.0`, and against a zero band `abs_error > 0` is already exact
equality, so the exactness rule **changes no verdict on the board today**; it
changes what happens once those tolerances are re-derived non-zero. The one case
where it would differ today is an integer above 2²⁴, where the float32 cast
collides — the largest values these five emit are expert indices, token counts
and position ids, none close to it.

**Not done, and it is what stands between this and the board:** `artifacts/05`
for the five problems is stale, so `artifacts/05/workloads/` still ships
`max_atol 0.0` and no published number has moved. Closing it needs
`calibrate_tolerance.py` re-run for those five on a GPU (10 seeds × 2 executions
× 76 workloads), then `apply_tolerances.py`, then task 06 re-run for `L2__049`
and `Quant__011` (their T_b may change: a compile variant rejected for one ULP
can now win), then a manifest rebuild. Until then the five problems' tolerances
are known-wrong and should not be read as results.

> **CORRECTION, attached rather than merged.** The original entry said the three
> all-integer problems "pass every compile variant". They do not. Recomputed
> from `artifacts/06/candidates/*.json`:
> ```
> L1__058  v1 16/16  v2 16/16  v3 16/16  v4 16/16  v5 0/16
> L1__028  v1 12/12  v2 12/12  v3 12/12  v4 12/12  v5 0/12
> L2__006  v1 16/16  v2 16/16  v3 16/16  v4 16/16  v5 0/16
> ```
> `v5_compile_contiguous` fails **every** workload of all three, and also 16/16
> on `L2__049` and `Quant__011`. **The inference the sentence was supporting
> survives**, because all of those v5 failures are `RUNTIME_ERROR` (a
> torch-dynamo `_wrap_fx_proxy` traceback) and not `INCORRECT_NUMERICAL`: the
> only `INCORRECT_NUMERICAL` failures among the five problems are `L2__049`
> (8 of 16 under v2_compile, the same 8 under v3) and `Quant__011` (3 of the 3
> workloads each of those variants reached) — **11 distinct workloads**, which
> is the ">= 11" of the original entry.

### The same leak between two float dtypes, 65536x wide {#d52b}

Found by the review of the [D52](#d52) fix, which closed the int/float leak and
left the float/float one open in the same function. `_dtype_floor` filtered the
integer outputs out and then read `tensors[0].dtype`, and summed the RMS scale
over every remaining output regardless of dtype.

**Scope, counted this session and not taken from any report.** Of 235
definitions, **17 declare more than one float output dtype**. Every one of the
17 is exactly `{bfloat16, float32}`, and in every one a **bfloat16 output comes
first**. `Quant__033` is deferred, so 16 are scoreable and they carry **396
workloads** (412 minus Quant__033's 16). The eight FlashInfer paged/ragged
attention problems (012–019, output bf16 + lse fp32) are 285 of the 396; the
rest are backward problems whose weight gradients are fp32 next to bf16
activations.

**Consequence, read from `artifacts/05` and NOT re-measured:** those problems'
fp32 outputs shipped **bf16's epsilon**. 0.0078125 against float32's 1.1920929e-07
is **65536x**, and with a bit-exact reference the floor IS the whole band —
`FlashInfer-Bench__012` atol 0.004542 / rtol 0.0078125, `L1__013` atol 0.010156,
`L1__051` atol 0.304172, `L2__044` atol 0.074598, each with `run_to_run.max_abs
0.0` and `_derivation` ending "floored at torch.bfloat16 epsilon". The RMS sum
was the same leak again in the scale term.

**Fixed:** `_dtype_floor` now groups the float outputs **by dtype** and derives
`{eps, RMS}` per group; nothing crosses a group. `_rms` is factored out for that
(same chunked `torch.where`, same [D13](#d13) `masked_select` avoidance).

**What could NOT be fixed, with the evidence.** The data model cannot hold a
per-output floor: `Workload.tolerance` is a single `ToleranceSpec`
(`src/sol_execbench/core/data/workload.py:117`) and `eval_driver` applies it to
every output. So the per-dtype floors are collapsed to one band, and **the
collapse is the MAX** — the min would hold a bf16 output to fp32's epsilon,
which is exactly the unpassable-by-construction failure D52 exists to remove. Be
plain about what that buys: since bf16 is the looser epsilon and is present in
all 17, the **applied rtol is unchanged, so the 65536x over-grant on their fp32
outputs survives.** What the fix removes is the cross-dtype RMS in atol and the
silence: `_dtype_floors` and `_floor_over_grant` are now recorded in the
tolerance, each output carries its `own_floor` in `entry["outputs"]`, and
`_derivation` names both floors and the collapse factor. The remainder is a
bound-quality figure of [D39](#d39)'s kind, recorded rather than invisible.
**Closing it properly is a schema change**, not improvised here (prime
directive 7).

**Also carried through:** the third comparison loop. `vs_golden` still maxed
over integer outputs, so an index difference and a weight difference shared one
number — the conflation D52 exists to remove, left in the one loop the fix did
not touch. Split, with `exact_outputs_max_abs` alongside, matching `run_to_run`.

**Also, downstream:** `apply_tolerances.py` rebuilt `tolerance` from a fixed key
list, so `_exact_outputs` (and now `_dtype_floors`, `_floor_over_grant`) never
reached `artifacts/05/workloads`. **Decision: carry them.** They are not
`ToleranceSpec` fields and pydantic's default `extra="ignore"` drops them at
load — verified in the container, `ToleranceSpec(**{..., "_exact_outputs": [0]})`
parses and `model_dump()` returns only the five declared fields — so carrying
them cannot change what the harness enforces, exactly as `_provenance` has
always been carried. Its nondeterminism triage table also printed only the float
`max_abs`, so a reference non-deterministic purely in its indices appeared as
`deterministic: False` with `max_abs 0.0`; the table now has a second column, and
prints "-" for an artifact that predates the key rather than a 0.0 nobody
measured.

Tests: `test_calibrate_tolerance.py` grew to 20 cases;
`tests/scripts/test_apply_tolerances.py` is new, 5 cases, running the real
script over a `tmp_path` tree with every output path redirected — it writes
nothing into `artifacts/` or `reference/`. Load-bearing check, against the
pre-D52b `_dtype_floor` restored in a scratch copy: **10 failed, 10 passed**.
Two of the ten fail on the *number* rather than a missing key: with bf16 outputs
at 1024 and fp32 outputs at 1, the old cross-dtype RMS gives atol **5.656857**
where the bf16 group's own floor is 8.0; with the magnitudes swapped it gives
the same 5.656857 where bf16's own floor is 0.0078125.

**Not done:** no artifact changed. The **396 workloads on disk still carry the
cross-dtype floor**, and the fix reaches nothing published until
`calibrate_tolerance.py` re-runs for those 16 problems on a GPU, then
`apply_tolerances.py`, then a manifest rebuild. The atol may move in **either
direction** — same epsilon, differently grouped RMS — so nothing is claimed
about which scores would change.

### The float64 goldens were drawn from a different RNG, and nothing noticed {#d53}

`gen_golden.py` drew its inputs at `device="cpu"` while
`calibrate_tolerance.py` drew at the `prepare_inputs` default `"cuda:0"`. Both
call `torch.manual_seed(0)` first, so the code reads as if it reproduces the
same inputs; it does not, because **CPU runs `at::mt19937` and CUDA/HIP runs
`Philox4_32_10`**. `L1__067` seed-0 CPU against seed-0 CUDA `hidden_states`
differ by **7.096e+00**. The golden was the right answer to a question nobody
asked, and the comparison was written to `entry["vs_golden"]` and never read —
**the check that would have caught [D51](#d51) was computed, recorded, and never
looked at.**

**Re-derived independently, from `artifacts/05/*.json` only** (235 files, no
`.pt` opened): 3957 workload entries, 2331 with `vs_golden`
(`{float64: 2136, native_cpu: 195}`); `max_abs > tolerance.max_atol` on
**2302 = 98.756% of vs_golden, across 164 problems**. Matches the prior
session's **2302 of 2331 (98.8%)** to the digit.

**Fixed in code, 2026-08-12, CPU-only** (three source files plus tests):

* `scripts/runners/_common.py` now owns the whole contract: `INPUT_DEVICE =
  "cuda:0"`, `GOLDEN_SEED = 0`, `GOLDEN_CONTRACT_VERSION = 2`,
  `golden_contract_stamp()`, `golden_stamp_matches()`. `prepare_inputs` defaults
  to `INPUT_DEVICE`.
* `scripts/gen_golden.py` draws at `--input-device` (default `INPUT_DEVICE`),
  then **moves** the inputs to the CPU — a copy, not a re-randomisation — and
  computes in float64 there. So **`gen_golden.py` now requires a GPU** for its
  default invocation. Every golden gets a `<key>.meta.json` sidecar carrying the
  stamp, written **after** the `.pt`; the `.pt` itself is written-then-renamed
  (`os.replace`), matching `_common.write_result`, so a worker killed mid-save
  cannot leave a truncated golden under a valid sidecar.
* `scripts/runners/calibrate_tolerance.py` applies **the same predicate as the
  writer** — `golden_stamp_matches`, i.e. version AND device AND seed — instead
  of its own one-field check on `input_device`. It also requires the `.pt` to be
  loaded before calling a golden comparable, so a sidecar whose `.pt` was
  deleted reads as `golden_available:false` / `golden_comparable:false` **with a
  note**, rather than `comparable:true` with nothing under it. Recorded, not
  acted on: no tolerance derivation changed.

Three further defects closed in the same pass:

* `reference_draws_rng` watched `torch.random.get_rng_state()`, the **CPU
  generator alone**, so a reference doing `torch.randn(..., device="cuda")`
  inside `run()` was stamped false — the exact misreading the flag exists to
  prevent. `_rng_fingerprint` now includes the device generators when CUDA/HIP
  is initialised, and a fingerprint that **gains** a generator counts as a draw
  (a first device draw initialises it).
* `artifacts/golden/_report.json` is a flat problem->report mapping again. The
  fix had added a top-level `"_contract"` key, which makes `len()` and key
  iteration see 236 entries for 235 problems; the contract is unchanged
  information because `gen_one` already stamps it into every per-problem report.
* `--jobs` is back to its pre-fix default of **32**. The fix had dropped it to
  1, justified by an unmeasured assertion about HIP context footprint, against a
  hazard (GPU 0) that `HIP_VISIBLE_DEVICES` pinning handles, and it would have
  serialised the float64-on-CPU execution that is the expensive half and needs
  no GPU at all. No GPU was available to measure the footprint, so the default
  is the status quo ante and **the measurement is recorded as owed** in
  `TODO.md`, with its procedure.

Tests: `tests/scripts/test_golden_input_contract.py` grew from 13 cases to 27.
The headline test used to read the default it was meant to pin out of the module
under test (`gen_golden.INPUT_DEVICE IS _common.INPUT_DEVICE` by import), so it
**compared an object to itself**. It now builds the parser — `main()`'s argparse
is factored into `build_parser()` for this — and asserts
`get_default("input_device") == "cuda:0"`, a literal. Mutation-checked both
halves: argparse default -> `"cpu"` gives 1 failed / 60 passed; reader predicate
-> `input_device` only gives 3 failed / 58 passed. The second was 0 failures
before this pass, because the reader tests called the shared predicate directly
rather than the function the artifact is written from; `golden_comparability` is
now module-level in `calibrate_tolerance.py` so the test reaches the real code
path.

Verified end-to-end, CPU-only (`--input-device cpu`, so no GPU): 94 L1 problems,
4 workers, element cap 200000 -> 18 problems with goldens, 18 `.pt` and 18
sidecars written, 0 `*.tmp` left behind, `_report.json` has 94 keys and none
beginning with `_`. Re-run: 18 cached, 76 empty (the cap).

**Not done:** the **165 `.pt` files under `artifacts/golden` (2331 workload
goldens, 143 GB by `du -sh`) were all drawn on the CPU and are meaningless.**
`ls artifacts/golden/*.meta.json | wc -l = 0` — none carries a stamp, so
`is_cached()` is false for every one and a regeneration run will redraw rather
than reuse. **Every existing golden stays invalid until regenerated on a GPU**,
and until then every `vs_golden` number in `artifacts/05` is a comparison
against a different input draw.

### The tolerance runner's memory profile, and one absurd allocation {#d9}

Twenty-seven workloads across five problems failed calibration with HIP OOM, and
the two causes are unrelated.

*Retention.* Relative error can only be measured once `atol` is known, and
`atol` is only known after the last seed, so the first implementation kept every
seed's outputs — `seeds × 2 × output_size` of device memory, **234 GiB of 252**
held at the point of failure. `--low-memory` keeps one seed's outputs and
re-runs the seed loop instead. Same derivation, twice the executions.

*Comparison width.* The comparison promoted whole tensors to float64 and
materialized a difference, peaking near **4x the output's size** — an 18 GiB
output cannot be compared to itself on a 252 GiB GPU. Now chunked at 64 Mi
elements. This changes no number: a maximum over chunks is the maximum.

Four of the five problems calibrate after those two fixes. What was left —
`L1/018` and `L1/026`, both "Tried to allocate 16781313.00 GiB" — was recorded
as **not an OOM and not yet explained**. **That is now explained: it is
[D13](#d13).**

### `masked_select` asks for 16781313 GiB above 2³² elements {#d13}

The workloads [D9](#d9) could not explain were not an OOM. **Boolean indexing on
ROCm 7.2 / torch 2.9.1 computes a garbage allocation size once the tensor has
more than 2³² elements.** Reproduced in isolation, on a flat tensor with nothing
else on the GPU:

```python
n = (1 << 32) + 1000
t = torch.ones(n, dtype=torch.float16, device="cuda")
t[torch.isfinite(t)]
# OutOfMemoryError: Tried to allocate 16781313.00 GiB
#                   (2**54 + 2**42 + 2**30 bytes), 70 GiB free
```

Promoting the same tensor to float64 and reducing it is fine, so it is the mask
path and not the size. The tolerance floor now accumulates over bounded chunks
with `torch.where`.

**What made it worth chasing rather than filing as an OOM:** the *same* absurd
number appeared **to the byte** on problems that share no operator. An allocator
under pressure does not do that. 16781313 GiB is 2⁵⁴ bytes.

With this closed, every non-NVFP4 workload has an AMD-derived tolerance — **3717
of 3957**, the missing 240 being exactly the 15 deferred NVFP4 problems × 16
workloads. Keep this entry as a **standing platform hazard**: any new code that
boolean-indexes a large tensor on this stack will hit it.

### `gen_golden.py` assumed a `get_inputs()` that does not exist {#f12}

It was written against the KernelBench convention; SOL-ExecBench problems
declare their inputs in `definition.json` and generate them through
`gen_inputs` / `load_safetensors` / `custom_inputs_entrypoint`. **As written it
would have failed on all 235 problems.** Rewritten against the real schema, per
workload, keyed by workload uuid — which is what task 05 compares against.

### fp64 promotion breaks dtype-literal references, and goldens now record their tier {#f13}

Promoting only the inputs to float64 raises on any reference that constructs
internal tensors at a literal dtype (`torch.zeros(..., bfloat16)`, weights made
inside `run`). **63 of 1480 L1 workloads.** It now falls back to a native-dtype
CPU run and **records which tier produced each golden** (`ok:float64` against
`ok:native_cpu`), because they are not equally strong evidence: a disagreement
against float64 is a bug, against native CPU it may be ordinary noise.

---

# 5. Agent runs, scoring and reward hacking

How the agent baselines were produced, what went wrong in producing them, and
the first exploits the corpus caught in the wild. `docs/agent-baseline.md` holds
the run-by-run description, the cost model and why `pilot8` is off the board;
this section holds the findings.

### The agent pilot billed the wrong gateway key, and looked fine doing it {#d16}

`~/.claude.json` carries an `env` block, and Claude Code applies it **over** the
process environment. On this host it sets

```
ANTHROPIC_CUSTOM_HEADERS = Ocp-Apim-Subscription-Key: 6838c76b…
```

so every header `agent_baseline.py` exported — including the `fb97d25…` key it
was told to use — was discarded. The `pilot8` run ($65.08 over 8 problems)
therefore authenticated with the **wrong AMD gateway subscription key**.

Nothing about the run looked wrong. It completed, produced kernels, reported
costs. **It surfaced only from a falsification test**: a session was run with a
deliberately invalid subscription key and **succeeded anyway**, which is
impossible if the exported header were being used. Directly against the gateway,
`fb97d25…` -> 200, `6838c76b…` -> 200, `0000…` -> **401**, so the gateway does
validate the key and the earlier success proves the override.

No personal Anthropic credential was ever involved: `ANTHROPIC_API_KEY` on this
host is the literal string `dummy` and there are no stored OAuth credentials.
Both keys are AMD gateway keys reaching `llm-api.amd.com/Anthropic`.

Fixed by passing `--settings` with an explicit `env` block, which does take
precedence — verified the same way round: an invalid key passed through
`--settings` makes the session fail. The run record now carries
`gateway_key_prefix`, so which key paid for a run is an artifact, not a
recollection.

### The scorer wrote into the container and scored every kernel zero {#d17}

`agent_score.py` passed the container a **host** path for `--out`. Only two
trees are bind-mounted — the repo at `/work`, and `SOLEXBENCH_SCRATCH` at its
own absolute path — so a run directory anywhere else (`artifacts/…` given as an
absolute path, or a scratch experiment under `$HOME`) resolves inside the
container to a directory the unprivileged user cannot create. The runner died
before writing anything, `retime()` discarded its stderr, and all eight problems
reported `0/0 passed, 0 scored`.

**That output is indistinguishable from eight kernels that genuinely failed**,
which is what makes it dangerous: a real result of "the agent achieved nothing"
was available for the taking. The artifact is now staged through scratch and
copied out, and a runner that never ran reports `RUNNER FAILED` with its stderr
instead of a zero score.

The earlier one-problem validation run passed only because `--run` was given as
a *relative* path, which happened to resolve against the container's `/work`
working directory.

### The first near-full-benchmark agent run, and 7 more bad bounds {#d31}

`agent-glm-sweep-2`: **192 of 220 scoreable problems**, GLM-5.2 driven by
codex-cli through the amdpilot fleet, one MI350X per job, re-timed on an idle
GPU 0 at 50 iterations. **2,760 workloads scored, mean S = 0.5975**, 0 flagged.
The largest agent run on the board at the time by a factor of eight and the
first that was not a pilot.

**Its 44 bound violations name 7 problems not previously known to have a wrong
T_SOL**: `L1__006`, `L1__057`, `L2__030`, `L2__035`, `L2__045`, `L2__068`,
`L2__073`, alongside the known `L1__005` and `L1__035` ([D21](#d21)). Total
across all runs on the board went to **10**, against the 3 that v1 ships marked.
This is the mechanism `CLAUDE.md` already names — a self-consistent bound and
anchor cannot detect a shared error, and only an independent kernel separates
them — firing seven more times as soon as a real optimizer was pointed at the
rest of the benchmark. Their scores are excluded from the mean, not zeroed:
including them would report 0.6135.

On the board it sits at **#3 in both scopes** — 0.5451 over what it ran, 0.4366
over the whole benchmark — ahead of both `torch.compile` variants and just under
the two eager baselines on the shared denominator.

**A capped session is not a failed one.** 154 of the 192 jobs were SIGTERM'd at
the fleet's 3,600 s wall clock rather than stopping on their own, and on the
authoritative re-times they score *higher* than the ones that finished:

```
stopped by the 1 h cap    154 problems   139 scored   mean S 0.6072
stopped on their own       38 problems    38 scored   mean S 0.5579

final kernel == reference  41 problems               mean S 0.4685
final kernel modified     151 problems               mean S 0.6352

every workload passed  172     partial 5     nothing passed 15
workloads 2760/2977 passed
```

The capped sessions used the whole hour, and it shows. **This is the opposite of
`pilot8`, whose cap produced survivorship** — there the mean was over the
fraction that finished; here every problem was attempted and every kernel
scored, including the ones that score 0. So it is on the board, with the cap
stated in its notes and recorded as its trial constraint.

**One real cost of the cap, not visible in the numbers above:** the prompt's
rule is "whatever is in `kernel.py` when you stop is your submission", and
SIGTERM does not wait for a good moment. **36 final kernels are broken or were
never evaluated, and 27 of those have a passing snapshot in `evals/` that is
discarded.** No substitution was made — best-of-N over a session's snapshots is
a different protocol and would have to be declared, not quietly applied — but it
means this run's score is a floor by roughly 27 problems' worth.

Trajectory, transcripts and effort came across via
`scripts/import_fleet_depth.py`: 189 trajectories, 1,264 evaluation steps, 1,052
of them scored, 189 transcripts.

**Superseded on coverage by [D31b](#d31b)**, which completed the same run to
220/220. This entry's claim that "`TODO.md`'s no-full-benchmark-agent-baseline
is now 87% closed" is **100% closed** as of D31b.

### The run is complete: 220/220, and it leads the board {#d31b}

`agent-glm-sweep-2` now covers **every scoreable problem**. A follow-up sweep
(`glm-sweep-2-fill`, 48 jobs, same harness, same model, **same 3,600 s cap** — a
different cap would have made the submission's own trial label false) was merged
in by `scripts/merge_agent_runs.py`, which **replaces the earlier attempt
outright wherever both exist. Not best-of-two**: keeping whichever scored higher
would make the submission a maximum over re-runs rather than a measurement of
the harness.

```
                              ran    whole | clean part fail none  workloads
codex-cli agent (GLM-5.2)  0.5921  0.5921 |   218    1    1    0  3618/3717
PyTorch eager              0.4536  0.4536 |   219    1    0    0  3707/3717
eager + contiguous         0.4518  0.4518 |   218    1    1    0  3688/3717
torch.compile              0.4216  0.4190 |   149   69    2    0  3171/3694
max-autotune               0.4104  0.4034 |   136   77    7    0  3041/3654
```

**3,690 workloads scored, mean S = 0.6083, 0 flagged.** The two scopes are
identical to four decimals because nothing is missing from the denominator —
the cleanest possible demonstration that the coverage is real.

What had been missing, and why: **26 = the whole FlashInfer-Bench category,
never ported** (`sbt select` sent L1 92, L2 82, Quant 18, FlashInfer 0 — an
omission in the selection, invisible because every problem it *did* send came
back); **2 L1 never ported** (92 of 94); **3 ported that produced no result**;
**20 killed mid-edit at the cap**, of which 11 had a fully-passing evaluated
snapshot discarded because `kernel.py` was mid-write when SIGTERM landed.

**The re-run settles the snapshot question [D31](#d31) left open.** Same cap,
fresh attempt: **mid-edit losses fell from 11/20 to 3/19**, and 15 of the 19
came back with a final kernel that passes everything. It was a coin flip on
where the kill landed, not something the model controls. At three problems in
220 the case for changing the submission rule has gone; the score is a floor by
about three problems and that is worth stating rather than engineering around.

The GPU-0 reservation **held for this sweep** — an active `manual` hold on
device 0, all seven contracts on 1–7, zero KFD processes on 0 throughout.
[D29](#d29) did not recur. The merged `run.json` still records agent GPUs
`[0..7]` because the *original* sweep used all eight; that is history, not this
run.

Two things worth keeping:

* **`rocm-smi --showuse` is not a usable signal on this node.** It reported
  GPU 0 at 97–100% busy while `--showpids` listed no process on it and all eight
  cards drew a uniform 235–244 W. A saturated 1000 W part does not sit at idle
  power. The KFD process list and the scheduler holds are what to trust.
* **A `failed` fleet job is not a failed kernel.**
  `FlashInfer-Bench__020` died with `harness_error` at 1898 s and the kernel it
  left behind re-times clean at 19/19.

### gpt-5.6-sol over all 220, and the counter that could not count {#d41}

**2026-08-10.** `agent-gpt56-220`: 220 of 220 problems, 3,717 workloads,
**3,701 scored, mean S = 0.6381**, 16 flagged, 4 problems whose bound a kernel
beat. Benchmark score **0.6332** at 99.2% coverage, against GLM-5.2-FP8's 0.5989
at 98.1%. `check_coverage.py` reports 235/235 with the 15 NVFP4 deferrals
accounted for. 180 problems ran through the fleet as J2 backfill in **11.9 h at
a mean concurrency of 6.93 of 7 cards**, zero failed, zero cancelled. The port
needed `scripts/port_via_fleet.py` because the queue's write guard had been
switched on ahead of its callers.

**The re-time was done 8-wide, and that is a departure with a measured cost.**
`artifacts/11/parallel-retime-validation.json`: eight problems with an existing
GPU-0 number, re-measured under full 8-card load, 98 workloads. Median ratio
**1.0139** — 8-wide is **1.4% slower, systematically**, which is 13x task 01's
sanctioned sibling interference. In score terms ~0.004 of S at S ≈ 0.6 against a
head-to-head gap of ~0.03: it does not flip the comparison and it is ~13% of it.
Every artifact it produced carries `concurrent_cards: 8`. 113 problems took
**10.5 min** this way; a serial re-time costs 1.9 h.

**Reverted on 2026-08-10. The published numbers are serial again.** All 220
problems re-timed one at a time on GPU 0 (PID 2528350, 66 min wall, the 113
8-wide artifacts backed up first):

| | 8-wide | serial | Δ |
|---|---|---|---|
| mean S over scored workloads | 0.6381 | **0.6391** | +0.0010 |
| benchmark score on the board | 0.6332 | **0.6341** | +0.0009 |

The direction is right — 8-wide measured slower, so removing it raises the score
— and the size is **a quarter of what the 1.4% median ratio predicted** (~0.004
of S). Two things account for the gap and neither is a correction to the 1.4%:
only 113 of 220 problems were 8-wide, and S is a **saturating** function of
`T_k`, so a fixed fractional slowdown moves S less where a submission is already
far from its bound. The projection was an upper bound and is now superseded by
the direct measurement.

**8-wide is a tool with a known cost, not a mistake.** `retime_parallel.py`
stays. Use it for exploration and for anything not being compared against a
serially-measured run; do not use it for a published number that shares a table
with one.

**Two more bad bounds, and that is the pattern not an accident.** `L1__018` and
`L1__042` had never been reached. 3 -> 5. **The count tracks how hard anything
has tried** — see [D31c](#d31c) and [D37](#d37).

**The flagged column could not have reported them.** `n_flagged` was computed
inside `leaderboard_rows`'s aggregate, whose `WHERE` is `status='PASSED' AND
score IS NOT NULL` — and a flagged workload has status `REWARD_HACK` and a NULL
score, so **each clause on its own excluded it**. The counter was structurally
incapable of returning anything but zero. The board read 0 from the day it was
built, `/methodology` asserted that zero in prose, and it read 0 for hours after
the harness caught 48 real ones. **A negative result guaranteed by construction
is not evidence of anything.** Counted on its own query now, with
`test_flagged_is_counted_over_every_result_not_just_the_scored_ones` building a
one-flagged-row board rather than trusting the artifacts, and mutation-checked:
putting the filter back makes it fail.

**Three operational mistakes worth recording.** Two `agent_score.py` processes
ran on GPU 0 at once, because the unattended finish script woke on the fleet
draining while a hand-started pass was still going — the exact collision this
node's discipline exists to prevent, made by the person who wrote the guard for
it. 52 of 112 measurements were taken on a card another team's container was
sharing, because the exclusivity check added that morning **recorded** and did
not **wait**; they were discarded and re-measured. And a host path was handed to
a container, which fails as `FileNotFoundError` on the kernel and reads like a
missing submission rather than a mount mistake. (The same class of failure
recurred as [D61](#d61); the guard is only a guard if it refuses.)

### The first reward hacks on this board, and they were already published {#d40}

**2026-08-10.** The tf32 guard added earlier the same day (see [D35](#d35)'s
allow_tf32 exposure) was run for the first time over real submissions, as part
of re-timing `agent-gpt56-40`'s 40 problems. **It fired immediately.**

```
agent-gpt56-40    L2__069   16 workloads   torch.set_float32_matmul_precision("high")   line 6
agent-glm-sweep-2 L1__001   16 workloads   torch.set_float32_matmul_precision("high")   line 5
agent-glm-sweep-2 Quant__006 16 workloads  allow_tf32 = True / matmul / allow_tf32 = False
agent-glm-run1    FIB__019   not re-timed  torch.set_float32_matmul_precision("high")   line 4
```

**48 of 19,310 recorded results.** These were **scored and published** —
`L2__069` at mean S 0.6203 since 2026-08-09, `L1__001` and `Quant__006` since
2026-08-08. Nothing checked for it until the guard existed. Run means barely
move (gpt56-40 0.6406 -> 0.6407, glm-sweep-2 0.6104 -> 0.6106) because the
flagged workloads scored near the mean; **the count is the finding, not the
mean.**

**Two of the three spell it `torch.set_float32_matmul_precision("high")`, not
`allow_tf32 = True`.** A guard matching source text would have missed both. This
one snapshots the backend flags the alias sets, so the spelling is irrelevant —
which is why it caught a form nobody had thought to write down.

**`Quant__006` is the one that matters for the design.** It sets the flag, runs
one matmul, sets it back, and returns. The check as first written compared the
flag before and after the call, so a submission that restores it is invisible —
and **restoring is not a weaker exploit, it is the same speedup with the
evidence cleaned up.** The check now records every *lowering write at the
setter* and fires on the attempt, so restoring is irrelevant. Mutation-checked:
with the recorder disabled that kernel's corpus twin passes cleanly, undetected.

**And the before/after check was catching it for the wrong reason, with a
victim.** Restoring `allow_tf32 = False` does not restore the knob it shadows:
measured here, `fp32_precision` goes `none -> tf32 -> ieee`. So "before != after"
fired on the **restore**, not the exploit — and would fire identically on an
honest kernel that defensively writes `allow_tf32 = False` and nothing else. The
comparison now asks whether a knob moved from full fp32 to something cheaper;
`none -> ieee` is not that. `test_defensively_disabling_tf32_is_not_a_hack`
asserts an honest submission still passes, which is the only kind of case in
this corpus that can catch a guard being **too eager**.

Three corpus cases added, all mutation-checked:
`test_matmul_precision_flag_restored_before_returning`,
`test_matmul_precision_via_the_public_alias`,
`test_defensively_disabling_tf32_is_not_a_hack`.

**Not re-timed:** `glm-run1`'s `FlashInfer-Bench__019`. That run is withdrawn
from the board so nothing of it is published; recorded so the omission is a
decision.

### The external fleet ran 34 jobs on GPU 0 {#d29}

`dash-overlay`'s J2 backfill sweep places one agent per GPU and takes a
scheduler hold on the authoritative device (`sbt`'s
`reserve_authoritative_gpu`). **The hold did not hold**: across the recorded J2
jobs the placement is 34 · 42 · 36 · 35 · 35 · 35 · 36 · 34 over GPUs 0–7 —
**GPU 0 took a full share.**

No published number is affected: no authoritative timing was running during
those jobs, and every score on the board was re-timed before the sweep began.
**The failure is that the property was not enforced**, so the next time the two
do overlap nothing will say so. Whichever side owns it, the check belongs where
the placement happens, not in a comment.

This is cited from code as `TODO.md D29` (`scripts/guard_authoritative_gpu.py:8`
and `:226`, `scripts/gpu_exclusive.py:7`) — the owed work lives there, the
finding lives here. It did not recur on the [D31b](#d31b) sweep, and it recurred
in a different form as [D61](#d61).

---

# 6. The leaderboard: a view that disagreed with its artifacts

The board is a *view*. Never edit `leaderboard/solbench.db`; change the artifact
and re-ingest. Every defect below is a view defect — **no measurement was
changed by any of them** — but several of them changed a published number,
because a mislabelled measurement is a wrong number to a reader.

## 6.1 Numbers the board got wrong

### The board scored a whole problem by one flag, under-reporting 1,239 passing workloads {#d28}

**No measurement is wrong. The board read 1,239 of them as failures that the
artifacts record as passes.**

`ingest_variants()` (`leaderboard/ingest.py`) wrote each variant row as
`"PASSED" if all_passed else "FAILED"`, where `all_passed` is a **per-problem**
flag on the variant. A problem where `torch.compile` matches on 13 of 20
workloads therefore landed on the board with all 13 marked `FAILED` and scored
`NULL` — even though `latency_ms_by_workload` holds exactly those 13 timings and
nothing else. The per-workload truth was available the whole time, one key
across: each variant also carries `failures: [{workload_uuid, status, log}]`,
which the ingest never read.

Board figure against the artifacts, over the 220 scoreable problems:

| variant | board says | artifacts say | delta |
|---|---|---|---|
| `v1_eager` | 3701 / 3717 | 3707 passed of 3717 attempted | 6 |
| `v4_contiguous` | 3687 / 3717 | 3688 of 3717 | 1 |
| `v2_compile` | 2586 / 3717 (69.6%) | **3171 of 3694** (85.8%) | 585 |
| `v3_compile_max_autotune` | 2394 / 3717 (64.4%) | **3030 of 3611** (83.9%) | 636 |

The problem counts (`149/220`, `136/220`) are *not* wrong — they are counts of
problems where the variant passed everything, which is what `all_passed` means.
It is the workload column that answers a different question than its heading.

**Worked example**, `L1__062_kv_cache_update_with_rope_backward`, `v2_compile`:

```
artifact  workloads=16  passed=15  all_passed=False
          latency_ms_by_workload: 15 entries      failures: 1
          overlap between them: 0   <- a timing exists only where it passed
board     15 rows, all FAILED, all score NULL, all with a latency
```

**Fixed 2026-08-07, and four published baseline numbers moved.**
`ingest_variants()` now builds a row per workload from **both** halves of the
artifact — `latency_ms_by_workload` for the passes, `failures` for the rest,
each carrying the status the harness recorded. A variant that died before
measuring anything (`error`, neither list) gets an `ERROR` row per workload,
because it was still run on them.

| variant | what it ran | whole benchmark | problems attempted |
|---|---|---|---|
| PyTorch eager | 0.4541 -> 0.4536 | 0.4528 -> **0.4536** | 220 -> 220 |
| eager + contiguous | 0.4553 -> 0.4518 | 0.4517 -> **0.4518** | 219 -> **220** |
| torch.compile | 0.4002 -> **0.4216** | 0.3414 -> **0.4190** | 218 -> **220** |
| max-autotune | 0.3880 -> **0.4104** | 0.3174 -> **0.4034** | 213 -> **220** |

**No measurement changed.** Every number moved because rows mislabelled `FAILED`
are now labelled with what actually happened, and the scored ones carry the
score they always had.

The problems column is the second half of the same defect and the easiest to
misread: `latency_ms_by_workload` holds only passes, so a problem where a
variant passed **nothing** produced no rows and read as *never attempted*. All
four variants were run on all 220 problems. The board said `torch.compile: 218`
and `max-autotune: 213` because they passed nothing on 2 and 7 problems —
`L2__036`, `L2__037`, `Quant__011`, `L1__094`, `L2__012`, `L2__070`, `L2__077`.

One thing this dislodged: `v5_compile_contiguous` is excluded from the board
because it passed zero workloads, and the exclusion tested "produced no rows".
That was the same statement only while failures were discarded — the moment they
were kept, v5 acquired 3,717 rows and walked back on with a score of 0.0000. The
test is now "passed nothing". The ingest's own drop guard caught the intermediate
build that had published it, which is what that guard is for.

**"Re-ingest" means re-running `leaderboard/ingest.py`.** It reads JSON that
already exists and writes SQLite. No GPU, no kernel runs, no re-measurement, and
*nothing converts from fail to success*. The three things that sound alike and
are not:

| | what it is | GPU cost |
|---|---|---|
| re-ingest | read `failures` per workload instead of the problem flag | none |
| authoritative re-time (89 problems) | these workloads already pass; their timing on record came from a sweep GPU, not GPU 0 | ~2¼ h |
| the genuine gaps (129 workloads) | never measured at all — no timing exists | see below |

**89 of the 220 problems have at least one variant with passing workloads but no
GPU-0 authoritative re-time**, because the authoritative pass only re-timed
variants that were clean (`v2` lacks 69, `v3` 76, `v1` 9, `v4` 8). Those rows
carry a sweep timing from GPUs 1–7, **a different measurement class**, and have
to be labelled as one. At the authoritative sweep's measured rate — 217 problems
in 323.3 min, ~1.5 min/problem (`artifacts/06/logs/authoritative.log`) —
re-timing all 89 is about 2¼ hours on GPU 0; a full re-derivation is ~5.4 h.
**This is still owed** and is in `TODO.md`.

The genuine gaps are **129 workloads across 8 problems** where the task-06 sweep
recorded fewer workloads than the problem has: `v2_compile` 23 across 2,
`v3_compile_max_autotune` 106 across 8. Causes are visible in the artifacts and
are not all the same — `L1__094` timed out at 3600 s, `L2__012` died with
`rc=-6` and no traces, and `L2__055`/`L2__059` recorded `all_passed=True` over 3
and 5 workloads while the problem has 15 and 16, **a driver that stopped early
and declared victory over what it had reached**.

Re-timed 2026-08-07 into **`artifacts/06/candidates-gapfill/`**, GPUs 1–7,
inside `env/solb`, 50 iterations — deliberately *not* into
`artifacts/06/candidates/`. That directory is frozen input to manifest v1, and
`T_b` is the fastest variant per workload: a new `v3` timing landing there could
lower `T_b`, which moves the bound, which moves every score on that workload.
Filling these gaps is a v1.1 change and needs the manifest regenerated and
re-published, not an artifact edited underneath it.

A first attempt ran the runner on the **host** rather than through `env/solb` —
python 3.11.7, ROCm 7.15, `torch.available: false` against the pinned 3.12.3 /
7.2.0 / torch 2.9.1. Caught on the provenance stamp of the first output file,
and those artifacts were deleted rather than kept. **This is exactly what prime
directive 6 is about and it took one command to get wrong.**

#### And the gap-fill found something worse than a gap

**10 of the 32 variant×problem cells re-run did not reproduce.** Not "produced a
slightly different latency" — changed verdict:

```
L1__094  v2_compile                16/9  -> timeout   WORSE
L2__037  v2_compile                 6/0  ->  6/6      verdict flipped
L2__037  v3_compile_max_autotune    6/0  ->  6/6      verdict flipped
L2__059  v3_compile_max_autotune    5/5  -> 16/16     coverage recovered
L2__070  v2_compile                16/8  -> 16/16     verdict flipped
L2__070  v3_compile_max_autotune     1/0 ->  1/1      verdict flipped
L2__077  v2_compile                16/8  -> 16/16     verdict flipped
L2__077  v3_compile_max_autotune     3/0 ->  3/3      verdict flipped
Quant__011 v2_compile               3/0  -> 16/16     coverage recovered
Quant__011 v3_compile_max_autotune  3/0  -> 16/16     coverage recovered

reproduced identically       22
same workloads, more passes   6   <- the verdict changed, not the coverage
more workloads measured       3
got worse                     1
```

**Do not generalise this to the whole sweep.** These 8 problems were *selected*
for having incomplete coverage — they are the already-suspect ones, and a 31%
non-reproduction rate among them says nothing directly about the 212 that
recorded full coverage.

> **The doubt this raised is now inverted, and the entry's own caution was
> pointed the wrong way.** The original text said the D28 rewrite "should not
> describe [the 523 v2 failures] as torch.compile disagreeing with eager beyond
> tolerance until a repeat run says they reproduce". A repeat run said they
> **do**: the verdict flips above are explained by [D50](#d50) (the 8-shape
> recompile cliff — several of those cells are exactly the boundary signature),
> and [D59](#d59)'s full re-sweep measured the compile variants losing **−499
> and −526** passes once the cliff was closed. So the 523 was a **floor**, not
> an over-count, and the direction of the doubt was backwards. What survives
> unchanged is the narrow, correct observation that **a recorded
> `INCORRECT_NUMERICAL` is not always a verdict about the kernel; some of it is
> the run** — `Quant__011`'s `passed=0 over 3 workloads` was a driver that died
> after three, not a variant that failed.

### A failed workload was carrying a score {#d22}

Found while building the submission × problem view, the first page that puts a
status and its score on the same row. `ingest.py` scored the four reference
variants unconditionally:

```python
rows.append((sub_id, pkey, uuid, "PASSED" if all_passed else "FAILED",
             ms, sol_score(ms, t_b, t_sol), 0, label))   # score even on FAILED
```

A variant that fails the correctness check still has a latency, and `sol_score`
turns it into a perfectly plausible number — **the speed at which the wrong
answer was produced.** `torch.compile` on
`L1__002_vae_conv3x3_groupnorm_silu_residual_fused` displayed
`FAILED … S = 0.4956` on thirteen rows, and the page's own summary card read
"best workload S = 0.5832" for a submission that passed nothing on that problem.

**No ranking was ever wrong.** Every aggregate filters `status='PASSED'`, and
the board totals are byte-identical before and after the fix. It was confined to
per-workload *display* — but `/api/problems/{key}` served it too, so anything
consuming the API would have inherited it.

The agent path never had this: `agent_score.py` leaves `score` at `None` unless
the workload passed. The variant path had simply diverged. Fixed at ingest
rather than in the templates, so every consumer gets one answer. The run page
additionally suppresses the T_b speedup on a failed row, for the same reason —
"0.99x vs optimized PyTorch" next to `FAILED` reads as near-parity.

**This is the second defect of exactly this shape** ([D21](#d21)'s ingest bug
was the first): a number that is real, plausible, and attached to the wrong
claim. Both were invisible until a page forced two facts next to each other.

### Three tables ranked means with different denominators {#d26}

`AVG(score)` skips NULL, and a `PASSED` result stores `score IS NULL` when the
kernel beat `T_SOL` — the bound is invalid there, so no score is defensible. The
denominator therefore varies per row, and the leaderboard put such rows next to
each other under one `mean S` heading, in a **sortable** column.

The real instance: `agent-pilot8` on
`FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1` passes 38
of 38 workloads, 25 of which beat the bound. Its mean was printed as **0.9899
over 13 workloads**, directly above `agent-glm-run1`'s **0.9430 over all 38** —
so **the run with a third of the evidence sorted to the top.** Same shape in
`problem_detail()`, the submission page, and the `peers` query.

All 31 such results repo-wide were traced and they map exactly onto defects
already recorded: **25 to [D18](#d18)** and **6 to [D21](#d21)** (`L1__005`
4/16, `L1__035` 2/16). Nothing new was found and no bound was silently adjusted.

**No number was changed.** Every mean is arithmetically correct and stays as it
was; what was missing is the count it was divided by, now printed under each one
(`over 13 of 38 — bound beaten on 25`). Changing the mean was the tempting
alternative and it is unavailable in both directions: scoring a beaten bound as
0 punishes a kernel for the bound being wrong, and dropping the workload inflates
the run. **Publishing the denominator is the only move that does not invent a
number.**

The trial switcher had the opposite bug and was fixed the other way: it used
`AVG(score)` where the run card used `score_sum / attempted` with
`COALESCE(score, 0)`, so the two disagreed on exactly these rows. The switcher
now matches the card and says why a clean sweep can still score low.

Guarded by `tests/leaderboard/test_score_denominator.py` (6 tests). The
denominator is printed on **every** row, not only the short ones — **a caveat
shown only when it applies is indistinguishable from one nobody checked.**

### A submitted kernel whose re-time timed out is invisible {#d23}

`glm-run1` has 24 kernels on disk and results for 23. The missing one,
`FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`, has a
`retimed/*.json` recording `TimeoutExpired` after 1200 s.

It produces no result rows, so every aggregate treats it as *not attempted* —
identical to a problem the agent never opened. The agent wrote 180 lines and the
harness could not measure them; that is a different fact, and it was not
recoverable from the board. Found only because ingesting kernels-from-disk
disagreed with kernels-from-results, 24 against 23.

The board now carries `run_kernel.retime_ok` / `retime_error` and shows the
state on both the run page and the submission page. **The score is unchanged and
should be**: no measurement exists, so there is nothing to score. The fix is to
stop the absence from being silent, not to fill it in.

The cause was [D33](#d33) — the inner timeout was never forwarded, and with it
forwarded the kernel re-times 30/30. **The budget question is still open**:
whether 1200 s is right for a paged-prefill problem of that size. Corroborated
by the two 5400 s timeouts in [D59](#d59)'s re-sweep.

### The same "dropped the external run" bug, three times {#d24}

`ingest.py` reads agent runs from `artifacts/10` unless told otherwise, so any
rebuild that omitted `--agent-runs` silently deleted every run kept outside the
repo — at the time, the $250 Opus run. Introduced three separate times:

1. The staleness banner told the reader to run a bare `ingest.py`. Fixed by
   printing the roots the build actually used.
2. `worker.py` shelled out to a bare `ingest.py` after scoring. **Caught in the
   first end-to-end test**: the job scored correctly, and the Opus run
   disappeared from the board. Fixed by reading `meta.input_extra_roots` back
   out of the database that is about to be replaced.
3. Any manual rebuild, still, if the flag was forgotten.

**The shape of the bug is that the *default* is lossy and the loss is silent** —
the board still renders, still looks complete, and only the person whose
submission vanished would notice. `worker.py` now diffs the submission set
across a rebuild and reports a drop rather than trusting the exit code.

**The semantics have since inverted, and this is the live trap.** The durable
fix was made: the roots now live in `leaderboard/sources.json` (untracked; see
`sources.json.example`) which `ingest.py` reads by default. `--agent-runs`
**overrides** that list rather than adding to it — so **passing the flag is now
the way to get the omission, not the way to avoid it.** Either way the ingest
refuses to publish a board that has lost a submission the current one has,
unless `--allow-drop` says so.

### A stale pre-split database that nothing serves {#d54}

`leaderboard/solbench.db` (built 2026-08-06T23:25Z, before the [D28](#d28) fix
landed on 08-07) really is inverted: **585 not-PASSED rows for
`baseline-v2-compile` against the artifacts' 523, intersection ZERO**, all 585
carrying a latency today's `ingest.py` cannot produce — so it was built by code
that no longer exists.

But `part_databases()` reads `db/solbench-<PART>.db` first and falls back to the
single file only for a part the per-part layout has not produced.
`db/solbench-MI350X.db` exists and has **523 not-PASSED matching the artifacts
exactly, 0 with a latency**. The running board confirms it: `L2__009` returns
8 `INCORRECT_NUMERICAL` / 8 `PASSED`. No re-ingest needed; the file is gitignored
and has never deployed.

**Left open: delete the leftover.** If `db/` were wiped, the fallback would
serve it and `run.sh`'s rebuild guard would count it as a database — the
freshness check is the only thing that catches it (it does: `stale: True`). Not
deleted at the time of writing; verify before relying on this.

> **CORRECTION, recorded because it changed what the maintainer was told.** The
> first version of this entry said **the live board was showing inverted
> statuses. It was not.** The leftover file was queried and taken for the served
> one, which is also why the failure population was first framed as "585 over 69
> problems" rather than the true **523 over 71**. A peer session caught it, and
> it was re-verified against the running board before correcting. Ground truth
> throughout is `artifacts/06/candidates`.

## 6.2 Data the board did not have, or had wrong

### The four baselines' pages said their code was not recorded {#d34}

Every reference-variant problem page — **880 of them** — rendered "No kernel was
recorded for this submission on this problem", plus the generic "no trajectory
was recorded" and "no per-problem cost was recorded". All three read as gaps in
what the harness captured. **None of them was.**

The source *is* recorded: `variant_source` holds **1,175 rows**, five transforms
× 235 problems, regenerated at ingest from each problem's own reference. The
pane was gated on `kernel`, which a variant has no row in because nothing was
authored, so it collapsed to the fallback while the code sat one query away.
`variants` in the same handler was no help — it lists whichever transforms won
T_b *on this problem*, the same list for every viewer, answering a different
question than "what did this submission run".

Fixed by resolving the submission's own transform through a new
`submission.variant` column **rather than reconstructing the name from the
slug**: `baseline-v3-compile-max-autotune -> v3_compile_max_autotune` happens to
round-trip today, and the failure mode when it stops is that the page shows a
*different* transform's real, valid, correctly-highlighted code — **not
detectable by reading the page.** The test fixture's slug is deliberately
`ref-v1-eager`, which does not round-trip, so a slug-derived lookup fails in CI
instead of shipping.

Where a variant is also the T_b anchor on some of a problem's workloads, the
pane now says so and says that those workloads **score exactly 0.5 by
construction** — the number on these pages most likely to be misread as a
result. The trajectory and cost sections get a `depth_note` instead of the
fallbacks: a variant is one deterministic transform compiled and timed once, so
there is no trajectory because nothing iterated and no cost because no model was
called. **Zero by construction, not unrecorded.**
`tests/leaderboard/test_variant_source_pane.py`, 5 checks.

### A category filter that filtered the rows and not their labels {#d44}

`/?category=X` filtered the rows and **not** their labels. The scope notes, the
"whole benchmark" segment and the coverage header quoted the manifest's
3,717 / 220 above a table divided by (L1) **1,480 / 94**. Fixed at the source:
`scoreable_totals()` is now the one implementation of the denominator, and the
page quotes what the rows used.

### An unknown category rendered a full board of zeros {#d45}

An unknown `?category=` matched nothing and rendered a full board of 0.0000
scores with empty coverage bars. Now a **400**, on both HTML routes and all four
JSON ones, like an unknown `?part=`. The two board filters (which problems /
which denominator) are one row, one colour each — blue for the set, red for the
denominator — each with its own label, `aria-label` and `aria-current`.

### Every per-workload `axes_json` in the database was empty {#d46}

All **3,957 rows** carried `{}`, so the problem page's axes column had rendered
empty since it was written. The manifest carries no axes — it is a scoring
artifact — but the dataset's own `workload.jsonl` does, and `ingest.py` now
reads it. **3,941 of 3,957 have axes**; the other 16 (`L1__016`) declare none in
the dataset and say so on the page.

The re-ingest that carried the fix is also the reference for what a correct
rebuild produces, which is worth having written down somewhere: **235 problems,
3,957 workloads, 22,357 results — unchanged counts, and no score moved.** A
rebuild that changes any of the three has done something other than what it
claimed to.

### A workload's parameters were only the axes that vary {#d47}

Three of seven on `L1__001`, because `workload.jsonl` carries only those. The
`const` values come from `definition.json` and the `expr` ones are now computed
from the rest by an **AST walk over a whitelist** (`eval_axis_expr`, **123
distinct expressions** in the dataset, none of them anything but arithmetic).
All 3,957 workloads now list their full parameter set, as upstream does, with
const/expr chips dimmed.

### Workload identity was an 8-char uuid prefix corresponding to nothing {#d48}

`dataset_index` is the position in the dataset's own `workload.jsonl`; checked
across all 235 problems, that is also the order upstream returns its workloads
in, so **#4 here is #4 there**. The table is ordered by it (the manifest sorts
by uuid).

### A board built from a bare clone failed silently in five different ways {#d49}

`data/` is gitignored, so a fresh clone has **no problem definitions**. Every
measured number is fine; description, reference source, inputs, outputs, axes,
workload parameters and dataset numbering are all absent. It failed **silently
and differently in five places**: 235 listing rows printing the literal string
`"None"`; empty header-only tables; an empty reference pane; "none declared"
against every workload — **a sentence about the dataset that was false**; and a
`#` column falling back to a loop counter over uuid-sorted rows, which looks
exactly like a dataset position and **is a different workload**.

`ingest.py` now counts definitions found (`meta.dataset_problems`) and warns
loudly at zero; the board carries one banner naming the cause, the fix and the
fact that no measurement is affected; and each of the five sites states absence
rather than inventing a value. A database built before the key existed raises no
alarm.

**Closed by `reference/dataset-meta.json`** — 2.2 MB, tracked, written by
`scripts/export_dataset_meta.py`: descriptions, references, inputs, outputs,
axes and per-workload axes/order, copied verbatim from the dataset. Descriptive
only; no timing, bound, tolerance or score, and a test asserts the key set.
`ingest.py` reads `data/` first and falls back to it, so a machine with the
dataset can never be served a stale copy; `--check` re-derives and compares byte
for byte. **Proven:** a board built with `DATASET` pointed at an empty directory
is **identical** to the real one on every dataset-derived column — description,
reference, axes, inputs, outputs, `dataset_index` and both B200 columns, 235
problems and 3,957 workloads.

Two more found by diffing the public deployment against local, both of the same
shape: `{% if meta.b200_matched %}` on the **string** `"0"` is truthy, so the
remote rendered the B200 switch over columns that were all blank; and
`{% if meta.problems_with_invalid_bound %}` on the string `"[]"` is truthy too —
an always-true guard under an always-present nav entry, **two bugs cancelling**.
Both now go through `|int` / the parsed list.

### The NVIDIA B200 overlay, and what it may and may not be used for {#b200-overlay}

`scripts/fetch_nvidia_b200_reference.py` pulls NVIDIA's published per-workload
baseline and SOL for all 235 problems from their public JSON API into
`reference/nvidia-b200/published.json`; the ingest matches **3,915 of 3,957** by
axes (unique matches only) into two new `workload` columns. **Display only** —
off by default, absent from `/api/v1`, **never an input to a bound, a tolerance
or a score.** Prime directive 2 is the reason the columns are tinted, captioned
and switchable rather than simply printed.

**Observed:** B200 SOL / AMD T_SOL agrees within 2x on **58%** of matched
scoreable workloads, and the disagreeing tail lands where `bound_quality`
already says the bound is loose or vacuous — an independent smell test that
mostly confirms the existing marking. Two exceptions, both `declared_traffic`,
are recorded against [D39](#d39) (`L2__007`, at 11.7x and 13.8x). **No bound was
changed and none may be changed on this evidence.** The full 5-band ratio table
over 3,675 matched workloads is in `TODO.md`.

## 6.3 Presentation defects that are correctness defects

### One number system for the board, and three figures inside it that were not measured {#d58}

**2026-08-12.** The formatting half first, because it is what the change is.
Every quantity the board prints now goes through one helper in
`leaderboard/app.py` — `dur`/`ratio`/`sci`/`cycles`/`n`/`usd`/`mins`/`pct`/
`bytes_h`, each with a `_text` twin — instead of `%.5f`, `|round`,
`'%.0f'|format(x/60)` and a truthiness check written per call site. Durations
ride an ns/µs/ms/s ladder at three significant figures, so the old `fmt_ms`
style switch at 1e-4 ms is gone and a T_SOL column no longer mixes two
notations. The unit goes in its own fixed-width slot so it forms a vertical
stripe down the column: measured by running the shipped `_laddered` over every
`workload.t_sol_ms` in `db/solbench-MI350X.db` (`mode=ro`), **104 of the 235
problem pages render their whole T_SOL column in one unit, 129 use two and 2 use
three** — so the stripe is carrying 131 pages, the majority, not a tail.

**Why that is a correctness change and not a cosmetic one: the sorter.**
`base.html` falls back to `parseFloat(text.replace(/[,%$×x*]/g,""))`, which turns
`"1.23k×"` into 1.23 and `"377 µs"` into 377 — **a plausible wrong number, not a
NaN**, so a mis-sorted column looks perfectly fine. Every helper that can emit a
non-numeric character therefore tags its wrapper `q-needs-sort` and the cell
must carry `data-sort`; `data-sort=` in `leaderboard/templates/` goes 23 -> 45.
`tests/leaderboard/test_number_format.py` is the gate, because the convention is
not the guarantee.

**Then the part that had to be undone.** Three numbers in that change were
written as measured facts and were not — prime directive 1, and the reason this
entry exists rather than a line in the previous one. All three were recomputed
against `db/solbench-MI350X.db` opened `mode=ro`:

```
claimed                                  measured
FlashInfer-Bench__016's T_SOL column     that problem's 15 t_sol_ms values
reads 0.37706 / 0.00060 / 3.08e-06       hold no 0.00060; the middle one is
(dur_text docstring + a test case)       0.00039923076923076924
125 of 235 pages span two decades,       104 / 129 / 2 (see above). The 131
6 span three (style.css)                 total was right; the split was not
3,744 cells sit at S = 0.5               3,758 -- SELECT COUNT(*) FROM
(score_text docstring)                   result WHERE score = 0.5, and the
                                         same with the board_visible join
```

The FlashInfer example is **deleted rather than restated**: an illustration that
has to be checked against a database to be trusted is a liability in a
docstring. The two counts are corrected in place and each now names the query
that produced it. `0.00060` itself is a real board value — 36 rows of
`b200_sol_ms` — so the test case survives, relabelled to its actual column;
every other case in that file is now labelled with the column it came from, or
with `synthetic` if it is a ladder-boundary probe and not on the board at all.

Four smaller things found in the same pass, all closed:

* `sortv` returns `""` for a value that is not there, and its docstring said
  `base.html` treats that as null. **It did not**: the read was
  `getAttribute("data-sort") || c.innerText`, and an empty attribute is falsy in
  JS. It gave the right answer only where the cell's text was the em-dash the
  next line maps to null, and the wrong one in `run.html`'s trajectory "at"
  column, whose text is "not recorded" — `parseFloat` NaN, then compared as a
  lowercase string against numbers, neither first nor last. **The behaviour was
  fixed, not the docstring**: 45 call sites now depend on `sortv` meaning what
  it says, and a comment that documents an accident is worth less than the
  invariant.
* "One spelling of the em-dash in value position" was asserted and not achieved
  — `problems.html`'s deferred cell and `run.html`'s trajectory pass count each
  spelled `&mdash;` themselves, one with no `q-na` and no title. Both now come
  from `app.EM_DASH`, through `MISSING` or the new `na(reason)`. Checked over
  305 real pages: **7,325 cells whose entire visible text is the dash, 0 not
  built from the constant.**
* Two ladder ends re-created the defects the ladder removed: `dur_text(1e-10)`
  gave `0.000 ns` (a real value displayed as zero, moved down twelve decades
  rather than eliminated) and `ratio_text(1e9)` gave `1000M×` (four digits,
  because the promotion is guarded by `i > 0` and cannot fire on the top rung).
  Neither occurs in today's data. Both now fall back to scientific at the rung
  they cannot leave, and both are pinned.
* `run.html`'s trajectory x-axis label still formatted a minute figure where it
  was printed. It goes through `mins_text` now.

**Test coverage had also narrowed without saying so.**
`test_bound_quality.py::test_every_scoreable_workload_on_the_board_is_marked`
skipped any cell not carrying a bare `×`, but `ratio` promotes to `k×` at 1e3
and `M×` at 1e6 — so it could no longer see one workload in the `vacuous` band,
**the band the file exists for** ([D39](#d39)). It kept passing because the
`assert checked` guard stayed non-zero on the `×` cells. The matcher takes all
three units again, and a second test drives the real board: the problem holding
the largest `bound_headroom` (`FlashInfer-Bench__014`, 1499133.450644357) must
render promoted cells and every one must be marked `bq-vacuous`.

**Left open, and surfaced rather than quietly corrected:** `run.html`'s comment
above the `.solbar` step edges reports the distribution those edges were cut
from as "all 12,883 scored cells … as of 2026-08-06", of which 3,744 are exactly
0.5. It is honestly dated and it is now **stale** — the same DB today has
**21,040** non-null `result.score` rows and **3,758** at 0.5. The edges are
therefore tuned to a distribution the board no longer holds. Re-cutting them is
a design decision about the scale, not a formatting fix, so nothing was changed.
The same three figures appear at three separate call sites in `run.html`
(`:305`, `:310`, `:219`) with different numbers; `TODO.md` carries the item.

No measurement, manifest, bound, tolerance or `.db` was touched.
`leaderboard/.venv/bin/python -m pytest tests/leaderboard -q` -> **303 passed,
1 skipped**.

### The board's coverage: one bar, in problems, over the whole benchmark {#coverage-bar}

`coverage` and `problems` asked overlapping questions in incompatible units —
workloads passed out of 3,717, and problems swept clean out of 220 — and between
them could not answer "how much of this benchmark has that submission seen".
Worse, coverage was rendered **twice, once per score scope**, which implied that
what a run was given depends on how you divide its score. It does not.

One stacked bar per row now, always over all 220 problems, four states: **swept
clean · partly passed · attempted and nothing passed · never attempted**, the
last in the track colour at the right end so the grey tail is the unrun
benchmark and is comparable by eye down the column. **Widths come from the
counts unrounded**: four widths rounded to one decimal do not sum to 100%, and
the remainder shows as a hairline of track at the end of a bar that should be
full. Verified: every row sums to exactly 100.000000%.

`flagged` is **deleted from the board**. Every cell was a dash, the column
carried no explanation of what a flag would mean, and a column that has never
once been non-empty is a column the reader has to decode for nothing. The
reward-hack count is unchanged in `/api/v1` and on the per-problem pages.
(Note the interaction with [D41](#d41): the column read 0 partly because
`n_flagged` was structurally incapable of returning anything else.)

`tests/leaderboard/test_coverage_bar.py` (9 tests): the four states partition
the benchmark exactly; `attempted == total − untouched` both ways; every
reference variant has `untouched == 0`; every bar sums to 100% with no
zero-width segment; the bar carries no scope-keyed markup; the key is present;
`flagged` is out of the header and still in the API.

### One score, two scopes: the headline is a switch {#score-scopes}

The board printed two four-decimal numbers side by side — `benchmark score` and
`mean (attempted)` — ranked by the first and read as either. **Two denominators
under adjacent headings is not a disclosure; it is a thing to misread.**
Replaced by one score column whose denominator is a state of the table, selected
by a segmented control:

* **what it ran** (default) — divided by the workloads the submission was
  actually given. A failed attempt is still a zero; only workloads it never saw
  leave the denominator.
* **whole benchmark** — divided by all 3,717. Never-attempted counts as zero
  exactly like failed, and the row carries the `*` Partial mark.

Both numbers, both ranks and both coverage figures are rendered server-side into
the same cells; the switch changes which is shown, re-points `data-sort` so a
column sort still sorts what is on screen, and restores that scope's ordering.
With JavaScript off the default scope is fully rendered and the switch does
nothing. **The denominator is never computed in the browser** — that would be a
second implementation of the score, disagreeing in the fourth decimal.

The `problems` column now counts **problems attempted**, with problems swept
clean on the sub-line. It counted only the clean sweeps, so `torch.compile` read
as having seen 149 problems when it had run 218.

**This changes the default ranking, and the change is worth looking at
directly.** Under *what it ran*, the four-problem Opus-5 run sits at #1 with
0.7011 and `PyTorch eager` at #4 with 0.4541; under *whole benchmark* those are
#6 with 0.0111 and #1 with 0.4528. Neither order is wrong and neither makes the
rows comparable — a 59-workload denominator and a 3,717-workload one are
different questions, and the short one is usually the easier one.

`/api/v1/leaderboard` is unchanged in shape and **still ordered by `rank`**, the
full-benchmark scope: a consumer reading position N must not start getting a
different row because a page default moved. `rank_attempted`,
`problems_attempted` and `coverage_attempted` are additive.
`tests/leaderboard/test_score_scope.py`, 8 tests.

### The grid ramp: distribution and contrast, closed {#grid-ramp}

**Distribution**, over 12,883 scored results: u5 9.0 · u4 9.0 · u3 13.3 · u2
19.5 · u1 17.1 · b1 29.3 · b2 0.6 · b3 0.6 · b4 0.7 · b5 0.9 %. The prior
89%-in-two-buckets bunching is gone. **b1's 29.3% is structural, not a binning
error**: the reference variant that wins `T_b` scores exactly S = 0.5 there by
construction. *(The 12,883 figure is dated 2026-08-06 and is now stale — see
[D58](#d58): the same DB holds 21,040 non-null scores and 3,758 at 0.5. The
ramp edges have not been re-cut.)*

**Contrast**, recomputed independently from the tokens in `style.css`: every
ratio the file documents reproduces to the digit — dark beat
2.70/3.68/5.03/6.96/9.47 and under 2.70/3.67/5.06/6.87/9.36 against
`--panel #161a21`; light beat 1.74/2.18/2.76/3.44/4.36 and under
1.74/2.24/2.86/3.68/4.68 against `#ffffff`. The light floor of **1.74:1** is
below WCAG 1.4.11's 3:1 for non-text objects and is carried by the border on
every scored cell instead, measured at **3.44:1 and 3.68:1**. Polarity is never
on colour: `.g-u*` are circles, `.g-b*` squares, and a `forced-colors` block
restates all of it in system colours.

### The code pane had no test at all {#d27}

`leaderboard/static/highlight.js` (syntax highlighting, copy button) and the
`<pre>` that feeds it were untested. The pane is the only place a reader sees
the kernel that produced a number, and the copy button hands `pre.textContent`
to the clipboard — **so a template that mangles the source does not look wrong,
it *exports* something that was never run.** Escaped-once and escaped-twice
render identically for most sources; the check has to be mechanical.

`tests/leaderboard/test_code_pane.py` (4 tests): unescape the served pane and
compare byte-for-byte against the database, on a constructed hostile source and
on every real kernel. Verified against the built board — **36/36 panes
byte-identical, and all 36 stored `sha256` values match their own source.**
Mutation-checked: removing the protective leading newline after `<pre>` and
adding a `|trim` filter each fail it.

**Two things are asserted here and two are not:**

* **Not the tokenizer.** This node has no JavaScript runtime (`node`, `deno`,
  `qjs`, `d8` all absent), so `highlight.js` is unexecuted. Its guarantees are
  structural in that file — every emitted chunk passes through `esc()`, sticky
  regexes that cannot match ahead of the offset, a total fallback branch, and
  `data-hl` so a block is never scanned twice — and were **read, not run**.
* **Not the `hip` branch of language detection.** `run.html` picks `hip` when
  the source starts with `#include` and `python` otherwise. Checked against an
  independent signal (`__global__`, `extern "C"`, `#pragma unroll`, `template<`
  against `import`/`def`/`class`): **36 of 36 kernels and all 1175
  `variant_source` rows are Python, 0 disagreements.** So the heuristic is
  correct on everything the board holds and **the `hip` path has never been
  exercised by real data. The first HIP kernel submitted is the test.**

### Two vertical scrollbars on every code pane {#d30}

`pre.code.has-ln` split the axes: `overflow-y:auto` on the `<pre>`,
`overflow-x:auto` on the `<code class="src">` inside it. But **CSS Overflow 3 §3
forbids one axis staying `visible` while the other is not**, so `.src`'s
`overflow-y` computed to `auto` as well. Under the default `align-items:stretch`
that gave `.src` a definite 608px height and a scrollbar of its own, while the
gutter — same stretch, no overflow — spilled past the 640px cap and gave the
`<pre>` a second one. Dragging the outer bar then moved the line numbers and not
the code, which is the blank space it appeared to reveal: the gutter's overhang.

Now one scroll container, the `<pre>`, both axes; `align-items:flex-start` so
neither child is stretched, and the gutter is `position:sticky; left:0` so a
200-column line still cannot push the numbers off the left edge. The copy button
moved from `right:10px` to `22px` — it is positioned against `.codewrap`,
outside the scrollport, so it sat under the scrollbar of every pane long enough
to have one.

**Found from a screenshot, not from a test, and it stays that way:** verifying
it needs layout, and this node has no browser. `test_code_pane.py` asserts what
the pane *contains*, which is the part that can be checked here.

### The section nav was laid out by the header's CSS {#d32}

`style.css` carried `nav{display:flex;gap:4px;margin-left:auto}` and three
sibling rules, written when the site had exactly one `<nav>` and matched **by
element name**. The section nav added the day before is also a `<nav>`. It
inherited all four, laid seven block links out in a **row** inside a 210px grid
column, overflowed, and showed two of the seven behind a horizontal scrollbar —
on both pages, at every window size. **Nothing in the sidenav's own rules was
wrong and none of its tests could see it: they assert the served HTML, and the
HTML was correct.**

Scoped to `header nav`, with `.sidenav nav{display:block}` stated rather than
assumed, because a landmark element attracts element selectors and the second
one always arrives. `tests/leaderboard/test_coverage_bar.py` greps the served
stylesheet for a bare `^nav {` — the only mechanical check available here for a
cascade collision, and enough for this one.

Two sizing faults went with it. `.sidenav` had `overflow-y:auto`, so its
`overflow-x` computed to `auto` and produced that scrollbar — **the same CSS
Overflow §3 rule as [D30](#d30), twice in two days.** And the column was a fixed
210px against an elastic page: a quarter of the content at a narrow window,
labels wrapping for no reason at a wide one. Now
`clamp(9.5rem, 15vw, 14rem)`, with the sub-1080px layout a wrapping row of chips.

### Section nav on the two long reference pages {#section-nav}

`/methodology` and `/problems/<key>` are eight and seven sections of reference
material with no way to see the shape of the page or jump within it. Both carry
a sticky left nav, **server-rendered** from `TOC_METHODOLOGY` / `TOC_PROBLEM` in
`app.py` — not scraped from the DOM, so it is there with JavaScript off and a
test can check it. The scroll spy that marks the current section is the only
part that needs JS, and its absence costs nothing but the highlight.

`tests/leaderboard/test_sidenav.py` (7 tests) asserts every `href="#x"` resolves
to an `id="x"` in the body, that the nav is in document order (the spy assumes
it), that no `h2` is missing from the nav, and that a page passing no `toc`
still renders single-column. **Two files with no compiler between them would
otherwise drift into links that render, look live, and scroll nowhere.**

The highlight later vanished mid-section on `/methodology`'s longest section:
the spy asked "which heading is inside a band 59px to 30% down" (an **event**)
where the question is "which heading did I last pass" (a **position**). Any
section taller than the band has an interval where the answer is "none" and the
nav renders nothing. **Its own comment described the position rule; the code
implemented the event rule.** Rewritten as geometry on a rAF-coalesced passive
scroll listener; fixes `/methodology`, `/problems/<key>` and the run page.

---

# 7. Dataset, packaging and the environment

### The dataset ships as parquet, not per-problem directories {#d1}

(Carried from session 1, still true.) The Hub publishes
`data/{L1,L2,Quant,FlashInfer-Bench}.parquet`, one row per problem.
`scripts/materialize_dataset.py` is the exact inverse of the dataset's own
converter and round-trip-verifies all 235.

**Found in session 2:** the materializer wrote `reference` only to
`reference.py`, **not into `definition.json`**. `Definition` declares `reference`
as a required field, so *every* problem failed to load with a pydantic
`Field required` error the first time a runner touched one. The audit described
the directory *contents*; it did not imply a different schema. Now written to
both places, with the round-trip check comparing them so they cannot drift.

The census is confirmed against the files, not taken from the paper: **L1 94,
L2 82, Quant 33, FlashInfer-Bench 26 = 235.**

### Nine FlashInfer-Bench problems need a second, separate dataset {#d5}

(Carried from session 1.) **304 blobs** from `flashinfer-ai/flashinfer-trace`,
fetched by `scripts/fetch_flashinfer_traces.py`, and `FLASHINFER_TRACE_DIR` must
be set. Both confirmed working on this node. Without those blobs, 9 of the 26
FlashInfer problems fail at run time as **ordinary runtime errors** — which
looks like a port defect and is not one. This is an operational precondition,
restated in `README.md` and `CLAUDE.md` §8.

### The vendored data-model package was never committed {#d6}

`.gitignore` contained `data/` **unanchored**, which matches
`src/sol_execbench/core/data/` and `tests/sol_execbench/core/data/` as readily
as it matches the dataset directory it was written for. Session 1's commit
therefore **silently omitted nine source files and five test files** that the
code imports — `Definition`, `Workload`, `Solution`, `Trace`, the dtype map, the
whole schema layer. Session 1's tests passed because the files existed in its
working tree; git simply never took them.

Recovered by re-vendoring from upstream at the pinned SHA (`a9fa080`) and
re-applying the AMD delta (`hip_cpp`/`ck`/`ck_tile`/`hipblaslt`/`miopen`/`aiter`
languages, `MI350X`/`MI355X` hardware, `.hip` entry points). Pattern changed to
`/data/`, anchored, with a comment saying why.

**Worth stating plainly because the failure mode generalizes:** the tests were
green, the working tree was correct, and the artifact that would have been
shipped was missing a third of the port. **Nothing in the session-1 workflow
could have caught it — only a fresh clone could.** (Compare [D49](#d49), which
is the same lesson arriving from the deployment side.)

### One of SOLAR's own torchview patches is malformed and unnecessary {#d7}

SOLAR ships two patches for torchview. `torchview-collect-attributes.patch` is
**corrupt** — its first hunk header declares 9 lines and supplies 8 — so
`git apply` and GNU `patch` both refuse it, and SOLAR's own `install.sh`
silently skips it on failure.

Investigated rather than skipped, because a silently-dropped patch that mattered
would have left reduction-op attributes (`dim`/`keepdim`) uncaptured and quietly
changed every analysis. **It does not matter: both of its changes are already
present upstream** in torchview at the commit SOLAR pins, written with
`functools.partial` where the patch used a lambda. `env/Dockerfile` now
**asserts** both changes are present rather than assuming, so a future torchview
bump that drops them fails the build instead of producing subtly wrong bounds.

### This node was exclusively ours — on 2026-08-03 {#d2}

The MI355X node was shared (another user, another container). This one was not:
no other logins, no KFD processes. The node-wide clock lock was therefore safe
to leave in place, and the sibling-power contamination flagging in
`clock_calibrate.py floor` reported no busy siblings during any tail window.

> **This is history, not a standing property, and reading it as one is how
> [D61](#d61) happened.** On **2026-08-12 at 18:39 and 18:40** two foreign
> tenants started on this node (`sglang::scheduler`, 194 GB resident, and
> `ray::MegatronTrainRayActor`), with GPUs 0–3 at 100% utilisation, and they
> invalidated two "solo GPU 0" measurements. **Exclusivity must be checked at
> the time of measurement, by `scripts/gpu_exclusive.py`, not inferred from this
> entry or from having seen the card idle earlier.**

### `roofline_probe.py` hardcoded MI355X spec peaks {#f14}

It printed "spec peak 2500 @ 2.4 GHz" on an MI350X, whose peak is **2307 @
2.2 GHz** — same die, lower clock. Comparing an MI350X measurement against the
MI355X peak understates the achieved fraction by 9%. Now resolved per part from
`solexbench_rocm/parts.py`, and the achieved fraction is written into the
artifact so no reader has to infer the denominator.

Rooflines at **default** clocks (reference points only — per task 00's guard
rails these are **not** scoring ceilings and must not be cited downstream):

| | MI350X (this node) | MI355X (session 1) |
|---|---|---|
| HBM copy | 4.53 TB/s (56.7% of 8.0 spec) | 4.87 TB/s (61%) |
| BF16 GEMM | 1168 TFLOPS (50.6% of 2307 spec @2.2 GHz) | 1433 TFLOPS (57% of 2500 @2.4 GHz) |

### `sol_bounds.py` tripped over scalar inputs and killed 41 of 94 L1 problems {#f15}

`get_input_shapes` returns `None` for a scalar input (e.g. a dropout
probability). Iterating that raised `TypeError: 'NoneType' object is not
iterable` and killed **41 of 94** L1 problems before any bound was computed.
Scalars are now passed as **Python numbers**, which is also semantically
required: the reference uses them in control flow, where a meta tensor would
silently change the traced graph. (The same fact drives half of [D14](#d14)'s
fix.)

### SOLAR's pipeline needs per-problem process isolation {#f16}

Stage 1 traces arbitrary reference code, and some references trace
pathologically. In a `ProcessPoolExecutor` **a stuck worker cannot be
cancelled**, so one bad problem stalls the sweep behind it — the classic way a
"finished" sweep silently covers 200 problems instead of 235. Each problem now
runs as a killable subprocess with a timeout, and **a timeout is recorded as a
result.**

### The static source screen could not see a startup hook {#f19}

`static_source_screen()` scanned file *contents* only. Python imports some names
automatically at interpreter startup — `sitecustomize.py`, `usercustomize.py`, a
`.pth` line beginning with `import`, and `conftest.py` under pytest — **before
any runtime guard has installed itself and outside every timed region.** A
submission shipping one of those executes code the harness never invoked, which
is the same escape as monkey-patching.

**A content scan cannot catch it, because the content does not have to look
suspicious.** The case that surfaced this shipped a two-line `sitecustomize.py`
defining `enum.StrEnum` to work around an interpreter older than
`requires-python`. It was not malicious, did not touch the numerics, and was
arguably a repair — **it was also the difference between a problem scoring 16/16
and not evaluating at all**, which is more leverage than any submission should
hold outside the record.

The screen now checks the filename as well as the contents. False-positive
anchors are in `reference/exploits/test_source_screen.py`: all 235 dataset
references still pass, as do near-misses like `my_conftest.py`,
`site_customize.py` and `path.py`.

`.pth` in `_PATH_HAZARDS` flags any such file, where the hazard is specific to
site directories and `.pth` is also the conventional PyTorch checkpoint suffix.
**Left as-is deliberately:** submissions here carry source text, a `.pth` among
them is never legitimate, and the screen reports rather than raises.

### Stale artifacts read as fresh findings {#d10}

Half an hour was spent classifying 52 "SOLAR failures" that turned out to be
records written before the dataset was re-materialized and before the container
image was fixed. Re-running one by hand produced a *different* error, which is
the only reason the staleness was noticed at all.

The scratch directory is keyed by problem, **not by (problem, code version)**,
so a failure recorded by an older build looks exactly like one recorded by the
current build. `--resume` re-runs failures precisely so they refresh — but
anything that *reads* the scratch directory mid-sweep is reading a mixture.

**Standing rule: triage failures from the artifact the sweep wrote at the end,
never from scratch state while it is still running.**

---

# 8. Checks that could not fail, and sweeps that could not report failure

Five separate gates in this repo have been found asserting nothing. They are
grouped because the pattern is the finding: **a check nobody has watched fail is
not evidence.** The countermeasure used throughout is a negative control — make
the check fail on purpose before believing it when it passes.

### A task-01 check that could not fail {#f17}

`f_lock_from_state()` matched `F_LOCK.*?(\d{3,4})` against the whole of
STATE.md — i.e. the first number following the first mention of F_LOCK anywhere:
a prose sentence, a table cell, a deviation write-up, whichever came first. That
is fine while the file documents one part. On a node whose STATE.md still
discussed the other part's bound it resolved to the wrong number and reported:

```
[PASS] F_LOCK recorded in STATE.md                1300 MHz
[PASS] F_LOCK at or below lowest observed floor   F_LOCK 1300 <= min p5 1724
```

**Both green, and neither could have failed:** 1300 clears a 1724 floor so
comfortably that no wrong answer would ever trip it. **The check protecting the
most consequential measurement in the project was inert.**

The match now requires a canonical bold `F_LOCK = <n> MHz` marker line in
STATE.md, and a new check compares it against `CLOCK_LOCK_PRESETS` — the value
the code actually applies and stamps. A document and a constant that disagree
about the frequency every bound is expressed at is exactly the failure nothing
downstream can detect.

**Constraint on anyone reformatting STATE.md:** that marker's exact form is
load-bearing — the pattern requires `MHz**` immediately after the digits, and it
must be the only occurrence of that form in the file. STATE.md says so at the
marker.

### The tightening failed its own check, and silently deleted a physics check {#f20}

The PR contributing [F17](#f17) stated that master's marker line already
matches. **It did not**: master's line read
`**F_LOCK = 1300 MHz achieved, at determinism setting 1600.**`, and the pattern
requires `MHz**` immediately after the digits. The regex matched nothing on
`b1c53dc`, on `654864c`, on the PR head itself, or on current master, **so
`--task 01` failed on the PR's own branch.** Fixed by restructuring the
*Decisions taken* line to lead with the bare marker.

**Worse, the tightening silently removed a working check.** The floor comparison
is guarded by `if p5s and fl:`, so a missing marker took `F_LOCK at or below
lowest observed floor` out of the run entirely — **task 01 went from 8 checks to
7, losing the one that catches a clock the GPU cannot hold.** It now falls back
to the preset, so a documentation defect can no longer delete a physics check.

### `check_06` asserted a schema that was never produced {#f22}

It required `artifacts/06/t_b.json` with a `problems` map, and
`anchor-verification.md`. Task 06 writes one file per problem under
`authoritative/` keyed by `winner_by_workload`, and the anchor result as
`.json`. So **`t_b.json exists` has failed on every run this repo has ever
had**, while STATE.md recorded task 06 as done. **The mirror image of
[F17](#f17): not a check that could not fail, but one that could not pass.**

Rewritten against the real layout and extended with [F18](#f18)'s one-clock
invariant at acceptance time. It now reports 220/220 problems anchored at a
single F_LOCK, and surfaces [D15](#d15) as a WARN (336/349) rather than silence.

**A defect introduced inside the audit, recorded because it took thirty seconds
to introduce:** while rewriting it, the anchor check was keyed on `n_failed`, a
field that does not exist in the artifact, which resolved to `None` and printed
"every checked workload within tolerance" **over 13 real failures** — the
audited defect, reproduced inside the audit.

### `check D` was a literal unconditional PASS {#f23}

The line was

```python
c.add(JUDGE if "PENDING" in text else PASS,
      "check D: T_SOL <= best measured", "needs task 06")
```

`cross-checks.md` contains no `PENDING` and no check-D section, so **this passed
always and compared nothing.** It is also **the one invariant that would have
caught [D18](#d18).** It now compares T_SOL against every measurement under
`artifacts/10/*/scored.json` and reports, correctly:

```
[FAIL] check D: no measurement beats its T_SOL   25 of 115 measured workloads are
       faster than T_SOL (worst 0.29x the bound) across 1 problem(s):
       FlashInfer-Bench__019_mla_paged_prefill — the bound is wrong (D18)
```

With no submissions on disk it reports JUDGE rather than PASS, **because the T_b
variants cannot falsify a bound that is too slow** — the reference over-reads
exactly the way the bound does.

This is the one gate that still fails today, and deliberately: it reads manifest
**v1**, the frozen release artifact, which is meant to go on reporting what v1
shipped. It is not a live signal about the board, which serves v1.2. **A second
failure anywhere is a regression.**

### `check_07` required a write-up, not evidence {#check-07}

It asserted `artifacts/07/fp8-validation.md`, never written, and so failed
permanently while the validation it stands for had been done. It now checks the
evidence — **all 18 non-NVFP4 Quant problems pass every workload in the task-02
reference sweep, 18/18** — and warns separately that no summary document exists.

### 38 dead tests behind a skip that read like a scheduling choice {#d19}

`pytest tests/` reported `75 skipped`. 63 of those carried the `timing_serial`
marker, which `conftest.py` auto-skipped **unconditionally**. Running them showed
**40 of 63 failing.** Two separate causes, neither visible from the skip line:

* **38 are CUPTI tests.** `timing.py` imports cupti *lazily* — correctly, so the
  module stays importable on ROCm — which means these collect fine here and then
  fail at call time with `ModuleNotFoundError` instead of being skipped.
  `_NVIDIA_ONLY_TEST_FILES` never caught them because the file imports cleanly.
  CUPTI has no ROCm build, so no hardware makes them pass; the AMD path is the
  task-04 rocprofiler shim. Now skipped by class, with that reason.
* **The marker's own instruction did not work.** The skip reason printed 63
  times said to run `pytest tests -m timing_serial -n 0`. `-n 0` is a
  pytest-xdist flag and **xdist is not installed in the pinned image**, so that
  command errors. **The one pointer at the largest block of unrun tests was
  dead.**

The blanket skip was also wrong in intent: `timing_serial` exists because these
measure GPU wall-clock and a co-scheduled worker corrupts them — a reason to
skip *under parallelism*, not always. Gated on actual xdist now, so on an idle
node they run. `pytest tests/` went from **483 passed / 75 skipped** to **503
passed / 55 skipped**, 0 failed. The two remaining non-CUPTI failures are
[D20](#d20).

### `shard_sweep` counted 235 crashes as "235 ok, 0 failed" {#d57}

Found by walking into it: a re-sweep was first launched on the host instead of
inside the measurement container, every one of the 235 runners raised
`ModuleNotFoundError: No module named 'pydantic'`, `run_guarded` did its job and
wrote the traceback into each artifact **and exited 0** — and `shard_sweep.py`
reported success, because it counted the exit status. **A sweep that failed
everywhere was indistinguishable from one that worked**, which is exactly the
omission `CLAUDE.md` §0 describes. `_run_on` now reads the artifact it just
wrote and counts `ok: false` as a failure.

**Audited afterwards, since the check is cheap:** every `*.json` under
`artifacts/` carrying `ok: false`. Nothing in `artifacts/00`–`artifacts/09`
does, so **no sweep-produced artifact in the tree was hidden this way.** The 44
hits are all legitimate recorded failures of a different kind — 43 are single
iterations inside agent `trajectory/` directories, where an agent's own kernel
failing to run IS the result, plus one `glm-run1/retimed` timeout at 1200 s
([D23](#d23)).

---

# Numeric index

Sorted numerically, for a reader arriving from a code comment. **D-numbers are
stable identifiers and are never renumbered or reused** — 29 of them are cited
from code across 168 call sites, and a few citations name `TODO.md` rather than
this file (notably D29), which is why `TODO.md` keeps its filename.

| D | heading | anchor |
|---|---|---|
| D1 | The dataset ships as parquet, not per-problem directories | [#d1](#d1) |
| D2 | This node was exclusively ours — on 2026-08-03 | [#d2](#d2) |
| D3 | D3 is the question D55 answers (pointer; no entry was ever written) | [#d3](#d3) |
| D4 | *— no entry; a gap in the numbering, cited from nowhere* | — |
| D5 | Nine FlashInfer-Bench problems need a second, separate dataset | [#d5](#d5) |
| D6 | The vendored data-model package was never committed | [#d6](#d6) |
| D7 | One of SOLAR's own torchview patches is malformed and unnecessary | [#d7](#d7) |
| D8 | Determinism mode does not do what its name suggests | [#d8](#d8) |
| D9 | The tolerance runner's memory profile, and one absurd allocation | [#d9](#d9) |
| D10 | Stale artifacts read as fresh findings | [#d10](#d10) |
| D11 | The shard runner could put two timing runs on one GPU | [#d11](#d11) |
| D12 | T_SOL was truncated to whole cycles, and eight truncated to zero | [#d12](#d12) |
| D13 | `masked_select` asks for 16781313 GiB above 2³² elements | [#d13](#d13) |
| D14 | The bound was priced at the vector-FP32 rate on 160 of 235 problems | [#d14](#d14) |
| D15 | One problem's T_b does not reproduce to 3% | [#d15](#d15) |
| D16 | The agent pilot billed the wrong gateway key | [#d16](#d16) |
| D17 | The scorer wrote into the container and scored every kernel zero | [#d17](#d17) |
| D18 | The bound prices the allocation, not the work — paged attention | [#d18](#d18) |
| D19 | 38 dead tests behind a skip that read like a scheduling choice | [#d19](#d19) |
| D20 | Matmul timing spread is bimodal, and the cause is unknown | [#d20](#d20) |
| D21 | Two bounds a real kernel beat, and neither was D18's mechanism | [#d21](#d21) |
| D22 | A failed workload was carrying a score | [#d22](#d22) |
| D23 | A submitted kernel whose re-time timed out is invisible | [#d23](#d23) |
| D24 | The same "dropped the external run" bug, three times | [#d24](#d24) |
| D25 | `f_lock_mhz: null` was blamed on a preset that exists | [#d25](#d25) |
| D26 | Three tables ranked means with different denominators | [#d26](#d26) |
| D27 | The code pane had no test at all | [#d27](#d27) |
| D28 | The board scored a whole problem by one flag | [#d28](#d28) |
| D29 | The external fleet ran 34 jobs on GPU 0 | [#d29](#d29) |
| D30 | Two vertical scrollbars on every code pane | [#d30](#d30) |
| D31 | The first near-full-benchmark agent run, and 7 more bad bounds | [#d31](#d31) |
| D31b | The run is complete: 220/220, and it leads the board | [#d31b](#d31b) |
| D31c | A second model found a bound that 220 problems of the first did not | [#d31c](#d31c) |
| D32 | The section nav was laid out by the header's CSS | [#d32](#d32) |
| D33 | `agent_score.py --timeout` never reached the evaluation | [#d33](#d33) |
| D34 | The four baselines' pages said their code was not recorded | [#d34](#d34) |
| D35 | F_LOCK is a floor, not a lock, and most "wrong bounds" were that | [#d35](#d35) |
| D36 | Manifest v1.1: what the two corrections actually moved | [#d36](#d36) |
| D37 | SOLAR prices a grouped convolution as a dense one | [#d37](#d37) |
| D38 | The timer never saw a submission's own stream | [#d38](#d38) |
| D39 | The defect class nothing checks: a bound far below anything achievable | [#d39](#d39) |
| D40 | The first reward hacks on this board, and they were already published | [#d40](#d40) |
| D41 | gpt-5.6-sol over all 220, and the counter that could not count | [#d41](#d41) |
| D42 | The five surviving bad bounds are two causes, and one is D18 again | [#d42](#d42) |
| D43 | BLOCKED: rocprofv3 counter collection hangs in this container | [#d43](#d43) |
| D44 | A category filter that filtered the rows and not their labels | [#d44](#d44) |
| D45 | An unknown category rendered a full board of zeros | [#d45](#d45) |
| D46 | Every per-workload `axes_json` in the database was empty | [#d46](#d46) |
| D47 | A workload's parameters were only the axes that vary | [#d47](#d47) |
| D48 | Workload identity was an 8-char uuid prefix corresponding to nothing | [#d48](#d48) |
| D49 | A board built from a bare clone failed silently in five different ways | [#d49](#d49) |
| D50 | torch.compile silently stopped compiling after the 8th shape | [#d50](#d50) |
| D51 | The tolerance is a bit-identity-with-eager test | [#d51](#d51) |
| D52 | A zero tolerance leaked from an integer output onto float outputs | [#d52](#d52) |
| D52b | The same leak between two float dtypes, 65536x wide | [#d52b](#d52b) |
| D53 | The float64 goldens were drawn from a different RNG | [#d53](#d53) |
| D54 | A stale pre-split database that nothing serves | [#d54](#d54) |
| D55 | Does the lock cost us anything? On this part, no — it never binds | [#d55](#d55) |
| D56 | The recompile cliff is fixed, and the failure count goes up | [#d56](#d56) |
| D57 | `shard_sweep` counted 235 crashes as "235 ok, 0 failed" | [#d57](#d57) |
| D58 | One number system for the board, and three figures that were not measured | [#d58](#d58) |
| D59 | A T_b re-sweep, and what it does and does not establish | [#d59](#d59) |
| D60 | What the T_b non-reproduction is NOT | [#d60](#d60) |
| D61 | CORRECTION: the re-measurements were not on an exclusive card | [#d61](#d61) |

**F-numbers** (fixes to scripts on first contact). F1–F11 are session-1 fixes on
MI355X; they all still hold and their detail is in git history.

| F | heading | anchor |
|---|---|---|
| F12 | `gen_golden.py` assumed a `get_inputs()` that does not exist | [#f12](#f12) |
| F13 | fp64 promotion breaks dtype-literal references; goldens record their tier | [#f13](#f13) |
| F14 | `roofline_probe.py` hardcoded MI355X spec peaks | [#f14](#f14) |
| F15 | `sol_bounds.py` tripped over scalar inputs and killed 41 of 94 L1 problems | [#f15](#f15) |
| F16 | SOLAR's pipeline needs per-problem process isolation | [#f16](#f16) |
| F17 | A task-01 check that could not fail | [#f17](#f17) |
| F18 | A manifest built from a directory holding two clocks | [#f18](#f18) |
| F19 | The static source screen could not see a startup hook | [#f19](#f19) |
| F20 | The tightening failed its own check, and deleted a physics check | [#f20](#f20) |
| F21 | The clock guard failed open off-GPU | [#f21](#f21) |
| F22 | `check_06` asserted a schema that was never produced | [#f22](#f22) |
| F23 | `check D` was a literal unconditional PASS | [#f23](#f23) |
| F24 | The determinism setpoint, read back off the hardware | [#f24](#f24) |

**Findings with no number**, listed so nothing is unreachable:

| topic | anchor |
|---|---|
| `check_07` required a write-up, not evidence | [#check-07](#check-07) |
| The NVIDIA B200 overlay, and what it may and may not be used for | [#b200-overlay](#b200-overlay) |
| The board's coverage: one bar, in problems | [#coverage-bar](#coverage-bar) |
| One score, two scopes: the headline is a switch | [#score-scopes](#score-scopes) |
| The grid ramp: distribution and contrast, closed | [#grid-ramp](#grid-ramp) |
| Section nav on the two long reference pages | [#section-nav](#section-nav) |

---

## Where the retractions are

Nine places in this file print a claim next to the evidence that withdrew it.
They are listed here because quoting the withdrawn half is the single easiest
mistake to make from an archive:

1. **[D59](#d59) / [D60](#d60) retracted by [D61](#d61)** — the 2.021x is not a
   reproduction gap. What survives: the old-SHA/new-SHA exoneration, and the
   pre-tenant 2.45x.
2. **[D37](#d37)** carries an in-place retraction of its own second half —
   `L2__036` was never in scope and is not the witness for anything.
3. **[D25](#d25)** *is* a retraction: three documents asserted a missing preset
   that has existed since 2cdb7b0. Its own count of 28 is now incomplete (37/45).
4. **[D54](#d54)** carries its own correction — the live board was never showing
   inverted statuses; the leftover file was.
5. **[D52](#d52)** carries an attached correction: the three all-integer
   problems do *not* pass every compile variant (v5 fails all of them), though
   the inference survives.
6. **[D28](#d28)**'s gap-fill doubt is inverted by [D50](#d50)/[D59](#d59) — the
   523 was a floor, not an over-count.
7. **[D36](#d36)** and **[D35](#d35)** each carry a dated in-place correction of
   a figure generalised from the wrong subset.
8. **[F18](#f18)** is retracted in part by **[F24](#f24)**, and carries the
   retraction inline.
9. **[D58](#d58)** retracts three figures that were stated as measured; the
   grid-ramp section's 12,883 is the one that is still stale downstream.
10. **[D31c](#d31c)**'s "the count going up is the finding" is narrowed by
    [D37](#d37): the worst bounds are the ones nothing can reach.
