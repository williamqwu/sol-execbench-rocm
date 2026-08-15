# MI355X bound quality — open issues

**Status of the benchmark itself: shipped.** `artifacts/09-MI355X/manifest-v3.json`
carries 220/235 problems and 3717 scoreable workloads, coverage is 235/235, and
seven of eight acceptance gates report zero failures. Nothing below blocks using
the benchmark. Everything below is about whether individual **bounds** are right.

**How to read this document.** Every number here was measured and the command
that produced it is given. The *interpretations* are flagged as such and you
should not inherit them — several of yesterday's confident interpretations were
wrong in ways that took an adversarial reviewer to find, and two of them were
wrong in the same direction. Re-derive from the evidence. Where this document
says "hypothesis", treat it as a thing to falsify, not a thing to implement.

Read `CLAUDE.md` first; its prime directives govern. The most relevant here:

> **A self-consistent bound and anchor cannot detect a shared error.** `T_b`
> comes from a PyTorch reference that over-reads exactly where the
> declared-traffic bound over-counts, so the `T_SOL <= T_b` gate passes while
> both are wrong. Only an independent kernel separates them.

That sentence is the whole subject of this document.

---

## Background you need, in one page

A score is `S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))`.

* `T_SOL` — the roofline bound. The fastest the hardware could possibly run this
  workload. **A lower bound on time.**
* `T_b` — a measured PyTorch reference implementation. The anchor: its own
  implementation scores 0.5.
* `T_k` — the measured time of a submitted kernel.

`T_SOL` is the max of two independently derived tiers:

| tier | file | what it prices |
|---|---|---|
| `solar_fused` | `artifacts/03-MI355X/t_sol.json` | SOLAR roofline over the traced graph |
| `declared_traffic` | `artifacts/03-MI355X/t_sol_traffic.json` | every declared input read once, every output written once, over DRAM bandwidth |

`scripts/build_manifest.py::combine_bounds` takes the max and **rejects** a tier
whose `T_SOL` exceeds the measured `T_b` (physically impossible).

### The two error directions are NOT symmetric

This is the single most important thing on this page.

**T_SOL too LARGE (over-counted work).** The bound claims a kernel cannot go
faster than X, and a kernel does. **Detectable** — a real submission falsifies
it, and `check D` fires. This is what D18 was.

**T_SOL too SMALL (under-counted work).** The bound is loose. `S` is inflated for
everyone. **Undetectable by any measurement**, because nothing is supposed to be
faster than the bound and nothing is. The `T_SOL <= T_b` gate still passes. The
only handle is an *a priori* floor — the declared traffic every correct kernel
must move — which is what `check A` exists for.

**Most of the remaining work is on the undetectable side.** Yesterday's session
spent most of its effort on the detectable side because that is the side that
shouts.

---

## Issue 1 — the SOLAR tier looks like it has D18's defect too

**This is the highest-value item and it is on the undetectable side.**

D18 was: the declared-traffic tier priced a paged KV cache at its **whole
allocation** (`num_pages` rows) when the workload only names `num_kv_indices` of
them. Fixed in `scripts/sol_gathered_traffic.py`, verified to nine significant
figures, check D went 102 violations → 28.

The evidence that SOLAR does the same thing, from the D18 investigator's own
residual-risk note (**not independently confirmed — confirm it first**):

```
FlashInfer-Bench__018:
  SOLAR memory_bytes            1,140,133,554
  traffic tier's own allocation 1,140,133,608     <- differ by 54 bytes
  proposed correction:  solar_bytes - (allocation - gathered) = 44,082
                        against the tier's corrected 44,136
  estimated blast radius: 47 workloads
```

Now that the traffic tier is corrected, `max_of_both` takes SOLAR's number on
these — so the fix moved the binding tier onto an uncorrected one.

**Why check D cannot see it:** no MI355X submission beats these bounds *yet*.
Absence of a falsifying kernel is not evidence the bound is right; it is the
definition of the undetectable direction.

**Questions to answer from first principles**
1. Does SOLAR resolve gathered/paged inputs at full allocation? Read the SOLAR
   bridge and the traced graph, do not infer it from the 54-byte coincidence.
2. If yes, is the right fix in the bridge, in the arch config, or a
   post-processing correction like the traffic tier got? Argue the choice.
3. What is the real blast radius? 47 is an estimate, not a count.

**Do not** apply the traffic tier's correction to SOLAR by analogy. The two
tiers count different things; SOLAR traces an actual graph.

---

## Issue 2 — five workloads are still falsified by real kernels

Real agent kernels, correctness-passed under AMD tolerances, running faster than
their own published lower bound. These bounds are wrong. Scores on these three
problems are not results.

```
0.706x  L2__050_vae_decoder_mid_block_attention_resnet          tier=solar_fused
        uuid=f009abdb-585e-590b-b599-1c3d73e4e6b6  harness=codex
        t_k=1.06753 ms   T_SOL=1.51224 ms   T_b=2.54835 ms

0.792x  FlashInfer-Bench__015_gqa_paged_prefill_causal_h32_kv8_d128_ps1
        uuid=a94c44ab-5899-419f-9c79-9898fda0e173  harness=codex   tier=declared_traffic
        t_k=0.02680 ms   T_SOL=0.0338189 ms   T_b=9.66659 ms

0.840x  FlashInfer-Bench__015 (same problem)
        uuid=3b672ff1-11ba-4d30-bd7e-303472aecf0b  harness=codex   tier=declared_traffic
        t_k=0.01812 ms   T_SOL=0.0215636 ms   T_b=3.83657 ms

0.920x  FlashInfer-Bench__014_gqa_paged_prefill_causal_h32_kv4_d128_ps1
        uuid=3e553162-e3e4-446c-b72d-546bbf07c495  harness=codex   tier=declared_traffic
        t_k=0.03110 ms   T_SOL=0.0338175 ms   T_b=6.83707 ms

0.971x  FlashInfer-Bench__014 (same problem)
        uuid=d14e12cc-4fd1-43d2-a156-4e784c4f252d  harness=codex   tier=declared_traffic
        t_k=0.0316405 ms  T_SOL=0.0325829 ms  T_b=69.628 ms
```

**Hypothesis on record for the four FlashInfer ones, from the D18 investigator
(unconfirmed):** after the paged correction these workloads are dominated by `q`
and the output at `total_q=16384` (~270 MB), and the kernel's implied bandwidth
against the corrected bound is 10.09 and 8.70 TB/s against an arch DRAM peak of
8.0 TB/s. A 270 MB working set **fits inside the 268 MB Infinity Cache**, so the
claim is that this is the LLC-bandwidth question (Issue 6), not an allocation
being mispriced.

Test that before acting on it. If true, the bound is using DRAM bandwidth for
traffic that never reaches DRAM, and the fix is a bandwidth model, not a byte
count. **Changing the roofline bandwidth constant is a methodology change**
(prime directive 7) — raise it, do not improvise it.

`L2__050` is `solar_fused` and is a separate question. It may be Issue 1.

**Reproduce**
```bash
env/solb python scripts/verify_artifacts.py --task 03 --part MI355X \
  --manifest manifest-v3.json          # check D
```

---

## Issue 3 — published bounds below the declared-traffic floor

The declared traffic is the minimum bytes any correct kernel must move: every
declared input read once, every output written once. A published bound below
that floor is below the unavoidable minimum and is not a bound.

**Two counts exist and they disagree. Establish which question you are asking
before you fix anything.**

| count | source | what it counts |
|---|---|---|
| **120** workloads | `check A-published` in `verify_artifacts.py --task 03` | excuses a floor that is itself refuted by measurement (floor > measured `T_b`) |
| **193** workloads / 61 problems | direct comparison of `t_sol_ms_published` against `t_sol_traffic.json` | no excusals |

By tier, on the 193: `max_of_both` 86, `solar_fused` 56, `declared_traffic` 51.

Worst offenders (ratio of published bound to its own floor):

```
0.0540x   L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion   solar_fused
0.0821x   L1__057   (same problem, 8 workloads affected)
0.1733x   L1__057
```

Most affected problems: `FlashInfer-Bench__017` (12), `__015` (10), `L1__057`
(8), `__014` (7), `__019` (7), `__021` (7).

**Note the contradiction worth chasing:** `L1__057` appears here as a bound
*below* its floor (too loose) and also appeared yesterday in the `T_SOL > T_b`
set (too tight). Both cannot be true of the same workload. Either the two
measurements are of different workloads within the problem, or one of the two
checks is wrong. Nobody has resolved this.

**Also unresolved:** `check A` reports 1021 workloads where SOLAR's memory term
alone is below declared traffic. An investigator concluded that **zero** of those
ship an invalid published bound, because `max_of_both` rescues them. An
adversarial reviewer agreed with that negative result. So `check A` is red by
construction and is a judgement item; `check A-published` is the gate. Verify
that reasoning rather than assuming it — it is load-bearing for the claim that
1021 red rows are harmless.

---

## Issue 4 — the two tiers express cycles at different clocks

Recorded as **D63** in `docs/findings.md`. Partially fixed; the underlying
inconsistency is still in the artifacts.

* `artifacts/03-MI355X/t_sol.json` — `t_sol_ms` is computed at **f_ref = 1.8 GHz**.
  Its sibling field is literally named `memory_cycles_at_f_ref`.
* The same file's header declares **`f_lock_mhz: 2400`**.
* `artifacts/03-MI355X/t_sol_traffic.json` converts at **2.4 GHz**.

So the file contradicts its own header, and the two tiers disagree with each
other, and nothing in either file says so.

This has already caused two separate defects:

1. `check B` divided `t_sol_cycles` by the arch's 2.4 GHz and reported **1327**
   workloads implying DRAM bandwidth above peak. Every reading was exactly
   1.33x = 2.4/1.8. All 1327 were phantoms. Fixed by comparing in time.
2. `combine_bounds` picked the binding tier by comparing **cycles** across the
   two tiers — i.e. comparing numbers on different clocks, favouring one by
   1.333. Fixed by comparing in time. Consequence, measured:

```
solar_fused 1541 -> 2282     max_of_both 1292 -> 589
published bounds: 475 moved DOWN (looser), 0 moved UP, 3226 unchanged
```

Most of the 475 is the legitimate D18 paged correction. The rest is the tier
swap. **All movement is in the loosening direction**, which is the undetectable
one, and it shipped without being stated until an adversarial reviewer diffed
`bound_sources`.

**The remaining work:** the artifacts still carry the inconsistency. A third
consumer will hit it. Either make `t_sol.json` state its own f_ref honestly, or
convert it, or stop publishing a cycle count that needs an out-of-band clock to
interpret. **Cycles on this part are only meaningful next to the clock they were
expressed at** — that is the general lesson and it is not yet enforced anywhere.

---

## Issue 5 — scoring and anchoring shard independently, so cards align by luck

`STATE.md` §4.4: `T_b` and `T_k` must be measured **on the same physical card**,
and scoring enforces it with no bypass.

But the authoritative T_b pass and the scoring pass each shard `plan[i::8]` over
**their own sorted plan**, and the plans are different. So a problem's anchor and
its kernel timing land on the same card only by coincidence.

Measured: backfilling scores against the merged anchor tree refused **178 of 220
problems** on card mismatch. Making `--tb-artifacts` repeatable so the anchor is
taken from whichever tree holds the matching card brought it to **139**. The
remaining 139 have no anchor on the card their `T_k` was measured on at all.

A re-anchoring run is in flight (`artifacts/06-MI355X/authoritative-oncard/`,
149 problems across 9 cards, staged from
`/var/tmp/solbench/reanchor-plan.json`). That is a workaround, not a fix.

**The fix is to make the two passes share one card assignment**, so that a
problem's anchor and its kernel timing are pinned to the same card by
construction. Nobody has designed that. Note this affects *scoring agent runs*,
not the benchmark's own bounds.

---

## Issue 6 — LLC bandwidth (the V2 judgement item)

MI355X has a **268 MB** Infinity Cache and the arch config prices all traffic at
the **8.0 TB/s** DRAM peak. A workload whose working set fits in cache can
legitimately exceed that, which makes the roofline a *lower* bound that is too
low — the undetectable direction again.

Measured roofline reference points on this part (default clocks, **not** scoring
ceilings): HBM **4.90 TB/s** achieved against 8.0 spec; GEMM **1430 TFLOPS** BF16
against 2517 at 2.4 GHz.

Note the tension: the *achieved* HBM number is well below spec, while the
suspect FlashInfer workloads imply **10.09 TB/s**. Both cannot be priced by one
constant.

This is recorded as an unresolved judgement item (V2) and is **not** a licence to
change the bandwidth constant. Deciding it requires an independent measurement of
achievable bandwidth as a function of working-set size on this part.

---

## Issue 7 — `provenance.stamp()` does not emit `part`

Small, and it has already caused three separate defects:

1. `score_solutions.py`'s part guard read `prov.get("part")`, which is never
   emitted, so the guard was dead code and MI350X bounds would have been used
   silently on MI355X.
2. `manifest-v2/v3` carry `_provenance.part: None`. The leaderboard resolves the
   part through a device-name *fallback* (`AMD Instinct MI355X` → `MI355X`),
   which works but is inference rather than statement.
3. Task 10 artifacts cannot be filtered by part at all; the only in-file evidence
   is a hostname buried in `card_check.reason`.

Fix at the source in `scripts/provenance.py`, then remove the fallbacks that
exist to work around it.

---

## What NOT to trust in the existing record

Yesterday's session produced eleven-plus defects that each **passed**. The
pattern is that the check, the report or the gate was measuring something other
than what it claimed. Specific traps, all real:

* **A gate can be auditing an artifact nobody published.** Task 06 read
  `artifacts/06-MI355X/authoritative` while the manifest was built from
  `authoritative-merged`, and reported "T_b covers only 208 of 220" against a
  manifest with 219. The manifest now records `sources`; check it.
* **A check can be blind and say so in words that sound like scheduling.** Task
  03's check D reported "no submissions on disk — untested" while 405 scored
  problems sat in the tree, because `ArtifactTree` resolved task 10 to
  `artifacts/10-MI355X`, which has never existed. Check D is the only check that
  can falsify a too-slow bound.
* **A constant from the other part.** `MAX_H_MIN = 0.066` was MI350X's
  locked-clock re-timing precision, applied unchanged to a part that cannot lock
  its clock. Search for more.
* **A "measurement" that never ran.** A T_b sweep launched without
  `SOLEXBENCH_CLOCK_BASIS=unlocked` stamps `clock_basis: locked` on a part that
  has never been locked, reports "6 ok, 0 failed" in 6.3 minutes, and is entirely
  unusable. `checked_clock_basis()` now refuses this; verify it is on your path.
* **A cross-check that audits one tier.** Sections A–D of
  `sol_cross_checks.md` audit the SOLAR tier alone. `A-published` and
  `D-published` audit the bound a score is actually computed against. The
  numbers differ a lot (D: 120 tier-level vs 0 published). Know which you are
  reading.
* **Counts that look like independent evidence but are one constant.** 1327
  "violations" that are all exactly 1.33x are one bug, not 1327 findings. If a
  population is suspiciously uniform, compute the ratio distribution first.

---

## Reproduction

```bash
cd /var/tmp/solbench/m2          # worktree, branch mi355x-bringup-refs-tolerances
git log --oneline -1             # 39bc7603 or later

# gates (MI355X, current manifest)
for t in 00 02 03 05 06 07 08 09; do
  env/solb python scripts/verify_artifacts.py --task $t --part MI355X \
      --manifest manifest-v3.json
done

# MI350X must stay at: 03 -> 1 failed (its own known one), 06 -> 0, 09 -> 0
for t in 03 06 09; do env/solb python scripts/verify_artifacts.py --task $t; done

# cross-checks, both tier-level and published
SOLEXBENCH_CLOCK_BASIS=unlocked env/solb python scripts/sol_cross_checks.py \
  --t-sol artifacts/03-MI355X/t_sol.json \
  --arch SOLAR/configs/arch/MI355X.yaml \
  --t-b artifacts/06-MI355X/authoritative-merged \
  --manifest artifacts/09-MI355X/manifest-v3.json \
  --out /tmp/xc.md                        # write to /tmp, never over the artifact

# rebuild a bound tier (device="meta", NO GPU)
env/solb python scripts/sol_traffic_floor.py --t-sol artifacts/03-MI355X/t_sol.json \
  --arch SOLAR/configs/arch/MI355X.yaml --t-b artifacts/06-MI355X/authoritative \
  --out /tmp/t_sol_traffic.json

# rebuild the manifest (CPU only)
SOLEXBENCH_CLOCK_BASIS=unlocked env/solb python scripts/build_manifest.py \
  --version v4 --out artifacts/09-MI355X/manifest-v4.json \
  --t-sol artifacts/03-MI355X/t_sol.json \
  --t-sol-traffic artifacts/03-MI355X/t_sol_traffic.json \
  --t-b artifacts/06-MI355X/authoritative-merged \
  --tolerances artifacts/05-MI355X --force

# coverage, after every sweep
python3 scripts/check_coverage.py --artifacts artifacts/06-MI355X/authoritative-merged

# the fleet. Look BEFORE launching anything: timing and exploration must not
# share a card, and a wedged process holds a card at idle power.
python3 scripts/fleet_monitor.py --nodes localhost,mia1-p02-g45,mia1-p02-g05

# tests: baseline 921 passed, 164 skipped
env/solb python -m pytest tests/ -q
```

**Almost all of this needs no GPU.** Bounds are derived on `device="meta"`, the
manifest is CPU, and `scripts/backfill_scores.py` recomputes scores from a newer
manifest without re-timing anything. Do not re-measure to make a bound change
visible — recomputing arithmetic is both cheaper and more correct, because a
re-measurement lands on a different day under different node conditions.

## Ground truth to preserve

* `artifacts/09-MI355X/manifest-v3.json` — 220/235 problems, 3717 workloads.
* Coverage 235/235 (L1 94, L2 82, Quant 18 of 33 with 15 NVFP4 deferred,
  FlashInfer-Bench 26).
* Gates: 00 13/0, 02 12/0, 03 14/**2**, 05 10/0, 06 12/0, 07 4/0, 08 4/0, 09 9/0.
  The two task-03 failures are Issues 2 and 3.
* Tests 921 passed, 164 skipped.
* **Never regenerate an unsuffixed `artifacts/NN/` file** — those are MI350X
  release artifacts. MI355X is `artifacts/NN-MI355X/`.
* This part is **never clock-locked**. Every measurement runs on
  `SOLEXBENCH_CLOCK_BASIS=unlocked` and carries its own bracketed clock.
