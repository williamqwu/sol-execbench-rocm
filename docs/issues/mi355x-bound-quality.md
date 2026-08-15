# MI355X bound quality — resolved-issue record, 2026-08-15

**This document has been rewritten.** It opened on 2026-08-15 as a list of seven
open issues and a set of hypotheses to falsify. Six of the seven were worked that
day. **Four of the seven hypotheses it stated were wrong**, and two of them
misdirected real work before an investigator refuted them. This version records
what the evidence actually was, what shipped, and what is still owed — and it is
blunt about the original's errors, because a hypothesis written confidently in a
brief gets inherited.

**Status of the benchmark on this part.** `artifacts/09-MI355X/manifest-v4.json`
carries **220/235 problems and 3717 scoreable workloads**, coverage 235/235,
`part: MI355X` stated at the top level and in `_provenance`. Gates: task 03 has
**one** failure (`L2__050`, below); every other MI355X gate is 0 failed.

The prime directive this whole document is about still governs:

> **A self-consistent bound and anchor cannot detect a shared error.** `T_b`
> comes from a PyTorch reference that over-reads exactly where the
> declared-traffic bound over-counts, so the `T_SOL <= T_b` gate passes while
> both are wrong. Only an independent kernel separates them.

Every finding referenced here lives in [`docs/findings.md`](../findings.md) under
a stable D-number. The owed work is in [`TODO.md`](../../TODO.md) and
[`docs/TODO-MI355X.md`](../TODO-MI355X.md); this file does not duplicate it.

---

## The two error directions, and the arithmetic this document previously got wrong

A score is `S = (T_b − T_SOL) / (T_b + T_k − 2·T_SOL)`.

**T_SOL too LARGE (over-counted work).** A real submission runs faster than the
bound and `check D` fires. **Detectable.** This is what D18 was.

**T_SOL too SMALL (under-counted work).** The bound is loose. **Undetectable by
any measurement**, because nothing is supposed to be faster than a bound and
nothing is. The `T_SOL ≤ T_b` gate still passes. The only handle is an *a priori*
floor — the declared traffic every correct kernel must move.

> **This document used to add "so `S` is inflated for everyone", and that is
> backwards for three quarters of the corpus.** The derivative is
> `dS/dT_SOL = (T_b − T_k) / (T_b + T_k − 2·T_SOL)²`, so the sign is decided by
> `T_b − T_k` alone: a bound moving **down deflates `S`** for any kernel faster
> than `T_b` and inflates it only for one slower. Measured: **1549 of 2078
> PASSED MI355X records (74.5%) are faster than their own `T_b`.** The wrong
> unqualified form propagated from this file into three agents' briefs on
> 2026-08-15 before an audit caught it. **Detectability, not the sign of `dS`,
> is what makes the too-small direction the dangerous one** — that part was
> always right. Full statement and evidence: [D69](../findings.md#d69).

---

## Issue 1 — the SOLAR tier and `FlashInfer-Bench__018`

**Verdict: the number was confirmed, the stated mechanism was refuted, and the
fix that shipped is a different one at a different site.** See
[D66](../findings.md#d66).

**Where this document was wrong.** It asserted that SOLAR "resolves gathered/paged
inputs at full allocation" — D18's defect in the SOLAR tier — and offered a
54-byte coincidence as the evidence. It also instructed the investigator to
confirm that first, which is the only reason the error cost hours rather than a
release.

**What the evidence is.** SOLAR charges a `__getitem__` its *output* size and
there is no path in it that charges a gather its source. On the traced graph for
`018/00cb2bc2` the gather reads **4,096 elements (8 rows × 512)** out of
506,710,528. The allocation enters through `Model.to` — the reference's own
`ckv_cache.squeeze(1).to(torch.float32)`, executed **before** the index — which
really does materialise a 1.14 GB fp32 copy to select 8 rows from it. A controlled
four-body experiment settles it: `table[idx]` prices 8,192 B, `table.to(f32)[idx]`
prices 1,024,000,000 B. The 54 bytes are not a coincidence either: they are
`kv_indptr`, `kv_indices` and `lse` priced at SOLAR's single
`bytes_per_element = 2`, derived to the byte.

So the bound is **a correct roofline for the reference's algorithm and a wrong one
for the problem**. That is a different defect with a different fix site than the
one this document named.

**What shipped.** MI350X v1.1 parity, which is not an improvisation because that
part already ships it: on workloads carrying a derived `gathered_axes`, discard
SOLAR's *memory* term and keep its *arithmetic* term. 47 bounds on `__018`, all
DOWN by 12.8x–25,832x, marked `bound_quality` (46 vacuous, 1 loose) and
`bound_headroom` on all 47.

**What did not ship.** The durable fix — a slice pushdown in SOLAR's
`graph_analyzer`, propagating a gather's demand backwards through a
shape-preserving elementwise producer — is a change to how a bound is derived.
**Recommended, not enacted** (prime directive 7). It needs an owner and a
methodology sign-off.

**Corrections to the record made while closing it.** The integration report said
"nothing on this part falsifies the old bound … the argument is semantic, not
empirical." True of MI355X, and it overstates: **36 real MI350X kernels falsify
the old bound** (manifest-v1, worst 0.2693x). Conversely that falsification
justifies lowering by at most 3.7x while the correction lowers by 12.8x–25,832x —
the best kernel anyone has written for this problem is **97.5x slower than the new
bound**. This trades a detectable error for an undetectable one, deliberately.

**Was unguarded; guarded later the same day, in code.** `_solar_arithmetic_only`
zeroed SOLAR's memory term on the mere presence of `gathered_axes`, with no check
that SOLAR saw nothing beyond the allocation — harmless on today's corpus
(measured 0.9609–0.99999995 on all 63 records, never reaching 1) but that is a
property of the data, not of the rule. Two counted guards now refuse, and both
leave the allocation-priced bound in place, which is the *detectable* way to be
wrong. **Not in the shipped `manifest-v4.json`**, which was not rebuilt.

---

## Issue 2 — the five falsified bounds

**Verdict: four of the five were one byte-count defect and are deleted. The
fifth, `L2__050`, is a precision-pricing decision and is still open.** See
[D64](../findings.md#d64) and [D67](../findings.md#d67).

**Where this document was wrong, twice.**

1. **The Infinity-Cache hypothesis is refuted by measurement.** This document
   claimed a ~270 MB working set "fits inside the 268 MB Infinity Cache" and that
   the fix was therefore a bandwidth model. The arithmetic does not work
   (270.5 MB against a 256 MiB = 268.4 MB cache — it overflows), the criterion has
   no discriminating power (two of the four violating workloads *do* fit and two
   do not, and all four violate by similar factors), and a dedicated probe on
   GPU 0 measures **7.24–7.31 TB/s at a ~258 MiB working set** against the
   10.09 TB/s the hypothesis requires, with **no knee at 256 MiB** and >8 TB/s
   only below ~64 MiB. The refutation runs in the inconvenient direction: the
   measured achievable ceiling at that size is *below* the 8.0 TB/s constant those
   bounds already use, so re-pricing at a measured LLC bandwidth would make
   `T_SOL` **larger** and the violations **worse**.
2. **Its list of five was 12 by the gate's own count, and 7 of those were clock
   phantoms.** The gate reported "12 of 2078 across 7 problems, worst 0.53x"
   because `check D` read the manifest's legacy `t_sol_ms` column rather than the
   bound a score is computed against. All seven extras have
   `published/t_sol_ms = 1800/f_published` to five decimals — one bug, seven rows
   ([D63](../findings.md#d63)). The document's five was the correct *published*
   set; its framing of the population was not, and a future session reads the
   gate.

**What the defect actually is.** The declared-traffic tier charged a full read of
`q` on causal paged prefill workloads whose own reference skips nearly every query
row (`if max_kv_idx <= 0: continue`, with `output` pre-zeroed and `lse` pre-set to
`-inf`). Live rows read from the workloads' own trace blobs: **1, 2, 3 and 25** out
of 10447, 16384, 16384 and 15783. Charging the streamed input at its live rows —
and going on charging `output` and `lse` in full — drops the bound by ~1.96x and
clears all four with 57–92% margin. It is D18's defect on a **masked stream**
rather than a **gathered allocation**.

**What shipped.** 64 of 3717 bounds, `FlashInfer-Bench__015` (36) and `__014` (28),
ratio 0.5039 … 0.5103 … 0.8081, 0 UP. The detector is by AST from each problem's
own reference with no name allowlist; over all 235 definitions it fires on exactly
those two problems. Recomputed independently from the raw safetensors by an
adversarial auditor: 68 of 68 live-row counts match, and the byte identity holds
exactly on all 68.

**`L2__050` is untouched and is the last real falsification on this part.** SOLAR's
compute count is exactly right — 118,382,133,248 MACs, reproduced by hand — but is
priced at `MAC_per_cycle_fp32_sm = 32768` while the submitted kernel legitimately
runs the graph under `torch.autocast(float16)` on the matrix cores, a 16x faster
datapath, which the workload's `max_rtol = 0.5583` admits. Two mutually exclusive
resolutions, both methodology changes: price the bound at the fastest precision the
tolerance admits (bound goes 1.512 → 0.0945 ms and becomes nearly meaningless), or
make precision part of the specification and the correctness gate enforce it (the
submission becomes INCORRECT). **Neither was chosen. It needs a maintainer.**
Exposed population: **22 workloads across 5 problems** where a loose tolerance meets
an fp32-priced compute bound.

**A residual risk that got worse, not better.** On `__014`/`__015` the correctness
gate has almost no discriminating power: `required_matched_ratio = 0.99` against
0.010–0.158% live rows, so a kernel that filled `(0, -inf)` everywhere would miss at
most 0.16% of elements and pass. **Lowering these bounds makes such a kernel score
better.** Raised three times now and still unaddressed.

---

## Issue 3 — "published bounds below the declared-traffic floor"

**Verdict: the population never existed. The 120 was a regex artefact and the
"73-row difference" it was compared against does not exist either.** See
[D68](../findings.md#d68).

**Where this document was wrong.** It presented 120 and 193 as two counts of the
same phenomenon that disagreed, and asked which question was being asked.

* The **120** is not check A-published at all. `verify_artifacts.py`'s
  unanchored regex ran 3123 characters past the end of the A-published section —
  because `sol_cross_checks.py` emits its `**N VIOLATIONS**` clause only when
  `N > 0` — and matched section **D**'s number instead. A-published's real verdict
  on the shipped artifact is **0 violations**, and zero is structural rather than
  lucky: `published < floor` iff the traffic tier was dropped, and it is dropped by
  exactly the rule that is A-published's own excusal condition.
* The 120 it *did* match is section D tier-level, reproduced exactly at 120 of
  2694 — and **120 of 120 are D63 clock phantoms**. Re-derived at each
  measurement's own bracket the count is **0 of 2694**.
* The two sets are **disjoint** (intersection 0), so the document's "contradiction
  worth chasing" — `L1__057` appearing as both too loose and too tight — is two
  different tiers seen through two different checks, not a contradiction. Its
  traffic tier was correctly rejected for exceeding `T_b`; its published SOLAR bound
  is separately loose.
* **Do not fix the 193.** 185 of it is integer-cycle quantization at two different
  clocks and 8 is `L1__057`'s correctly-rejected tier.

**The load-bearing negative result survives, and its stated reason did not.**
Zero of the 1021 `check A` rows ship an invalid published bound — but "`max_of_both`
rescues them" is true of only 370 of 960 rescued rows. The rescue is
`_reclock_terms` publishing `memory_bytes = max(solar, traffic)` regardless of the
`t_sol_source` label, so 533 records labelled `solar_fused` publish on the traffic
tier's byte count. **An auditor filtering on `t_sol_source == "max_of_both"` reaches
the wrong conclusion on 590 of 960 rows.**

**What shipped:** the section-scoped regex, and `sol_cross_checks.py` now always
emits an explicit `0 VIOLATIONS` so "no match" stops meaning two things. Task 03
MI355X went 2 failures → 1. Later the same day the gate was also **bound to its
evidence**: the report records the manifest and every input by digest, and
A-published REFUSES rather than passing when the binding fails. **Consequence you
will trip over: regenerate `cross-checks.md` after any manifest rebuild — the
binding is by sha256, not path — or task 03 goes red.**

**Three audit findings; two closed late in the day, one still live:**

1. ~~`cross-checks.md` still publishes section D's "120 VIOLATIONS, each one a
   config error".~~ **Closed 2026-08-15 09:40 UTC.** Section D now re-derives at
   each anchor's own clock bracket and reads `2694/2694 workloads satisfy
   T_SOL <= T_b`. It was false and published for most of a day *after* the only
   gate reading it went green. Keep the lesson: **a gate turning green does not
   retract the artifact it was reading.**
2. ~~check A-published is not bound to the manifest it is asked about.~~
   **Closed 2026-08-15.** Task 03 now carries two binding guards — *check
   A-published is bound to the manifest under audit* (sha256 of the report's
   manifest against the one being audited) and *cross-checks report's other
   inputs are the ones on disk* — and A-published itself REFUSES rather than
   passing when they fail. Verified: `--manifest manifest-v1.json` now FAILs the
   binding check and refuses A-published, where before it returned a green light
   off a report generated against v4.
3. On exactly the 64 workloads corrected by Issue 2, it is a **self-comparison** —
   the floor is read from the same `memory_bytes` the causal mask halved. The
   traffic-causal-mask report's "identical before and after, so the fix does not
   ratify itself" is backwards as evidence.

**Four separate floor defects were found and none is fixed**, on `L1__018` (a
preallocated cache declared as both input and output, scatter-written on a slice —
4× declared, 90.4% of the total), `L1__057` (an embedding table gathered through a
2-D derived index that `gathered_axes` cannot see — 92.7%), `Quant__023` (a declared
output that is a zero-copy `expand()` view — 99.8%) and `L1__042` (declared inputs
the reference never reads — 49.2%). Every one implies a bandwidth above 8.0 TB/s
and none is rescued by Issue 6.

---

## Issue 4 (D63) — the two tiers expressed cycles at different clocks

**Verdict: the analysis half was right, the enforcement half shipped the same day,
and the largest consequence was in the direction nobody was looking.** See
[D63](../findings.md#d63) and [D65](../findings.md#d65).

**Where this document was incomplete rather than wrong.** It did not know
`t_sol.json` is **internally mixed** — 2902 of 2998 records at 1.8 GHz and 96 (the
six D37 grouped-conv problems, re-run) at 2.4 — under a header declaring
`f_lock_mhz: 2400`, a clock this part can never be locked to. And it expected a
direct 1.333x error in the published bounds. There is none: the published bound is
re-derived per measurement from three clock-free terms, so no published bound is a
stored cycle count divided by a stored `f_ref`.

**The damage ran through the rejection gate instead, and was larger.**
`combine_bounds` compared each tier's own stored `t_sol_ms` to decide whether
`T_SOL > T_b`. The SOLAR tier read 1.333x too slow at 1.8 GHz, was judged
physically impossible against a `T_b` measured near 2.4 GHz, and was dropped —
so **127 bounds across 13 problems fell back to the declared-traffic floor alone,
4.58x to 249x too small, median 39.5x**. 1.8 GHz is refuted by the measurements
themselves on that exact population: `f ≥ compute_cycles/T_b` puts a floor of
1809–2306 MHz (median 2032) under all 127.

**What shipped:** the comparison is made in time at the measurement's own bracket,
from clock-free terms; every T_SOL record now states its own `f_ref_mhz`; the tier
header is `null` unless every record in the body agrees with the arch clock; a
`--resume` can no longer restamp a body it did not recompute.
`solar_rejected_above_t_b` went 127 → 0, one for one.

**What the release note must not say.** These are **not** 127 independent findings.
The acceptance inequality forces every member into `(1800/f_pub, 1.0]` of `T_b`;
measured, `published/T_b` is 0.7592–0.9641 and headroom is "narrow" (<2x) on
**127 of 127**, against a corpus p50 of 0.082. And this correction **inflates `S`
on 43 records** (max `dS` +0.163, `L1__035/81f42cda` 0.51174 → 0.67466 at bound
×11.33) — the integration report's "36 records move, all DOWN, max −0.0028" was
measured on a population that card-refuses every affected problem.

**Still open:** `bound_quality`/`bound_headroom` are emitted on 0 of the 127 in
the **shipped** artifact, all of which are now narrow — the build was corrected
later the same day to band every published bound (a scratch rebuild moved no
existing field on any of 3957 records) but `manifest-v4.json` was not rebuilt; `artifacts/03-MI355X/t_sol.json` is still internally mixed
on disk (deliberately not re-derived — a SOLAR sweep nobody was authorised to run,
and stamping a recovered `f_ref` would write an inference in as a statement); and
every downstream consumer except `score_solutions.py` still reads the legacy
`t_sol_ms` column, which now diverges from the published bound by >1% on 1622 of
3717 workloads.

---

## Issue 5 — scoring and anchoring shard independently

**Verdict: worse than this document said, a module exists, nothing consumes it,
and closing the gap is what cost the board 63% of its scored records.** See
[D70](../findings.md#d70).

**Where this document was incomplete.** It said both passes shard `plan[i::8]` over
their own sorted plan and so align by coincidence. The stronger true statement is
that **the assignment is not a function of the problem name at all**, though both
call sites document it as one: it is a function of position in a list whose
membership is data-dependent. Measured — one candidate file gaining a passing
variant moved **100 of 217** problems to a different card. And the assignment is an
integer slot, not a card identity, which cannot name a card across a three-node
fleet whose nodes share a BDF→torch-index map.

Worse: **the enforcement checked a different object from the one it protects.** The
card check reads `card_identity` out of a `T_b` *tree*, while the `T_b` that enters
`S` comes from the *manifest*, built from `authoritative-merged` — 221 problems
assembled from **24 cards across 3 nodes**. The repeatable-`--tb-artifacts`
workaround that moved the refusal count 178 → 139 exploits exactly that gap, and
yields a **lower** `S` on 543 of 831 records. It is worse than the defect.

**What shipped:** `scripts/card_assignment.py` and a tracked
`artifacts/06-MI355X/card-assignment.json` — identity (`hostname|bdf|uuid`), not
slot; keyed on the 235-problem dataset census, not on the candidates plan; fleet
enumerated rather than counted; the map authoritative and the rule only recorded;
tamper-evident by digest. **Nothing consumes it yet.** The five call-site diffs are
specified and were left unapplied.

**What it cost, and it is the thing that got worse.** With the backfill's anchor
tree corrected to the manifest's own `sources.t_b`, `full-01` went `sol_score_v1`
**1619 → 594** and `quant-fill` **230 → 70**. None of it is the manifest — a control
backfill with HEAD code and manifest-v3 collapses identically. 1025 records lost
their score because `T_b` and `T_k` were measured on different physical cards, which
`STATE.md` §4.4 forbids with no bypass. Publish it as *"these records cannot support
the claim they make"*, not as *"undoing inflation"*. Remedy measured: **45 problems /
90 session files ≈ 2.1 card-hours** re-scored on the assigned card, **0 anchors need
re-measuring**.

**The re-anchoring run did not die** — it was staged and abandoned; only the one
problem assigned to a local card was launched, and it succeeded.

**The recovery ran the same day and finished.** Re-scoring on the assigned card
needs no new anchor. `full-01` `sol_score_v1` **594 → 1750** (pre-session 1619),
`sol_headroom` 1254 → 100, mean `S` **0.6584**; `quant-fill` 70 → **230**, mean
`S` **0.3787**. The board now carries more scored records than before the session
and **every one of them has its `T_b` and its `T_k` on the same physical card**,
which was not true of the 1619. That is what Issue 5 was actually about, and it
is invisible in either headline number. The recovery does not retract the loss —
it restores only records that can now support the claim they make, and 100 stay
`sol_headroom`.

---

## Issue 6 — LLC bandwidth (the V2 judgement item)

**Verdict: measured, and the premise is false in the band it was raised about.**
See [D67](../findings.md#d67).

This document's premise — "a workload whose working set fits in cache can
legitimately exceed 8.0 TB/s" — is **true only below ~64 MiB** and **false** at the
268 MB band. The streaming envelope is 7.24–7.30 TB/s across 128–256 MiB with no
step at the cache capacity, and the DRAM asymptote beyond 2 GiB is 6.4–6.6 TB/s
(82% of spec). The document also noted a "tension" between an achieved HBM figure of
4.90 TB/s and a suspect 10.09 TB/s; both halves are now superseded — 4.886 TB/s
under-reads the real asymptote by 25% (it is a launch-limited `copy_` kernel), and
the 10.09 was a byte count, not a bandwidth.

Three measurement traps the probe caught **in itself** are written up in
[D67](../findings.md#d67) as reusable warnings: loop-invariant hoisting (inflates by
exactly `N_PASSES`), reuse distance measuring L1 (produces a flat curve, i.e. no
signal), and rotation degeneracy (inflates only sizes whose arithmetic divides
evenly — falsified by a 2.9x swing on a knob that changes no bytes).

**Nothing was enacted.** The arch config and `parts.py` are untouched. V2 remains a
judgement item, and it is now a *narrower* one: what is unresolved is not "is the
roofline bandwidth right" but "does `llc_capacity = 256 MiB` /
`llc_bytes_per_sec = 17.0e12` describe this part at all" — both values are
byte-identical to the MI300X row and carry `[PLACEHOLDER - verify]`. That question
also sizes `flush_buffer_bytes`, so it is not roofline-only.

---

## Issue 7 — `provenance.stamp()` does not emit `part`

**Verdict: fixed at the source. Two of its three stated consequences were
misstated, and its closing instruction would have made things worse.**

**Claim 1 (correct).** `score_solutions.py`'s part guard read a key that was never
emitted, so the guard was dead code. Fixed: `part` is stamped at the source, and
`manifest-v4.json` carries `part: MI355X` at the top level *and* in `_provenance`,
with `part_source: "declared"` — v2 and v3 had no `part` key at all.

**Claim 2 (misstated).** It said `manifest-v2/v3` carry `_provenance.part: None`.
They carry no `part` key whatever, which is a different failure mode: `get("part")`
and an explicit null are indistinguishable to a reader and not to a schema check.

**Claim 3 (misstated).** It said task-10 artifacts "cannot be filtered by part at
all; the only in-file evidence is a hostname buried in `card_check.reason`". In fact
their `_provenance` carries the full device list (eight `AMD Instinct MI355X`) and
the host, so part *was* recoverable — by inference, which is the actual defect, not
by nothing. The MI355X gate reads the right scores out of the unsuffixed
`artifacts/10` today for a reason **no code states**, and an MI350X score tree
landing there would be silently mixed in. That is the real exposure and it is
unchanged.

**The closing instruction — "then remove the fallbacks that exist to work around
it" — must not be followed.** Removing the device-name fallback would have blinded
`check D`, which resolves task 10 to the unsuffixed tree and is the only check that
can falsify a too-slow bound. `agent_score` is now fail-closed on a missing
manifest and `leaderboard/worker.py` refuses rather than guessing a default, which
is the correct shape: refuse, do not fall back, and do not delete the inference
that is currently load-bearing until something states the fact instead.

---

## What NOT to trust in the existing record

The traps below all recurred on 2026-08-15, which is why they are kept.

* **A gate can be auditing an artifact nobody published.** Task 06 read
  `authoritative` while the manifest was built from `authoritative-merged`. It
  happened again this session, in `sol_traffic_floor.py` (gated against the wrong
  tree, 237 records shipped ungated) and in `backfill_scores.py` (card-checked
  against the wrong tree, 126 vs 594 records scored). **Check `sources`.**
* **A check can be blind and say so in words that sound like scheduling.**
* **A check can be green for the wrong reason.** `check A-published` now passes and
  is not bound to the manifest it is asked about.
* **A check can be a self-comparison.** If a fix moves both sides of a comparison,
  "the count is identical before and after" is guaranteed by construction and is not
  evidence.
* **A statistic can name the opposite population.** `tier_compared_at_reference_clock`
  was incremented in an `except` branch: it read 348 where the true count was 3369,
  and the 348 it named were exactly the *unfixed* records.
* **A constant from the other part.** `MAX_H_MIN = 0.066` was MI350X's locked-clock
  precision. `llc_capacity` and `llc_bytes_per_sec` in `parts.py` are byte-identical
  to the MI300X row. Keep searching.
* **A "measurement" that never ran.** A sweep launched without
  `SOLEXBENCH_CLOCK_BASIS=unlocked` stamps `clock_basis: locked` on a part that has
  never been locked. `checked_clock_basis()` refuses it; verify it is on your path.
* **A cross-check that audits one tier.** Sections A–D of `cross-checks.md` audit
  the SOLAR tier alone; `A-published` / `D-published` audit the bound a score is
  computed against. Know which you are reading.
* **Counts that look like independent evidence but are one constant.** 1327 "check B
  violations" at exactly 1.33x were one bug. So were the 120 (Issue 3), the 7 phantom
  check-D rows (Issue 2), and — in a subtler form — the 127 of [D65](../findings.md#d65),
  which are one selection band, not 127 discoveries.
* **A provenance header can attest to a different run.** 417 score artifacts
  rewritten today still carry yesterday's `utc` and `git_sha`, because a fresh stamp
  was merged *under* the document that already had one.

---

## Reproduction

```bash
cd /var/tmp/solbench/m2          # branch mi355x-bringup-refs-tolerances

# gates (MI355X, current manifest) — expect 0 failed everywhere except task 03,
# which has exactly one (L2__050). Check COUNTS moved 13->14->18 on task 03 in a
# single day as guards were added; read the failed column, not the count.
for t in 00 02 03 05 06 07 08 09; do
  SOLEXBENCH_CLOCK_BASIS=unlocked env/solb python scripts/verify_artifacts.py \
      --task $t --part MI355X --manifest manifest-v4.json
done

# MI350X must stay at: 03 -> 1 failed (144 of 7840, its own known one), 06 -> 0, 09 -> 0
for t in 03 06 09; do env/solb python scripts/verify_artifacts.py --task $t; done

# rebuild the manifest (CPU only, device="meta")
SOLEXBENCH_CLOCK_BASIS=unlocked env/solb python scripts/build_manifest.py \
  --version v4 --out artifacts/09-MI355X/manifest-v4.json \
  --t-sol artifacts/03-MI355X/t_sol.json \
  --t-sol-traffic artifacts/03-MI355X/t_sol_traffic.json \
  --t-b artifacts/06-MI355X/authoritative-merged \
  --tolerances artifacts/05-MI355X --force

# coverage, after every sweep
python3 scripts/check_coverage.py --artifacts artifacts/06-MI355X/authoritative-merged

# the fleet. Look BEFORE launching anything.
python3 scripts/fleet_monitor.py --nodes localhost,mia1-p02-g45,mia1-p02-g05

# tests
env/solb python -m pytest tests/ -q      # 1066+ passed, 165 skipped
```

**Almost none of this needs a GPU.** Bounds derive on `device="meta"`, the manifest
is CPU, and `backfill_scores.py` recomputes scores from a newer manifest without
re-timing anything. Do not re-measure to make a bound change visible — a
re-measurement lands on a different day under different node conditions.

## Ground truth to preserve

* `artifacts/09-MI355X/manifest-v4.json` — 220/235 problems, 3717 scoreable
  workloads, 3957 workload records, `clock_basis: unlocked`,
  `sources.t_b = artifacts/06-MI355X/authoritative-merged`.
* Coverage 235/235 (L1 94, L2 82, Quant 18 of 33 with 15 NVFP4 deferred,
  FlashInfer-Bench 26).
* Gates, 2026-08-15 09:52 UTC: MI355X 00 13/0, 02 12/0, 03 18/**1** (+1 WARN),
  05 10/0, 06 12/0, 07 4/0, 08 4/0, 09 9/0. The one failure is `L2__050`. Check
  D's denominator was moving that afternoon as the on-card re-scoring rewrote
  `artifacts/10` — re-run before quoting it. **The count of failures is the
  stable claim; the denominator and the check count are not.**
* MI350X 03 → 1 failed, 06 → 0, 09 → 0. **Any movement there is a regression.**
* Tests 1066 passed, 165 skipped (`tests/`). The `tests/leaderboard` suite was
  **not run** — `leaderboard/.venv` does not exist in this worktree and fastapi is
  not importable from any python on this node.
* **Never regenerate an unsuffixed `artifacts/NN/` file** — those are MI350X release
  artifacts. `artifacts/10` is the one exception and holds MI355X data.
* This part is **never clock-locked**. Every command carries
  `SOLEXBENCH_CLOCK_BASIS=unlocked` and every measurement carries its own bracket.
