# PLAN — what is next

*Last reviewed 2026-08-10.*

**This file is only forward-looking.** Three neighbours, so nobody has to guess
which one to open:

| file | answers |
|---|---|
| `README.md` | what this benchmark is and how to run it |
| `STATE.md` | what was done, when, and what it measured — the ledger |
| `TODO.md` | every known gap, listed as a defect |
| **`PLAN.md`** (this file) | **which of those gaps to work on next, in order, and what "fixed" would mean** |

`TODO.md` and this file overlap on purpose and from different ends: `TODO.md` is
the complete list and makes no claim about priority; this is a short ordered plan
and is not complete. If they disagree about a *fact*, `TODO.md` and `STATE.md`
win.

The plan written before the port began is archived at
[`docs/plan-2026-07-31.md`](docs/plan-2026-07-31.md) — the upstream audit, the
component mapping and the risk register are still worth reading for *why* the
port is shaped as it is. Nothing in it is maintained.

---

## The state, in five lines

* All ten work items in `tasks/` are **done**. The port is not what is open.
* **220 of 235** problems scoreable, **3,717** workloads, measured on 8× MI350X
  at **F_LOCK 1300 MHz**. 15 NVFP4 deferred under the sanctioned contingency.
* Three manifests: v1 frozen, v1.1, **v1.2 — the one the board serves**.
* **Five T_SOL bounds are known wrong**, all diagnosed (D42), **none corrected**.
* **Nothing has been measured on MI355X.**

---

## What is open

The port is done. What is open is **the accuracy of the bounds the port scores
against**, and the parts of the deliverable that were scoped but never reached.
Ordered by what a wrong answer costs.

The organising fact behind most of this:

> **The benchmark enforces exactly one automatic check on a bound — that nothing
> may beat it — and that check is one-sided.** It catches a `T_SOL` too *large*
> and is blind to one too *small*, because a weak lower bound breaks no rule. So
> every defect the project has found so far is from the half of the space that
> happens to be checkable, and it was found by a *submission*, never by a check.

---

### P1 — Fix the declared-traffic tier, not another problem  *(v1.3)*

**Why first.** Three of the five open bound violations are one defect, and it is
D18 seen again: the declared-traffic tier prices **every declared input at its
full allocation**, regardless of whether the kernel reads all of it, part of it,
or none of it. v1.1 fixed that *for the two paged FlashInfer problems* rather
than at the tier. **328 workloads across 38 problems** still rest on it, p50
1.50× and max 128.9×.

Most of those produce no violation today. That is not evidence they are right —
it is evidence nothing has got close enough yet, and P1 exists so the next good
submission does not discover them one at a time.

Three sub-classes, and they need different work:

| class | example | what it needs |
|---|---|---|
| **never read** | `L1__013` (`x`), `L2__044` (`hidden_states`, only `.shape`) | drop the term. Mechanical. `scripts/bounds/scan_unread_inputs.py` already names all 6 of these that are on the tier |
| **read as a slice** | `L1__018` (262,144-slot KV cache, `seq_len` touched), `L1__057` (157,184-row table, `B×S` rows gathered) | a per-workload derivation from the index tensor. The scan deliberately does **not** count these, because a slice does move bytes |
| **read whole** | the rest | nothing to fix |

**Acceptance.** `L1__018`, `L1__042` and `L1__057` come off the violation list
*because their bounds were re-derived*, each with an exact ratio recorded, and
the other 325 workloads move only where the derivation says they should.
`scripts/bounds/diagnose_bad_bounds.py` re-run reports 2 remaining, not 5.

**Do not** adjust a bound until a violation disappears. Every correction so far
(D18, D35, D37) landed on an exact ratio first.

---

### P2 — An independent measurement, because the hand check cannot be the check

P1's derivations are hand computations against the same declared shapes that
produced the wrong numbers. `CLAUDE.md` §6 is explicit that this cannot detect a
shared error: a self-consistent bound and anchor pass the `T_SOL ≤ T_b` gate
while both are wrong.

The counter route is **blocked** — `rocprofv3 --pmc` hangs in this container on
a three-kernel script (D43), and it was left alone rather than worked around,
because a rebuild or a capability change alters the image every baseline was
measured in.

**So do it the way the three known-bad bounds were actually found: write a
minimal independent kernel** that moves only the traffic the problem requires,
time it on GPU 0, and compare. Slower, needs no counters, and it is the only
method with a track record here. Start with `L1__018` — memory-bound, 2.1×
headroom, an 18.6× over-count to confirm.

---

### P3 — `L2__045`, and the trap inside it

SOLAR prices the Q-Former and projector over all `ceil(S/15)` windows; the
reference reads only the first `ceil(N/40)`. **21.5× of the counted MACs never
reach the output.**

**The trap:** a second, independent error is masking it. Those fp32 einsums are
priced at the **bf16** MAC rate — exactly 1/16, on all sixteen workloads — which
makes the bound *smaller* and so breaks no rule and is reported by nothing.
Fixing only the dead-work term leaves the bound **16× tighter than intended**.
Both, or neither, and say which in the manifest.

---

### P4 — Decide what a bound that cannot be modelled is worth  *(needs the maintainer)*

827 workloads (22.3%) sit above 100× headroom, where `S` collapses toward
`T_b/(T_b + T_k)` and carries no roofline content. `L2__006` is the extreme at a
median **115,005×**.

They are **marked** (`bound_quality`: narrow / ok / loose / vacuous) and that is
all — the marking changes no score and asserts nothing about any individual
bound. Level 1 of three, and the remaining two are:

2. Derive an arithmetic term by hand for the worst cases, **only where the
   mechanism is exact.** `L2__036` is the worked example of why they are loose:
   its reference runs a 128-iteration Python loop of unfold-and-reduce that the
   traced graph barely contains, so the arithmetic term is near zero and the
   bound is whatever tiny traffic term survives. Expect the same in `L2__006`
   and `L1__016`.
3. **Decide whether a problem whose bound cannot be modelled belongs in the
   scored set at all**, or in `artifacts/deferred.json` with a reason. This is a
   scope decision, not an engineering one, and it should not be made by whoever
   happens to be fixing bounds that week.

Be honest about the shape of this while working on it: the worst offenders are
at the top *because v1.1 was right*. Pricing a paged KV cache at the pages it
gathers took `FlashInfer-Bench__018` from 185,274 cycles to **8** — correct, and
vacuous. The fix exposed the gap; it did not cause it.

The opposite tail is real too. 504 workloads (13.6%) sit under 2× headroom,
where run-to-run variance is a material share of the score. `L1__035` at 1.008×
is the visible end of that band, not an outlier.

---

### P5 — Promote `bound_quality` into the manifest

It is derived at ingest today, because the manifest could not safely be rebuilt
while a scorer was reading it. **A consumer of `manifest-v1.2.json` alone cannot
see it**, which means the honesty is a property of the website rather than of
the artifact. Small, and it should ride along with the next manifest version.

---

### P6 — MI355X

**Nothing has been measured there.** The port needs no work; every number does —
F_LOCK (and whether the 0.83 request-to-achieved ratio even holds on a 1400 W
liquid-cooled part), tolerances, `T_b`, and therefore every score. The database
is already per-part (`leaderboard/db/solbench-<PART>.db`) precisely so that a
MI350X number and a MI355X number cannot be averaged by accident.

`docs/TODO-MI355X.md` is the checklist. The blocker is node access, not code.

---

### P7 — Deferred and unexplained, carried

* **15 NVFP4 problems** (`tasks/07`). NVFP4 ≠ MXFP4: block 16 vs 32, FP8-E4M3
  scales vs E8M0. They need **re-specification, not translation**, and shipping
  220 was the sanctioned contingency, not an accident.
* **D20** — 0.13% of matmul iterations cost 3.9–4.5×. The clock hypothesis was
  tested and *falsified*. Two upstream tests are skipped behind it.
* **D43** — `rocprofv3 --pmc` hangs. See P2.
* **The task-06 sweep does not fully reproduce** (D28). 10 of 32 re-run
  variant×problem cells changed verdict. Nothing should describe the 523 `v2`
  and 581 `v3` numerical failures as "torch.compile disagreeing with eager"
  until a repeat sweep says they reproduce.

---

### Not on this list, deliberately

**Fleet concurrency at 14.** Worth ~6 hours on a 180-problem sweep: the
scheduler already keeps the node full (6.93 of 7 mean concurrency over 180 jobs
in 11.9 h), but each leased card is only **56% busy**, because the agent spends
the rest waiting on the model. 14 is where two independently measured limits
meet — 14 concurrent through the broker landed within ±2.5% of an idle card, and
the gateway caps in-flight requests at 16. The change lives in `dash-overlay/`,
which is a different repo with other owners. It is *their*
backlog item, not this project's, and the one hard constraint from here is that
**a `reservation` must stay absolute** — if GPU 0 ever becomes shareable, every
published number after that is suspect.

**Chasing the bad-bound count to zero.** The count is a function of how good the
submissions are. Driving it to zero by fixing the five that are visible would
leave the tier defect in place for the next good kernel to find, which is why P1
is written as "fix the tier" and not as "fix five problems".

