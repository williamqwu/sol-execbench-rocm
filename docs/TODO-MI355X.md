# TODO-MI355X — bringing the benchmark up on an 8× MI355X node, **unlocked**

This is what an agent landing on an 8× MI355X node executes, in order. Every
script, flag, artifact path and acceptance command below exists on `master` at
the time of writing, and every number in it is either read out of a tracked file
here or quoted with its source. Nothing is estimated.

**The port needs no work. Every number does.** MI355X and MI350X are the same
CDNA4 die (`gfx950`), so the harness, the tolerance machinery, the shim, the
exploit corpus and the manifest builder are all part-independent. What is not
transferable is anything with a millisecond or a megahertz in it.

**The one thing that changed since the previous version of this file: the clock
policy.** MI355X will be measured **without a clock lock**. That decision is
made; §1 records the evidence behind it and §10 records what it does *not*
settle. The rest of this document exists because the repo was built on the
assumption that a single `F_LOCK` constant describes a run, and on an unlocked
node it does not.

**And one thing has changed in source since then**, which §3.3 and §3.4 cover
in full and which you will hit whether or not you read them:
`ClockPreset.f_lock_mhz` no longer falls back to the requested clock, so
`get_clock_preset("AMD Instinct MI355X").f_lock_mhz` is now **`None`**, not
1650. Nothing on MI355X stamps a clock any more, and `build_manifest.py`
refuses to build until you supply a measured one. That is the intended
behaviour.

---

## 0. Read first, in this order

| | why |
|---|---|
| `CLAUDE.md` | the eight prime directives. Directives 1, 2 and 3 are the ones this task will tempt you to break. |
| `docs/methodology.md` §3 and §5 | §3 is PR #2's MI355X clock measurement, including two corrections its author left visible; §5 *Both terms of the roofline, not only their max* is the mechanism that replaces `F_LOCK`. |
| `STATE.md` D55 and D61 | D55 is the MI350X mirror image of §3. D61 is a retraction, and the worked example in §8.2. |
| `TODO.md` | the open defect list. Most of it follows you to MI355X (§9). |
| `tasks/DEPENDENCIES.md` + `tasks/NN-*.md` | the specification of each acceptance check. `tasks/01`, `02` and `03` were authored in session 1 *on MI355X*; `master` then executed them on MI350X. |

---

## 1. The policy: unlocked, and why

**Decided. Do not relitigate it here; §10 lists what is still open.**

The evidence, quoted from `docs/methodology.md` §3 (*That question, asked on a
second MI355X node*), measured on `mia1-p02-g10` with
`scripts/gpu_parity_check.py` and `scripts/unlocked_clock_probe.py`, neither of
which imports anything from the benchmark harness:

* Unlocked, all eight cards on `perf_level=auto` under one saturating 8192³ BF16
  GEMM: **1454–1500 TFLOPS (3.0% spread), 1725–1829 MHz, 1381–1399 W**.
* The same run with `--setperfdeterminism 1660` on all eight: throughput spread
  **21.0%**, six or seven cards ~320 MHz below the frequency they acknowledged,
  at ~980 W against a 1400 W cap, with `amd-smi` reporting no violation of any
  kind.
* Measured three times over five days **including across a hard reboot**:
  unlocked 3.4 / 3.0 / 3.0%, locked 21.2 / 21.0 / 21.3%.
* Per-card, all eight, two setpoints: **15 of 16 measurements land at
  0.795–0.864× of the setpoint while drawing 734–999 W; the single one that
  reached its setpoint drew 1272 W.** So the failure is the card not raising its
  power state, not clock control refusing a number.
* Unlocked/locked, side by side: drift over 8 min sustained **0.7% / not
  measured**; sensitivity to seven busy neighbours **1.0% / up to −15%**;
  per-card behaviour stable run to run **yes / no**.

The opposite result on the other part, quoted from `STATE.md` D55 (MI350X,
`gbt350-odcdh1-a08-1`): across twelve loads spanning 305–900 W and 1303–1586
MHz, locked vs unlocked vs cap-raised-to-2200 are indistinguishable. D55's four
geomeans against the locked arm are **1.0012** (3 s holds), **0.9957**
(locked-at-2200), **1.0005** (saturating set) and **1.0024** (60 s soak), and
its own between-block spreads are **1.33% locked1600 / 0.92% unlocked / 0.40%
locked2200**, plus **0.6%** on the saturating set — so no ratio clears the
spread of the blocks it was computed from. The reason is in the telemetry, not
the ratios: the 1600 setpoint never binds on a part already sitting at its
power-limited operating point.

**Carry D55's own caveat with the result** (`STATE.md`, D55's closing paragraph,
"UNRESOLVED, and recorded rather than smoothed over"): `artifacts/01` has an
unlocked GPU 0 reaching **1390 MHz at 1001 W**, and that session never got the
card above 900 W on any load. Duration was tested and ruled out; the remaining
explanation — that the earlier probe's loop differs from `torch.mm` in a way
that draws the last 100 W — was not identified. So "the cap never binds" is
true of everything measured there and **not proven in general**, which is a
second reason not to carry the result onto this part.

Two parts, one API, opposite failures. That is why the policy is **per part**
and not a repo-wide constant, and why neither result may be carried across. The
module docstring of `src/solexbench_rocm/t_sol_at.py` states the same thing and
is the short version to hand a reviewer.

**What unlocked costs you, stated up front.** Frequency becomes a property of
the kernel: `artifacts/01/unlocked-clock.json` (host `mia1-p02-g10`) records a
**27.9% spread across workload types** on that node — a dense GEMM pushed to
~1730 MHz against the 1400 W cap, a memory-bound kernel boosting to ~2394 MHz.
So there is no single frequency to divide a cycle count by, and §3 is about
everything in the repo that currently assumes there is.

---

## 2. What is already on `master` for MI355X — and the trap it created

PR #2 is merged (`697749f0`, *Merge PR #2: the MI355X clock finding, and the
machinery to score without a lock*). Merged with it:

| | what it is |
|---|---|
| `scripts/gpu_parity_check.py` | one fixed 8192³ BF16 GEMM on all eight cards, wall-clock throughput, plus a per-card setpoint sweep. `--n-gpus 8 --seconds 45 --setpoint 1660 --sweep-gpus … --sweep-setpoints …`, out to `artifacts/00/gpu-parity.json`. |
| `scripts/unlocked_clock_probe.py` | varies workload, duration and neighbour load unlocked. `--gpu --n-gpus --seconds --drift-seconds`, out to `artifacts/01/unlocked-clock.json`. |
| `scripts/burst_clock_probe.py` | per-iteration time vs burst length, which answers "was the clock the same?" without telemetry. `--gpu` (default **1**) `--mode unlocked\|locked --setpoint`. |
| `env/solb-native`, `env/check_stack.py` | a daemonless path that *asserts* the pinned torch/HIP instead of assuming it. |
| `src/solexbench_rocm/t_sol_at.py` + `tests/scripts/test_t_sol_at.py` | re-max both roofline terms at an arbitrary clock (§4). |
| `scripts/sol_bounds.py` | now emits `compute_cycles`, `memory_cycles_at_f_ref`, `mac_per_cycle`, `dram_byte_per_sec` alongside the max. |
| `scripts/clock_calibrate.py` | per-device SMI calls now go through `smi_device_index()` → `gpu_map.torch_to_amdsmi()`. |

**`artifacts/00/` and `artifacts/01/` now hold two parts' files side by side,
and nothing in the tree records which is which.** Verified by reading
`_provenance` out of each file — and note how the part had to be recovered,
because it is §10 item 7's defect happening here too: only
`artifacts/01/unlocked-clock.json` carries an explicit `_provenance.part`
(`"MI355X"`). For the other three the part is an *inference* off the torch
device-name list at `_provenance.torch.devices[0]`, `"AMD Instinct MI355X"`.

| file | host | part | stamped `f_lock_mhz` |
|---|---|---|---|
| `artifacts/00/gpu-parity.json` | `mia1-p02-g10` | MI355X *(inferred)* | 1650 |
| `artifacts/01/unlocked-clock.json` | `mia1-p02-g10` | MI355X *(explicit)* | 1650 |
| `artifacts/01/burst-clock.json` | `mia1-p02-g10` | MI355X *(inferred)* | 1650 |
| `artifacts/01/burst-clock-locked.json` | `mia1-p02-g10` | MI355X *(inferred)* | 1650 |
| `artifacts/00/node-report.json` | `gbt350-…` | MI350X *(inferred)* | null |
| `artifacts/01/floor-gpu{0,1,2}.json`, `det1600-gpu{0..7}.json` | `gbt350-…` | MI350X *(inferred)* | null |

Two consequences, both of which will bite during bring-up:

1. `scripts/verify_artifacts.py:246` collects clock floors with
   `(ART/"01").glob("floor-gpu*.json")` — **no part filter** — and then requires
   `F_LOCK <= min(p5 over all matching files)`. The first MI355X `floor-gpu*.json`
   written into `artifacts/01/` silently gates the MI350X F_LOCK against MI355X
   floors. Move the MI350X tree aside first (§5 step 3).
2. Those four MI355X files are stamped `f_lock_mhz: 1650`, which is the number
   §3 is about, and which the node they were taken on did not run at.

**Still not on `master`:** `origin/feat/agent-scoreboard`, 30 commits ahead of
HEAD (`git rev-list --count HEAD..origin/feat/agent-scoreboard` → 30). It still
carries `scripts/run_pipeline.sh`, `scripts/score_solutions.py` and
`scripts/backfill_scores.py`, none of which exist here. Its measurements are a
colleague's lab notebook: its `T_b` was taken while a 404-session agent sweep
saturated the node's CPUs and `S` was retracted rather than published. Read it;
adopt nothing from it. Decide explicitly, and record in `STATE.md`, whether you
cherry-pick the pipeline scripts or re-implement — silently re-implementing
`run_pipeline.sh` is the expensive option.

---

## 3. What "unlocked" actually breaks, file by file

`F_LOCK` is not a documentation convenience. It is a divisor, a stamp, a guard,
and a headline number on the board. Each of these is a real site, read this
session.

### 3.1 The bound divisor

`scripts/sol_bounds.py:410` — `"t_sol_ms": cycles / (freq_ghz * 1e6)`, where
`freq_ghz` comes from the required `--freq-mhz`. This is the only place T_SOL
becomes a time. Unlocked there is no single correct value to pass here. §4 is
the replacement; until then, whatever you pass is a *reference* clock, not a
measurement clock, and the artifact must say so.

`scripts/rebuild_manifest_v11.py` and `scripts/rebuild_manifest_v12.py` hardcode
MI350X's clock, and **the module constant is not the only literal**:

* `rebuild_manifest_v11.py:68` `F_LOCK_MHZ = 1300.0`, used at `:207` as
  `max(F_LOCK_MHZ, measured)` — i.e. 1300 is a hard *floor* on the per-datapath
  clock — and at `:290`.
* `rebuild_manifest_v11.py:284` opens with a **second, bare** `f = 1300.0`
  before the datapath override, and `:296` computes
  `w["t_sol_ms"] = w["t_sol_cycles"] / (f * 1e3)`. Editing line 68 alone leaves
  this in place.
* `rebuild_manifest_v12.py:72` `F_LOCK_MHZ = 1300.0`, `:107-108` the same clamp,
  `:186` `f = F_LOCK_MHZ`.

Both read `artifacts/09/manifest-v1.1.json` and are MI350X rebuilders in
practice, but **neither asserts a part**. Pointed at an MI355X manifest they
produce a plausible wrong answer. Do not reuse them on MI355X; the v1.1/v1.2
corrections must be re-derived, not replayed.

### 3.2 The per-datapath divisors (D35)

`artifacts/01/f_lock_by_datapath.json` (MI350X, host `gbt350-…`) answers in its
own words: *"Every matrix-core path sits at F_LOCK -- bf16 1296, fp16 1299, fp8
1314. The fp32 vector path sustains 1441 MHz, 10.8% above it… Every
compute-bound fp32 bound in manifest v1 is therefore 10.8% too large in
milliseconds."* That correction is a per-part, per-datapath measurement and
**does not transfer**. On an unlocked node it is also the wrong shape: the datapath
is a proxy for the clock, and unlocked you can measure the clock instead.

### 3.3 The stamp

`scripts/provenance.py:88-119`, `f_lock_mhz()`. It resolves from
`SOLEXBENCH_F_LOCK_MHZ`, else from `get_clock_preset(torch.cuda.get_device_name(0))`.
**It never consults a device.** `stamp()` writes one value per *artifact*, not
per measurement.

**This has been fixed in source, and not the way an earlier version of this
document proposed.** The preset used to be

```python
"AMD Instinct MI355X": ClockPreset(gpu_clk_mhz=1650, dram_clk_mhz=None),
```

with `f_lock_mhz` reading `achieved_gpu_clk_mhz or gpu_clk_mhz` — so an entry
with no measured `achieved_gpu_clk_mhz` handed back the **requested** 1650 and
labelled it achieved. `src/sol_execbench/core/bench/config/device_config.py`
today keeps the entry and adds a flag instead:

```python
    requested_is_achieved: bool = True          # :67, the NVIDIA case

    @property
    def f_lock_mhz(self) -> Optional[int]:      # @property :69, def :70,
                                                # docstring :71-88, body :89-91
        if self.achieved_gpu_clk_mhz is not None:
            return self.achieved_gpu_clk_mhz
        return self.gpu_clk_mhz if self.requested_is_achieved else None

    "AMD Instinct MI355X": ClockPreset(         # :135-137
        gpu_clk_mhz=1650, dram_clk_mhz=None, requested_is_achieved=False
    ),
```

So today **`get_clock_preset("AMD Instinct MI355X").f_lock_mhz` is `None`**.
MI350X, which carries a real `achieved_gpu_clk_mhz=1300` beside its
`gpu_clk_mhz=1600`, still returns 1300; the three NVIDIA entries take the
`True` default and are unchanged. All four are pinned by
`tests/sol_execbench/core/bench/config/test_clock_preset_f_lock.py`.

**Keeping the entry is the point, not a compromise.** `1650` is still recorded
as the *request*, which is the only thing it ever was, and it is still what
gets applied: `clock_lock.py:110-112` reads `preset.gpu_clk_mhz`, not
`f_lock_mhz` (the assignment opens on :110, names the field on :111 and closes
on :112), so a deliberate lock still works and nobody has to rediscover the
number.
What is gone is the fallback that turned a request into a measurement.

**Why the fallback was worth removing, from PR #2's own numbers.** Apply the
sourced band to that 1650: at 0.795–0.864× of the setpoint the card holds
1650 × 0.795 = 1312 MHz to 1650 × 0.864 = 1426 MHz. Stamping 1650 across that
band overstates the clock by 1650/1426 = 1.157 to 1650/1312 = 1.258 — **16% to
26% high**. (Do not quote a single figure for this; the per-card table in
`docs/methodology.md` §3 is explicitly labelled a snapshot, not a taxonomy.)

**And that is the undetectable direction.** A bound divided by 1650 while the
silicon holds 1312–1426 MHz comes out **1.157–1.258× too small** — that is
13.6% to 20.5% *below* the correct value, **not** the clock's own 16–26%. The
two figures are different arithmetic and must not be swapped: a clock that is
1.157–1.258× high divides the bound by that factor, so the correct bound is
1.157–1.258× the one you get (equivalently, 1 − 0.864 = 13.6% and
1 − 0.795 = 20.5% short). And a `T_b` measured on that same slow
silicon is slow by the same factor, so task 03's `T_SOL <= T_b` gate
passes *more* easily. This is exactly the shared-error case `CLAUDE.md` §6 says
a self-consistent bound and anchor cannot detect, which is why the fix is in
the property rather than in a comment.

**What the operator must now do**, because a `None` is not free — 3.4 has the
consequences in full, and the short version is that the manifest build refuses
outright:

1. **Measure a clock on your node and pass it as `SOLEXBENCH_F_LOCK_MHZ`.**
   `provenance.f_lock_mhz()` checks that first, so it is the supported way to
   build off-preset, and `build_manifest.py:252-257` names it in its own error
   message. It must be a number you measured, in this session, on this node —
   an env var is exactly as capable of carrying an invented measurement as a
   table was.
2. **Or measure the part properly and add `achieved_gpu_clk_mhz`** to the
   MI355X entry. That is a source edit, it is legitimate, and §1 is the reason
   it is also not sufficient: unlocked there is no single achieved clock to put
   there — 27.9% across workload types on the one node that has been probed.
3. **Or take §4.3's route and stop dividing by one clock at all**, which is the
   direction the rest of this document argues for.

Whichever you pick, write it into `STATE.md` before measuring (§5 step 6). A
`null` stamp is a known, visible gap and the board already renders it (3.5);
1650 was an invented measurement; a number from an env var nobody sourced would
be an invented measurement with an extra step.

### 3.4 The guards

* `scripts/build_manifest.py` has **two** F_LOCK guards, they sit at different
  levels, and they treat `None` in opposite ways. Getting them the wrong way
  round is what an earlier version of this section did, so read both.

  **The per-artifact comparison, `:162`,** inside `collect_t_b`:

  ```python
  measured_at = (doc.get("_provenance") or {}).get("f_lock_mhz")
  if f_lock_mhz is not None and measured_at not in (None, f_lock_mhz):
      foreign.append((f.name, measured_at))          # rejected
  ```

  This rejects any T_b artifact whose *stamped* clock disagrees with the
  expected one, and **admits a stamped `None`** — deliberately, per the comment
  at `:158-160`: a null stamp means the artifact predates F_LOCK stamping,
  which is a different problem from being measured at the wrong clock, and
  `check_06` requires provenance separately. That admission is about the
  artifact's own `_provenance`, not about the F_LOCK the build resolved.

  **The top-level resolution, `:243-257`,** which runs first:

  ```python
  expected_f_lock = _f_lock()
  if expected_f_lock is None:
      raise SystemExit("cannot resolve F_LOCK: no GPU preset and no "
                       "SOLEXBENCH_F_LOCK_MHZ. ...")
  ```

  Its comment says why in as many words: `collect_t_b(..., None)` would admit
  artifacts from any clock, "refusing is the only safe reading: an unknown
  clock is not a permissive one."

  **So on MI355X today the manifest build refuses outright**, because 3.3 makes
  `provenance.f_lock_mhz()` resolve to `None` there. That is the intended
  behaviour and it is a *feature* of the shipped state, not a hole: the failure
  mode is a hard exit before anything is written, not a silently permissive
  guard. The escape hatch is the one the error message names —
  `SOLEXBENCH_F_LOCK_MHZ=<measured MHz>` — and taking it re-arms `:162` at
  whatever number you supply, so it is only as good as that measurement.
  What is still missing, and what §4.3 is about, is that neither guard has any
  notion of a *per-measurement* clock; both compare one number per artifact.
* `scripts/verify_artifacts.py` `check_01` (`:205-`) — as written it **cannot
  pass on an unlocked node**. It requires `F_LOCK recorded in STATE.md`, then
  `F_LOCK present in CLOCK_LOCK_PRESETS`, then
  `STATE.md and CLOCK_LOCK_PRESETS agree on F_LOCK`, then `every GPU is at the
  preset's determinism setpoint`. All of those presuppose a lock, and after 3.3
  it fails at the second one *concretely* rather than in principle:
  `f_lock_from_preset()` (`:141-159`) returns `preset.f_lock_mhz`, which is
  `None` on MI355X, so the check reports "no preset for this device" even
  though the entry exists. Note the split — `requested_clock_from_preset()`
  (`:127-138`) still returns `gpu_clk_mhz`, so the setpoint check keeps
  comparing against 1650. Amending task
  01's acceptance check is a **methodology change** and therefore a maintainer
  decision (prime directive 7, `CLAUDE.md` §1: *a task is complete only when its
  acceptance check passes*). Do not quietly weaken it; §10 carries it as open.
* `scripts/verify_artifacts.py:172` — `len(gpus) == 8`. Both target nodes are
  8×, so it will not bite today; it is a node property sitting in a gate.
* `scripts/verify_artifacts.py:246` — the unfiltered `floor-gpu*.json` glob, §2.

### 3.5 The board

`f_lock_mhz` is a headline, not an internal: `leaderboard/models.py:43`,
`leaderboard/ingest.py:323` (copies it straight out of `_provenance`), and six
rendered sites — `templates/index.html:21` (a stat card labelled *locked clock
(achieved)*) and `:226`, `templates/methodology.html:30`, `:65`, `:159`,
`templates/problem.html:138`. `templates/base.html:207` already renders a
`null` stamp as **"not stamped"** with an explanatory tooltip, so the honest
empty state exists — but *"locked clock (achieved)"* is the wrong label for an
unlocked part, the methodology page's *"F_LOCK … is what the part…"* sentence
becomes false, and `base.html:207`'s tooltip states one specific *cause* for a
null stamp — *"the artifact was written by a process without torch"* — which is
the MI350X cause and not the MI355X one. On MI355X the stamp is null because
the preset resolves to `None` by design (3.3), which is a different fact about
a different problem, and a tooltip that names the wrong reason is worse than a
generic one. These are template edits, cheap, and they should land before the
first MI355X board is published rather than after.

`leaderboard/ingest.py:74` — `MANIFEST = ROOT/"artifacts"/"09"/"manifest-v1.2.json"`,
a module constant with **no `--manifest` flag** (`--db --part --agent-runs
--sources --allow-drop` are the whole CLI). To produce `db/solbench-MI355X.db`
the MI355X manifest has to *be* that path. `--part` **asserts**, it does not
relabel: `manifest_part()` reads the part from the manifest's own provenance and
a disagreeing `--part` is a hard exit. Adding a `--manifest` flag **is already
in the contract** — `leaderboard/DESIGN-v2.md` §6, *Amendment — the storage
contract cannot be executed as written* (:369-390), names exactly two missing
pieces: "1. A `--manifest PATH` argument, defaulting to today's constant so no
existing invocation changes. 2. Deriving the output database name from the
manifest's own part when `--db` is not given." So implement it to that spec
rather than raising it as an open question. (The amendment's own citation has
drifted: :375 quotes the constant as `manifest-v1.json  # ingest.py:50`; the
real one is `manifest-v1.2.json` at `ingest.py:74`, as above. The flag it asks
for is unaffected.)

### 3.6 The artifact tree

`scripts/build_manifest.py:216-223` defaults are all task-keyed and part-blind:
`artifacts/09/manifest-v1.json`, `artifacts/03/t_sol.json`,
`artifacts/06/authoritative`, `artifacts/05`, `artifacts/deferred.json`. An
MI355X run at defaults overwrites the MI350X release record that manifest v1
cites. §5 step 3 handles it by moving directories; a prose instruction is not a
guard, and adding a part dimension to `artifacts/` is on the open list (§10).

---

## 4. The mechanism that replaces F_LOCK — and exactly what is missing

### 4.1 What exists and is correct

`scripts/sol_bounds.py` now keeps both roofline terms rather than only their
max, because they scale oppositely with the clock (its own comment, and
`docs/methodology.md` §5):

* `MAC_per_cycle` is **architectural** and frequency-independent, so the compute
  term is a fixed number of **cycles** and its *time* goes as 1/F.
* `DRAM_byte_per_cycle` is derived as `bytes_per_sec / freq`, so the memory term
  is a fixed **time** and its *cycle count* scales with F.

So it emits `compute_cycles`, `memory_cycles_at_f_ref`, `mac_per_cycle` and
`dram_byte_per_sec` per workload, and `src/solexbench_rocm/t_sol_at.py` provides
`t_sol_cycles_at(w, f_mhz)`, `t_sol_ms_at(w, f_mhz)` and `bottleneck_at(w, f_mhz)`.
`t_sol_cycles_at` reproduces `sol_bounds`' `max(1, ceil(...))` rounding
deliberately, and `bottleneck_at` is reported rather than assumed because **the
bottleneck can flip as F moves**. Records written before the split raise
`MissingBoundTerms` rather than being inferred from `bottleneck`. The tests are
CPU-only and pass here (`tests/scripts/test_t_sol_at.py`).

This is the right machinery and it is part-independent. Land its use on MI350X
too if you can — it makes the bound model explicit and testable.

### 4.2 What is missing, in order of how much it blocks

**(a) No existing bound record can be re-clocked.** `t_sol_at.REQUIRED_FIELDS =
("compute_cycles", "memory_bytes", "dram_byte_per_sec")`. The tracked
`artifacts/03/t_sol.json` predates the split and carries none of
`compute_cycles` / `dram_byte_per_sec`, so every record raises. **This is not a
problem for MI355X**, because you are re-running `sol_bounds.py` from scratch
anyway and it costs no GPU (SOLAR runs on `device="meta"`). It *is* the reason
you cannot shortcut by converting the MI350X artifact.

**(b) The declared-traffic tier emits none of those fields.**
`scripts/sol_traffic_floor.py` writes a memory-only bound with no
`dram_byte_per_sec`, so `t_sol_at` refuses all of it. For a *pure*
declared-traffic bound that is only cosmetically bad — a memory-only bound is
clock-invariant in time, so the answer is "unchanged". For workloads whose bound
is the max of both tiers it is a real gap: both tiers must be re-evaluated and
re-maxed at the new clock, and there is no code for that. Count the affected
workloads on *your* manifest before deciding how much it matters; the MI350X
counts do not transfer.

**(c) `build_manifest.py` does not propagate the four new fields** into the
manifest's per-workload record. Until it does, the scorer can never see them,
however correct `t_sol_at` is.

**(d) — the one that decides whether any of this works — nothing measures a
clock per measurement.** There is no clock field anywhere in a T_b candidate
artifact. `provenance.f_lock_mhz()` is an env var or a table lookup (3.3), one
value per artifact either way, and on MI355X the table half now yields nothing.
Getting a per-measurement clock means: add a sampler inside
`time_runnable`, choose a reduction, thread it into the T_b artifact, through
`build_manifest`, and into the scorer. PR #2 ships the last mathematical step
and none of that plumbing.

**And the plumbing may not be buildable as specified**, which is the honest part.
From `docs/methodology.md` §7 (*How short is the timed window?*):
`BenchmarkConfig` defaults to `warmup_runs=10, iterations=50`, so `time_runnable`
times 60 back-to-back executions — for most of this corpus a **1–13 ms window,
shorter than any telemetry sampler can observe**. The finest in-loop sampler in
this repo is `scripts/probe_stall_clock.py`, whose `--period` defaults to
`0.001` (`:75`) and which `STATE.md` D20 records actually achieving **~860 Hz**
inside the timing loop — about 1.16 ms a sample, so a 1–13 ms window is roughly
**1 to 11 points**. That is the finest instrument the repo has, and eleven
points is not a clock measurement; the conclusion is unchanged and now the
arithmetic matches the instrument. Worse, that window is not steady
state: per-iteration time at 60 iterations relative to a 50,000-iteration
sustained loop is **1.217× (GEMM 4096³), 2.040× (GEMM 1024³), 1.042×
(elementwise)** unlocked. So a clock read *during* the window is undersampled and
a clock read *around* it is not the window's clock. Note §7 also finds the bias
**slightly worse locked than unlocked** in all three shapes — unlocking does not
cause this and does not fix it.

### 4.3 The honest alternative, if per-measurement clock is not achievable

State it in `STATE.md` as a methodology decision, with its cost, before
measuring anything. In descending order of fidelity:

1. **Lengthen the timed window until the clock is samplable.** §7 says
   convergence needs ~10,000 iterations. This also removes the short-window
   bias. It is explicitly declined upstream-compatibility-wise —
   *"50 is upstream's methodology; changing it makes these numbers incomparable
   with upstream's and requires re-timing everything"* — so it is a decision
   about what the benchmark is for, and it makes the two changes a package
   rather than two independent improvements.
2. **Bracket the window.** Sample the clock immediately before and after the 60
   iterations, record both, and *refuse* the measurement if they disagree by
   more than a stated threshold. This does not give you the window's clock; it
   gives you a bound on how wrong assuming one is, which is honest and cheap.
   Record the threshold and the refusal count as first-class artifact fields.
3. **A per-kernel reference clock, measured separately.** Time the same kernel
   at 10,000 iterations once, off the scoring path, read the clock there, and use
   it for that workload's bound. Defensible only if you also record that the
   scoring window ran **1.04–2.04× off steady state, unlocked** (the same
   `docs/methodology.md` §7 row as above: elementwise 1.042, GEMM 4096³ 1.217,
   GEMM 1024³ 2.040 — the locked arm is a shade worse at 1.044 / 1.241 / 2.090,
   and mixing the two arms into one range is a mistake this document used to
   make) and therefore at a possibly different clock — i.e. it is an approximation with a stated error, not a
   measurement.
4. **Publish `T_b`, `T_k` and the two roofline terms, and no `S` at all.**
   Speedups (`T_b/T_k`) survive without any clock. `S` does not, because
   `T_SOL` is analytic and unbiased while both measurements carry the window
   bias, so the bias does **not** divide out of `S` — §7's own conclusion is
   that SOL efficiency is *systematically understated*. Shipping a board of
   speedups and headroom, with `S` marked unavailable, is a legitimate v1 for
   this part and is strictly better than shipping an `S` whose denominator is
   invented.

**Do not** pick option 5, which is the tempting one: assume the unlocked GEMM
clock (~1730 MHz on that node) applies to everything. §1's own data says the
spread across kernel types is 27.9%.

### 4.4 One more thing nothing in the repo normalizes

`S = 1/(1 + (T_k − T_SOL)/(T_b − T_SOL))` has three terms, and re-clocking
addresses one. Unlocked, `T_b` was measured at whatever clock *its* kernel pulled
and `T_k` at whatever clock the candidate's kernel pulls, and per
`artifacts/01/unlocked-clock.json` those differ by up to 27.9% across kernel
types. A candidate that turns a compute-bound kernel into a memory-bound one is
then rewarded twice: once for the real speedup, once for boosting. Nothing in
the repo normalizes `T_b` against `T_k`. Decide before you publish whether the
two are re-timed back to back in one session (so the clocks are at least
similar) or whether `S` becomes explicitly a two-clock quantity. This is the
difference between a workable unlocked methodology and a scoreboard an agent can
farm by lowering power draw.

---

## 5. Bring-up sequence

Ordered. Each step names its acceptance check and what it costs. **Start GPU
work first and do CPU work in its shadow** — the node is the scarce resource.

### Step 1 — verify the stack, before anything

```bash
env/solb bash -lc 'python -c "import torch; print(torch.__version__, torch.version.hip)"'
# no docker on the node?  Take env/solb-native, not a hand-rolled venv:
env/solb-native python -c 'import torch; print(torch.__version__)'
```

`env/solb-native` calls `env/check_stack.py`, which exits non-zero when torch or
HIP does not match the pinned `SOLB_WANT_TORCH` / `SOLB_WANT_ROCM` — *"a drifted
stack does not fail loudly; it produces numbers that look exactly as
authoritative as the pinned ones"* (prime directive 6). Check the interpreter
version first: an earlier MI355X node ran Python 3.10 and all 33 Quant
references import `StrEnum` (3.11+), so they fail before any submission is
involved. The container image is py3.12.

**Acceptance:** `env/solb bash -lc 'python -m pytest tests/ -q'` — expect
**519 passed, 75 skipped** (`CLAUDE.md` §7; read a drop in *passed* as the
regression, not a change in the skip count). **Cost:** minutes.

Environment gotchas, each of which cost a session somewhere:

* **Container GPU access needs the host's numeric video/render GIDs.**
  `--group-add render` resolves against the *container's* `/etc/group` and grants
  nothing. Symptom: `torch.cuda.device_count()` returns 8 while any real HIP
  context raises `No HIP GPUs are available`. `env/solb` already does this; do
  not "simplify" it.
* **The container mounts a generated `/etc/passwd`.** Without it `torch.compile`
  dies in `getpass.getuser()` with `uid not found`, breaking every submission
  that compiles.
* **`FLASHINFER_TRACE_DIR=/work` must be set** — the eval driver resolves trace
  paths against the *staging* directory, not the CWD.
* **Anything large goes to `/var/tmp/solbench`**, mounted at the same absolute
  path inside and outside the container so a path in an artifact means the same
  thing in both. `/home` is NFS. A runner told to write outside `/work` and
  outside scratch dies before writing anything (D17).
* **`env/solb` recreates the container when the image ID changes**, unless a
  sweep is running inside it. If you rebuild mid-sweep, read the warning it
  prints rather than working around it.
* **`env/solb-root` exists only for the clock lock**, and under this policy you
  should not need it. Note the trap it documents anyway: a stock container does
  not *fail* to set clocks — `rocm-smi --setperfdeterminism` exits 0 having done
  nothing, because `/sys` is read-only.

### Step 2 — node parity, all eight cards, unlocked

```bash
env/solb bash -lc 'python scripts/gpu_parity_check.py --n-gpus 8 --seconds 45 \
    --out artifacts/00/gpu-parity-<HOST>.json'
```

Do this **before** the artifact-tree move, and write it to a host-suffixed name,
because `artifacts/00/gpu-parity.json` is already the `mia1-p02-g10` file (§2).
Wall-clock throughput only, no harness, no telemetry dependency.

**Acceptance, and it is a judgement not a gate:** unlocked spread across eight
cards. The reference point is `mia1-p02-g10`'s **3.0–3.4%**. A materially wider
unlocked spread means you have a different node problem — a weak card, a cooling
imbalance — and §1's policy argument does not automatically apply to it.
**Cost:** minutes. **Record the firmware** (`amd-smi` VBIOS/SMC/MEC/RLC/SOS,
ROCm-SMI-LIB, amdgpu) in `STATE.md`: §1's finding is firmware-level and yours is
not comparable to it without that.

Also run, in the same sitting, because it is cheap and it is the input to every
later decision about what a clock number means:

```bash
env/solb bash -lc 'python scripts/unlocked_clock_probe.py --gpu 1 --n-gpus 8 \
    --out artifacts/01/unlocked-clock-<HOST>.json'
env/solb bash -lc 'python scripts/burst_clock_probe.py --gpu 1 --mode unlocked \
    --out artifacts/01/burst-clock-<HOST>.json'
```

`burst_clock_probe.py` defaults to `--gpu 1` and issues node-wide `rocm-smi`
perf-level changes in its locked mode; on a node where GPU 0 is authoritative,
that will move GPU 0's policy under a timing run. Use `--mode unlocked` and, if
you need the locked arm, run it while nothing authoritative is in flight.

### Step 3 — move the MI350X record aside

```bash
git mv artifacts/00 artifacts/00-MI350X && git mv artifacts/01 artifacts/01-MI350X
mkdir -p artifacts/00 artifacts/01/logs
# then move the four MI355X files identified in §2 back into the new tree
```

Do not delete: the MI350X manifest cites `artifacts/00` and `artifacts/01`, and
`verify_artifacts.py:246`'s unfiltered `floor-gpu*.json` glob is one MI355X floor
run away from corrupting the MI350X task-01 gate. **Acceptance:**
`git status` shows the move, and `python scripts/verify_artifacts.py --task 00`
against the new tree. **Cost:** minutes.

### Step 4 — node acceptance

```bash
env/solb bash -lc 'bash scripts/node_acceptance.sh'
env/solb bash -lc 'python scripts/roofline_probe.py --gpu 0 --out artifacts/00/roofline-gpu0.json'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'
```

Expect 8× `gfx950`, 288 GiB, **1400 W** cap and a **2400 MHz** ceiling
(`src/solexbench_rocm/parts.py`, which is the single source of truth and already
has a complete MI355X entry) — if the report says 1000 W / 2200 MHz you are on an
MI350X node and this file does not apply. `roofline_probe.py` looks the spec peak
up per part, so the achieved fraction is against MI355X's peak, not MI350X's.
These are **default-clock reference points and not scoring ceilings**; do not
cite them downstream. **Cost:** under an hour. **Acceptance:** task 00 gate, and
`CLAUDE.md`'s dataset census confirmed against the files: L1 94, L2 82, Quant 33,
FlashInfer-Bench 26.

### Step 5 — dataset and traces

```bash
env/solb bash -lc '
  python -c "
from huggingface_hub import snapshot_download
snapshot_download(\"nvidia/SOL-ExecBench\", repo_type=\"dataset\",
                  local_dir=\"/var/tmp/solbench/data-hf\")"
  python scripts/materialize_dataset.py \
      --parquet-dir /var/tmp/solbench/data-hf/data \
      --out data/SOL-ExecBench/benchmark'
env/solb bash -lc 'python scripts/fetch_flashinfer_traces.py'
```

`materialize_dataset.py` is the exact inverse of the dataset's own converter and
round-trip-verifies all 235. Without the FlashInfer blobs, 9 of the 26 FlashInfer
problems fail at run time as ordinary runtime errors — which reads as a port
defect and is not one. **Acceptance:** the round-trip verification, and 235
problem directories. **Cost:** bandwidth-bound.

### Step 6 — **the first real decision point**

Everything above is part-agnostic setup that cannot be got wrong in an
undetectable way. Everything below divides by a clock.

**Before task 02, you must have written into `STATE.md`:**

1. Where F_LOCK comes from, given that the MI355X `ClockPreset` already
   resolves to `None` (§3.3): a measured `SOLEXBENCH_F_LOCK_MHZ`, a measured
   `achieved_gpu_clk_mhz` added to the preset, or neither — in which case say
   so, and know that `build_manifest.py` will refuse at step 12 (§3.4). Do not
   discover that at step 12.
2. Which of §4.3's options is the clock methodology, with its cost.
3. What task 01's acceptance check becomes on an unlocked node (§3.4), agreed
   with the maintainer. Task 01 is a hard blocker for 03, 05 and 06, so
   "unlocked, therefore task 01 does not apply" is not available — the tasks'
   dependency graph does not care why the check cannot pass.
4. Whether `verify_anchor.py`'s and `verify_artifacts.py`'s anchor gates apply
   unchanged (§10).

Reaching this point with 1–4 unwritten and starting the sweeps anyway is how a
port ships numbers nobody can defend. It costs a day of node time to stop here;
it costs the release not to.

### Step 7 — task 02, the reference sweep

```bash
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task references \
    --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
    --out artifacts/02/references/ > artifacts/02/logs/references.log 2>&1 &'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/02/references'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 02'
```

**All four categories.** An omitted `--category` is the realistic way scope
silently shrinks (`CLAUDE.md` §0). Resumable by construction: `shard_sweep.py`
skips a problem whose output already parses as JSON, so re-invoking the *exact
same command* continues. Never restart with different settings (prime directive
7). **Cost:** the MI350X elapsed for this sweep was 0.6 h — a bound on the wall
clock on a different part, not a prediction for yours. It was **not** 7-way:
`_provenance.visible_devices` across the 235 files in `artifacts/02/references`
spans all eight cards, GPU 0 included (0:32, 1:32, 2:29, 3:30, 4:26, 5:30,
6:27, 7:29), which §7 reserves for authoritative timing. Run yours `--gpus 1-7`
as written above and expect it to take longer than 0.6 h.

### Step 8 — task 05, tolerances

```bash
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task tolerances \
    --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
    --out artifacts/05-MI355X/ -- --seeds 10 --margin 1.25 --low-memory \
    > artifacts/05-MI355X/logs/sweep.log 2>&1 &'
env/solb bash -lc 'python scripts/apply_tolerances.py --calibration artifacts/05-MI355X \
    --out-workloads artifacts/05-MI355X/workloads --out-triage artifacts/05-MI355X/triage.md'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/05-MI355X/workloads \
    --pattern workload.jsonl'
```

Pass every path explicitly. `apply_tolerances.py` defaults input *and* output to
`artifacts/05`, which is the MI350X tolerance set; inheriting by default value is
the omission-shaped failure `CLAUDE.md` §0 describes. **Whether to re-derive
tolerances at all is a live question** — see §6. **Cost:** MI350X elapsed 5.3 h,
genuinely 7-way: `_provenance.visible_devices` over the 235 files in
`artifacts/05` is GPUs 1–7 only (31–35 problems each), with GPU 0 untouched.

### Step 9 — task 03, bounds (no GPU; run it in the shadow of 05/06)

```bash
env/solb bash -lc 'python src/solexbench_rocm/solar/gen_arch_yaml.py \
    --part MI355X --freq-ghz <F_REF> -o SOLAR/configs/arch/MI355X.yaml'
env/solb bash -lc 'python scripts/sol_bounds.py --part MI355X --freq-mhz <F_REF> \
    --arch-yaml SOLAR/configs/arch/MI355X.yaml --out artifacts/03-MI355X/t_sol.json \
    --jobs 32 --resume'
```

`gen_arch_yaml.py` requires `--part` (choices from `PARTS`) and `--freq-ghz`;
`sol_bounds.py` requires `--part` and `--freq-mhz` and **aborts if the arch
YAML's `freq_GHz` disagrees with `--freq-mhz`**, so the 13 SOLAR numbers cannot
be silently reused at the wrong clock. That guard is the reason `--freq-mhz` is
not optional even though, unlocked, it is a *reference* clock rather than a
measurement clock. Whatever you pass, say so in the artifact and in `STATE.md`:
the milliseconds in that file are at `F_REF`, and the fields that matter are
`compute_cycles` / `memory_bytes` / `dram_byte_per_sec`, which `t_sol_at` will
re-max at whatever clock you end up with.

Then the traffic tier and the cross-checks:

```bash
env/solb bash -lc 'python scripts/sol_traffic_floor.py \
    --arch SOLAR/configs/arch/MI355X.yaml --t-b artifacts/06-MI355X/authoritative \
    --out artifacts/03-MI355X/t_sol_traffic.json'
env/solb bash -lc 'python scripts/sol_cross_checks.py \
    --arch SOLAR/configs/arch/MI355X.yaml --t-b artifacts/06-MI355X/authoritative'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/03-MI355X/t_sol.json'
```

**Cost:** CPU-only, 32-way. Not estimable in advance: SOLAR's per-problem timeout
is 900 s and the total is dominated by how many problems fail their trace rather
than how many succeed.

### Step 10 — task 06, T_b

```bash
env/solb bash -lc 'nohup python scripts/shard_sweep.py --task tb-candidates \
    --gpus 1-7 --out artifacts/06-MI355X/candidates/ \
    > artifacts/06-MI355X/logs/candidates.log 2>&1 &'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/06-MI355X/candidates'
# then, on a VERIFIED-EXCLUSIVE node, one GPU, serially:
env/solb bash -lc 'python scripts/gpu_exclusive.py --gpu 0'      # MUST report exclusive
env/solb bash -lc 'python scripts/authoritative_tb.py \
    --candidates artifacts/06-MI355X/candidates --out artifacts/06-MI355X/authoritative --gpu 0'
```

`--gpu` sets `HIP_VISIBLE_DEVICES` for each child itself; do not also set it
outside. Run `gpu_exclusive.py` **before and after** the authoritative pass, not
just before — §8.2 is a worked example of what happens otherwise. **Cost:** MI350X
elapsed 10.2 h for candidates — **8-way, with GPU 0 in the pool**, which §7
forbids: the top-level `gpu` field over the 235 files in
`artifacts/06/candidates` counts 0:31, 1:44, 2:34, 3:19, 4:24, 5:27, 6:29,
7:27. Budget more than 10.2 h for a 7-way rerun. Then 11.4 h for the
authoritative pass (elapsed, overlapping the candidate sweep, so not exclusive
GPU time).

### Step 11 — tasks 04, 07, 08: re-run, do not re-derive

```bash
env/solb bash -lc 'python scripts/shard_sweep.py --task methodology-compare \
    --category L1 --gpus 1-7 --out artifacts/04-MI355X/compare/'
env/solb bash -lc 'python scripts/mxfp4_spike.py --out artifacts/07-MI355X/spike.json'
env/solb bash -lc 'python -m pytest reference/exploits/ -q'
for t in 04 07 08; do env/solb bash -lc "python scripts/verify_artifacts.py --task $t"; done
```

The 15 NVFP4 Quant problems stay deferred with evidence
(`artifacts/deferred.json`: `dataset_total 235, deferred_total 15,
shipped_total 220`). Re-run `mxfp4_spike.py` — the software path may have moved —
but do not quietly change the count: 220 means 220 everywhere.

### Step 12 — task 09, the manifest, and the board

```bash
env/solb bash -lc 'python scripts/build_manifest.py \
    --out artifacts/09-MI355X/manifest-v1.json \
    --t-sol artifacts/03-MI355X/t_sol.json \
    --t-sol-traffic artifacts/03-MI355X/t_sol_traffic.json \
    --t-b artifacts/06-MI355X/authoritative \
    --tolerances artifacts/05-MI355X'
env/solb bash -lc 'python scripts/verify_anchor.py --sample 20'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 09 --full'
# roots in leaderboard/sources.json; pass --agent-runs only to OVERRIDE it
leaderboard/.venv/bin/python leaderboard/ingest.py --part MI355X
leaderboard/.venv/bin/python -m pytest tests/leaderboard -q
```

Every path explicit (§3.6). **`build_manifest.py` will hard-exit on the first
line unless `provenance.f_lock_mhz()` resolves** — on MI355X that means
`SOLEXBENCH_F_LOCK_MHZ` is exported from a real measurement, per §3.4 and the
decision written at step 6. Note also `ingest.py`'s fixed `MANIFEST` constant
(§3.5) — resolve that before you get here.

On `--agent-runs`, the risk has **inverted** since D24 and the old advice is
wrong. `ingest.py:1501` gives `--sources` a default of
`leaderboard/sources.json`, which the ingest reads whenever `--agent-runs` is
absent, so omitting the flag no longer drops the roots listed there. What drops
them is passing it: `--agent-runs` **overrides** `sources.json` rather than
adding to it (`:1496-1500`, and the precedence comment at `:1512-1515`), so an
incomplete `--agent-runs` is now the way to get D24, not the way to avoid it.
`SOLEXBENCH_AGENT_RUNS` is additive on top of whichever won. The safe move on a
new part is to put the roots in `sources.json` and pass no flag; the ingest
prints every root with where it came from either way, and refuses to publish a
board that has lost a submission unless `--allow-drop` says so.

Until `db/solbench-MI355X.db` exists the part switch lands on the honest empty
state — a first-class page saying nothing has been measured on MI355X, which
`leaderboard/app.py` already resolves against this file as that part's runbook
(`todo_runbook`). That rendering is correct and must not be faked.

`check_coverage.py` after **every** sweep, not at the end. A gap listed in
`artifacts/deferred.json` with a reason is a decision; a gap without one is a
bug.

---

## 6. What transfers from MI350X, and what does not

"Transfers" means use it as-is. "Confirm" means it is expected to hold and is
cheap to check. Nothing with a millisecond or a megahertz in it is in the first
column.

| Result | Transfers? | Why |
|---|---|---|
| The harness port (`src/sol_execbench/`) | **Yes** | Vendor logic keys off `gfx950` / `torch.version.hip`, not the SKU. `problem_packager.py` maps both parts to `gfx950`. |
| `src/solexbench_rocm/activity/` and the rocprofiler shim (task 04) | **Yes** | CPU-verified, mutation-tested; the HSA clock domain is not part-specific |
| `src/solexbench_rocm/parts.py` | **Yes, it is the model** | Already a complete MI355X entry, `ARCHITECTURAL`/`PART`/`MEASURED` tagged per field, and `detect_part()` raises rather than defaulting |
| `src/solexbench_rocm/t_sol_at.py` and the split bound terms | **Yes** | Pure arithmetic, part-independent, CPU tests green |
| `--offload-arch=gfx950`, `-ffast-math`, `-lamdhip64` | **Yes** | Same ISA target, same toolchain |
| `LLC_BYTES[gfx950] = 256 MiB` → 512 MiB flush | **Yes, confirm** | Same die. `roofline_probe.py --llc-sweep`; the cliff should land near 256 MiB |
| Dataset census (235: 94/82/33/26) | **Yes** | A property of the dataset |
| The exploit corpus and static source screen (task 08) | **Yes** | torch-level; re-run it, do not re-derive it |
| `scripts/gpu_map.py`'s PCI translation | **Yes** | Resolves live through PCI bus identity; the hostname table in its header is a comment |
| **Any `F_LOCK`** | **NO** | §1. And 1650 was never one: it is a request, and since §3.3 the code says so — `f_lock_mhz` returns `None` for MI355X. |
| `artifacts/01/` (MI350X floors, `det1600-*`, `f_lock_by_datapath.json`) | **NO** | Every file is a clock on a 1000 W air-cooled part |
| `artifacts/00/` MI350X roofline and node report | **NO** | Default clocks, different spec peak, different power budget |
| `artifacts/03/` `t_sol_ms` | **NO** | Contains a clock |
| `artifacts/03/` `t_sol_cycles` | **NO, not in general** | See below — this is widely misstated in this repo |
| `artifacts/03/` `macs`, `memory_bytes`, `precision` | **Yes** | `device="meta"` trace, identical die. This is the big saver |
| `artifacts/05/` tolerances | **Yes, with a caveat — but see below** | Numerics are a property of the gfx950 ISA and the torch build, not the clock |
| `artifacts/06/` every `T_b` | **NO** | A wall-clock time at MI350X's clock on MI350X silicon |
| `artifacts/09/` every manifest (v1, v1.1, v1.2) | **NO** | Every `t_b` is at the wrong clock, and v1.1/v1.2's rebuilders hardcode 1300 (§3.1) |
| `artifacts/10/`, `artifacts/11/`, `artifacts/12/` | **NO as numbers, YES as questions** | Agent runs, bound diagnostics and clock A/B on the other part |
| `leaderboard/db/solbench-MI350X.db` | **NO** | One database per part, enforced in the filesystem |
| Script fixes and bug fixes | **Yes** | Real bugs, fixed in code |

**The tolerance caveat is bigger than it looks.** `artifacts/05` was *inherited*
rather than re-derived once already, on the argument above. That argument is
falsifiable cheaply — run task 02's reference sweep against the MI350X
tolerances and see whether anything fails — and doing so is worth an hour before
committing to a 5-hour recalibration. But note the sharper reason to re-derive
on this part: `TODO.md` records that the tolerance derivation measures
run-to-run spread **in-process only**, and eager PyTorch is not deterministic
*across* processes because hipBLASLt/MIOpen algorithm selection can change. An
unlocked node adds a second source of cross-process variation on top of that.
Decide deliberately and write the decision down.

### The T_SOL shortcut, stated correctly

`CLAUDE.md` §6 and `gen_arch_yaml.py`'s docstring both say *"T_SOL in cycles is
invariant to F_LOCK — compute it once, convert to milliseconds by one
division."* **That is true only for compute-bound workloads.**
`DRAM_byte_per_cycle` is *derived* as `bytes_per_sec / F`, so:

```python
exact_cycles = max(macs / MAC_per_cycle,             # cycles: invariant in F
                   memory_bytes / DRAM_byte_per_cycle)   # cycles: proportional to F
```

A memory-bound workload's bound is invariant in **milliseconds** and scales with
F in **cycles** — the opposite of the shortcut. In MI350X's
`artifacts/03/t_sol.json` today, of 3739 workload records **1163 are
`"bottleneck": "memory"`, 1835 compute, 741 with no bottleneck recorded** — so
memory is 38.8% of the 2998 that have one. Rescaling their cycle counts to a new
clock would inflate their bounds by the clock ratio, and the bottleneck can flip
as F moves, so **neither column is safe on its own**. This is precisely why
`sol_bounds.py` now emits both terms (§4.1), and it is the one place where the
unlocked-node requirement and a pre-existing MI350X documentation defect are the
same fix.

Re-run `sol_bounds.py` (§5 step 9) rather than converting the MI350X artifact.
It is CPU-only, runs 32-way, needs no GPU, and the MI350X artifact predates the
split so `t_sol_at` refuses it anyway.

---

## 7. GPU discipline on this part

**GPU 0 for authoritative timing only. Everything else on 1–7.** Pin with
`HIP_VISIBLE_DEVICES` and record which GPU produced every timing artifact.

Two things specific to MI355X:

* **Sibling interference must be re-measured and must not be inherited.** MI350X
  measured **−0.11%** and concluded sweeps and authoritative timing may share the
  node. That verdict is a measurement, not a rule. On `mia1-p02-g10`,
  `docs/methodology.md` §3 records unlocked sensitivity to seven busy neighbours
  at **1.0%** — but that is a different quantity measured a different way, and
  the same section records **up to −15%** under a lock. Also note that a burst
  probe's duty cycle can keep the card on the fast branch: whatever you validate
  with is what you have validated.
* **CPU contention is the resource that actually bit.** An earlier MI355X
  session's `T_b` was voided not by GPU interference but by running the
  authoritative pass on GPU 0 while an agent sweep saturated 120 CPUs.
  `torch.compile` and Triton autotuning are CPU-bound, so a compile-heavy timing
  run beside compile-heavy agents measures the scheduler. Quiet means quiet on
  both sides of the PCIe bus.

---

## 8. The traps

### 8.1 The PCI-vs-torch device ordering scramble

`rocm-smi` and `amd-smi` order devices by PCI bus. torch does not. Passing a
torch index to `-d` sets one card and measures another, and **this repository has
lost three findings to it**: `STATE.md` D11, D20's clock alignment, and PR #2's
own first version of §3 — which reported the setpoint as a *no-op on two cards*
because the request went to a neighbour while the loaded card sat at its own
boost clock. The correction is kept visible in `docs/methodology.md` §3 and in
`src/solexbench_rocm/t_sol_at.py`'s docstring rather than edited away.

`scripts/gpu_map.py` resolves this live through PCI bus identity and stays
correct under `HIP_VISIBLE_DEVICES`: `torch_to_amdsmi()` at `:39`,
`amdsmi_handle(torch_index)` at `:76`, `torch_to_rocm_smi()` at `:113`.
`scripts/clock_calibrate.py:80` (`smi_device_index`) and
`scripts/gpu_parity_check.py:191` now route through it.

**Two call sites still do not**, and they are in the module that applies the lock
during an actual run: `src/sol_execbench/core/bench/device/amd.py` builds
`["rocm-smi", "--setperfdeterminism", str(gpu_mhz)] + ["-d", str(gpu)]` with a
**torch** index in `lock_clocks()`, and `unlock_clocks()` does the same with
`rocm-smi -r -d <torch index>`. Under this policy you should not be calling
either — but if anything in your path does, it is addressing the wrong card. Fix
it or assert it is unreachable; do not leave it ambiguous.

Also note the tool choice is *not* what fixes this: `docs/methodology.md` §3 says
`amd-smi` and `rocm-smi` enumerate identically, and both differ from torch.
Prefer `amd-smi` for other reasons, and translate regardless.

### 8.2 Exclusivity — a worked example of how easy it is to lose

`STATE.md` **D61** is a retraction, written this month, by a session that knew
the rule. Quoting it:

> At **18:39 and 18:40 on 2026-08-12** two foreign tenants started on this node:
> `sglang::scheduler` (pid 2617421, 194 GB resident) and
> `ray::MegatronTrainRayActor` (pid 2638981). Both were still running at 21:00,
> with GPUs 0-3 reading 100% utilisation.

Both of that session's "solo GPU 0" re-times — 19:32 and 20:00 — postdated them,
and `scripts/gpu_exclusive.py --gpu 0` afterwards reported
`gpu_id 36538 is NOT exclusive -- 2 foreign process(es)`. The headline claim
built on those two numbers (a published anchor not reproducing at 2.021×) was
retracted, not because it is known false, but because it is **not established**.
What D61 states about the cost, in its own words: both re-times are marked
`CONTAMINATED`, the 2.021× "does not survive", and the re-time it owes has to
be redone on a verified-exclusive card. D61 also records that the tenants ran
past 21:00 and that a candidate sweep spanning 16:13–20:10 has "roughly the
last third" suspect, so the loss is not confined to the two re-times.

The rule that follows: **run `scripts/gpu_exclusive.py --gpu 0` before *and*
after every authoritative pass, and record both outputs in the artifact.** A
check run only at the start cannot see a tenant that arrives in the middle.
`scripts/runners/time_tb_candidates.py` has no exclusivity check at all —
invoking it directly routes around `gpu_exclusive.py`,
`guard_authoritative_gpu.py` and `retime_parallel.py`'s `foreign_on()` check
alike, which is exactly how D61 happened. Adding a refusal there is the one item
on the whole list that prevents a *future* invalid number rather than correcting
a past one.

### 8.3 Container-vs-image staleness

`env/solb` recreates the container when the image ID changes — **unless a sweep
is running inside it**, in which case it warns and keeps the old container. A
long sweep started before an image rebuild therefore finishes on the *old*
stack while the repo, and anything you run afterwards, is on the new one. Read
that warning; do not work around it. On a daemonless node the equivalent risk is
worse because there is no image ID at all, which is why `env/solb-native` calls
`env/check_stack.py` and refuses rather than assuming. And prime directive 6:
if the pinned ROCm/torch combination is incompatible with something, record the
incompatibility — silently upgrading torch changes every measured baseline.

### 8.4 The FlashInfer trace blobs

`scripts/fetch_flashinfer_traces.py` fetches external blobs that are not in git
and not in the dataset. Without them, **9 of the 26 FlashInfer problems fail at
run time as ordinary runtime errors** — which reads as a port defect and is not
one. `FLASHINFER_TRACE_DIR=/work` must also be set, because the eval driver
resolves trace paths against the staging directory rather than the CWD;
`env/solb` sets it and a hand-rolled environment will not.

### 8.5 Two smaller ones

* **`determinism-sweep` never resets the setpoint.**
  `scripts/clock_calibrate.py`'s `determinism-sweep` applies a setpoint per step
  and leaves the last one in place. `STATE.md` records the consequence without
  naming the part — `STATE.md:2244-2246`, "an unreset determinism sweep left
  **a node** at a 1900 MHz setpoint, `provenance.f_lock_mhz()` returned the
  preset's 1640 without reading a device, and 143 artifacts measured at
  ~1860 MHz were stamped 1640"; and `STATE.md:2312-2314`, "**On another node**
  an unreset sweep left a 1900 MHz setpoint while the preset returned 1640;
  143 artifacts were measured at ~1860 and stamped 1640". Eleven hours of
  measurement, every value about 12% faster than the number it claimed. Which
  part, and which kind of artifact, is not recorded anywhere in the repo, and
  neither is a preset of 1640: `build_manifest.py:139-143` retells the same
  story in the same words, `CLOCK_LOCK_PRESETS` (`device_config.py:94`) has no
  1640 entry (`git log -S1640` over that file is empty), and every other 1640
  in tracked `.py` is a synthetic value in
  `tests/scripts/test_build_manifest.py`. So do not assume it was this part.
  If you run it at all —
  and under this policy you should only run it to *document* the derate, not to
  choose a lock — bracket it with `rocm-smi -r` and verify. **The table is not
  the hardware.**
* **`determinism-sweep` needs the privileged wrapper and writes its artifact as
  root** into an NFS-mounted repo. `chown` it back, or the next unprivileged run
  cannot overwrite it.

---

## 9. What will still be open when you are done

None of these are MI355X problems. They follow the port, and each will be wrong
on MI355X in the same way and for the same reason.

* **D18 / D42 — the declared-traffic tier over-counts.** It prices every declared
  input at its full allocation regardless of what the kernel reads. `CLAUDE.md`
  states the MI350X blast radius as **328 workloads across 38 problems**, still
  resting on that tier after v1.1 fixed two paged problems individually rather
  than fixing the tier. The defect is in the traffic model, not the clock, so it
  crosses parts intact. Worse, it interacts with unlocked scoring in the unsafe
  direction: an over-counted *memory* term is also the term that does **not**
  shrink with clock, so on a boosting card it wins the `max` more often than it
  did at a pinned clock, and the over-count propagates to more workloads. If you
  can fix the tier before bringing up MI355X, do that first.
* **D39 — the bound check is one-sided.** Nothing may beat a bound; nothing checks
  that a bound is tight. `CLAUDE.md`: 827 workloads (22.3%) sit above 100×
  headroom, where `S` is a PyTorch comparison with no roofline content. Marked
  (`bound_quality`) at ingest, which means a consumer of the manifest alone
  cannot see it.
* **D43 — `rocprofv3 --pmc` hangs in this container**, so the counter path to an
  *independent* traffic measurement is closed. This matters more than it sounds:
  `CLAUDE.md` §6 says a self-consistent bound and anchor cannot detect a shared
  error, and this is the tool that would break the tie. The shim is not
  implicated. The counter-free route is a minimal independent kernel, timed.
* **D20 — matmul timing is bimodal and unexplained** on MI350X: 0.13% of
  iterations cost 3.9–4.5×; the clock hypothesis was tested and falsified;
  hipBLASLt kernel selection is the untested suspect. Two upstream tests are
  skipped behind it because their thresholds were measured on RTX 4090 / B200.
  Re-deriving them on MI355X (`scripts/derive_timing_variance.py`) is a second
  data point on an open question and is worth the twenty minutes.
  `docs/methodology.md` §7 adds a control D20 did not have: the short-window cost
  is 21.2 µs on a 4096³ GEMM, 13.5 µs on a 1024³ one, and **0.6 µs** on
  elementwise `a + b` — a 33–36× spread that puts it on the GEMM path, not the
  clock and not launch overhead generally.
* **The 15 NVFP4 Quant problems** stay deferred with evidence. NVFP4 has no ROCm
  kernel path and an MXFP4 twin is a re-specification, not a translation.
* **`scripts/verify_artifacts.py` has no test coverage**, and it is the
  acceptance gate for all ten tasks. A bug in it does not fail loudly; it passes
  quietly. Highest leverage-per-hour item on the list.
* **`origin/feat/agent-scoreboard`, 30 commits, unmerged** — §2.
* **No full-benchmark agent baseline on either part.** `docs/agent-baseline.md`
  prices it.

---

## 10. What is explicitly NOT decided

Each of these needs data or a maintainer, and each must be revisited before the
first MI355X `S` is published. Do not resolve any of them by picking a
convenient default.

1. **What task 01's acceptance check becomes on an unlocked node.** As written it
   requires a preset, agreement between the preset and `STATE.md`, and every GPU
   at the preset's setpoint (§3.4). All three presuppose a lock, and task 01 is a
   hard blocker for 03, 05 and 06. Amending it is a methodology change
   (prime directive 7).
2. **Where the per-measurement clock comes from — or which of §4.3's
   alternatives replaces it.** This is the load-bearing one. The 1–13 ms timed
   window is shorter than any available sampler and runs 1.04–2.04× off steady
   state unlocked (§4.3), so "read the clock back per measurement" is currently
   a design intention, not a capability.
3. **Whether the timed window is lengthened.** §4.3 option 1 fixes both the clock
   sampling and the short-window bias, and breaks comparability with upstream's
   50 iterations. A single decision with two consequences.
4. **Whether `S` is published at all for MI355X v1**, or only speedups and
   headroom (§4.3 option 4).
5. **How `T_b` and `T_k` are normalized when they run at different clocks**
   (§4.4). Unresolved, and it is the difference between a methodology and a
   farmable scoreboard.
6. **Whether `artifacts/` gains a part dimension** (`artifacts/<part>/<task>/`).
   Two parts are already mixed inside `artifacts/00` and `artifacts/01` (§2), and
   `verify_artifacts.py:246` is one MI355X floor run away from silently
   corrupting the MI350X task-01 gate. §5 step 3 works around it; it does not fix
   it.
7. **Whether the MI350X artifacts get retro-stamped with an explicit
   `_provenance.part`.** Today `manifest-v1.2.json`'s part is an *inference* from
   a device string — its `_provenance.part` and `_provenance.host` are both
   `None` — while `leaderboard/ingest.py`'s `manifest_part()` already prefers an
   explicit field. Cheap to add going forward; changing tracked bytes is a
   maintainer call.
8. **Whether `--setperfdeterminism` is reliable on *any* CDNA4 part.** D55 says
   the setpoint never binds on MI350X; PR #2 says it binds destructively on
   MI355X. Same API, same die, opposite failures, and neither has been tested at a
   setpoint clearly *below* the clock the card would otherwise reach. Half of
   this has already been conceded in source: `requested_is_achieved` (§3.3)
   says per part whether a request may be *read* as an achieved clock, and it
   is `False` on MI355X. What it does not settle is whether `gpu_clk_mhz` — a
   clock you request and then apply — is worth keeping on CDNA4 at all. If the
   answer is "no", the concept is the thing to remove, not a number to update.
9. **Whether the tolerance set is inherited or re-derived** (§6), and if
   re-derived, the written-down ceiling rule that stops a re-derivation from
   becoming "loosen it until the kernel passes" (task 05 forbids exactly that).
10. **Whether `verify_anchor.py`'s and `verify_artifacts.py`'s anchor gates still
    mean the same thing.** PR #2 changed what `total` counts in
    `anchor-verification.json`, and `verify_artifacts.py` reads `passing`/`total`
    and requires `>= 0.95`. The tracked artifact predates the change, so the
    loosening is invisible until the next `verify_anchor` run — on either part.
    Decide explicitly whether the exemption applies, and teach the gate about it
    either way.

---

## 11. Checklist

Each line is a real command. Paste its real output into `STATE.md`.

```bash
# stack
env/solb bash -lc 'python -m pytest tests/ -q'                       # 519 passed, 75 skipped
env/solb-native python env/check_stack.py                            # on a daemonless node

# node, unlocked
env/solb bash -lc 'python scripts/gpu_parity_check.py --n-gpus 8 --out artifacts/00/gpu-parity-<HOST>.json'
env/solb bash -lc 'python scripts/unlocked_clock_probe.py --out artifacts/01/unlocked-clock-<HOST>.json'
env/solb bash -lc 'python scripts/burst_clock_probe.py --mode unlocked --out artifacts/01/burst-clock-<HOST>.json'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 00'

# THE DECISION POINT -- four written decisions in STATE.md before this line (§5 step 6)

# the port, on this part
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/02/references'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 02'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 04'

# bounds, tolerances, anchors -- every path explicit
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/03-MI355X/t_sol.json'
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/05-MI355X/workloads --pattern workload.jsonl'
env/solb bash -lc 'python scripts/gpu_exclusive.py --gpu 0'          # before AND after task 06
env/solb bash -lc 'python scripts/check_coverage.py --artifacts artifacts/06-MI355X/candidates'

# quant, red team, release
env/solb bash -lc 'python scripts/verify_artifacts.py --task 07'
env/solb bash -lc 'python -m pytest reference/exploits/ -q'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 08'
env/solb bash -lc 'python scripts/verify_artifacts.py --task 09 --full'

# the board
leaderboard/.venv/bin/python leaderboard/ingest.py --part MI355X   # roots via sources.json
leaderboard/.venv/bin/python -m pytest tests/leaderboard -q
```

**Done means:** `--task 09 --full` passes or its failures are enumerated in
`STATE.md` with reasons; `check_coverage` exits zero for every sweep; every
deferral is in `artifacts/deferred.json` with a reason and the same count appears
in every document; `db/solbench-MI355X.db` exists so the part switch lands on
data; and every one of §10's ten open items is either resolved in writing or
still listed there. Anything short of that is recorded as a blocker, not
smoothed over.

---

## 12. Two bound defects found in `manifest-v2` (2026-08-15)

Both are fixed in code / re-derived; **neither is in a published artifact yet.**
`artifacts/09-MI355X/manifest-v2.json` still carries both.

### 12.1 The `T_SOL <= T_b` gate had no effect under the unlocked basis

41 workloads across 4 problems shipped in v2 with a **published T_SOL above
their own measured T_b** — impossible for a lower bound, and it inverts the
score: with `T_b - T_SOL < 0`, `S = 1/(1 + (T_k - T_SOL)/(T_b - T_SOL))` rises
with `T_k`, so a slower kernel scores higher, with a pole at
`T_k = 2*T_SOL - T_b`.

| problem | workloads | worst `T_SOL/T_b` | tier rejected, and leaked |
|---|---|---|---|
| `L1__029_mamba_conv1d_with_gating` | 15/16 | 3.04 | solar_fused |
| `L1__006_hyena_depthwise_conv1d_split_gate` | 12/16 | 4.06 | solar_fused |
| `L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion` | 8/16 | 3.43 | declared_traffic |
| `L2__035_convnextv2_block_with_grn` | 6/16 | 2.20 | declared_traffic |

**Mechanism.** `combine_bounds` rejects a tier above T_b by setting its
`t_sol_cycles` to `None` — but it then passed *both* raw tier records to
`_reclock_terms`, which unions `max(compute_cycles)` and `max(memory_bytes)`.
Under the unlocked basis the *published* bound is re-evaluated from that union
at the bracket's minimum clock, so the rejected tier's term won and the shipped
number **was the rejected bound**, wearing the surviving tier's `t_sol_source`
label. That is why the records read `t_sol_source: declared_traffic` while
`t_sol_cycles_published == compute_cycles` of the rejected SOLAR tier. Under the
locked basis (MI350X v1/v1.1/v1.2) the point bound is taken from the surviving
tier directly, so those manifests are not affected.

**Fix.** `scripts/build_manifest.py`: a rejected tier's terms are dropped with
it, and the record now carries `t_sol_tier_rejected_above_t_b`. Regression test:
`tests/scripts/test_build_manifest_reject_leak.py`. Rebuild of v2 with the fix
alone: **41 violations -> 0**, 173 workloads across 17 problems get a smaller
(always smaller) bound, `219/235` problems and `3701` workloads unchanged,
`no_valid_bound` still 0 — nothing loses scoreability.
Candidate: `artifacts/09-MI355X/candidate-v3-gatefix.json`.

### 12.2 `artifacts/03-MI355X/t_sol.json` is pre-D37: grouped conv priced as dense

The SOLAR tier only sat above T_b on those conv-flavoured problems because it is
the **uncorrected** arithmetic. `scripts/sol_bounds.py` applies
`solexbench_rocm.solar.conv_groups`, but the artifact dated 2026-08-14T19:18
does not carry the correction: its `macs` are identical to MI350X's pre-D37
values. Re-running today's code on the same arch YAML reproduces exactly the
corrected `macs` in `artifacts/11/d37/` (ratio 1.000000 on all 6 problems,
96 workloads):

| problem | `t_sol.json` / corrected macs |
|---|---|
| `L1__005_conv_gated_projection_with_causal_conv` | 1.999 |
| `L1__006_hyena_depthwise_conv1d_split_gate` | 768.000 |
| `L1__029_mamba_conv1d_with_gating` | 4.999 |
| `L2__035_convnextv2_block_with_grn` | 6.698–7.069 |
| `L2__051_...hyena_complete_forward_block` | 3.173–3.256 |
| `L2__058_mamba2_selective_scan` | 4.663–4.754 |

The declared-traffic tier is **not** implicated: it charges each declared tensor
exactly once (`L1__029` at `batch=16, seq=512` sums to 1,208,205,312 B by hand,
to the byte). No conv weight or state tensor is priced per timestep.

Re-derived bounds: `artifacts/03-MI355X/d37/*.json` (CPU, `device="meta"`, no
measurement), merged copy `artifacts/03-MI355X/t_sol-d37.json`, manifest
candidate `artifacts/09-MI355X/candidate-v3-gatefix-d37.json` — also 0
violations, and it moves bounds in *both* directions (SOLAR shrinks, but it now
survives the gate and wins on problems where only the loose traffic tier had).

**Owed:** decide whether v2 is withdrawn or superseded, then rebuild from
`t_sol-d37.json` with the gate fix. Until then, treat scores on those 4 (bound
inverted) + 6 (grouped conv) problems as not results.
