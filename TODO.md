# TODO — what is not yet built

Everything here is a known gap, deliberately left rather than forgotten. Nothing
in this repo silently pretends to be finished.

**Rewritten 2026-08-06.** The previous version described the pre-hardware state:
five scripts "never run on hardware", four sweep runners "not written at all",
the harness port "not written", `reference/b200-tolerances.json` missing,
tb-candidates and the exploit corpus listed as prep work, and F_LOCK unmeasured.
All ten tasks are now `done` and every one of those items exists. A TODO that
lists finished work as pending is worse than no TODO: it hides the real gaps
among items a reader will check, find complete, and stop trusting the file over.

For per-item detail see `STATE.md`; the D-numbers below point into it.

---

## Wrong numbers that ship in v1

These are in the manifest and produce scores. The scores are not usable.

**Fixed in manifest v1.1 on 2026-08-10 — 13 down to 6. Read `STATE.md` D36 for what moved, then D35 for why.** The table below describes v1, which is frozen and still reports 13. Under v1.1 the paged problems (D18) and five of the compute-bound ones are corrected; what remains is `L1__005`, `L1__006`, `L1__054`, `L1__057`, `L2__045` and a 1% residue on `L2__073`.

**Read `STATE.md` D35 before this table.** The thirteen were diagnosed on
2026-08-09 and they are not thirteen defects. Six of them are one cause that is
not a SOLAR error at all — `T_SOL_ms` divides cycles by a single F_LOCK of
1300 MHz, and 1300 MHz is the clock the card holds under a dense bf16
matrix-core load and not under everything. Light fp32 work clocks to
1439–1586 MHz, so those bounds are 10–22% too large and the six are simply
where a kernel got good enough to cross one. Three kernels that have *never*
been beaten clock just as high, one of them higher than any violation on the
board, so the list below is not the set of inflated bounds — it is the visible
part of it. **All 759 compute-bound fp32 workloads are scored against a bound
10–22% too generous.** Two of the rows below (D18) are confirmed to four
significant figures. What is genuinely undiagnosed is five problems, marked
below.

| what | scope | where |
|---|---|---|
| **Paged-attention T_SOL over-counts traffic.** The declared-traffic tier prices a paged KV cache at full allocation; the kernel gathers 34 pages of 989,669. | 6 problems, **249 scoreable workloads** | D18 |
| **`L1__005` bound beaten by 1.09–1.15×.** Compute-bound SOLAR roofline ~15% too slow. Not paged; D18 does not explain it. | 4 of 16 workloads | D21 |
| **`L1__035` bound beaten by 1.003–1.013×.** Total headroom is 1.008, so there is almost no scoring range; may be a bound too tight to measure against rather than a wrong one. Needs separating from `L1__005` — they are probably different defects. | 2 of 16 workloads | D21 |
| **Nine more bounds beaten**, found as a real optimizer covered the benchmark: `L1__006`, `L1__057`, `L2__030`, `L2__035`, `L2__045`, `L2__068`, `L2__073`, and — once coverage reached 220 — `FlashInfer-Bench__018` and `L1__054`. Undiagnosed; not yet separated into D18-style over-counting vs D21-style roofline error. | 72 workloads across 11 problems in that run | D31, D31b |
| **`L2__051` bound beaten**, found by `gpt56-40` on 2026-08-09. A different model, on 40 problems rather than 220, exposed one that 220 problems of GLM-5.2 did not. Undiagnosed. | 4 workloads | D31c |

The v1.1 fix for D18 is to derive paged traffic from the page table rather than
from `num_pages`. D21 has no fix yet, only a diagnosis, and the eight new ones
have neither. **v1 marks three of these thirteen**; the other ten are marked
nowhere in the shipped manifest and are known only from this file, `STATE.md`
D31/D31b/D31c, and the run pages that found them.

The count has risen every time a stronger optimizer covered more of the
benchmark — 3, then 10, then 12, then 13 — which is the finding, not an
accident of this run. A bound is only shown to be wrong by a kernel that beats
it, so the number of *known* bad bounds is a lower bound on the number of bad
bounds, and it tracks how hard anything has tried rather than how many there
are.

`L2__051` sharpens that, because it is not more coverage finding more: it is
**less** coverage finding more. `gpt56-40` attempted 40 problems where
`glm-sweep-2` attempted 220, and still turned up a bound the larger run missed.
So the count does not track effort along one axis that can be exhausted — a
second optimizer with different habits is its own search direction. Any estimate
of how many bad bounds remain that is extrapolated from one model's sweep is
extrapolating from one direction.

## The task-06 sweep does not fully reproduce

Re-running the 8 problems whose candidate sweep left a variant short,
**10 of 32 variant×problem cells changed verdict** — 6 went from failing to
passing every workload at identical coverage, 1 went from 9 passes to a
timeout. `Quant__011`'s recorded `passed=0 over 3 workloads` was a driver that
died after three, not a variant that failed.

Those 8 were *selected* for being incomplete, so this is not a rate for the
whole sweep. But it means the 523 `v2` and 581 `v3` `INCORRECT_NUMERICAL`
failures **have not been shown to be stable**, and nothing should describe them
as "torch.compile disagreeing with eager" until a repeat sweep says they
reproduce. New artifacts in `artifacts/06/candidates-gapfill/`, kept out of
`candidates/` because a new timing there can move `T_b` and every score under
it. See `STATE.md` D28.

## `f_lock_mhz: null` in the roll-up artifacts

**This is not a missing clock preset, and it is not a task-01 gate failure.**
Earlier versions of this file, `CLAUDE.md` and `DESIGN-v2.md` all said it was.
`CLOCK_LOCK_PRESETS` has carried
`"AMD Instinct MI350X": ClockPreset(gpu_clk_mhz=1600, dram_clk_mhz=None,
achieved_gpu_clk_mhz=1300)` since commit **2cdb7b0** (2026-08-03 20:36 UTC),
and `verify_artifacts.py --task 01` reports *11 checks, 0 failed* (re-run
2026-08-06). The only failing gate anywhere is task 03's.

What is actually true is two separate things, neither of which loses a
measurement:

**(a) Artifacts written before 2cdb7b0 landed.** Everything in `artifacts/00/`
and `artifacts/01/` was written between 18:53 and 20:30 UTC on 2026-08-03 —
before 20:36. Their provenance shows `torch.available: true` and eight
`AMD Instinct MI350X` devices, so `get_clock_preset()` was called and returned
`None` because the table genuinely had no entry yet. 20 files. They are
history and are correct as history; the clock they were taken at is what task
01 was in the middle of establishing.

**(b) Artifacts written by a host-python process, after the preset existed.**
8 files: `artifacts/10/{pilot8,glm-run1,submitted-apitest}/scored.json`,
`artifacts/10/pilot8/{run,cost-report}.json`, and
`artifacts/02/timing-{variance-amd,stall-probe,stall-clock}.json`. Their
provenance says `python: 3.11.7`, `torch: {"available": false}`,
`rocm.version: 7.15.0` — the *host* interpreter, not the pinned container
(`python 3.12.3`, `torch 2.9.1+rocm7.2.0`, `rocm 7.2.0`). `python3 -c "import
torch"` on this host raises `ModuleNotFoundError`, so `provenance.f_lock_mhz()`
falls through its `except Exception` and returns `None`. Verified: with
`SOLEXBENCH_F_LOCK_MHZ=1300` exported, the same host call returns 1300.

Running on the host is deliberate, not a mistake. `agent_score.py`,
`agent_cost_report.py` and `agent_baseline.py` orchestrate: they shell each
kernel into the container through `env/solb` and never touch a GPU themselves
(`agent_score.py`'s own comment says so — it loads `sol_score.py` by file path
precisely because host python has no pydantic). The measurements those runs
contain are stamped correctly: every `artifacts/10/*/retimed/*.json` — written
*inside* the container by `agent_eval.py` — carries `f_lock_mhz: 1300`,
`python: 3.12.3`, `visible_devices: "0"`. Only the outer roll-up, which
aggregates rather than measures, is unstamped. The leaderboard header is
likewise unaffected: `meta.f_lock_mhz` is `'1300'`, read from the manifest.

The fix, in order of preference:

1. Have the three host-side scripts export `SOLEXBENCH_F_LOCK_MHZ` (or pass it
   through to `stamp()`) from the F_LOCK the container reported in the
   per-workload artifacts they just collected. That keeps the value measured
   rather than asserted.
2. Failing that, `provenance.f_lock_mhz()` could fall back to reading the
   device name from `amd-smi` instead of `torch.cuda.get_device_name(0)`, which
   would work in any process that can see the card. Note this only resolves the
   name → preset lookup; it does not observe the running clock.

What would be wrong is defaulting the field to 1300 in `stamp()`. A roll-up
written on a different part would then claim a clock it was never measured at,
and nothing downstream could detect it.

Adjacent, and a different gap: `artifacts/10/glm-run1/run.json` and
`artifacts/10/submitted-apitest/run.json` have **no `_provenance` block at
all**. They are hand-assembled / worker-assembled run descriptors rather than
`stamp()`ed artifacts, so the question above does not even arise for them. Only
`pilot8/run.json` went through `agent_baseline.py`.

## Never measured on MI355X

The port needs nothing; every measurement needs redoing. `tasks/01` first — it
blocks 03, 05 and 06 exactly as it did on MI350X. The `MI355X: 1650` preset is
from an earlier session on a different node and is labelled, not trusted.

`origin/feat/agent-scoreboard` carries 24 commits of MI355X work and is **not
merged**. Its data is deliberately absent from the leaderboard: its `T_b` is not
anchor-verified, so it has no `S` to publish.

## Coverage gaps that are not deferrals

* **Five backends accepted by the schema, never built through** — `ck`,
  `ck_tile`, `hipblaslt`, `miopen`, `aiter`. See `docs/backend-coverage.md`,
  which also lists the three defects the one `hip_cpp` seed found.
* **~~No full-benchmark agent baseline.~~ Closed 2026-08-08.**
  `agent-glm-sweep-2` covers **220 of 220** problems and all 3,717 scoreable
  workloads: 3,690 scored, **mean S = 0.6083**, 218 problems swept clean. It
  leads the board on the shared denominator (0.5921 against eager's 0.4536).
  Upstream's median SOL of 0.732 on B200 is **not** the comparison -- these are
  AMD-derived bounds and no cross-vendor number comparison is defensible; what
  is now available is a full-coverage agent result on this part.
  Two caveats travel with it: 168 of 220 sessions were stopped by the harness's
  1 h cap rather than choosing to stop, and 3 submitted a kernel that was
  mid-edit when the kill landed while a passing snapshot went unused -- so the
  figure is a floor by about three problems.
* **Golden references are capped by tensor size.** 165 `.pt` files; the rest of
  the 235 problems have workloads recorded as `skipped: N elements > cap` in
  `artifacts/golden/_report.json`. The report covers all 235 — the skips are
  recorded, not missing — but a size-capped golden set cannot check the largest
  workloads, which are the ones most likely to expose a layout bug.
* **15 NVFP4 Quant problems**, deferred with evidence for v1.1. Not a gap in the
  port: NVFP4 has no ROCm kernel path, and an MXFP4 twin is a re-specification,
  not a translation. `tasks/07`, `artifacts/deferred.json`.
* **~~D28 — the board under-reports 1,239 passing baseline workloads.~~
  Fixed 2026-08-07.** `ingest_variants()` now reads the per-workload `failures`
  list instead of painting a problem with its `all_passed` flag. torch.compile
  went 0.3414 → 0.4190 on the whole-benchmark scope and max-autotune 0.3174 →
  0.4034; all four variants now show the 220 problems they actually attempted,
  where two of them read as 218 and 213. No measurement changed. **What is
  still open**: 89 problems have passing workloads whose only timing came off a
  sweep GPU rather than GPU 0, so the board now scores some sweep timings
  alongside authoritative ones — labelled per row in `note`, but a re-time
  would remove the mixture. **≈2¼ h on GPU 0** at the measured 1.5 min/problem.

## Unexplained

* **D20 — matmul timing spread is bimodal on MI350X and the cause is unknown.**
  0.13% of iterations cost 3.9–4.5×. The clock hypothesis was tested and
  **falsified** (in-call spread 1.04× against a required 3.9×; the clock is
  steady at ~1450 MHz through the measurement). hipBLASLt kernel selection is
  the untested suspect. Two upstream tests are skipped behind this because their
  PASS thresholds were measured on RTX 4090 / B200 and no defensible AMD
  constant could be derived — re-specifying them needs the cause.
* **`mm[2048]`'s ~21× outlier did not reproduce** — 3 events in 3,600
  iterations, then 0 in 12,000.
* **D23 — `FlashInfer-Bench__014`'s authoritative re-time timed out at 1200 s**
  for `glm-run1`. Whether that budget is simply too small for a paged-prefill
  problem of that size, or whether it is another instance of D18's trouble on
  the same family, was not investigated.

## Service and tooling

* **`scripts/verify_artifacts.py` has no test coverage.** It is the acceptance
  gate for all ten tasks. A bug in it does not fail loudly — it passes quietly.
* **D24 — `ingest.py`'s default is lossy.** Without `--agent-runs`, every run
  kept outside the repo is silently dropped from the board. This has been
  introduced three separate times and patched at three call sites. The durable
  fix is a config the ingest reads by default, so "rebuild" cannot mean two
  different things.
* **The submission service has no sandbox.** `env/solb` is a reproducibility
  boundary, not a security one: submitted kernels run as the invoking user with
  the repo bind-mounted read-write. Authenticated internal users only; do not
  expose the port. `leaderboard/submit.py` states the boundary in full.
* **D29 — the external fleet's GPU-0 hold does not hold.** `dash-overlay`'s J2
  sweep placed 34 jobs on GPU 0 despite taking a scheduler reservation on it.
  Nothing published is affected — no authoritative timing overlapped them — but
  the property is unenforced, so the next overlap will be silent.
* **Nothing publishes an external run to the board.** The `dash-overlay` fleet
  writes the leaderboard's address into every job's payload, which reads as a
  pipeline and is not one: `sbt collect` → `scripts/agent_score.py` →
  `ingest.py` are three manual steps, and until all three are run the kernels
  exist only in `~/.jobd/jobs/<id>/kernel.py` — 281 of them on disk as of
  2026-08-07, none of them on the board.
* **Agent evaluation times the reference on every call.** `agent_eval.py` sets
  `benchmark_reference=True` so the agent can see its speedup. Measuring the
  reference once per sandbox would roughly halve every evaluation, which is what
  limits agentic optimization on the expensive FlashInfer problems. Changing it
  mid-programme would make runs incomparable, so it is a change for a clean
  batch.
