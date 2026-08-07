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

| what | scope | where |
|---|---|---|
| **Paged-attention T_SOL over-counts traffic.** The declared-traffic tier prices a paged KV cache at full allocation; the kernel gathers 34 pages of 989,669. | 6 problems, **249 scoreable workloads** | D18 |
| **`L1__005` bound beaten by 1.09–1.15×.** Compute-bound SOLAR roofline ~15% too slow. Not paged; D18 does not explain it. | 4 of 16 workloads | D21 |
| **`L1__035` bound beaten by 1.003–1.013×.** Total headroom is 1.008, so there is almost no scoring range; may be a bound too tight to measure against rather than a wrong one. Needs separating from `L1__005` — they are probably different defects. | 2 of 16 workloads | D21 |

The v1.1 fix for D18 is to derive paged traffic from the page table rather than
from `num_pages`. D21 has no fix yet, only a diagnosis.

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
* **No full-benchmark agent baseline.** Three runs exist (4, 8 and 24 problems);
  the largest covers 10.3% of scoreable workloads. Upstream's median SOL of
  0.732 is a result over the whole benchmark and is still not replicated.
  `docs/agent-baseline.md` prices what closing it would take.
* **Golden references are capped by tensor size.** 165 `.pt` files; the rest of
  the 235 problems have workloads recorded as `skipped: N elements > cap` in
  `artifacts/golden/_report.json`. The report covers all 235 — the skips are
  recorded, not missing — but a size-capped golden set cannot check the largest
  workloads, which are the ones most likely to expose a layout bug.
* **15 NVFP4 Quant problems**, deferred with evidence for v1.1. Not a gap in the
  port: NVFP4 has no ROCm kernel path, and an MXFP4 twin is a re-specification,
  not a translation. `tasks/07`, `artifacts/deferred.json`.
* **D28 — the board under-reports 1,239 passing baseline workloads.** Not a gap
  in what was measured: `ingest_variants()` paints every workload of a problem
  `FAILED` when the variant's per-problem `all_passed` is false, discarding the
  per-workload `failures` list the artifacts already carry. `torch.compile`
  really passed 3171 of 3694, not 2586 of 3717. Fixing the read costs no GPU
  time; putting the recovered rows on the board wants an authoritative re-time
  for the 89 problems that never got one, **≈2¼ h on GPU 0** at the measured
  1.5 min/problem. Full detail and the per-variant table in `STATE.md` D28.

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
