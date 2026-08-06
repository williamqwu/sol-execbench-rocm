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

## Blocking a clean task-01 acceptance

`CLOCK_LOCK_PRESETS` has **no MI350X entry**, so `lock_clocks()` refuses and
every artifact stamps `f_lock_mhz: null` — including both agent runs and
everything the submission worker produces. F_LOCK *was* measured (1300 MHz, in
`STATE.md` and the manifest); it was never added to the table the code reads.

This is the one remaining `--task 01` gate failure. It is a two-line change and
it has been deferred rather than done, which is why it is here and not silently
carried.

Related: `provenance.f_lock_mhz()` could stamp the *observed* clock instead of
the configured preset, which would make the field meaningful even where no
preset exists.

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
* **Agent evaluation times the reference on every call.** `agent_eval.py` sets
  `benchmark_reference=True` so the agent can see its speedup. Measuring the
  reference once per sandbox would roughly halve every evaluation, which is what
  limits agentic optimization on the expensive FlashInfer problems. Changing it
  mid-programme would make runs incomparable, so it is a change for a clean
  batch.
