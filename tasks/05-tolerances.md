# Task 05 — Tolerance calibration

**Goal:** AMD-derived `(atol, rtol, matched_ratio)` for every workload instance.

**This is mandatory, not optional.** Upstream's tolerances were calibrated by
repeated reference probing *on B200* with a 1.25× safety margin. MFMA
accumulation order, fast-math behaviour and SIMD width all shift the empirical
error distribution on CDNA4.

Copying B200 tolerances fails in both directions at once: false failures on
correct kernels, and — far worse for benchmark integrity — tolerances loose
enough to reward kernels that are wrong but fast.

## Preconditions

- Tasks 01 and 02 done.

## Method

Mirror upstream's procedure so the numbers mean the same thing:

1. Run each problem's reference across many seeds.
2. Perturb (re-run, different allocation, different launch order) and record the
   empirical error distribution between runs.
3. Take max observed error × **1.25**.
4. Emit an AMD `workload.jsonl` per problem.

### Use the float64 CPU golden references

Where `artifacts/golden/` has a float64 CPU reference for a problem, compare
against it as well as run-to-run. This separates two very different things:

- *AMD differs from NVIDIA* — expected, benign, a numerics difference.
- *AMD differs from correct math* — a bug, and one that run-to-run comparison
  alone cannot see because a deterministic wrong answer looks perfectly stable.

If goldens are missing, generate them (CPU-only, no GPU needed —
`scripts/gen_golden.py`) rather than skipping the check.

## Steps

```bash
# Long sweep. Shard it, log it, make it resumable, then go do task 03/04/07.
# ALL FOUR CATEGORIES. Quant needs tolerances exactly as much as L1 does —
# arguably more, since quantized outputs have the widest error distributions.
nohup python scripts/shard_sweep.py --task tolerances \
    --category L1 L2 Quant FlashInfer-Bench --gpus 1-7 \
    --seeds 10 --margin 1.25 \
    --out artifacts/05/ > artifacts/05/logs/sweep.log 2>&1 &
```

Then triage:

- **Nondeterministic references.** Some ops (atomics-based scatter, parts of MoE
  dispatch) may be structurally nondeterministic on ROCm. Where variance is
  structural rather than a bug, widen `matched_ratio` deliberately and document
  *per problem* why. Do not widen `atol` to hide it — that is the change that
  lets wrong kernels through.
- **Problems needing much looser tolerances than B200.** Each one is a finding.
  Investigate before accepting; a 10× looser tolerance usually means something
  is wrong, not that CDNA4 is noisy.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 05
```

Passes when: **all 235 problems covered** (per-category, enforced); every
workload has AMD-derived tolerances with provenance; no
tolerance was copied from upstream (the checker diffs against B200 values and
flags exact matches); every problem needing >2× B200's tolerance is individually
justified in `artifacts/05/triage.md`; golden-reference comparison recorded as
run or explicitly not-applicable per problem.

## Guard rails

- **Never loosen a tolerance to make a kernel pass.** Tolerances are derived
  from reference-vs-reference variance and nothing else. If a submission fails,
  that is the system working.
- Never reuse a B200 tolerance, even when it looks reasonable.
- Do not calibrate on a node whose clocks are unlocked or that is thermally
  unstable — re-check that task 01's lock is still in force before starting, and
  again after.
- If the sweep dies partway, **resume; do not restart with different settings.**
  Mixed-settings tolerance data is unusable.

## Outputs

- `artifacts/05/workloads/<problem>/workload.jsonl` — AMD tolerances
- `artifacts/05/triage.md` — per-problem justifications
- `artifacts/05/logs/`
