# Task 06 — Scoring baselines (T_b)

**Goal:** measure T_b, the optimized-PyTorch anchor, for every problem on
MI355X at F_LOCK.

T_b is the S=0.5 point in

```
S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))
```

so it sets the entire score scale. It must be **re-tuned and re-measured**, not
ported: the fastest PyTorch expression of a problem differs across platforms
because Inductor makes different decisions and different fusions are profitable.

## Preconditions

- Tasks 01 and 02 done. Task 05 helpful but not blocking (a candidate can be
  timed before its final tolerances exist).

## Method

Mirror upstream's optimization policy exactly, so T_b means the same thing on
both platforms:

- eager vs `torch.compile`, whichever is faster
- contiguity and layout hygiene
- **no handwritten kernels** — T_b is the PyTorch anchor, not a tuned kernel

Candidate variants are pre-authored in `reference/tb-candidates/` (they are
platform-independent PyTorch, so they were written without hardware). Your job
is measurement and selection, not authoring — this is a batch job, not a
creative one. Add variants only where the pre-authored set is clearly missing an
obvious formulation, and record what you added.

## Steps

1. **Sweep all candidates**, sharded:
   ```bash
   nohup python scripts/shard_sweep.py --task tb-candidates \
       --gpus 1-7 --out artifacts/06/candidates/ \
       > artifacts/06/logs/candidates.log 2>&1 &
   ```

2. **Authoritative pass on the winners.** Re-time the fastest variant per
   problem under full harness conditions — locked clocks, LLC flush, shifting
   allocator, subprocess isolation.

   **Consult task 01's interference verdict before scheduling this.** If sibling
   load measurably perturbs timings, this pass needs an otherwise-quiet node and
   cannot overlap the sweeps.

3. **Freeze** `t_b` per workload into the scoring manifest.

4. **Verify the anchor property.** This is the check that proves the scale is
   real:
   - submitting T_b's own implementation must score **0.5 ± 0.03**
   - the plain reference must score **< 0.5**

   Run on a ≥20-problem sample. If the anchor property fails, either T_b or
   T_SOL is wrong — do not ship a manifest that fails it.

5. **Feed task 03's cross-check #4**: `T_SOL ≤ best measured` for every problem.
   A violation is always a SOL config error, never a fast kernel.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 06
```

Passes when: `t_b` exists for **all 235 problems** (or the difference is in
`artifacts/deferred.json` with reasons) with provenance and the
winning variant recorded; authoritative pass ran under documented conditions
(including GPU id and node quiet/busy state); anchor property holds on the
sample; T_SOL ≤ best-measured holds everywhere, or each violation is
individually explained.

## Guard rails

- **T_b is an anchor, not a target.** Do not tune it to make scores land in a
  pleasing range. If the median agent score comes out unlike upstream's ~0.732,
  that is a result to report, not a number to engineer.
- No handwritten kernels in T_b. That would make S=0.5 mean something different
  on AMD than on NVIDIA and quietly break cross-platform interpretation.
- Record the winning variant per problem. "Optimized PyTorch" is not
  reproducible; a named variant is.
- If task 01 found >3% interference and you time the authoritative pass on a
  busy node anyway, the numbers are contaminated. Wait for the quiet window.

## Outputs

- `artifacts/06/candidates/` — all variants timed
- `artifacts/06/t_b.json` — frozen anchors + winning variant per problem
- `artifacts/06/anchor-verification.md`
