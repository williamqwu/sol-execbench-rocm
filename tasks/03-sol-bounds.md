# Task 03 — SOL bounds (T_SOL)

**Goal:** an analytically derived Speed-of-Light runtime for every problem and
workload instance on MI355X at F_LOCK.

Barely a GPU task. SOLAR runs on `device="meta"`; the only thing that needs
silicon is the final cross-check against measured times.

## Preconditions

- Task 01 done (F_LOCK known). Nothing else.

## Steps

1. **Generate the arch config.**
   ```bash
   python src/solexbench_rocm/solar/gen_arch_yaml.py \
       --part MI355X --freq-ghz <F_LOCK/1000> -o SOLAR/configs/arch/MI355X.yaml
   ```
   It is a generator, not a static file, because `MAC_per_cycle` is
   architectural (frequency-**in**dependent) while `*_byte_per_cycle` is derived
   (bandwidth is fixed in bytes/second, so it must be rescaled when F_LOCK
   changes). Hand-editing `freq_GHz` corrupts the roofline balance point.

2. **Resolve the three flagged unknowns** marked `V1`/`V2`/`V3` in the
   generator's header comment:
   - **V1 TF32 matrix support on CDNA4.** CDNA3 had it; CDNA4 reportedly
     dropped it. Confirm against the CDNA4 ISA guide. If absent, decide the
     fallback for any problem SOLAR tags `tf32` (recommendation: bf16 rate) and
     record the decision.
   - **V2 Infinity Cache bandwidth.** Capacity (256 MB) is published; bandwidth
     is a placeholder in the generator and drives the buffer-aware memory bound
     for fused L2-category problems. Measure it (`scripts/roofline_probe.py`
     has a cache-resident mode) or source it authoritatively.
   - **V3 MXFP4 dense vs sparsity.** AMD's spec sheet lists 10.1 PFLOPS for
     MXFP4/MXFP6 dense *and* 10.1 for FP8 **with sparsity**. Do not conflate the
     two rows.

3. **Run SOLAR over all problems** → `artifacts/03/t_sol.json`, keyed by
   (problem, workload instance).

   Emit **T_SOL in cycles as well as milliseconds.** The cycle figure is
   invariant to F_LOCK; if F_LOCK ever changes, the millisecond column is one
   scalar division away rather than a full re-run.

4. **Cross-checks — all four must pass.**
   - Memory-bound problems (most of L1: norms, RoPE, embeddings, SwiGLU) should
     land within a few percent of B200's published SOL, because both machines
     are 8 TB/s at the roof. A large divergence means a config error.
   - BF16 compute-bound SOLs should scale by ≈ (2.5/2.25) × the F_LOCK derate
     ratio relative to B200.
   - Hand-audit ~20 problems: compute FLOPs and bytes independently and compare.
   - **T_SOL ≤ best measured time, for every problem.** A violation means the
     "lower bound" is above something real — a config error, always. This one
     needs task 06's output; run it as soon as 06 lands.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 03
```

Passes when: `MI355X.yaml` generated at F_LOCK; V1/V2/V3 each resolved and
recorded; `t_sol.json` covers every problem/workload with both cycles and ms;
cross-checks 1–3 pass; check 4 recorded as pending-06 or passing.

## Guard rails

- **Never copy a B200 SOL number.** Comparison is a sanity check, not a source.
- Keep the theoretical-peak convention (not achieved-STREAM) for methodological
  parity with upstream — but publish the measured ceilings from task 00/01
  alongside, so consumers can see the achievable-fraction difference between
  platforms. This is how "MFMA utilization ceilings differ across vendors" gets
  handled honestly instead of by forking the methodology.
- SOLAR needs **no code changes** for MXFP4: it resolves precision by string
  lookup and maps `float4_e2m1fn_x2 → "nvfp4"`, and the generator emits an
  `nvfp4` alias at the MXFP4 rate. If you find yourself patching SOLAR, stop and
  re-read the generator's comment.

## Outputs

- `SOLAR/configs/arch/MI355X.yaml`
- `artifacts/03/t_sol.json` (cycles + ms)
- `artifacts/03/cross-checks.md`
- `STATE.md`: V1/V2/V3 resolutions
