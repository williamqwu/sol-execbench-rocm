# Task 07 — Quant category (33 problems)

**Goal:** port the quantized problems. Two sub-families with very different
risk profiles. **This is the highest-uncertainty task in the project** — resolve
it early enough that the fallback is still available.

## Preconditions

- Task 02 done.

## 7a. FP8 blockwise — 18 problems, expected to port directly

CDNA4 supports **OCP FP8** (`e4m3fn` / `e5m2`) — the same formats as B200. The
problem definitions therefore carry over unchanged. (It is CDNA3/MI300X that has
the `fnuz` mismatch; that is a MI300X-tier concern, not yours.)

Risk here is software maturity, not hardware capability. Verify end-to-end:

- `torch.float8_e4m3fn` tensor creation and manipulation
- `torch._scaled_mm` on ROCm
- hipBLASLt blockwise-scaled GEMM for the optimized baseline

Gate each problem on a reference-correctness check and record which ones need
workarounds.

## 7b. NVFP4 → MXFP4 — 15 problems, respec required

**These formats are not interchangeable, and translation is the wrong mental
model.**

| | NVFP4 | MXFP4 |
|---|---|---|
| Elements | E2M1 | E2M1 |
| Block size | **16** | **32** |
| Scale format | **FP8 E4M3** | **E8M0** (power-of-two) |
| Origin | NVIDIA proprietary | OCP standard |

Different block granularity and a different scale format mean different
quantization error. Re-scaling NVFP4 data into MXFP4 does not produce an
equivalent problem.

**Do this instead — author an MXFP4 twin per problem:**

- same tensor shapes, same fusion structure, same workload axes
- quantize/dequantize reference rewritten for MX semantics
- tolerances calibrated fresh (task 05 procedure)
- filed under `quant_mxfp4/` with provenance metadata linking to the NVFP4
  original

Expressed in PyTorch via `float4_e2m1fn_x2` plus E8M0 scale tensors; kernel-side
support comes from hipBLASLt/CK MXFP4 paths on gfx950 and Triton's scaled-dot.

### Feasibility spike first — timebox 1 week

Before authoring 15 problems, confirm the software path exists at all:

```bash
python scripts/mxfp4_spike.py --out artifacts/07/spike.json
```

Probes: `float4_e2m1fn_x2` support in the installed torch, E8M0 scale handling,
`torch._scaled_mm` MXFP4 path, hipBLASLt MXFP4 GEMM, Triton scaled-dot.

**Fallback if the spike fails:** ship v1 with **220 problems** (18 FP8 included,
15 MXFP4 deferred to v1.1) and document the omission prominently. A
well-documented 220-problem benchmark is a real deliverable; 15 problems built
on a broken software path are not.

Take this decision explicitly and record it in `STATE.md` — do not let it happen
by drift.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 07
```

Passes when: all 18 FP8 problems pass reference correctness (or each failure is
recorded); the MXFP4 spike has a clear verdict; **either** 15 MXFP4 twins exist
with fresh tolerances and provenance links, **or** the deferral is recorded with
its reasoning and the problem count is documented everywhere it appears.

## Guard rails

- **Do not present an MXFP4 twin as the NVFP4 problem.** They are different
  problems that answer the same question on different hardware. Provenance
  metadata must make that legible to anyone comparing leaderboards.
- Do not simulate MXFP4 in higher precision to make a problem "work". A
  simulated quant problem measures nothing.
- Do not conflate MXFP4-dense with FP8-with-sparsity when reading AMD's spec
  sheet — both are quoted at 10.1 PFLOPS and they are different rows.

## Outputs

- `artifacts/07/spike.json` + verdict
- `artifacts/07/fp8-validation.md`
- `data/quant_mxfp4/` (or a recorded deferral)
