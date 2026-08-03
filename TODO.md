# TODO — what is not yet built

Everything here is a known gap, deliberately left rather than forgotten. Nothing
in this repo silently pretends to be finished.

## Scripts written but never run on hardware

These are structurally reviewed and syntax-clean, but no MI355X has executed
them. Expect small first-contact fixes — exact SMI field names, sysfs paths,
library entry points. **When you fix one, note it in `STATE.md`**, so a later
session knows whether a given measurement predates the fix.

| Script | First needed | Likely fix |
|---|---|---|
| `scripts/clock_calibrate.py` | task 01 | `amdsmi` field names; `rocm-smi` JSON shape |
| `scripts/roofline_probe.py` | task 00 | none expected; pure torch |
| `scripts/build_node_report.py` | task 00 | `amdsmi` power/temp accessors |
| `scripts/mxfp4_spike.py` | task 07 | `_scaled_mm` signature; hipBLASLt entry point |
| `scripts/gen_golden.py` | task 05 | input construction when a problem has no `get_inputs()` |

## Not written at all

### Sweep runners — `scripts/runners/`
`shard_sweep.py` dispatches to these; each takes `--problem <dir> --out <file>`.
Write each as part of the task that needs it.

| Runner | Task |
|---|---|
| `run_reference.py` | 02 |
| `compare_methodology.py` | 04 |
| `calibrate_tolerance.py` | 05 |
| `time_tb_candidates.py` | 06 |

Contract: **on failure, still write an output file recording the error.** A
missing file means "not yet done" and will be redone; a failure recorded as an
artifact is a result. Prime directive 1.

### The port itself
Task 02 is the actual harness port against `reference/upstream-audit.md`. The
audit locates every call site; the code is not written.

### rocprofiler shim
Task 04. Contract fully specified in `reference/contracts/rocprof_shim.md`.

### `reference/b200-tolerances.json`
Used by `verify_artifacts.py --task 05` to detect tolerances copied verbatim
from upstream (prime directive 2). Extract from the upstream dataset as
`{"<problem>:<workload>": [atol, rtol, matched_ratio]}`. Without it the
copy-detection check degrades to a warning.

## Still doable without a GPU — do these first if the node is not ready

Highest value first:

1. **`reference/tb-candidates/`** — pre-authored optimized-PyTorch variants per
   problem. Platform-independent, so every one written here is measurement time
   saved on the node, and it converts task 06 from a creative task into a batch
   job. This is the single biggest remaining prep win.
2. **`reference/exploits/`** — the replay corpus for task 08. All torch-level.
3. **Golden references** — `scripts/gen_golden.py` needs only CPU and the
   dataset. Run it anywhere.
4. **Static op-coverage scan** — parse all 235 references, extract the torch ops
   used, cross-reference against known ROCm gaps. Tells you which problems will
   break before you touch a GPU.
5. **ROCm Docker image** — build and push. Image builds need no GPU.

## Unverified assumptions

Carried into this repo from research, never confirmed against hardware or a
live download. Confirm early; record what you actually find.

- **Dataset accessibility and layout.** `huggingface-cli download
  nvidia/SOL-ExecBench` was never run (the build environment was
  network-restricted). Category counts L1=94, L2=82, Quant=33,
  FlashInfer-Bench=26 come from the paper, not from the files.
- **Infinity Cache bandwidth** — placeholder in the arch generator (V2). Capacity
  (256 MB) is published; bandwidth is not, and it feeds the SOLAR buffer model.
- **TF32 on CDNA4** (V1) — reportedly dropped versus CDNA3. Confirm in the ISA
  guide and decide the fallback.
- **MXFP4 dense vs sparsity** (V3) — AMD quotes 10.1 PFLOPS for MXFP4/MXFP6
  dense *and* for FP8 with sparsity. Different rows; do not conflate.
- **F_LOCK** — no estimate is carried anywhere on purpose. Task 01 measures it.
