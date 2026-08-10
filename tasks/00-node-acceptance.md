# Task 00 — Node acceptance

**Goal:** establish that this node is what we think it is, before anything
depends on it. Budget: under an hour.

The failure this prevents is discovering on day three that the node has seven
healthy GPUs and one that throttles, or a ROCm version that does not match the
container, and that two days of measurements are contaminated.

## Preconditions

None. This is the first task.

## Steps

1. **Inventory.**
   ```bash
   bash scripts/node_acceptance.sh 2>&1 | tee artifacts/00/acceptance.log
   ```
   Records: hostname, GPU count and model, gfx arch, VBIOS, ROCm version,
   amdgpu driver version, HBM capacity per GPU, torch version and whether
   `torch.cuda.is_available()` (ROCm reports through the same API), per-GPU
   temperature and power at idle.

2. **Confirm all eight GPUs are equivalent.** Not just present — equivalent.
   Compare per-GPU: capacity, reported max clock, power cap, temperature at
   idle. A GPU with a lower power cap or a hotter idle will quietly produce
   different timings.

3. **Measure the two rooflines.** These are the empirical ceilings that will sit
   alongside the analytic SOL bounds in the scoring manifest (see `docs/plan-2026-07-31.md` §7.1).
   ```bash
   python scripts/roofline_probe.py --gpu 0 --out artifacts/00/roofline-gpu0.json
   ```
   Produces: achieved HBM copy bandwidth (TB/s) and achieved BF16 GEMM
   throughput (TFLOPS), each at default clocks. These are *reference points*,
   not yet calibrated numbers — task 01 re-measures at F_LOCK.

4. **Repo self-test.**
   ```bash
   pytest tests/ -q
   ```
   Must be green. This exercises the CPU-verified activity package and needs no
   GPU; if it fails, the checkout is wrong, not the node.

5. **Dataset.**
   ```bash
   huggingface-cli download nvidia/SOL-ExecBench --repo-type dataset --local-dir data/
   ```
   Then confirm the category layout matches `reference/upstream-audit.md`
   (expect `L1/` 94, `L2/` 82, `Quant/` 33, `FlashInfer-Bench/` 26, each problem
   a directory with `definition.json` + `workload.jsonl`).

   **This download was never verified from the build environment.** If it is
   gated, or the layout differs, record what you actually find in `STATE.md`
   before adapting anything to it.

## Acceptance check

```bash
python scripts/verify_artifacts.py --task 00
```

Passes when: `artifacts/00/node-report.json` exists with provenance, reports
exactly 8 GPUs all `gfx950`, no GPU deviating >5% from the median on idle power
or power cap, both roofline probes present and non-zero, `pytest tests/` green,
and dataset presence recorded either way.

## Guard rails

- **Do not proceed to task 01 with a degraded GPU.** Record it and ask. Seven
  good GPUs is a fine node to work on; seven good GPUs *believed* to be eight is
  not.
- Do not install or upgrade anything to make a probe pass. Record the version
  mismatch instead — see prime directive 6.
- Roofline numbers here are at *default* clocks and are not the scoring
  ceilings. Do not cite them anywhere downstream.

## Outputs

- `artifacts/00/node-report.json` — the environment record everything cites
- `artifacts/00/roofline-gpu0.json`
- `artifacts/00/acceptance.log`
- `STATE.md` environment table filled in
