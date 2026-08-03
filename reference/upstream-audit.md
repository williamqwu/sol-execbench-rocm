# Upstream audit — where the NVIDIA coupling actually is

Audit of `nvidia/sol-execbench` **v1.0.2**. Every NVIDIA-specific call site,
located. Task 02 works through this list.

The headline: the harness core is ~2,800 lines of Python and most of it is
vendor-neutral by accident. The genuinely NVIDIA-locked surface is small and
concentrated.

## Port map

| Component | Path | Verdict |
|---|---|---|
| Problem schemas | `core/data/` (Pydantic v2) | Portable. Add enum members only. |
| Reference impls | dataset, `run()` per problem | Pure PyTorch — run unmodified. |
| Correctness | `core/bench/correctness.py` | Portable. **Tolerances must be recalibrated** (task 05). |
| Timing — events | `core/bench/timing.py::bench_time_with_cuda_events` | Works on ROCm via HIP events. |
| Timing — CUPTI | `core/bench/timing.py`, `cupti_utils.py` | **NVIDIA-only.** Task 04. Selection logic already extracted + CPU-verified. |
| Cache hygiene | `timing.py::_get_empty_cache_for_benchmark` | Sizing wrong for AMD; `cudaCtxResetPersistingL2Cache` → no-op. |
| Anti-hack | `core/bench/reward_hack.py`, `io.py::ShiftingMemoryPoolAllocator` | All torch-level. Portable; **re-validate** (task 08). |
| Clock lock | `core/bench/clock_lock.py`, `config/device_config.py` | Rewrite on `amdsmi`/`rocm-smi`. |
| Build path | `driver/problem_packager.py`, `templates/build_ext.py` | Mechanism portable (auto-hipify); flags/arch/includes need AMD variants. |
| Eval driver | `driver/templates/eval_driver.py` (~660 lines) | Mostly portable; per-language dispatch needs AMD backends. |
| Languages | `core/data/solution.py` enums | Needs AMD mapping (below). |
| Docker | `docker/Dockerfile` | New ROCm image. |
| Deps | `pyproject.toml` | NVIDIA wheels → optional `[nvidia]` extra. |
| Score | `sol_score.py` | **Unchanged.** Needs AMD T_SOL / T_b. |
| SOL bounds | `NVlabs/SOLAR`, `configs/arch/*.yaml` | Arch-agnostic until final stage. Add MI355X config (task 03). |

## Specific call sites

### `core/bench/config/device_config.py`
```python
CLOCK_LOCK_PRESETS = {
    "NVIDIA B200": ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996),
    "NVIDIA H100": ClockPreset(gpu_clk_mhz=1410, dram_clk_mhz=1593),
    "NVIDIA A100": ClockPreset(gpu_clk_mhz=1065, dram_clk_mhz=1215),
}
```
Add MI355X at measured F_LOCK. `dram_clk_mhz=None` — Instinct parts do not
expose independent memory-clock locking. Add a separate explicit `LLC_BYTES`
table; do not derive flush size from device properties.

### `core/bench/clock_lock.py`
`sudo nvidia-smi -lgc/-lmc`, reset `-rgc/-rmc`, verify via
`nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory`, 50 MHz
tolerance, `SOL_EXECBENCH_GPU_CLK_MHZ` env override.

AMD: `amdsmi` preferred, `rocm-smi --setperfdeterminism <mhz>` fallback, reset
`rocm-smi -r`. Keep the env override and the verify-with-tolerance loop. Reuse
`scripts/clock_calibrate.py` rather than writing a second implementation.

### `core/bench/timing.py`
- `_get_empty_cache_for_benchmark` — sizes at `2 × props.L2_cache_size`. On
  MI355X the structure that matters is the 256 MB Infinity Cache → **512 MB**
  buffer from the explicit table.
- `_reset_persisting_l2_cache` — `cudaCtxResetPersistingL2Cache` via
  `cuda.bindings.runtime`. No CDNA equivalent → vendor no-op.
- `bench_gpu_time_with_cupti` — replace record source only; selection logic is
  already extracted.

### `driver/problem_packager.py`
- `_get_local_sm()` shells `nvidia-smi --query-gpu=compute_cap` → replace with
  `torch.cuda.get_device_properties(0).gcnArchName`.
- `_sm_to_gencode()` → `--offload-arch=gfx950`.
- `_BLACKWELL_HARDWARE = {SupportedHardware.B200}` → add MI355X.
- `_CPP_LANGUAGES = {CUDA_CPP, CUTLASS, CUDNN, CUBLAS}` → add HIP_CPP, CK,
  MIOPEN, HIPBLASLT.

### `driver/templates/build_ext.py`
- `CUTLASS_DIR` default `/usr/local/cutlass` → `CK_DIR`.
- Source glob `(.cu .cpp .cc .cxx .c)` → add `.hip`.
- Defaults: `cuda_cflags=["-O3","--use_fast_math"]` → `["-O3","-ffast-math"]`;
  `ld_flags=["-lcuda"]` → `["-lamdhip64"]`.

### `core/data/solution.py`
`SupportedHardware`: currently `B200 | LOCAL` — add `MI355X`. The enum already
anticipates multiple targets, which is why the vendor layer is plausibly
upstreamable.

## Language mapping

| NVIDIA | AMD | Note |
|---|---|---|
| `pytorch` | `pytorch` | unchanged |
| `triton` | `triton` | official AMD backend; the most important shared language |
| `cuda_cpp` | **`hip_cpp`** | native HIP + inline GCN asm |
| `cublas` | **`hipblaslt`** | + rocBLAS |
| `cudnn`, `cudnn_frontend` | **`miopen`** | narrower fused-attention coverage than cuDNN 9 |
| `cutlass` | **`ck`** | Composable Kernel |
| `cute_dsl` | **`ck_tile`** | |
| `cutile` | — | no equivalent; Triton covers the niche. Reject on AMD with a clear error. |
| — | **`aiter`** | AMD inference kernels; natural backend for FlashInfer-Bench |

**Languages are properties of solutions, not problems.** Every problem's
reference is pure PyTorch, so dropping `cutile` loses zero problems — only
NVIDIA-specific *example solutions*, which get AMD-native counterparts.

## Dataset

`nvidia/SOL-ExecBench` on HuggingFace. Per problem: `definition.json`,
`workload.jsonl`, optional `reference.py` / `solution.json`.

| Category | Count | Port |
|---|---|---|
| L1 | 94 | direct |
| L2 | 82 | direct |
| Quant | 33 | 18 FP8 direct (CDNA4 = OCP FP8, same as B200); 15 NVFP4 → MXFP4 respec |
| FlashInfer-Bench | 26 | direct; AITER as backend |

~16 workload instances per problem; 78 problems (33%) use custom `get_inputs()`
for structured inputs (paged KV caches, sparse masks) — pure PyTorch, expected
to run as-is, but verify determinism under fixed seeds since task 05 depends on
it.

**Not verified from the build environment** (network-restricted). Confirm the
layout on first contact and record any difference in `STATE.md`.

## Reference stacks

| | B200 (upstream) | MI355X (target) |
|---|---|---|
| Toolkit | CUDA 13.1.1, cuDNN 9.17.1 | ROCm ≥7.2 |
| Driver | 580.95 | amdgpu (record actual) |
| torch | 2.9.0 | 2.9.0 ROCm build |
| Container | `nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04` | `rocm/pytorch` |
| Template lib | CUTLASS v4.4.1 | Composable Kernel (pin a tag) |
