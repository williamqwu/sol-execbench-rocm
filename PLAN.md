# Porting SOL-ExecBench to AMD ROCm (MI355X)

**Engineering plan — full-parity port of NVIDIA's SOL-ExecBench GPU-kernel benchmark to AMD Instinct MI355X / ROCm**

*Prepared July 31, 2026. Based on an audit of [`nvidia/sol-execbench` v1.0.2](https://github.com/nvidia/sol-execbench) (Apache-2.0), the [SOL-ExecBench paper (arXiv:2603.19173)](https://arxiv.org/abs/2603.19173), the [SOLAR analyzer](https://github.com/NVlabs/SOLAR), and the [HuggingFace dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench).*

---

## 1. Executive summary

SOL-ExecBench evaluates GPU kernels not by speedup over a software baseline but by proximity to an analytically derived hardware "Speed-of-Light" (SOL) bound. It ships three separable pieces: a **dataset** of 235 kernel problems extracted from 124 production AI models (PyTorch reference + workloads + tolerances), an **evaluation harness** (correctness + anti-reward-hacking + high-fidelity timing), and a **scoring methodology** (SOLAR-computed roofline bounds for a clock-locked B200, plus an anchored score formula).

The good news for an AMD port: the benchmark is architected in a way that is mostly vendor-portable by accident. Problem definitions are pure PyTorch; the harness is written against `torch.cuda.*` APIs, which PyTorch's ROCm backend implements directly; the score formula is hardware-agnostic; and SOLAR's hardware model lives in per-architecture YAML configs. The genuinely NVIDIA-specific surface is small and concentrated: CUPTI-based timing, `nvidia-smi` clock locking, nvcc arch flags, the NVIDIA library backends (cuBLAS/cuDNN/CUTLASS/CuTe/cuTile), the NVFP4 quantization format, and the B200 roofline numbers.

The plan below ports the benchmark at full parity to **MI355X (CDNA4, gfx950)** — the correct B200 peer: identical 8 TB/s memory bandwidth, comparable dense-matrix throughput, and native OCP FP8 + MXFP4 support, which makes the Quant category mappable rather than droppable. Estimated effort: **~14–18 engineer-weeks** across six phases, with a usable L1/L2 harness on ROCm after ~4 weeks and the full 235-problem benchmark with AMD SOL bounds after ~3–4 months.

One framing decision to make explicit up front: an AMD SOL score and an NVIDIA SOL score are each *within-platform* measures of "fraction of hardware headroom reclaimed." They are directly comparable in spirit (r = 0.981 against headroom reclaimed, per the paper) but should not be advertised as a cross-vendor horse race, because analytic peaks are reachable to different degrees on different microarchitectures. The deliverable is "SOL-ExecBench-ROCm": same problems, same score semantics, AMD-calibrated bounds.

---

## 2. What exactly has to be ported (anatomy of the benchmark)

From the repo audit, the moving parts and their portability:

| Component | Where | What it does | Portability verdict |
|---|---|---|---|
| Problem schemas | `core/data/` (Pydantic v2: `Definition`, `Workload`, `Solution`, `Trace`) | Kernel contract, symbolic axes, tensor specs, tolerances, source packaging | ✅ Portable as-is; needs new enum members (hardware, languages) |
| Reference implementations | dataset (HF), `run()` per problem | Pure PyTorch, self-contained; 78 problems have custom `get_inputs()` | ✅ Run unmodified on ROCm PyTorch |
| Correctness checking | `core/bench/correctness.py` | (atol, rtol, matched_ratio) tuples per workload, NaN/Inf and degenerate-output rejection | ✅ Portable; **tolerances must be recalibrated on AMD** |
| Timing — CUDA events path | `core/bench/timing.py` (`bench_time_with_cuda_events`, derived from Triton `do_bench`) | Event-pair timing, cache clear each iter, setup-callback cloning | ✅ Works on ROCm (`torch.cuda.Event` → HIP events) |
| Timing — CUPTI path (default) | `core/bench/timing.py` + `cupti_utils.py` (`cupti-python`) | Kernel-activity windows with ns timestamps; excludes CPU launch overhead; discovers expected kernel sequence and matches it per iteration | ❌ NVIDIA-only; needs rocprofiler-sdk equivalent (largest single engineering item) |
| L2/cache hygiene | `timing.py` | Zeroes a 2×L2-size buffer per iteration + `cudaCtxResetPersistingL2Cache` | ⚠️ Concept portable, sizing wrong for AMD (must cover 256 MB Infinity Cache, not just L2); persisting-L2 reset becomes a no-op |
| Anti-reward-hacking | `core/bench/reward_hack.py` + `ShiftingMemoryPoolAllocator` in `io.py` | Monkey-patch detection via function identity, thread-injection check, FakeTensor rejection, eval-integrity snapshots, per-iteration randomized pointer shifts | ✅ All torch-level; portable with re-validation |
| Clock locking | `core/bench/clock_lock.py` + `config/device_config.py` | `sudo nvidia-smi -lgc/-lmc` with per-device presets (B200: 1500/3996 MHz), verify within 50 MHz | ❌ Rewrite on `amd-smi`/`rocm-smi` perf-determinism mode |
| Build path | `driver/problem_packager.py` + `templates/build_ext.py` | `torch.utils.cpp_extension.load`, auto-injects `-gencode …sm_100a`, CUTLASS include dirs, defaults `-O3 --use_fast_math`, `-lcuda` | ⚠️ Mechanism portable (cpp_extension auto-hipifies on ROCm); flags/arch/includes need AMD variants |
| Eval driver | `driver/templates/eval_driver.py` (~660 lines, runs in subprocess) | Loads inputs, dispatches per-language, DPS/return conventions, emits JSONL traces | ⚠️ Mostly portable; language-specific import/dispatch needs AMD backends |
| Language support matrix | `core/data/solution.py` enums | pytorch, triton, cute_dsl, cutile, cudnn_frontend, cutlass, cudnn, cublas, cuda_cpp | ⚠️ Needs AMD mapping (Section 5) |
| Docker | `docker/Dockerfile` (`nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04`, CUTLASS v4.4.1, uv) | Reproducible eval env; entrypoint locks clocks | ❌ New ROCm image |
| Python deps | `pyproject.toml` (torch 2.9.0, cuda-bindings, cupti-python, cuda-tile, cupy-cuda13x, nvidia-cutlass-dsl, cudnn-frontend…) | | ⚠️ Swap NVIDIA wheels for ROCm equivalents |
| Score | `sol_score.py` — S(T_k) = 1 / (1 + (T_k − T_SOL)/(T_b − T_SOL)) | S=0.5 at optimized-PyTorch baseline, S=1 at SOL | ✅ Unchanged; needs AMD T_SOL and T_b values |
| SOL bounds | [SOLAR](https://github.com/NVlabs/SOLAR) (separate repo, Apache-2.0), arch YAMLs in `configs/arch/` | Graph→einsum→roofline pipeline; B200 bound at locked 1500 MHz; Orojenesis-style buffer-aware memory bounds | ⚠️ Pipeline is hardware-independent until the last stage; add an MI355X arch config |
| Dataset | HF `nvidia/SOL-ExecBench`: L1 (94), L2 (82), Quant (33: 18 FP8-blockwise + 15 NVFP4), FlashInfer-Bench (26) | ~16 workload instances per problem | ⚠️ L1/L2/FlashInfer portable; Quant needs format remapping (Section 6) |

The harness core is ~2,800 lines of Python — this is a *small, tractable* port. The effort is dominated not by code volume but by four hard problems: (1) kernel-level timing without CUPTI, (2) trustworthy AMD SOL numbers, (3) quantized-format remapping, and (4) re-validating the anti-cheating measures on a new platform.

---

## 3. Target platform

| | NVIDIA B200 (benchmark ref) | AMD MI355X (port target) |
|---|---|---|
| Architecture | Blackwell, sm_100a | CDNA4, **gfx950** |
| Memory | 192 GB HBM3e | 288 GB HBM3E |
| Memory bandwidth | 8 TB/s | **8 TB/s** ([AMD spec](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)) |
| Dense BF16/FP16 matrix | ~2.25 PFLOPS | 2.5 PFLOPS |
| Dense FP8 | ~4.5 PFLOPS | 5 PFLOPS (OCP FP8 & MXFP8) |
| 4-bit float | ~9 PFLOPS (NVFP4) | 10.1 PFLOPS (MXFP4; also MXFP6 at 10.1) |
| Last-level cache | ~126 MB L2 | 32 MB L2 + **256 MB Infinity Cache (LLC)** |
| Peak / locked clock | ~1.97 GHz boost / **locked 1500 MHz** for benchmark | 2.4 GHz peak / lock value TBD (calibration item C1) |
| TDP | 1000 W | 1400 W |
| Software | CUDA 13.1.1, cuDNN 9.17, PyTorch 2.9 | ROCm ≥7.2, PyTorch 2.9 ROCm wheels, Triton (official AMD support) |

Why not MI300X: no FP4 datapath, fnuz (non-OCP) FP8 variants, and 5.3 TB/s bandwidth would silently change problem character (some compute-bound problems become memory-bound). MI355X preserves the benchmark's intent. A later MI300X *tier* can reuse everything here minus the Quant-FP4 subset (the harness changes are identical; only the device config, SOL numbers, and FP8-format handling differ — MI300X would need fnuz remapping of the 18 FP8 problems).

Fallback if MI355X allocation is a bottleneck early on: Phases 0–2 (harness port) can be developed and CI-tested entirely on MI300X or even consumer RDNA hardware, since they exercise mechanism, not numbers.

---

## 4. Harness port — component by component

### 4.1 Device abstraction rather than find-and-replace

The cleanest structure — and the one worth proposing upstream, since the repo already has a `SupportedHardware` enum with only `B200 | LOCAL` — is a small vendor/device layer:

```
core/bench/device/
    __init__.py       # detect_vendor() → "nvidia" | "amd" (torch.version.hip is not None)
    nvidia.py         # existing behavior, unchanged
    amd.py            # everything below
```

with the device config table extended:

```python
CLOCK_LOCK_PRESETS = {
    "NVIDIA B200": ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996),
    ...
    "AMD Instinct MI355X": ClockPreset(gpu_clk_mhz=<F_LOCK>, dram_clk_mhz=None),  # C1
    "AMD Instinct MI300X": ClockPreset(gpu_clk_mhz=1900,     dram_clk_mhz=None),
}
LLC_BYTES = {"AMD Instinct MI355X": 256 * 2**20, ...}   # explicit, do not trust device props
```

`SupportedHardware` gains `MI355X` (and later `MI300X`); `problem_packager` maps it to `--offload-arch=gfx950` the same way `B200` maps to `sm_100a`.

### 4.2 Clock locking (`clock_lock.py`)

Replace the `nvidia-smi -lgc/-lmc` calls with AMD's deterministic-performance mode. On MI300X the documented command is `rocm-smi --setperfdeterminism 1900` (caps the soft max clock below the 2100 MHz default so power-control events don't perturb attainable clocks; reset with `rocm-smi -r`) — see [AMD's MI300X system-optimization guide](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html). The port should:

1. Prefer the **`amdsmi` Python library** (AMD's NVML-equivalent, pip-installable) for set/verify/reset, with `amd-smi` / `rocm-smi` subprocess fallback — mirroring the existing sudo-probe design.
2. **Calibration item C1:** choose the MI355X lock frequency empirically — run a sustained MFMA-saturating workload at 1400 W, observe the clock floor under PCC/thermal steady state, and lock slightly below it (the B200 preset's 1500 MHz vs ~1.97 GHz boost is exactly this kind of derate, ~76%; expect something like ~1.8–2.0 GHz on MI355X, but measure, don't guess). All SOL peak numbers derive from this choice (Section 7), so it must be fixed before SOL computation and recorded in every trace.
3. HBM clock: Instinct parts don't expose independent memory-clock locking the way `-lmc` does; verify the memory clock is stable at max under load and record it, rather than locking it. Keep the existing verify-within-tolerance loop (query via `amdsmi`).
4. Keep the `SOL_EXECBENCH_GPU_CLK_MHZ` env override so operators can adjust without code changes.

### 4.3 Cache hygiene (`timing.py`)

Two changes:

- **Sizing:** `_get_empty_cache_for_benchmark` sizes the flush buffer at 2× `props.L2_cache_size`. On MI355X the relevant last-level structure is the 256 MB Infinity Cache, and what HIP reports in `l2CacheSize` must be treated as unreliable for this purpose (verification item V3). Use the explicit `LLC_BYTES` table → a **512 MB flush buffer** (2× LLC). Cheap relative to 288 GB VRAM.
- **Persisting L2:** `cudaCtxResetPersistingL2Cache` has no CDNA equivalent (no L2 persistence API); make it a vendor no-op.

The rest of the timing scaffolding — warmup/rep structure, `ShiftingMemoryPoolAllocator` (randomized 2 KB-granular pointer shifts per iteration), input cloning, subprocess isolation with 300 s timeout — is `torch`-level and carries over unchanged. One check: the allocator's alignment assumptions (256 B-class shifts) are fine on AMD (HIP allocations are 256 B-aligned), but re-run its unit tests on ROCm.

### 4.4 Timing methodology — the CUPTI problem

The harness's default methodology is not CUDA events but **CUPTI activity tracing**: it records kernel/memcpy/memset activities with device timestamps, discovers the submission's expected kernel sequence after warmup, then per timed iteration selects exactly that activity window and reports `max(end) − min(start)`. This (a) excludes CPU launch overhead, (b) is robust to setup work, and (c) is itself an anti-cheating measure (work hidden outside the recorded sequence is caught by the count assertion).

ROCm's equivalent layer is **[rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)** (the library behind `rocprofv3`), which provides kernel-dispatch buffer tracing with begin/end device timestamps — functionally the CONCURRENT_KERNEL + MEMCPY activity feed. Unlike `cupti-python`, there is no mature pip-installable Python binding, so plan for a **thin C++ shim** (pybind11 or a small torch extension, a few hundred lines): configure a tracing session for kernel-dispatch + memory-copy records, expose `start()/stop()/drain() → list[(name, start_ns, end_ns, correlation_id)]` and a `get_timestamp()` for the CPU-window bracketing the existing code does with `cupti.get_timestamp()`. The sequence-matching logic in `cupti_utils.py` (~230 lines) is pure Python over `(name, start, end, correlation_id)` tuples and ports unchanged once the record source is swapped.

Phasing: ship **Phase 1 with the `cuda_events` methodology** (already implemented, works via HIP events, matches Triton's `do_bench` lineage) and treat the rocprofiler shim as **Phase 2 parity work**. Two consequences to manage meanwhile: event timing includes some launch/synchronization slop for very short kernels (SOL scores for tiny workloads read slightly low), and one anti-cheating layer (activity-sequence verification) is absent — mitigations in 4.6. Record the methodology in every trace (`methodology: "hip_events" | "rocprof"`), and after the shim lands, cross-validate the two paths on the full L1 set (expect ≤1–2% divergence except on μs-scale kernels).

### 4.5 Build path

PyTorch's `torch.utils.cpp_extension` **auto-hipifies** CUDA sources on ROCm builds (this is long-standing upstream functionality — see PyTorch [PR #22091](https://github.com/pytorch/pytorch/pull/22091), [PR #149245](https://github.com/pytorch/pytorch/pull/149245)), so `build_ext.py` largely works, but make the AMD path explicit and first-class rather than relying on transparent hipification of user code:

- Accept `.hip` alongside `.cu/.cpp` in the source glob; submissions targeting AMD should be free to write native HIP without a hipify pass mangling it.
- Flag translation in `problem_packager`: `-gencode arch=…,code=sm_100a` → `--offload-arch=gfx950`; default `cuda_cflags = ["-O3", "--use_fast_math"]` → `["-O3", "-ffast-math"]`; `ld_flags = ["-lcuda"]` → `["-lamdhip64"]`. Keep these in a per-vendor defaults table on `CompileOptions`.
- Include paths: `CUTLASS_DIR` → `CK_DIR` (Composable Kernel checkout) for the `ck` language; add `AITER` install for the inference-primitive backend.

### 4.6 Anti-reward-hacking on ROCm

Every mechanism in `reward_hack.py` is torch-level and portable: `torch.cuda.Event.elapsed_time` identity capture works identically on ROCm (same class, HIP-backed), thread-count checks, strict `type(t) is torch.Tensor` FakeTensor rejection, and critical-function snapshots. The port work here is **red-teaming, not code**: re-run the paper's three exploit families (hidden streams/`torch.jit.fork`, state caching keyed on data pointers, timing monkey-patching) on ROCm and check the detectors still fire — 14.5% of agent submissions were flagged on NVIDIA, so this layer is load-bearing. AMD-specific additions:

- The stream-disallowing policy must cover HIP streams however exposed (`torch.cuda.Stream` on ROCm, raw `hipStreamCreate` in C++ submissions — scan sources for it as the CUDA path does for its equivalents).
- Block `amd-smi`/`rocm-smi`/`amdsmi` invocation from inside submission subprocesses (a submission that *raises* the clock cap mid-run would beat the locked-clock SOL calibration). The NVIDIA image has the same latent issue with sudo'd `nvidia-smi`; on the AMD image, scope passwordless sudo to the entrypoint only.
- Until the rocprofiler shim lands, add an interim guard for the events-only phase: assert `torch.cuda.current_stream()` is default and thread count unchanged (already present), and run the LLM-judge static screen (leaderboard-side, per the paper) on all submissions.

### 4.7 Eval driver, CLI, Docker, dependencies

`eval_driver.py` needs: vendor detect at startup, per-language import guards for AMD backends (Section 5), and the device-layer calls above; DPS/return-style conventions, JSONL trace emission, and the Pydantic models are untouched. The Docker image rebases onto **`rocm/pytorch`** (ROCm ≥7.2, Ubuntu 24.04, torch 2.9 ROCm): install Composable Kernel (pinned tag), AITER, hipBLASLt dev headers, MIOpen, rocprofiler-sdk, `amdsmi`; entrypoint runs perf-determinism lock + verify and sets `SOL_EXECBENCH_CLOCKS_LOCKED`. In `pyproject.toml`, the NVIDIA wheel set (`cuda-bindings`, `cuda-python`, `cupti-python`, `cuda-tile`, `cupy-cuda13x`, `nvidia-cutlass-dsl`, `nvidia-cudnn-frontend`, `nvidia-libcal-cu12`, `nvidia-cupynumeric`) moves behind an optional `[nvidia]` extra; an `[amd]` extra carries `amdsmi` and the rocprofiler shim. `torch==2.9.0` stays, sourced from the ROCm index.

---

## 5. Language / backend mapping

The `SupportedLanguages` enum and its example problems map as follows:

| NVIDIA backend | AMD equivalent | Notes |
|---|---|---|
| `pytorch` | `pytorch` | Unchanged; `torch.compile` works on ROCm (Inductor→Triton) |
| `triton` | `triton` | Official AMD backend; same submission format. The single most important shared language — most agent-generated kernels use it |
| `cuda_cpp` | **`hip_cpp`** (new) | Native HIP C++ with inline GCN asm permitted, mirroring "CUDA C++ with inline PTX." Also accept `cuda_cpp` sources on AMD via auto-hipify for cross-platform submissions, flagged as such |
| `cublas` | **`hipblaslt`** (new; plus rocBLAS) | hipBLASLt is the cuBLASLt-class library with FP8/MX-format epilogue support on gfx950 |
| `cudnn` / `cudnn_frontend` | **`miopen`** (new) | MIOpen is the cuDNN equivalent. Coverage of fused attention graphs is narrower than cuDNN 9 frontend; expect some cudnn-frontend example solutions to have no direct analog (AITER usually fills the gap) |
| `cutlass` | **`ck`** (new — [Composable Kernel](https://github.com/ROCm/composable_kernel)) | CK is AMD's CUTLASS-class C++ template library |
| `cute_dsl` | **`ck_tile`** (new) | CK-Tile is the CuTe-DSL-class tile programming layer |
| `cutile` | — (no equivalent) | Drop on AMD; the niche (Pythonic tile programming) is covered by Triton. Harness rejects `cutile` solutions on AMD targets with a clear error |
| (n/a) | **`aiter`** (new — [AITER](https://github.com/ROCm/aiter)) | AMD's inference-kernel library; the natural backend for the FlashInfer-Bench category. FlashInfer itself now has a [ROCm backend built on AITER](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html), so `flashinfer`-style solutions can also run directly |

Two important scoping notes. First, **languages are properties of *solutions*, not *problems***: since every problem's reference is pure PyTorch, zero problems are lost by dropping a language — only NVIDIA-specific *example solutions* (in `examples/`) fail to port, and those get AMD-native counterparts (one seed example per AMD backend, used as CI fixtures). Second, keep the solution JSON schema byte-compatible where possible so agent scaffolds built for the NVIDIA benchmark retarget AMD by changing only `target_hardware` and `languages`.

---

## 6. Dataset port (235 problems)

### 6.1 L1 (94), L2 (82) — direct port

References are self-contained PyTorch with symbolic-axis workloads; they run unmodified on ROCm. Work items: run all references across all ~16 workload instances each on MI355X (an automated sweep, ~1 day of GPU time); fix any op-coverage gaps in ROCm PyTorch (expect near-zero at torch 2.9/ROCm 7.2 for these mainstream ops; budget for a handful of workarounds); port the 78 custom `get_inputs()` generators (pure PyTorch — paged KV layouts, sparse masks — expected to run as-is; verify determinism under fixed seeds so tolerance calibration is stable).

**Tolerance recalibration is mandatory, not optional.** The (atol, rtol, matched_ratio) tuples in `workload.jsonl` were calibrated by repeated reference probing *on B200* with a 1.25× safety margin. MFMA accumulation order, fast-math differences, and different SIMD widths shift the empirical error distribution on CDNA4. Re-run the calibration procedure on MI355X (probe reference vs perturbed-reference across seeds, take the max observed error × 1.25) and emit an AMD `workload.jsonl`. Copying B200 tolerances would produce both false failures and — worse for benchmark integrity — tolerances loose enough to reward wrong-but-fast kernels.

### 6.2 Quant (33) — the real dataset work

Two sub-families:

- **18 FP8-blockwise problems:** CDNA4 natively supports **OCP FP8 (e4m3fn / e5m2)** — the same formats as B200, so *problem definitions carry over unchanged* (unlike a MI300X port, which would face the fnuz mismatch). Verify the reference path end-to-end: `torch.float8_e4m3fn` tensors, `torch._scaled_mm` on ROCm, hipBLASLt blockwise-scaled GEMM for optimized baselines. Risk is software maturity, not hardware capability; gate each problem on a reference-correctness CI check.
- **15 NVFP4 problems:** NVFP4 (e2m1 elements, **FP8-e4m3 block scales, block size 16**) is NVIDIA-proprietary. The AMD-native analog is **MXFP4** (e2m1 elements, **E8M0 power-of-two shared scales, block size 32**, per the OCP MX spec). These are *not* numerically interchangeable — different scale format, different block granularity, different quantization error. The correct move is a **respec, not a translation**: for each NVFP4 problem, author an MXFP4 twin (same tensor shapes, same fusion structure, same workload axes; quantization/dequantization reference rewritten for MX semantics; tolerances calibrated fresh). Dataset marks these `quant_mxfp4/` with provenance metadata linking to the NVFP4 original. PyTorch's `float4_e2m1fn_x2` dtype (already in the harness's dtype table) plus E8M0 scale tensors express this; kernel-side support comes from hipBLASLt/CK MXFP4 paths on gfx950 and Triton's scaled-dot support. This is the highest-uncertainty dataset item — schedule a 1-week feasibility spike early (Phase 0 exit criterion), with the fallback of shipping v1 with 18/33 Quant problems and adding MXFP4 in v1.1.

### 6.3 FlashInfer-Bench (26)

Problems follow the FlashInfer Trace schema but their references are, again, PyTorch — portable. The category's *purpose* (production inference primitives with structured inputs) is served on AMD by **AITER**, and FlashInfer's own ROCm/AITER backend means even FlashInfer-API solutions can run. Port the category as-is, seed one AITER example solution, and validate paged-KV input generators carefully (layout assumptions are the likeliest silent breakage).

---

## 7. SOL bounds and baselines for MI355X

This is the scientific heart of the port; get it wrong and the scores are noise.

### 7.1 Computing T_SOL

SOLAR's pipeline (graph → einsum → hardware-independent analysis → roofline prediction) is architecture-agnostic until its final stage, which reads per-arch YAML from `configs/arch/`. Work items:

1. **Author `MI355X.yaml`** with: peak dense matrix throughput per dtype (BF16 2.5 PF, FP8/MXFP8 5 PF, MXFP4/MXFP6 10.1 PF — *derated to the locked clock*, i.e., × F_LOCK/2400 MHz, mirroring how the B200 config represents 1500 MHz rather than boost), vector throughput, HBM bandwidth 8 TB/s, and the on-chip capacity hierarchy for the Orojenesis-style buffer analysis (LDS 160 KB/CU-class, 32 MB L2, 256 MB LLC — extract exact CDNA4 numbers from the ISA guide; the LLC materially tightens memory-traffic lower bounds for fused L2-category problems, same as B200's L2 does).
2. **Decide the bandwidth convention consciously.** The B200 bounds use theoretical peaks at locked clocks. Keep the *same convention* on AMD (theoretical, not achieved-STREAM) for methodological parity — but publish, alongside, the measured ceilings (hipBLASLt GEMM roof, HBM copy roof at F_LOCK) so score consumers can see the achievable-fraction difference between platforms. This is the honest way to handle "MFMA utilization ceilings differ across vendors" without forking the methodology.
3. **Run SOLAR over all 235 problem graphs** → `t_sol` table per (problem, workload instance). Spot-audit ~20 problems by hand-computing FLOPs/bytes.
4. **Sanity cross-checks:** memory-bound problems (most of L1 — norms, RoPE, embeddings, SwiGLU) should have MI355X SOLs within a few percent of B200's, because both machines are 8 TB/s at the roof; BF16 compute-bound SOLs should scale by ≈ (2.5/2.25) × (F_LOCK-derate ratio). Any problem violating these expectations gets audited. Also require T_SOL ≤ best measured kernel time on every problem (an *upper* bound violation means a config error).

### 7.2 Re-measuring T_b

T_b (the S=0.5 anchor) is an *optimized PyTorch implementation* per problem, timed on the target. These must be **re-tuned and re-measured on MI355X**, not copied: the fastest PyTorch expression of a problem differs across platforms (different `torch.compile`/Inductor decisions, different fusion profitability). Procedure: take the reference, apply the same optimization policy NVIDIA used (their optimized-PyTorch baselines ship as the scoring anchor — mirror the policy: eager-vs-compile choice, contiguity/layout hygiene, no handwritten kernels), measure under the full harness (locked clocks, cache clears, shifting allocator), and freeze `t_b` per workload. Then verify the anchor property empirically: submitting T_b's implementation must score ≈0.5, and the reference must score <0.5.

### 7.3 Score integrity across platforms

Publish per-problem (T_SOL, T_b, tolerances, F_LOCK, ROCm/driver/torch versions) as a versioned scoring manifest, exactly as the NVIDIA leaderboard does implicitly. Scores are valid *within* a manifest version; any stack upgrade that shifts T_b re-issues the manifest.

---

## 8. Phased roadmap

| Phase | Duration | Deliverable | Exit criteria |
|---|---|---|---|
| **0 — Bring-up & spikes** | 1–2 wk | ROCm 7.2 + torch 2.9 container; harness runs unmodified `pytorch`-language solutions on 5 L1 problems via `cuda_events` path on an MI300X/MI355X box; **spikes:** MXFP4 feasibility (§6.2), rocprofiler-sdk record fidelity (§4.4), C1 clock calibration protocol | All 5 problems produce traces; spike reports with go/no-go on MXFP4 |
| **1 — Harness port** | 2–4 wk | Vendor device layer; amd-smi clock locking + verification; LLC-sized cache clears; gfx950 build path (`hip_cpp`); enum/schema extensions; ROCm Dockerfile; CI on an AMD runner (harness unit tests + 10-problem smoke) | Full L1+L2 reference sweep passes correctness on MI355X; timing CV < 2% on locked clocks |
| **2 — Timing parity & anti-hack re-validation** | 2–3 wk | rocprofiler-sdk shim + ported sequence-matching (`rocprof` methodology); red-team pass on ROCm (stream hiding, cache exploits, patching, smi-from-submission) | events-vs-rocprof divergence ≤2% on L1 (excl. μs kernels, reported separately); all known exploit reproductions detected |
| **3 — SOL bounds + baselines** | 3–5 wk (overlaps 2) | `MI355X.yaml` for SOLAR; F_LOCK frozen; T_SOL for all problems; measured-ceiling companion report; re-tuned T_b set; scoring manifest v1 | Cross-checks in §7.1 pass; anchor property (§7.2) holds on 20-problem sample |
| **4 — Dataset completion** | 2–4 wk (overlaps 3) | AMD-calibrated tolerances for all workloads; 78 input generators verified; Quant: 18 FP8 validated + 15 MXFP4 respecs (or documented v1.1 deferral); FlashInfer category + AITER seed solutions; AMD examples per backend (`ck`, `ck_tile`, `hipblaslt`, `miopen`, `aiter`, `hip_cpp`, `triton`) | 235 (or 220) problems pass reference eval end-to-end; `SOL-ExecBench-ROCm` dataset published (HF) |
| **5 — Baselines, leaderboard, release** | 2 wk + ongoing | Agent-optimizer baseline sweep (the paper's median-0.732 analog for MI355X); LLM-judge screening in the submission path; docs; upstream PR conversation with NVIDIA for the vendor layer vs. maintained fork; public leaderboard | Public release with baseline scores and methodology writeup |

Total: **~14–18 engineer-weeks** (≈3–4 months for one strong systems engineer plus a part-time perf engineer for Phases 3–4; Phases 2/3/4 parallelize well across two people into ~2–2.5 months).

---

## 9. Risks and open questions

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **MXFP4 software path immaturity** on gfx950 (torch scaled-mm coverage, hipBLASLt/CK kernels, Triton scaled-dot) blocks 15 Quant respecs | High | Phase-0 spike; ship-without-and-add-later fallback (220-problem v1); engage AMD AITER/hipBLASLt teams |
| R2 | **rocprofiler-sdk shim** fidelity (timestamp semantics, records under heavy dispatch rates, correlation-id behavior) differs from CUPTI in corner cases | Med-High | Phase-0 spike; keep `hip_events` as always-available fallback with methodology recorded per trace; cross-validation gate in Phase 2 |
| R3 | **Analytic-peak reachability differs across vendors** → cross-platform score comparisons over-interpreted | Med (reputational) | Publish measured ceilings alongside T_SOL (§7.1.2); document scores as within-platform; never market a cross-vendor leaderboard delta without the caveat |
| R4 | **MI355X access** for CI + calibration (scarcer than MI300X in clouds) | Med | AMD Developer Cloud / partner allocation; develop Phases 0–2 on MI300X; only Phases 3–4 truly require MI355X |
| R5 | `l2CacheSize`/device-props semantics on ROCm mislead cache-clear sizing (V3) and any props-derived logic | Low-Med | Explicit per-device LLC table (§4.3); verify empirically (bandwidth cliff test on buffer sweep) |
| R6 | **Tolerance recalibration reveals reference nondeterminism** on ROCm (atomics-based ops, e.g., some MoE dispatch/scatter paths) | Med | Calibration across seeds and runs; where variance is structural, widen matched_ratio deliberately and document per problem |
| R7 | **Thermal/power variance at 1400 W** undermines run-to-run stability even at locked clocks | Med | C1 protocol locks below PCC floor; require timing CV gate in CI; monitor clock residency in traces and reject perturbed trials |
| R8 | New AMD-specific reward hacks (e.g., XCD partition tricks, smi from submission, LDS-resident state across iterations) | Med | Red-team phase 2; lock down smi/sudo in eval containers; pointer-shifting allocator already defeats address-keyed caching |
| R9 | Upstream churn: NVIDIA repo evolves (v1.0.2 today); fork drifts | Low-Med | Propose the vendor device layer upstream early (the enum design invites it); keep AMD deltas as a thin layer, rebase weekly |
| R10 | MIOpen/cudnn-frontend gap leaves some *example* solutions without analogs | Low | Problems unaffected (references are PyTorch); document backend-coverage matrix honestly |

**Open questions to resolve with stakeholders:** (a) Fork identity — upstream `sol-execbench` PR with `--vendor amd`, or a sibling `SOL-ExecBench-ROCm` repo? Recommendation: build as a fork structured for upstreaming, decide at Phase 5 based on NVIDIA's receptiveness. (b) Should an MI300X tier ship (huge accessibility win for the community, at the cost of a second SOL manifest and fnuz FP8 handling)? (c) Leaderboard hosting and compute sponsorship for continuous evaluation.

---

## 10. Verification plan (acceptance criteria)

1. **Mechanism:** all harness unit tests green on ROCm; 235-problem reference sweep passes correctness; subprocess isolation, timeouts, and trace schema byte-compatible with upstream tooling.
2. **Timing:** locked-clock timing CV < 2% across trials on a 30-problem sample; `hip_events` vs `rocprof` ≤ 2% median divergence (μs-scale kernels reported under a separate flag); flush-buffer sweep shows the expected bandwidth cliff at LLC capacity (validates §4.3).
3. **Scoring:** T_SOL ≤ best measured time on every problem; T_b implementations score 0.5 ± 0.03; references score < 0.5; memory-bound SOL parity with B200 within 5% on the 8 TB/s-roof subset; a deliberately reward-hacked submission set (replayed from the paper's three families) is 100% detected.
4. **End-to-end:** an off-the-shelf kernel-optimizing agent, retargeted only by changing `target_hardware`, completes a full run and produces a score distribution qualitatively comparable to the paper's (median in the ~0.6–0.8 band, headroom-reclaimed correlation reproduced on AMD).

---

## 11. Prior art worth leveraging

The AMD kernel-optimization ecosystem has matured fast: AMD's $100K kernel Developer Challenge and the GPU MODE leaderboards established MI300X evaluation infrastructure and a practitioner community; [AITER](https://github.com/ROCm/aiter) consolidates production inference kernels (its MLA/attention kernels are direct baselines for many FlashInfer-Bench problems); the [vLLM ROCm attention work](https://blog.vllm.ai/2026/02/27/rocm-attention-backend.html) and [FlashInfer-on-ROCm](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html) demonstrate the AITER-backed path this plan assumes. None of these do what SOL-ExecBench does — hardware-bound-anchored scoring — which is exactly why this port is worth doing: it would give the AMD ecosystem its first headroom-grounded kernel benchmark, and the community above is the natural early-adopter base.

---

## Appendix A — Quick-reference mapping table

| NVIDIA | AMD (this port) |
|---|---|
| B200 (sm_100a) | MI355X (gfx950) |
| CUDA 13.1 / nvcc | ROCm ≥7.2 / hipcc (amdclang++) |
| `nvidia-smi -lgc 1500 -lmc 3996` | `amdsmi` / perf-determinism at F_LOCK (C1) |
| CUPTI (`cupti-python`) | rocprofiler-sdk + custom shim |
| `cudaCtxResetPersistingL2Cache` | no-op (no CDNA equivalent) |
| 2× L2 (126 MB-class) flush buffer | 2× LLC = 512 MB flush buffer |
| `-gencode …sm_100a` | `--offload-arch=gfx950` |
| `--use_fast_math`, `-lcuda` | `-ffast-math`, `-lamdhip64` |
| cuBLAS(Lt) | hipBLASLt / rocBLAS |
| cuDNN (+frontend) | MIOpen |
| CUTLASS | Composable Kernel (CK) |
| CuTe DSL | CK-Tile |
| cuTile | — (Triton covers the niche) |
| FlashInfer | FlashInfer-ROCm / AITER |
| NVFP4 (e2m1, FP8-e4m3 scales, block 16) | MXFP4 (e2m1, E8M0 scales, block 32) — respec, not translate |
| OCP FP8 e4m3fn/e5m2 | identical on CDNA4 (fnuz only on CDNA3) |
| nvidia/cuda:13.1.1-cudnn-devel | rocm/pytorch (ROCm ≥7.2) |
| `cuda-bindings`, `cupy-cuda13x`, … | `amdsmi`, rocprofiler shim, CK/AITER installs |

## Appendix B — Key sources

- Benchmark: [problems page](https://research.nvidia.com/benchmarks/sol-execbench/problems) · [GitHub](https://github.com/nvidia/sol-execbench) · [dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) · [paper, arXiv:2603.19173](https://arxiv.org/abs/2603.19173) · [SOLAR](https://github.com/NVlabs/SOLAR)
- AMD: [MI355X specs](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) · [ROCm 7.2](https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html) · [MI300X system optimization (perf determinism)](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html) · [rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/) · [AITER](https://github.com/ROCm/aiter) · [Composable Kernel](https://github.com/ROCm/composable_kernel) · [FlashInfer on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html) · PyTorch ROCm cpp_extension: [#22091](https://github.com/pytorch/pytorch/pull/22091), [#149245](https://github.com/pytorch/pytorch/pull/149245)
