# Backend coverage

Which submission languages this port **accepts**, and which have a **seed
example that was actually compiled and run on gfx950**. The distinction is the
point: a language in the schema is a promise, and until something has been
built through it the promise is untested.

| language | schema | build path | seed example, run on hardware |
|---|---|---|---|
| `pytorch` | ✅ | n/a | ✅ every reference and every T_b variant (235 problems) |
| `triton` | ✅ | n/a | ✅ `artifacts/07/spike.json` — `dot_scaled` MXFP4, compiled, launched, numerically verified |
| `hip_cpp` | ✅ | hipcc, `--offload-arch=gfx950` | ✅ [`reference/seeds/hip_cpp__025_rmsnorm_h4096.json`](../reference/seeds/hip_cpp__025_rmsnorm_h4096.json) |
| `aiter` | ✅ | Python import | ✅ [`reference/seeds/aiter__025_rmsnorm_h4096.json`](../reference/seeds/aiter__025_rmsnorm_h4096.json) |
| `ck` | ✅ | needs a **writable** CK header tree (`CK_DIR`), and `-U__HIP_NO_HALF_CONVERSIONS__` | ✅ [`reference/seeds/ck__004_gemm_n128_k2048.json`](../reference/seeds/ck__004_gemm_n128_k2048.json) |
| `ck_tile` | ✅ | same as `ck` | ✅ [`reference/seeds/ck_tile__025_rmsnorm_h4096.json`](../reference/seeds/ck_tile__025_rmsnorm_h4096.json) |
| `hipblaslt` | ✅ | hipcc + `-lhipblaslt` | ✅ [`reference/seeds/hipblaslt__004_gemm_n128_k2048.json`](../reference/seeds/hipblaslt__004_gemm_n128_k2048.json) |
| `miopen` | ✅ | hipcc + `-lMIOpen` | ✅ [`reference/seeds/miopen__025_rmsnorm_h4096.json`](../reference/seeds/miopen__025_rmsnorm_h4096.json) |
| `cuda_cpp`, `cutlass`, `cudnn`, `cublas` | ✅ | NVIDIA only | n/a on this node — kept working as the regression reference |

## What "exercised" means here, exactly

Every ✅ in the last column above means: the seed was run through
`scripts/agent_eval.py`, which is the **real** `ProblemPackager` +
`eval_driver` path — the same compile, the same staging directory, the same
anti-reward-hack guards, the same output comparison — against the AMD-derived
tolerances in `artifacts/05/workloads/`, and **every workload of the problem
passed**. Results, with provenance, are in `artifacts/backends/*.json`; the
console output of each run is in `artifacts/backends/logs/`.

    env/solb python scripts/agent_eval.py \
      --problem data/SOL-ExecBench/benchmark/FlashInfer-Bench/<problem> \
      --solution reference/seeds/<seed>.json --out artifacts/backends/<lang>.json

(with `HIP_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=8`,
`SOLEXBENCH_WORKLOADS_ROOT=/work/artifacts/05/workloads`.)

`agent_eval.py` grew a `--solution` flag for this: `--kernel` can only carry a
bare source file, and three of the five seeds have to set `compile_options`
(`-lhipblaslt`, `-lMIOpen`, the `-U` defines CK needs). The flag changes
nothing about how a solution is evaluated — only where the `Solution` object
comes from.

Result of the last clean pass, GPU 0 of `mia1-p02-g46` (MI355X, ROCm 7.2.0,
torch 2.9.1+rocm7.2.0, driver 6.16.6), one run at a time on an otherwise idle
GPU:

| seed | problem | workloads passed | worst rel. error | vs. reference |
|---|---|---|---|---|
| `hip_cpp` | 025_rmsnorm_h4096 | 14/14 | 7.8e-3 | 2.75× |
| `aiter` | 025_rmsnorm_h4096 | 14/14 | 7.8e-3 | 6.11× |
| `miopen` | 025_rmsnorm_h4096 | 14/14 | 7.8e-3 | 1.97× |
| `ck_tile` | 025_rmsnorm_h4096 | 14/14 | 7.8e-3 | 5.35× |
| `hipblaslt` | 004_gemm_n128_k2048 | 25/25 | 0 (bit-exact) | 0.99× |
| `ck` | 004_gemm_n128_k2048 | 25/25 | 1.2e-3 | 0.18× |

The worst relative error on every bf16 seed is one bf16 ulp (2⁻⁷ = 7.8e-3),
which is the floor the tolerance itself is clamped at — these kernels are as
correct as bf16 permits, not merely inside a loose bound. `hipblaslt` is
bit-identical to the torch reference, which is the strongest available check
that its column-major layout mapping is right, since torch's own fp16 GEMM
lands on hipBLASLt too.

**The speedup column is not a measurement of anything.** It is `agent_eval`'s
feedback ratio, taken without a locked clock (`f_lock_mhz: null` in every one
of those artifacts). It is here to show the seeds are not accidentally
pathological, and for nothing else. The `ck` seed at 0.18× is expected: it is
the stock 256×128 tile on a problem whose N is 128 and whose M is often 1.

## What the seeds bought

### The first one, `hip_cpp`

`hip_cpp__025_rmsnorm_h4096` is a deliberately unoptimized wavefront-per-row
RMSNorm. It exists to exercise the path, not to be fast, and it found three
defects that would have hit the first real C++ submission — each of which
compiled or validated cleanly right up to the point it failed:

1. **hipify writes into the include tree it is given.** `build_ext.py` passed
   `/opt/rocm/include` (via `CK_DIR/..`) on every AMD build, and torch's ROCm
   `load()` hipifies every directory in `extra_include_paths` *in place*:

   ```
   PermissionError: '/opt/rocm/include/thrust/system/hip/detail/core'
   ```

   An unprivileged user could not compile anything. CK paths are now added only
   for `ck`/`ck_tile`, which is also the only language that needs them.

2. **`-lcuda` on ROCm.** `CompileOptions.ld_flags` defaults to `["-lcuda"]`.
   That default only materializes once `compile_options` exists — which the
   packager creates for every LOCAL submission when it injects
   `--offload-arch` — so *every* C++ submission on ROCm failed at the link
   step with `cannot find -lcuda`, after compiling cleanly. Now `-lamdhip64`.

3. **The AMD C++ languages were not in the driver's C++ set.** They were added
   to the schema's `cpp_languages` but not to `eval_driver`'s `_CPP_LANGUAGES`,
   so a `hip_cpp` submission took the Python branch and died on
   `ModuleNotFoundError: No module named 'kernel'` — with a correctly built
   `benchmark_kernel.so` sitting beside it.

Also fixed while here: with `PYTORCH_ROCM_ARCH` unset, torch compiles for all
eleven architectures its wheel targets, and the build fails on the first one
that dislikes the source (`1 error generated when compiling for gfx1030`) even
though gfx950 is fine. The template now reads the architecture off the device.

### The other five

Two more packaging defects, both of the same shape as defect 2 above and both
found the same way — a build failure on a kernel that was fine:

4. **`--use_fast_math` is nvcc-only.** `CompileOptions.cuda_cflags` defaults to
   `["-O3", "--use_fast_math"]`, and hipcc hands the flag straight to clang++:

   ```
   clang++: error: unknown argument: '--use_fast_math'
   ```

   Hit by the `hipblaslt` seed, whose only crime was asking for a link flag.
   On AMD the packager now substitutes `-ffast-math`, clang's spelling of the
   same intent.

5. **Defect 2's fix only covered half the cases.** It read
   `if not compile_options.get("ld_flags")` — but `-lcuda` is a pydantic
   *default*, so it is present in the dumped dict whenever the submission set
   any *other* build flag. The emptiness test therefore skipped exactly the
   submissions that had asked for something. Hit by the `ck` seed, which sets
   only `cuda_cflags` and still died on `cannot find -lcuda` after compiling a
   complete CK GEMM instance. It is now a substitution, not a fill-in.

   Both are in `ProblemPackager._inject_gencode_flags`, both are covered by
   `test_amd_translates_nvcc_only_defaults`.

And three environment facts that a submission has to know, none of which are
defects in this repo:

6. **CK will not compile under torch's half-precision defines.** torch's ROCm
   build passes `-D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1`,
   which strip `__half`'s converting constructors; `ck/utility/math_v2.hpp`
   does `static_cast<__half>(half_t)` unconditionally and fails with

   ```
   error: no matching conversion for static_cast from 'half_t' (aka '_Float16') to '__half'
   ```

   ×6. A `ck`/`ck_tile` submission must undo them in its own `cuda_cflags`
   (`-U__HIP_NO_HALF_CONVERSIONS__ -U__HIP_NO_HALF_OPERATORS__`), which is
   what both seeds do. This is deliberately **not** patched into the packager:
   it changes the meaning of `__half` for the submission's own code, and that
   is the submission's decision to make.

7. **MIOpen's norm entry points are behind `MIOPEN_BETA_API`.** Without
   `#define MIOPEN_BETA_API 1` before `<miopen/miopen.h>`,
   `miopenT5LayerNormForward` is not declared at all.

8. **MIOpen and hipBLASLt each keep their own stream.** Neither picks up the
   harness's. `miopenSetStream()` / the `hipblasLtMatmul` stream argument are
   not optional: without them the work is issued outside the timing bracket,
   which does not fail — it measures the wrong thing.

## Notes on the individual seeds

- **`aiter`** targets `aiter.rmsnorm2d_fwd`. For bf16, hidden ≤ 8192, 2-D,
  non-T5 input its dispatcher (`_rms_norm_fwd_dispatch`) takes the fast HIP
  branch — a JIT-built C++ module, `module_rmsnorm_quant`. So this seed does
  **not** touch aiter's gluon kernels, and therefore does not exercise the
  `AITER_USE_SYSTEM_TRITON=1` compromise (aiter at the pinned SHA wants
  triton ≥ 3.6.0 against this stack's 3.5.1; the flag downgrades its hard
  raise to a warning). **A gluon-dispatching aiter op is still untested**, and
  an agent that reaches one is running against an older Triton than aiter was
  written for. Whether that is safe is not established here.
- **`ck`** instantiates the stock `DeviceGemm_Xdl_CShuffle` fp16 instance with
  `GemmSpecialization::MNKPadding`, which is required, not stylistic: M is a
  benchmark axis and reaches 1, far under `MPerBlock`, and the unpadded
  specialization would be refused at run time by `IsSupportedArgument()` one
  workload at a time.
- **`ck_tile`** assembles `Rmsnorm2dFwd` from the shipped pipeline and
  epilogue templates. ROCm installs no host launcher for it — upstream keeps
  that in an example directory that is not packaged — so the seed writes the
  launch itself. Two traps: `Kernel::BlockSize()` is a host function (it calls
  the runtime `is_wave32()`) and so cannot be a template argument, and the
  LDS size must come from `Kernel::GetSmemSize()`, since the one-pass pipeline
  reduces across warps through LDS and passing 0 launches happily.
- **`hipblaslt`** and **`ck`** both hit the same layout question: the problem
  is row-major `C = A @ Bᵀ`, and both libraries are column-major. A row-major
  M×K buffer read column-major is K×M, which makes the mapping mechanical —
  but a mistake here produces numbers rather than an error, and only the
  tolerance would catch it.

## What is still not covered

- Every seed above targets **one** problem in FlashInfer-Bench (an RMSNorm or
  a small GEMM), chosen because a well-conditioned problem isolates the
  toolchain from the kernel. "The `ck` path builds and runs" is now
  established; "CK covers the shapes across the benchmark" is not, and is a
  different claim.
- No seed is tuned, none is a `T_b` candidate, and none has been timed under a
  locked clock. Nothing here belongs in a scoring artifact.
- `aiter`'s gluon path (see above) and MIOpen beyond `T5LayerNorm`.
