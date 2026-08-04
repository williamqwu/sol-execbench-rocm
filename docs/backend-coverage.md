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
| `aiter` | ✅ | Python import | ❌ not exercised |
| `ck` | ✅ | needs a **writable** CK header tree (`CK_DIR`) | ❌ not exercised |
| `ck_tile` | ✅ | same as `ck` | ❌ not exercised |
| `hipblaslt` | ✅ | hipcc + `-lhipblaslt` | ❌ not exercised |
| `miopen` | ✅ | hipcc + `-lMIOpen` | ❌ not exercised |
| `cuda_cpp`, `cutlass`, `cudnn`, `cublas` | ✅ | NVIDIA only | n/a on this node — kept working as the regression reference |

## What the one seed bought

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

## The gap, stated plainly

Five backends — `ck`, `ck_tile`, `hipblaslt`, `miopen`, `aiter` — are accepted
by the schema and have a plausible build path, and **nothing has been built
through them**. On the evidence of the three defects above, the probability
that all five work untested is low. Each needs one seed kernel before it can be
called supported.

`ck`/`ck_tile` additionally need `CK_DIR` pointed at a writable copy of the CK
headers, because hipify cannot process the read-only system tree.
