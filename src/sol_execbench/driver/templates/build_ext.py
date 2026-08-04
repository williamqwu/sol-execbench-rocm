# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

import torch.utils.cpp_extension as ext

from sol_execbench.core import Solution

HERE = Path.cwd().resolve()

# Parse solution — validates sources (e.g. forbidden keywords) at compile time.
solution = Solution(**json.loads((HERE / "solution.json").read_text()))
compile_options = solution.spec.compile_options

# set flags
cuda_cflags = compile_options.cuda_cflags if compile_options else []
cflags = compile_options.cflags if compile_options else []
ld_flags = compile_options.ld_flags if compile_options else []

from sol_execbench.core.bench.device import detect_vendor

_IS_AMD = detect_vendor() == "amd"

# Collect C/C++/CUDA/HIP source files from current directory.
# AMD: `.hip` is accepted so a submission can be written natively as HIP.
# torch.utils.cpp_extension auto-hipifies `.cu` on a ROCm build, which is
# convenient for ported CUDA but mangles source that is already HIP.
sources = [
    str(p)
    for p in HERE.iterdir()
    if p.suffix in (".cu", ".hip", ".cpp", ".cc", ".cxx", ".c") and p.is_file()
]
if not sources:
    # Wording keeps upstream's exact "No CUDA/C++ source files" substring, which
    # test_build_ext matches on.
    raise RuntimeError(
        "No CUDA/C++ source files found in working directory (.hip also accepted)"
    )

if _IS_AMD:
    # AMD: Composable Kernel is the CUTLASS counterpart -- but it is added
    # ONLY for the languages that need it.
    #
    # torch's ROCm `load()` runs hipify over every directory in
    # extra_include_paths and writes the hipified headers back *into that
    # directory*. Passing the ROCm include tree therefore fails for any
    # unprivileged user:
    #
    #   PermissionError: '/opt/rocm/include/thrust/system/hip/detail/core'
    #
    # which surfaces as a compile failure on a perfectly good kernel and has
    # nothing to do with the kernel. hipcc already has the ROCm includes on
    # its default search path, so a hip_cpp submission needs none of this.
    # Without this, torch compiles for ELEVEN architectures -- every gfx target
    # its wheel was built for, gfx908 through gfx1201 -- because
    # PYTORCH_ROCM_ARCH is unset. That is slow, and it fails the whole build on
    # an architecture the submission never targets: observed here as
    # "1 error generated when compiling for gfx1030" on a kernel that compiles
    # cleanly for gfx950. The arch is read from the device rather than
    # hard-coded, so the same template serves MI350X, MI355X and MI300X.
    if not os.environ.get("PYTORCH_ROCM_ARCH"):
        try:
            import torch

            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            if arch:
                os.environ["PYTORCH_ROCM_ARCH"] = arch
        except Exception:                                  # noqa: BLE001
            pass                                           # leave torch's default

    langs = {str(getattr(x, "value", x)) for x in solution.spec.languages}
    extra_include_paths = [str(HERE)]
    if langs & {"ck", "ck_tile"}:
        # Point CK_DIR at a WRITABLE copy of the CK headers. The default is
        # the read-only system tree, which hipify cannot process; that is a
        # loud failure rather than a silent one, and this comment is where it
        # is explained.
        ck_dir = os.environ.get("CK_DIR", "/opt/rocm/include/ck")
        extra_include_paths += [ck_dir, f"{ck_dir}/../"]
else:
    cutlass_dir = os.environ.get("CUTLASS_DIR", "/usr/local/cutlass")
    extra_include_paths = [
        str(HERE),
        f"{cutlass_dir}/include",
        f"{cutlass_dir}/tools/util/include",
    ]

ext.load(
    name="benchmark_kernel",
    sources=sources,
    extra_cuda_cflags=cuda_cflags,
    extra_cflags=cflags,
    extra_ldflags=ld_flags,
    extra_include_paths=extra_include_paths,
    build_directory=str(HERE),
    verbose=True,
)

# Rename platform-suffixed .so → benchmark_kernel.so
so_files = [
    f for f in HERE.glob("benchmark_kernel*.so") if f.name != "benchmark_kernel.so"
]
if so_files:
    so_files[0].rename("benchmark_kernel.so")
elif not (HERE / "benchmark_kernel.so").exists():
    raise FileNotFoundError("benchmark_kernel.so not produced by compilation")

print("benchmark_kernel.so ready", flush=True)
