#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the rocprofiler-sdk activity shim (task 04).

    python src/solexbench_rocm/shim/build.py

Produces `_rocprof_shim*.so` next to this file. Deliberately a plain
setuptools/pybind11 build rather than a torch extension: the shim links
rocprofiler-sdk, not torch, and pulling the whole torch extension machinery in
would make a link error here look like a torch problem.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCM = Path("/opt/rocm")


def main() -> int:
    import pybind11

    src = HERE / "rocprof_shim.cpp"
    out = HERE / f"_rocprof_shim{sysconfig.get_config_var('EXT_SUFFIX')}"

    include = ROCM / "include"
    lib = ROCM / "lib"
    if not (include / "rocprofiler-sdk" / "registration.h").exists():
        print(f"rocprofiler-sdk headers not found under {include}.\n"
              f"This is a hard stop, not something to work around: without the "
              f"SDK the AMD timing path stays on hip_events, which is correct "
              f"but includes launch overhead.", file=sys.stderr)
        return 2
    if not (lib / "librocprofiler-sdk.so").exists():
        print(f"librocprofiler-sdk.so not found under {lib}", file=sys.stderr)
        return 2

    cmd = [
        "g++", "-O3", "-Wall", "-shared", "-std=c++17", "-fPIC",
        f"-I{include}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        str(src),
        f"-L{lib}", "-lrocprofiler-sdk",
        # rpath so the extension finds the SDK without LD_LIBRARY_PATH, which a
        # subprocess-isolated eval driver would not inherit reliably.
        f"-Wl,-rpath,{lib}",
        "-o", str(out),
    ]
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return proc.returncode
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
