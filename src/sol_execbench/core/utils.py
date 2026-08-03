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

from __future__ import annotations

import os
import platform
import sys
from typing import TYPE_CHECKING, Dict, List, Optional


if TYPE_CHECKING:
    from sol_execbench.core.data import Environment


def is_cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def list_cuda_devices() -> List[str]:
    import torch

    n = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(n)]


def env_snapshot(device: str, methodology: Optional[str] = None) -> "Environment":
    # AMD: *methodology* is the timing methodology actually used for this
    # evaluation. Passed in rather than re-derived here, so the trace records
    # what ran and not what the default would have been -- an explicit
    # `methodology=` argument at the call site would otherwise be recorded
    # wrongly. Defaults to the vendor's default when the caller does not know.
    import torch

    from sol_execbench.core.bench import device as device_layer
    from sol_execbench.core.data import Environment

    libs: Dict[str, str] = {"torch": torch.__version__}
    try:
        import triton as _tr

        libs["triton"] = getattr(_tr, "__version__", "unknown")
    except Exception:
        pass

    try:
        import torch.version as tv

        if getattr(tv, "cuda", None):
            libs["cuda"] = tv.cuda
    except Exception:
        pass
    if methodology is None:
        try:
            methodology = device_layer.default_timing_methodology()
        except Exception:
            methodology = None

    # AMD: compute-partition mode belongs in libs rather than as another
    # top-level field -- it is environment detail, not a schema concept, and
    # upstream has no analogue. It must be recorded because SPX and CPX give a
    # kernel different amounts of the GPU, so two traces taken under different
    # modes are not comparable and nothing else in the trace would say so.
    try:
        from sol_execbench.core.bench.reward_hack import compute_partition_mode

        partition = compute_partition_mode()
        if partition:
            libs["compute_partition"] = partition
    except Exception:
        pass

    return Environment(
        hardware=hardware_from_device(device), libs=libs, methodology=methodology
    )


def hardware_from_device(device: str) -> str:
    import torch

    d = torch.device(device)
    if d.type == "cuda":
        return torch.cuda.get_device_name(d.index)
    if d.type == "cpu":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return platform.processor() or platform.machine() or "CPU"
    if d.type == "mps":
        return "Apple GPU (MPS)"
    if d.type == "xpu" and hasattr(torch, "xpu"):
        try:
            return torch.xpu.get_device_name(d.index)
        except Exception:
            return "Intel XPU"
    return d.type


def redirect_stdio_to_file(log_path: str) -> tuple[int, int]:
    """Redirect stdout/stderr to log file.

    Returns original stdout and stderr file descriptors for printing to terminal.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    fd = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
    return original_stdout_fd, original_stderr_fd


def flush_stdio_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
