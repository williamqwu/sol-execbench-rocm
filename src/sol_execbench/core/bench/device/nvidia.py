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

"""NVIDIA backend for the vendor device layer.

Behaviour-preserving: every function here reproduces what the harness did
before the device layer existed. This path is the regression reference for the
AMD port -- when an AMD result looks wrong, running the same code on NVIDIA is
what distinguishes "the refactor broke something" from "AMD genuinely differs".
"""

from __future__ import annotations

VENDOR = "nvidia"


def llc_bytes(device=None) -> int:
    """L2 cache size in bytes, as reported by the driver."""
    import torch

    return torch.cuda.get_device_properties(device).L2_cache_size


def flush_buffer_bytes(device=None) -> int:
    """Twice the L2 size -- the long-standing upstream sizing."""
    return llc_bytes(device) * 2


def reset_persisting_l2_cache(device=None) -> None:
    """Reset persisting L2 cache lines to normal status."""
    import torch
    from cuda.bindings import runtime as cuda_runtime

    def reset_current_device() -> None:
        result = cuda_runtime.cudaCtxResetPersistingL2Cache()
        if isinstance(result, tuple):
            result = result[0]
        torch.cuda.check_error(int(result))

    if device is not None:
        with torch.cuda.device(device):
            reset_current_device()
        return
    reset_current_device()


def arch_flags(hardware=None) -> list[str]:
    """`-gencode` flags for the target hardware.

    Kept in the packager, which owns the hardware-to-SM mapping; exposed here
    so the AMD backend can answer the same question.
    """
    from sol_execbench.driver.problem_packager import gencode_flags_for_hardware

    return gencode_flags_for_hardware(hardware)


def default_device_cflags() -> list[str]:
    return ["-O3", "--use_fast_math"]


def default_ld_flags() -> list[str]:
    return ["-lcuda"]
