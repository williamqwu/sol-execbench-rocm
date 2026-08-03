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

"""Vendor device layer.

Everything in the harness that is genuinely vendor-specific is reached through
this module, so the rest of the code can stay written against one API. The
NVIDIA backend reproduces the previous behaviour exactly; the AMD backend
implements the same contract for ROCm / CDNA.

Both backends stay importable on either platform -- selecting a backend must
never require the other vendor's libraries to be installed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

Vendor = Literal["nvidia", "amd"]


@lru_cache(maxsize=1)
def detect_vendor() -> Vendor:
    """Return the vendor of the current torch build.

    ``torch.version.hip`` is set on ROCm builds and ``None`` on CUDA builds.
    This is the ROCm-recommended discriminator: a ROCm torch reports through
    the ``torch.cuda`` API, so ``torch.cuda.is_available()`` cannot tell the
    two apart.

    Cached: this is consulted from Pydantic default factories, so it runs on
    every ``CompileOptions`` construction, and the answer cannot change within
    a process.

    The attribute walk is deliberately defensive. A torch that is absent or
    stubbed (as in ``test_build_ext``, which execs the build template against a
    fake torch module) means "not a ROCm build", not a crash.
    """
    try:
        import torch
    except ImportError:
        return "nvidia"

    return "amd" if getattr(getattr(torch, "version", None), "hip", None) else "nvidia"


def get_backend(vendor: Vendor | None = None):
    """Return the backend module for *vendor* (default: the detected one)."""
    vendor = vendor or detect_vendor()
    if vendor == "amd":
        from . import amd

        return amd
    from . import nvidia

    return nvidia


# -- Facade -----------------------------------------------------------------
# Thin pass-throughs so call sites read as `device.llc_bytes(...)` rather than
# repeating the backend lookup.


def llc_bytes(device=None) -> int:
    """Bytes of last-level cache that a benchmark must flush to run cold."""
    return get_backend().llc_bytes(device)


def flush_buffer_bytes(device=None) -> int:
    """Size of the buffer used to evict the LLC before a timed iteration."""
    return get_backend().flush_buffer_bytes(device)


def reset_persisting_l2_cache(device=None) -> None:
    """Reset persisting L2 lines, where the concept exists."""
    return get_backend().reset_persisting_l2_cache(device)


def arch_flags(hardware=None) -> list[str]:
    """Compiler flags selecting the target architecture."""
    return get_backend().arch_flags(hardware)


def default_device_cflags() -> list[str]:
    """Default device-compiler flags for this vendor."""
    return get_backend().default_device_cflags()


def default_ld_flags() -> list[str]:
    """Default linker flags for this vendor."""
    return get_backend().default_ld_flags()


__all__ = [
    "Vendor",
    "detect_vendor",
    "get_backend",
    "llc_bytes",
    "flush_buffer_bytes",
    "reset_persisting_l2_cache",
    "arch_flags",
    "default_device_cflags",
    "default_ld_flags",
]
