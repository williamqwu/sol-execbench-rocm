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

"""AMD / ROCm backend for the vendor device layer."""

from __future__ import annotations

VENDOR = "amd"

MIB = 1024 * 1024

# Last-level cache per architecture, in bytes.
#
# EXPLICIT TABLE, DELIBERATELY NOT DERIVED FROM DEVICE PROPERTIES.
# On MI355X, torch reports `L2_cache_size == 4 MiB` -- the per-XCD L2, which is
# real but is not the structure a benchmark has to defeat to run cold. The
# cache that actually holds a working set is the 256 MiB Infinity Cache (MALL).
# Sizing the flush buffer at `2 x L2_cache_size` therefore yields 8 MiB, 64x
# too small: every "cold cache" iteration would in fact run warm out of
# Infinity Cache, and memory-bound kernels would look far faster than they are.
# Nothing downstream could detect that -- the numbers stay plausible.
LLC_BYTES: dict[str, int] = {
    "gfx950": 256 * MIB,   # MI355X / MI350X (CDNA4)
}

# Multiplier applied to LLC capacity to size the eviction buffer.
FLUSH_MULTIPLIER = 2


def _arch(device=None) -> str:
    """Base gfx target for *device*, without feature flags.

    ``gcnArchName`` is reported as e.g. ``gfx950:sramecc+:xnack-``; the feature
    suffixes are irrelevant to cache geometry.
    """
    import torch

    name = torch.cuda.get_device_properties(device).gcnArchName
    return name.split(":", 1)[0]


def llc_bytes(device=None) -> int:
    """Last-level cache bytes from the explicit table.

    Raises rather than falling back to device properties: a silently wrong
    flush size produces measurements that look fine and are not.
    """
    arch = _arch(device)
    try:
        return LLC_BYTES[arch]
    except KeyError:
        raise NotImplementedError(
            f"No LLC_BYTES entry for {arch!r}. Add one measured for that part "
            f"-- do NOT fall back to torch's L2_cache_size, which reports the "
            f"per-XCD L2 and would undersize the cache flush by ~64x, making "
            f"every memory-bound kernel appear faster than it is."
        ) from None


def flush_buffer_bytes(device=None) -> int:
    return llc_bytes(device) * FLUSH_MULTIPLIER


def reset_persisting_l2_cache(device=None) -> None:
    """No-op.

    CDNA has no L2-persistence API: there is no equivalent of
    ``cudaCtxResetPersistingL2Cache`` because there is no way to mark lines
    persisting in the first place. The cache flush in ``timing.py`` is what
    provides cold-cache semantics here.
    """
    return None


PERF_LEVEL_GLOB = "/sys/class/drm/card*/device/power_dpm_force_performance_level"


def perf_levels() -> dict[str, str]:
    """Current ``power_dpm_force_performance_level`` for every card."""
    import glob
    from pathlib import Path

    out = {}
    for f in sorted(glob.glob(PERF_LEVEL_GLOB)):
        try:
            out[f] = Path(f).read_text().strip()
        except OSError as e:
            out[f] = f"<unreadable: {e}>"
    return out


def lock_clocks(gpu_mhz: int, gpu: int | None = None) -> bool:
    """Pin the GFX clock via AMD's performance-determinism mode.

    ``rocm-smi --setperfdeterminism <mhz>`` caps the soft maximum clock so that
    power-management events cannot push the attainable frequency around. There
    is deliberately no DRAM-clock argument: Instinct parts do not expose
    independent memory-clock locking the way ``nvidia-smi -lmc`` does. The
    memory clock is *verified* stable instead -- see ``memory_clock_mhz``.

    Returns True only if a card actually reports ``perf_determinism``.
    ``rocm-smi`` exits 0 having done nothing when /sys is read-only (as it is
    in a stock container), and an unverified lock means every subsequent timing
    is taken at an unknown clock while the artifacts claim otherwise.
    """
    import subprocess

    cmd = ["rocm-smi", "--setperfdeterminism", str(gpu_mhz)]
    if gpu is not None:
        cmd += ["-d", str(gpu)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False

    after = perf_levels()
    locked = [f for f in after if after[f] == "perf_determinism"]
    if not locked:
        return False
    # A partial lock is the dangerous case: some GPUs pinned, others boosting,
    # and nothing in the artifacts to tell them apart.
    if gpu is None and len(locked) != len(after):
        return False
    return True


def unlock_clocks(gpu: int | None = None) -> bool:
    """Reset clocks to their default (``auto``) governor."""
    import subprocess

    cmd = ["rocm-smi", "-r"]
    if gpu is not None:
        cmd += ["-d", str(gpu)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def _clock_info(device, domain: str) -> int | None:
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        idx = 0 if device is None else int(getattr(device, "index", device))
        handles = amdsmi.amdsmi_get_processor_handles()
        info = amdsmi.amdsmi_get_clock_info(
            handles[_amdsmi_index(idx)], getattr(amdsmi.AmdSmiClkType, domain)
        )
        clk = info.get("clk")
        return int(clk) if isinstance(clk, (int, float)) else None
    except Exception:
        return None


def _amdsmi_index(torch_index: int) -> int:
    """Map a torch device index to an amdsmi handle index by PCI identity.

    These orderings are NOT the same -- on an 8x MI355X node the observed map
    is torch [0..7] -> amdsmi [3,0,2,1,7,4,6,5]. Indexing handles positionally
    reads a different physical GPU's telemetry, which during clock calibration
    means reporting an idle GPU's frequency as the loaded one's.
    """
    import re

    import amdsmi
    import torch

    amdsmi.amdsmi_init()
    by_bus = {}
    for i, h in enumerate(amdsmi.amdsmi_get_processor_handles()):
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(h)
        m = re.match(r"[0-9A-Fa-f]{4}:([0-9A-Fa-f]{2}):", str(bdf))
        if m:
            by_bus[int(m.group(1), 16)] = i
    bus = int(torch.cuda.get_device_properties(torch_index).pci_bus_id)
    return by_bus[bus]


def current_clock_mhz(device=None) -> int | None:
    """Current GFX clock, in MHz."""
    return _clock_info(device, "GFX")


def memory_clock_mhz(device=None) -> int | None:
    """Current memory clock, in MHz.

    Recorded rather than locked: there is no Instinct equivalent of
    ``nvidia-smi -lmc``. Task 02 verifies it sits at max under load instead.
    """
    return _clock_info(device, "MEM")


def arch_flags(hardware=None) -> list[str]:
    """Offload-arch flags for the target hardware."""
    from sol_execbench.driver.problem_packager import gencode_flags_for_hardware

    return gencode_flags_for_hardware(hardware)


def default_device_cflags() -> list[str]:
    # hipcc spells fast-math the clang way; `--use_fast_math` is nvcc-only.
    return ["-O3", "-ffast-math"]


def default_ld_flags() -> list[str]:
    return ["-lamdhip64"]
