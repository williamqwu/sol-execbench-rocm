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

"""Device-specific configuration for SOL ExecBench benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClockPreset:
    """GPU and DRAM clock frequencies for stable benchmarking.

    ``dram_clk_mhz`` is Optional because Instinct parts do not expose an
    independent memory-clock lock (no equivalent of ``nvidia-smi -lmc``). None
    means "this vendor cannot lock DRAM"; the AMD path verifies HBM is stable
    at max under load and records it instead.
    """

    gpu_clk_mhz: int
    dram_clk_mhz: Optional[int]


CLOCK_LOCK_PRESETS: dict[str, ClockPreset] = {
    "NVIDIA B200": ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996),
    "NVIDIA H100": ClockPreset(gpu_clk_mhz=1410, dram_clk_mhz=1593),
    "NVIDIA A100": ClockPreset(gpu_clk_mhz=1065, dram_clk_mhz=1215),
    # AMD. MEASURED on mia1-p02-g10, not derived from any NVIDIA ratio and not
    # taken from a spec sheet:
    #
    #   sustained floors under saturating BF16 GEMM (p5 of the final 5 min of
    #   a 15 min run) were 1725 / 1734 / 1757 MHz on GPUs 0 / 1 / 2, and
    #   1728 MHz on GPU 0 with all seven siblings loaded. 1650 is the round
    #   number >=50 MHz below the lowest of those, so the cap sits under the
    #   floor even on the worst GPU in the worst node condition.
    #
    #   Verified at 1648 MHz median under sustained load; timing CV 0.0015.
    #   See tasks/01 and STATE.md. Raising this later invalidates every
    #   measurement taken at 1650.
    #
    # For scale: the B200 ratio (1500/1970 ~ 76%) would imply ~1830 MHz here,
    # which is ABOVE the measured floor and would throttle continuously. The
    # MI355X derate is milder at 1650/2400 ~ 69%.
    "AMD Instinct MI355X": ClockPreset(gpu_clk_mhz=1650, dram_clk_mhz=None),
    #
    # NO "AMD Instinct MI350X" ENTRY, DELIBERATELY.
    #
    # MI350X is the same CDNA4 die and the same gfx950 target, so it shares the
    # build path and the 256 MiB LLC sizing. It does NOT share this clock. It
    # is the air-cooled part with a materially lower power budget, and the
    # 1650 MHz above was derived from floors measured on a 1400 W liquid-cooled
    # MI355X that sat pinned at ~1400 W / 59-63 C throughout. A part with less
    # power and less cooling headroom will settle at a different -- almost
    # certainly lower -- sustained floor.
    #
    # Copying 1650 here because "it's the same architecture" is the same class
    # of error as copying a B200 constant into an AMD artifact: the number
    # would be plausible, nothing downstream could detect it, and every T_SOL
    # and T_b derived from it would be silently wrong.
    #
    # Re-run tasks/01 on the MI350X node. Absent an entry, `lock_clocks` logs
    # "No GPU clock preset" and returns False unless SOL_EXECBENCH_GPU_CLK_MHZ
    # is set explicitly -- a loud stop, which is the intended behaviour.
}


def get_clock_preset(device_name: str) -> Optional[ClockPreset]:
    """Get the clock preset for a given GPU device name.

    Returns None if the device is not in the preset table.

    Parameters
    ----------
    device_name : str
        The GPU device name string (e.g., from torch.cuda.get_device_name()).

    Returns
    -------
    Optional[ClockPreset]
        Clock preset with GPU and DRAM frequencies, or None if not in presets.
    """
    for key, preset in CLOCK_LOCK_PRESETS.items():
        if key in device_name:
            return preset
    return None
