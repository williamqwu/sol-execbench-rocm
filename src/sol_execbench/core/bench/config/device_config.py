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
    """The clock to REQUEST. On NVIDIA this is also what you get."""

    dram_clk_mhz: Optional[int]

    # AMD: what the part actually HOLDS when the request above is applied.
    #
    # On NVIDIA, `nvidia-smi -lgc 1500` pins the clock at 1500 and the two
    # numbers are the same, so upstream never needed to distinguish them. On
    # MI350X they are not the same and not close: `rocm-smi
    # --setperfdeterminism 1600` yields a rock-steady 1303 MHz under sustained
    # load -- about 81% of the request -- while drawing 885 W of a 1000 W cap,
    # so it is the determinism mode setting the operating point, not power.
    #
    # Both numbers are load-bearing and neither substitutes for the other:
    #   gpu_clk_mhz          is what `lock_clocks` must pass to rocm-smi.
    #   achieved_gpu_clk_mhz is F_LOCK -- the frequency every T_SOL and T_b is
    #                        expressed at. Using the requested value here
    #                        would overstate the clock by ~23% and make every
    #                        analytic bound wrong in the direction that looks
    #                        entirely plausible.
    #
    # None means "requested == achieved" (the NVIDIA case).
    achieved_gpu_clk_mhz: Optional[int] = None

    @property
    def f_lock_mhz(self) -> int:
        """The frequency measurements are actually taken at."""
        return self.achieved_gpu_clk_mhz or self.gpu_clk_mhz


CLOCK_LOCK_PRESETS: dict[str, ClockPreset] = {
    "NVIDIA B200": ClockPreset(gpu_clk_mhz=1500, dram_clk_mhz=3996),
    "NVIDIA H100": ClockPreset(gpu_clk_mhz=1410, dram_clk_mhz=1593),
    "NVIDIA A100": ClockPreset(gpu_clk_mhz=1065, dram_clk_mhz=1215),
    # AMD. MEASURED on mia1-p02-g10, not derived from any NVIDIA ratio and not
    # taken from a spec sheet:
    #
    #   sustained floors under saturating BF16 GEMM (p5 of the final 5 min of
    #   a 15 min run) were 1725 / 1734 / 1751 MHz on GPUs 0 / 1 / 2, and
    #   1724 MHz on GPU 0 with all seven siblings loaded. 1650 is the round
    #   number >=50 MHz below the lowest of those, so the cap sits under the
    #   floor even on the worst GPU in the worst node condition.
    #
    #   Verified at 1644 MHz median (min 1642) under sustained load on GPU 0,
    #   drawing 1269 W of a 1400 W cap -- so the setting binds, not the power
    #   limit, which is what makes it reproducible. Timing CV 0.0041.
    #
    # achieved_gpu_clk_mhz is 1640: a round number 0.24% below GPU 0's measured
    # median and below its measured minimum, which makes every T_SOL marginally
    # conservative rather than marginally optimistic.
    #
    # **This is a GPU-0 number, and only GPU 0 alone on an idle node.** At this
    # same setting the eight GPUs hold wildly different clocks -- 1644, 1643,
    # 1318, 1341, 1370, 1357, 1352, 1327 -- and not because of power (the slow
    # ones draw 950-995 W of 1400) nor because of weak silicon (unlocked, GPU 2
    # sustains 1756 MHz, the fastest on the node).
    #
    # Worse, it is not stable even for GPU 0: re-measured after a clean reset it
    # holds 1647 MHz alone but only 1394 MHz while its siblings are loaded, at
    # this same setting. The value below is therefore a claim about one GPU in
    # one node condition, and `provenance.assert_clock_lock()` exists because
    # nothing else in the pipeline could tell you the condition was not met.
    # See STATE.md D27. That is why authoritative timing is pinned to GPU 0, why
    # it requires an idle node, and why every timing artifact records its GPU.
    #
    # For scale: the B200 ratio (1500/1970 ~ 76%) would imply ~1830 MHz here,
    # which is ABOVE the measured floor and would throttle continuously. The
    # MI355X derate is milder at 1650/2400 ~ 69%.
    # MEASURED on mia1-p02-g10. Setpoint 1660, achieved 1655 MHz, on **GPU 1**.
    #
    # 1660 rather than 1650 because 1650 is bistable on this node: at that setpoint
    # GPU 0 held 1644 MHz in one measurement and 1397 in another. 1660 reproduces.
    #
    # F_LOCK 1650 is a round number below the 1655 measured median (min 1652), so
    # every T_SOL is marginally conservative rather than marginally optimistic.
    #
    # **Only ONE GPU on this node holds a determinism setpoint, and it is GPU 1.**
    # With all eight loaded at setpoint 1660:
    #
    #   torch 1 -> 1655 MHz (0.997)  1276 W   <- pinned
    #   torch 0 -> 1419 MHz (0.855)   989 W
    #   torch 2 -> 1324 .. torch 7 -> 1341    945-983 W
    #
    # The six slow cards are not power-limited; they draw ~950-980 W of a 1400 W
    # cap. Three ways of parallelizing authoritative timing were measured and all
    # three fail, which is recorded in STATE.md D29 because the negative result is
    # the useful part:
    #
    #   * Per-GPU setpoints equalizing the SATURATED clock (1471-1490 MHz across
    #     seven cards, 1.28%) do not pin it. Determinism caps the soft max at the
    #     setpoint, so a card set to 1910 to reach 1480 under load runs up to
    #     ~1890 on short or bursty kernels. Rejected on measurement: the clock
    #     monitor read 1888 MHz against an F_LOCK of 1480, +27.6%.
    #   * GPU 0 + GPU 1 at a common pinned setpoint: GPU 0 destabilizes under
    #     concurrent load -- 1654 alone, 1414 with one sibling, 1419 with seven.
    #   * A node-wide setpoint: only GPU 1 obeys it, per the table above.
    #
    # So authoritative timing is serial here, and the parallelism goes where the
    # clock does not have to be pinned: the agent sweep, where a kernel's own
    # timings are feedback for the agent and every score is re-measured on GPU 1.
    "AMD Instinct MI355X": ClockPreset(
        gpu_clk_mhz=1660, dram_clk_mhz=None, achieved_gpu_clk_mhz=1650
    ),
    #
"AMD Instinct MI350X": ClockPreset(
        gpu_clk_mhz=1600, dram_clk_mhz=None, achieved_gpu_clk_mhz=1300
    ),
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
