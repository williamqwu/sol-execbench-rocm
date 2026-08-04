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
    "AMD Instinct MI355X": ClockPreset(
        gpu_clk_mhz=1650, dram_clk_mhz=None, achieved_gpu_clk_mhz=1640
    ),
    #
    # MI350X. MEASURED on gbt350-odcdh1-a08-1 (tasks/01, 2026-08-03). Same
    # CDNA4 die and the same gfx950 target as the MI355X above; a different
    # part, and emphatically NOT the same clock. Copying 1650 down from the
    # line above would have been the same class of error as copying a B200
    # constant into an AMD artifact.
    #
    # How different: MI350X is the air-cooled 1000 W part (MI355X is
    # liquid-cooled at 1400 W) with a 2200 MHz ceiling (MI355X: 2400). Its
    # sustained UNLOCKED floors under saturating BF16 GEMM were 1390 / 1367 /
    # 1335 MHz on GPUs 0 / 1 / 2 -- versus 1725-1757 on MI355X.
    #
    # The request/achieved split is the part of this that does not transfer at
    # all. `--setperfdeterminism 1600` holds 1303 MHz on GPU 0 (min 1296 over
    # 20 samples) at 885 W, i.e. ~110 W below the cap, so the lock binds and
    # not the power limit -- which is what makes it reproducible. Requesting
    # more stops helping: at 1900 and at 2200 the part pins to the 1000 W cap
    # and lands on the same ~1400 MHz, at the mercy of ambient temperature.
    #
    # 1600 was chosen over 1700 (which gives 1380 MHz) for exactly that
    # margin: 1700 sits at 947 W on two of three GPUs sampled, and a setting
    # that is one warm afternoon away from becoming power-bound is not a lock.
    #
    # Per-GPU achieved at this setting, all eight measured:
    #   1303 1295 1264 1307 1279 1296 1285 1242  (median MHz, spread 65)
    # The spread is why authoritative timing is pinned to one GPU and why every
    # timing artifact records which GPU produced it.
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
