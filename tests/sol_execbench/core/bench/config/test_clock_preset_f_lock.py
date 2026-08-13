# SPDX-License-Identifier: Apache-2.0
"""`ClockPreset.f_lock_mhz` must never hand back a request as a measurement.

The defect this pins is not hypothetical and it is not loud. `f_lock_mhz` was
``achieved_gpu_clk_mhz or gpu_clk_mhz``, so a part whose achieved clock had
never been measured got stamped with the clock somebody ASKED for. Every
artifact then carries it, `build_manifest`'s guard compares the stamp against
the same table that produced it, and the one automated cross-check that could
notice -- `T_SOL <= T_b` -- gets *easier*: a bound divided by a clock that is
too high, and a T_b measured on silicon that is too slow, are wrong in the same
direction.

On MI355X that is a ~24% error (1650 stamped against ~1330 measured), which is
why these tests exist per part rather than only in the abstract.
"""

from __future__ import annotations

import pytest

from sol_execbench.core.bench.config import get_clock_preset
from sol_execbench.core.bench.config.device_config import ClockPreset


def test_mi355x_refuses_to_supply_an_f_lock_nobody_measured():
    """The whole defect in one assertion."""
    preset = get_clock_preset("AMD Instinct MI355X")
    assert preset is not None, "the entry is kept, not deleted"
    assert preset.gpu_clk_mhz == 1650, "the request is still recorded"
    assert preset.achieved_gpu_clk_mhz is None, "nothing has measured it"
    assert preset.f_lock_mhz is None, (
        "MI355X must not report an F_LOCK. Returning the 1650 request here "
        "makes every bound ~24% too tight, in the direction the T_SOL <= T_b "
        "gate cannot catch."
    )


def test_mi350x_still_reports_its_measured_achieved_clock():
    """The control: the part that HAS a measurement is unaffected."""
    preset = get_clock_preset("AMD Instinct MI350X")
    assert preset.gpu_clk_mhz == 1600
    assert preset.achieved_gpu_clk_mhz == 1300
    assert preset.f_lock_mhz == 1300


@pytest.mark.parametrize("name,expected", [
    ("NVIDIA B200", 1500),
    ("NVIDIA H100", 1410),
    ("NVIDIA A100", 1065),
])
def test_nvidia_parts_keep_requested_equals_achieved(name, expected):
    """`requested_is_achieved` defaults True, so NVIDIA behaviour is unchanged."""
    preset = get_clock_preset(name)
    assert preset.requested_is_achieved is True
    assert preset.f_lock_mhz == expected


def test_an_unmeasured_amd_part_cannot_be_added_by_accident():
    """A new entry that forgets the flag still fails closed if it declares it.

    This is the guard against the next MI3xx entry repeating the mistake: the
    author has to make a positive claim that the request is achieved, and on
    any AMD part measured so far that claim is false.
    """
    unmeasured = ClockPreset(gpu_clk_mhz=1800, dram_clk_mhz=None,
                             requested_is_achieved=False)
    assert unmeasured.f_lock_mhz is None

    measured = ClockPreset(gpu_clk_mhz=1800, dram_clk_mhz=None,
                           achieved_gpu_clk_mhz=1500,
                           requested_is_achieved=False)
    assert measured.f_lock_mhz == 1500, (
        "an explicit measurement always wins; the flag only governs the "
        "fallback"
    )
