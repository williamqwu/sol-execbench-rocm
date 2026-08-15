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

"""GPU clock locking for stable SOL ExecBench benchmark timing.

Clock locking pins the GPU to a fixed frequency, eliminating the 10-30% latency
variance caused by GPU boost clock fluctuations under thermal pressure.

Entry points:

- ``probe_clock_lock_available()`` — probes whether ``sudo nvidia-smi`` is
  available.  Useful for testing and startup diagnostics.

- ``lock_clocks(device_name)`` — locks GPU and DRAM clocks.  Called once at
  Docker entrypoint or server startup.

- ``unlock_clocks()`` — resets GPU and DRAM clocks.  Best-effort cleanup.

- ``are_clocks_locked()`` — reads ``SOL_EXECBENCH_CLOCKS_LOCKED`` env var to check
  whether clocks were locked at startup.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time

from .config.device_config import get_clock_preset

logger = logging.getLogger(__name__)

# Seconds to wait after locking before verifying actual clock frequencies.
VERIFY_DELAY_S = 3


def _vendor() -> str:
    """Which vendor's clock tooling applies."""
    from sol_execbench.core.bench.device import detect_vendor

    return detect_vendor()


def probe_clock_lock_available() -> bool:
    """Probe whether GPU clock locking is available via ``sudo nvidia-smi``.

    Runs ``sudo -n nvidia-smi -lgc 1`` and immediately resets the lock.  Returns
    ``True`` if the command exits with code 0 (passwordless sudo is configured for
    nvidia-smi), ``False`` otherwise.
    """
    # AMD: probe by reading the perf-level sysfs nodes. Deliberately does NOT
    # try a set-and-reset: `rocm-smi --setperfdeterminism` exits 0 when /sys is
    # read-only, so a set/reset probe would report "available" on exactly the
    # configuration where locking silently does nothing.
    if _vendor() == "amd":
        from sol_execbench.core.bench.device import amd as amd_device

        levels = amd_device.perf_levels()
        return bool(levels) and all(
            not v.startswith("<unreadable") for v in levels.values()
        ) and os.access(next(iter(levels)), os.W_OK)

    try:
        probe = subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lgc", "1"],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    available = probe.returncode == 0
    if available:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
    return available


def lock_clocks(device_name: str) -> bool:
    """Lock GPU and DRAM clocks for the given device.

    Looks up the device in the preset table, then calls ``sudo nvidia-smi -lgc``
    and ``sudo nvidia-smi -lmc``.  GPU and DRAM frequencies can be overridden
    via ``SOL_EXECBENCH_GPU_CLK_MHZ`` and ``SOL_EXECBENCH_DRAM_CLK_MHZ`` env vars.

    Parameters
    ----------
    device_name : str
        GPU device name (e.g. from ``nvidia-smi --query-gpu=name``).

    Returns
    -------
    bool
        True if both GPU and DRAM clocks were locked successfully.
    """
    preset = get_clock_preset(device_name)
    gpu_mhz_str = os.environ.get("SOL_EXECBENCH_GPU_CLK_MHZ")
    dram_mhz_str = os.environ.get("SOL_EXECBENCH_DRAM_CLK_MHZ")

    gpu_mhz = (
        int(gpu_mhz_str) if gpu_mhz_str else (preset.gpu_clk_mhz if preset else None)
    )
    dram_mhz = (
        int(dram_mhz_str) if dram_mhz_str else (preset.dram_clk_mhz if preset else None)
    )

    if gpu_mhz is None:
        logger.warning(
            f"No GPU clock preset for '{device_name}' and SOL_EXECBENCH_GPU_CLK_MHZ not set"
        )
        return False

    # AMD: Instinct parts expose no independent memory-clock lock (no
    # equivalent of `nvidia-smi -lmc`), so a missing DRAM clock is expected
    # rather than fatal. HBM is verified stable at max under load instead --
    # requiring dram_mhz here would make every AMD lock attempt "fail".
    if _vendor() == "amd":
        from sol_execbench.core.bench.device import amd as amd_device

        if not amd_device.lock_clocks(gpu_mhz):
            logger.warning(
                "Failed to enable performance determinism at "
                f"{gpu_mhz} MHz. Note that `rocm-smi --setperfdeterminism` "
                "exits 0 without effect when /sys is read-only, so this is "
                "reported by verifying the perf level, not the exit status."
            )
            return False
        logger.info(f"GPU clocks locked to {gpu_mhz} MHz (perf determinism)")

        logger.info(f"Waiting {VERIFY_DELAY_S}s for clocks to stabilize...")
        time.sleep(VERIFY_DELAY_S)
        if not verify_clocks(gpu_mhz, dram_mhz):
            logger.warning("Clock verification failed after locking — unlocking")
            unlock_clocks()
            return False
        return True

    if dram_mhz is None:
        logger.warning(
            f"No DRAM clock preset for '{device_name}' and SOL_EXECBENCH_DRAM_CLK_MHZ not set"
        )
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lgc", str(gpu_mhz)],
            check=True,
            capture_output=True,
        )
        logger.info(f"GPU clocks locked to {gpu_mhz} MHz")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Failed to lock GPU clocks: {e}")
        return False

    try:
        subprocess.run(
            ["sudo", "-n", "nvidia-smi", "-lmc", str(dram_mhz)],
            check=True,
            capture_output=True,
        )
        logger.info(f"DRAM clocks locked to {dram_mhz} MHz")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Failed to lock DRAM clocks: {e}")
        # Unlock GPU clocks since we couldn't lock DRAM
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
        return False

    # Verify clocks are actually held after a brief stabilization delay
    logger.info(f"Waiting {VERIFY_DELAY_S}s for clocks to stabilize...")
    time.sleep(VERIFY_DELAY_S)
    if not verify_clocks(gpu_mhz, dram_mhz):
        logger.warning("Clock verification failed after locking — unlocking")
        unlock_clocks()
        return False

    return True


def verify_clocks(
    expected_gpu_mhz: int, expected_dram_mhz: int, tolerance_mhz: int = 50
) -> bool:
    """Verify current GPU/DRAM clocks match expected frequencies via nvidia-smi.

    Queries all GPUs and checks that current frequencies are within *tolerance_mhz*
    of the expected values.

    Parameters
    ----------
    expected_gpu_mhz : int
        Expected GPU core clock in MHz.
    expected_dram_mhz : int
        Expected DRAM (memory) clock in MHz.
    tolerance_mhz : int
        Allowed deviation in MHz (default 50).

    Returns
    -------
    bool
        True if all GPUs report clocks within tolerance.
    """
    # AMD: read clocks from amdsmi. Only the GFX clock is compared against an
    # expectation -- HBM has no lock to verify, so its value is logged for the
    # record (task 02 asks for it to be recorded, not enforced).
    #
    # NOTE: an idle GPU reports a clock *below* any cap, so this check passes
    # trivially when nothing is running. Verification is only meaningful under
    # sustained load; see `clock_calibrate.py verify --under-load`.
    if _vendor() == "amd":
        import torch

        from sol_execbench.core.bench.device import amd as amd_device

        ok = True
        for i in range(torch.cuda.device_count()):
            gfx = amd_device.current_clock_mhz(i)
            mem = amd_device.memory_clock_mhz(i)
            if gfx is None:
                logger.warning(f"GPU {i}: could not read GFX clock")
                return False
            logger.info(f"GPU {i}: GFX {gfx} MHz, HBM {mem} MHz (HBM not locked)")
            if gfx - expected_gpu_mhz > tolerance_mhz:
                logger.warning(
                    f"GPU {i}: GFX clock {gfx} MHz EXCEEDS the {expected_gpu_mhz} "
                    f"MHz cap by more than {tolerance_mhz} MHz — the lock is "
                    f"not holding"
                )
                ok = False
        return ok

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"nvidia-smi clock query failed: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        logger.warning("nvidia-smi not found — cannot verify clocks")
        return False

    lines = result.stdout.strip().splitlines()
    if not lines:
        logger.warning("nvidia-smi returned no GPU data")
        return False

    for i, line in enumerate(lines):
        parts = line.split(",")
        if len(parts) < 2:
            logger.warning(f"GPU {i}: unexpected nvidia-smi output: {line!r}")
            return False
        try:
            actual_gpu = int(parts[0].strip())
            actual_dram = int(parts[1].strip())
        except ValueError:
            logger.warning(f"GPU {i}: could not parse clock values: {line!r}")
            return False

        if abs(actual_gpu - expected_gpu_mhz) > tolerance_mhz:
            logger.warning(
                f"GPU {i}: GPU clock mismatch — expected {expected_gpu_mhz} MHz, got {actual_gpu} MHz"
            )
            return False
        if abs(actual_dram - expected_dram_mhz) > tolerance_mhz:
            logger.warning(
                f"GPU {i}: DRAM clock mismatch — expected {expected_dram_mhz} MHz, got {actual_dram} MHz"
            )
            return False

        logger.info(
            f"GPU {i}: clocks verified — GPU {actual_gpu} MHz, DRAM {actual_dram} MHz"
        )

    return True


def unlock_clocks() -> None:
    """Reset GPU and DRAM clocks.  Best-effort — errors are logged but not raised."""
    # AMD: one reset covers both domains; there is no separate DRAM lock.
    if _vendor() == "amd":
        from sol_execbench.core.bench.device import amd as amd_device

        try:
            if amd_device.unlock_clocks():
                logger.info("GPU clocks reset to auto")
            else:
                logger.warning("Failed to reset GPU clocks")
        except Exception as e:
            logger.warning(f"Failed to reset GPU clocks: {e}")
        return

    try:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
        logger.info("GPU clocks unlocked")
    except Exception as e:
        logger.warning(f"Failed to unlock GPU clocks: {e}")

    try:
        subprocess.run(["sudo", "-n", "nvidia-smi", "-rmc"], capture_output=True)
        logger.info("DRAM clocks unlocked")
    except Exception as e:
        logger.warning(f"Failed to unlock DRAM clocks: {e}")


def are_clocks_locked() -> bool:
    """Check whether clocks were locked at startup.

    Reads the ``SOL_EXECBENCH_CLOCKS_LOCKED`` environment variable set by the Docker
    entrypoint or server startup.
    """
    return os.environ.get("SOL_EXECBENCH_CLOCKS_LOCKED", "0") == "1"
