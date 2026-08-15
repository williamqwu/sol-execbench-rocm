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

import os

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


_AMDSMI_INITIALISED = False


def _amdsmi_init_once() -> None:
    """``amdsmi_init()``, at most once per process and never concurrently.

    Two separate hazards, both of which produce a *missing artifact* rather than
    an error anyone can read:

    * **Repetition.** Every telemetry read in this module used to call
      ``amdsmi_init()`` first. At two samples per workload across a sweep that is
      thousands of initialisations of a library the most consequential
      measurement in the project depends on.
    * **Concurrency.** ``amdsmi_init()`` has been observed to **SIGSEGV** when
      several processes initialise the library at the same moment -- which is the
      normal condition here, because the sweep shards 8-way and every shard's
      eval driver brackets its own timing. A SIGSEGV is a C-level fault: it does
      not raise, ``except Exception`` cannot see it, the process dies before
      ``run_guarded`` writes its artifact, and the problem silently re-enters the
      pending set. That is the worst failure shape this repo has -- it looks like
      "not run yet" forever.

    The guard is an exclusive ``flock`` held across the init call only, so at most
    one process in the node is inside ``amdsmi_init()`` at a time. The lock file
    lives in ``SOLEXBENCH_SCRATCH`` when set (the sweep sets it) and in the system
    temp dir otherwise; both are node-local, which is what the hazard is.

    **Honest limitation: this fix is unverified.** The segfault could not be
    reproduced from this session -- reproducing it needs concurrent GPU work on
    cards 1-7, which are running live sweeps and are out of bounds. The
    serialisation is the same remedy applied to the same symptom elsewhere in
    this repo, and it is cheap and side-effect-free, but nobody has watched it
    prevent the crash. Do not record it as confirmed.

    Failure to take the lock is not fatal: an unwritable scratch dir must not stop
    telemetry, so it falls through to an unserialised init and accepts the
    original risk rather than losing the measurement outright.
    """
    global _AMDSMI_INITIALISED
    if _AMDSMI_INITIALISED:
        return
    import amdsmi

    try:
        import fcntl
        import tempfile
        from pathlib import Path

        scratch = Path(os.environ.get("SOLEXBENCH_SCRATCH") or tempfile.gettempdir())
        scratch.mkdir(parents=True, exist_ok=True)
        lock_path = scratch / "solb-amdsmi-init.lock"
        with open(lock_path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                amdsmi.amdsmi_init()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Cannot lock (read-only scratch, exotic filesystem). Serialising was
        # best-effort; losing telemetry entirely is worse than the race.
        amdsmi.amdsmi_init()
    _AMDSMI_INITIALISED = True


def torch_index_of(device) -> int:
    """The torch device index *device* names, whatever spelling it arrived in.

    This function exists because of a bug that cost a whole sweep, and the shape
    of it is worth keeping written down. The original line was::

        idx = 0 if device is None else int(getattr(device, "index", device))

    Correct for ``None``, for an ``int``, and for a ``torch.device`` (whose
    ``.index`` is the integer). Silently wrong for the **string** ``"cuda:0"`` --
    because ``str`` HAS an ``index`` attribute, the ``str.index`` *method*.
    ``getattr`` finds it, the default is never reached, and
    ``int(<built-in method index>)`` raises ``TypeError`` inside a bare
    ``except Exception: return None``. The eval driver passes exactly that
    string (``eval_driver.py:351``, ``_device = "cuda:0"``), so every clock
    sample in the first bracketed T_b sweep came back ``None``, every
    measurement was refused for absent clock evidence, and nothing raised.

    Parsing is therefore explicit and total: every accepted spelling is matched
    deliberately and anything else **raises**. It must never fall back to 0. A
    default of 0 would read *some* card and return a plausible number, which is
    the §8.1 ordering failure in a second costume -- and that one would be
    undetectable rather than merely total.
    """
    if device is None:
        return 0
    if isinstance(device, bool):                    # bool is an int subclass
        raise ValueError(f"not a device: {device!r}")
    if isinstance(device, int):
        return device
    # torch.device and anything else exposing an integer `.index`. Checked by
    # TYPE, before the string branch, precisely because `str` answers
    # hasattr(..., "index") with a method.
    idx = getattr(device, "index", None)
    if isinstance(idx, int) and not isinstance(idx, bool):
        return idx
    if isinstance(device, str):
        s = device.strip()
        if s.isdigit():
            return int(s)
        # "cuda", "cuda:0", "hip:3". A bare device type means "the current
        # device", which is index 0 under the per-worker HIP_VISIBLE_DEVICES
        # pinning every sweep in this repo uses.
        kind, sep, tail = s.partition(":")
        if kind in ("cuda", "hip", "gpu"):
            # No colon at all is a bare device type. A colon with nothing after
            # it ("cuda:") is malformed and falls through to the raise -- it is
            # far more likely to be a truncated f-string than an intent to mean
            # device 0.
            if not sep:
                return 0
            if tail.isdigit():
                return int(tail)
        raise ValueError(
            f"cannot resolve {device!r} to a torch device index; refusing to "
            f"guess, because guessing 0 would read a different card and return "
            f"a perfectly plausible number")
    raise ValueError(f"cannot resolve {device!r} ({type(device).__name__}) to a "
                     f"torch device index")


def _clock_info(device, domain: str) -> int | None:
    # Resolved OUTSIDE the try. A malformed device is a programming error and
    # must raise; only genuine telemetry failure (no amdsmi, no permission, an
    # unsupported domain) degrades to None. Collapsing the two is what made the
    # bug above invisible -- "nobody could read the clock" and "we asked the
    # wrong question" produced the same answer.
    idx = torch_index_of(device)
    try:
        import amdsmi

        _amdsmi_init_once()
        handles = amdsmi.amdsmi_get_processor_handles()
        info = amdsmi.amdsmi_get_clock_info(
            handles[_amdsmi_index(idx)], getattr(amdsmi.AmdSmiClkType, domain)
        )
        clk = info.get("clk")
        return int(clk) if isinstance(clk, (int, float)) else None
    except Exception:
        return None


_AMDSMI_INDEX: dict[int, int] = {}


def _amdsmi_index(torch_index: int) -> int:
    """Map a torch device index to an amdsmi handle index by PCI identity.

    These orderings are NOT the same -- on an 8x MI355X node the observed map
    is torch [0..7] -> amdsmi [3,0,2,1,7,4,6,5]. Indexing handles positionally
    reads a different physical GPU's telemetry, which during clock calibration
    means reporting an idle GPU's frequency as the loaded one's.

    Cached: the PCI topology does not change within a process, and rebuilding it
    per sample means an ``amdsmi_init()`` and a full BDF walk on the hot path of
    every clock bracket.
    """
    if torch_index in _AMDSMI_INDEX:
        return _AMDSMI_INDEX[torch_index]

    import re

    import amdsmi
    import torch

    _amdsmi_init_once()
    by_bus = {}
    for i, h in enumerate(amdsmi.amdsmi_get_processor_handles()):
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(h)
        m = re.match(r"[0-9A-Fa-f]{4}:([0-9A-Fa-f]{2}):", str(bdf))
        if m:
            by_bus[int(m.group(1), 16)] = i
    bus = int(torch.cuda.get_device_properties(torch_index).pci_bus_id)
    idx = by_bus[bus]
    _AMDSMI_INDEX[torch_index] = idx
    return idx


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
