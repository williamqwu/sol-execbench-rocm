#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""AMD Instinct part specifications — one source of truth for the whole port.

Why this module exists
----------------------
SOL-ExecBench-ROCm targets more than one CDNA4 part. MI350X and MI355X are the
**same die** (gfx950, 256 CU, 288 GB HBM3E at 8 TB/s, 256 MiB Infinity Cache)
in **different chassis**: MI355X is the liquid-cooled 1400 W part clocked to
2.4 GHz, MI350X the air-cooled 1000 W part clocked to 2.2 GHz.

That distinction is easy to get wrong in the direction that does damage. Every
figure below is therefore tagged with which kind of quantity it is:

* ``ARCHITECTURAL`` — a property of the die. **Shared** between the two parts,
  and provably so: the published peak-FLOPS figures for both parts fall out of
  one MAC/cycle table multiplied by their respective clocks (see the
  cross-check in ``MAC_PER_CYCLE`` below). Copying these across parts is
  correct.
* ``PART`` — a property of the part and its chassis. Peak clock, power cap.
  These differ and must never be shared.
* ``MEASURED`` — determined on silicon, per part, per node. F_LOCK above all.
  Never inferred from the other part, however similar the architecture.
  See ``core/bench/config/device_config.py``.

The rule this encodes: an architectural constant may be shared because it is
derivable and checkable; a measured one may not, because nothing downstream can
detect that it was carried over from different hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MIB = 2**20
GIB = 2**30


# ---------------------------------------------------------------------------
# ARCHITECTURAL — CDNA4 (gfx950). Shared by MI350X and MI355X.
#
# Derivation, from 256 CU x 4 matrix cores = 1024 matrix cores:
#   BF16/FP16 dense: 1024 cores x 512 MAC = 524,288 MAC/cycle
#   Each subsequent step doubles: FP8/INT8 2x, MXFP6/MXFP4 4x.
#   Vector FP32: 16384 stream processors, packed v_pk_fma_f32 -> 32,768.
#   FP64: 16,384.
#
# CROSS-CHECK against AMD's published dense peaks — the same table reproduces
# BOTH parts, which is the evidence that it is architectural and not a
# per-part number in disguise:
#
#   BF16   524288 x 2 x 2.4e9 = 2.52 PFLOPS  vs MI355X spec 2.5   ok
#          524288 x 2 x 2.2e9 = 2.31 PFLOPS  vs MI350X spec 2.3   ok
#   FP8   1048576 x 2 x 2.4e9 = 5.03 PFLOPS  vs MI355X spec 5.0   ok
#         1048576 x 2 x 2.2e9 = 4.61 PFLOPS  vs MI350X spec 4.6   ok
#   MXFP4 2097152 x 2 x 2.4e9 = 10.1 PFLOPS  vs MI355X spec 10.1  ok
#         2097152 x 2 x 2.2e9 =  9.2 PFLOPS  vs MI350X spec 9.2   ok
#   FP32   32768  x 2 x 2.4e9 = 157.3 TFLOPS vs MI355X spec 157.3 ok
#   FP64   16384  x 2 x 2.4e9 =  78.6 TFLOPS vs MI355X spec 78.6  ok
#
# UNRESOLVED (task 03):
#   V1 TF32 matrix support. CDNA3 had it; CDNA4 is reported to have dropped it.
#      Deliberately ABSENT from this table rather than guessed -- a missing key
#      raises, a wrong key computes a plausible bound. Resolve in task 03.
#   V3 The MXFP4 rate above is the DENSE row of AMD's spec sheet. AMD also
#      quotes 10.1 PFLOPS for FP8 *with sparsity*; they are different rows and
#      must not be conflated.
# ---------------------------------------------------------------------------
MAC_PER_CYCLE_CDNA4: dict[str, int] = {
    "fp64_tc": 16384,
    "fp32_sm": 32768,
    "bf16_tc": 524288,
    "fp16_tc": 524288,
    "fp8_tc": 1048576,      # OCP e4m3fn / e5m2, and MXFP8
    "int8_tc": 1048576,
    "mxfp6_tc": 2097152,
    "mxfp4_tc": 2097152,
}

MAC_PER_CYCLE_CDNA3: dict[str, int] = {
    "fp64_tc": 40960,
    "fp32_sm": 40960,
    "bf16_tc": 311296,
    "fp16_tc": 311296,
    "fp8_tc": 622592,       # fnuz variants on CDNA3 -- NOT OCP FP8
    "int8_tc": 622592,
}


@dataclass(frozen=True)
class Part:
    """One AMD Instinct part."""

    name: str
    """Short name, e.g. "MI350X"."""

    torch_device_name: str
    """Substring match against ``torch.cuda.get_device_name()``."""

    gfx: str
    """ISA target. Shared between MI350X and MI355X: both are gfx950."""

    # -- PART properties --
    peak_freq_ghz: float
    """Spec-sheet boost clock. NOT the benchmarking clock -- see F_LOCK."""

    power_cap_w: int
    """Default socket power cap. The reason the two parts clock differently."""

    cooling: str
    """"air" or "liquid". Determines how much of the power cap is usable."""

    # -- ARCHITECTURAL properties --
    compute_units: int
    llc_capacity: int
    dram_capacity: int
    dram_bytes_per_sec: float
    llc_bytes_per_sec: float
    """PLACEHOLDER on CDNA4 until measured -- see V2 in task 03."""

    mac_per_cycle: dict[str, int]
    aliases: dict[str, str] = field(default_factory=dict)

    def peak_flops(self, key: str, freq_ghz: float | None = None) -> float:
        """Peak FLOP/s for a precision at *freq_ghz* (default: spec peak)."""
        hz = (freq_ghz if freq_ghz is not None else self.peak_freq_ghz) * 1e9
        return self.mac_per_cycle[key] * 2 * hz


# SOLAR resolves precision by literal string lookup
# (f"MAC_per_cycle_{precision}_tc") and its dtype map sends
# float4_e2m1fn_x2 -> "nvfp4". Emitting an nvfp4 alias at the MXFP4 rate lets
# SOLAR run unmodified on an AMD config. It is an alias, not an equivalence
# claim: NVFP4 and MXFP4 are different formats (block 16 vs 32, E4M3 vs E8M0
# scales) and task 07 respecs those problems rather than translating them.
_CDNA4_ALIASES = {"nvfp4_tc": "mxfp4_tc"}


PARTS: dict[str, Part] = {
    "MI355X": Part(
        name="MI355X",
        torch_device_name="AMD Instinct MI355X",
        gfx="gfx950",
        peak_freq_ghz=2.4,
        power_cap_w=1400,
        cooling="liquid",
        compute_units=256,
        llc_capacity=256 * MIB,
        dram_capacity=288 * GIB,
        dram_bytes_per_sec=8.0e12,
        llc_bytes_per_sec=17.0e12,
        mac_per_cycle=MAC_PER_CYCLE_CDNA4,
        aliases=_CDNA4_ALIASES,
    ),
    "MI350X": Part(
        name="MI350X",
        torch_device_name="AMD Instinct MI350X",
        gfx="gfx950",
        peak_freq_ghz=2.2,
        power_cap_w=1000,
        cooling="air",
        compute_units=256,
        llc_capacity=256 * MIB,
        dram_capacity=288 * GIB,
        dram_bytes_per_sec=8.0e12,
        llc_bytes_per_sec=17.0e12,
        mac_per_cycle=MAC_PER_CYCLE_CDNA4,
        aliases=_CDNA4_ALIASES,
    ),
    "MI300X": Part(
        name="MI300X",
        torch_device_name="AMD Instinct MI300X",
        gfx="gfx942",
        peak_freq_ghz=2.1,
        power_cap_w=750,
        cooling="air",
        compute_units=304,
        llc_capacity=256 * MIB,
        dram_capacity=192 * GIB,
        dram_bytes_per_sec=5.3e12,
        llc_bytes_per_sec=17.0e12,
        mac_per_cycle=MAC_PER_CYCLE_CDNA3,
        aliases={},
    ),
}


def detect_part(device: int = 0) -> Part:
    """The Part this process is running on, from torch's device name.

    Raises rather than defaulting. A wrong part means wrong peak-FLOPS
    denominators, a wrong clock ceiling and a wrong power budget in every
    artifact that cites them -- all of which stay plausible.
    """
    import torch

    name = torch.cuda.get_device_name(device)
    for part in PARTS.values():
        if part.torch_device_name in name:
            return part
    raise NotImplementedError(
        f"No Part entry for {name!r}. Add one with its spec-sheet peak clock "
        f"and power cap, and MEASURE its F_LOCK (tasks/01) -- do not reuse "
        f"another part's, even on the same die."
    )
