#!/usr/bin/env python3
"""Generate a SOLAR arch YAML for AMD Instinct parts at an arbitrary locked clock.

Why a generator and not a hand-written YAML:

SOLAR's arch config mixes two kinds of quantity, and they scale differently
when you change the locked clock:

  * MAC_per_cycle_*   -- ARCHITECTURAL. Fixed by the matrix-core datapath
                         width. Frequency-INDEPENDENT. Do not touch when
                         F_LOCK changes.
  * *_byte_per_cycle  -- DERIVED. Real bandwidth is fixed in bytes/second
                         (HBM runs on its own clock domain), so the per-cycle
                         figure MUST be rescaled as bytes_per_sec / F_LOCK
                         whenever F_LOCK changes.

Hand-editing freq_GHz without rescaling the bandwidth terms silently changes
the machine's roofline balance point and corrupts every T_SOL. Hence: generate.

Consequence worth exploiting: because MAC_per_cycle is frequency-independent
and bandwidth is frequency-independent in absolute terms, the SOL bound in
CYCLES is invariant to F_LOCK. You can run SOLAR over all 235 problems today,
cache T_SOL in cycles, and convert to milliseconds with a single scalar
division once F_LOCK is measured on real silicon.

Usage:
    python gen_arch_yaml.py --part MI355X --freq-ghz 1.9 -o MI355X.yaml
    python gen_arch_yaml.py --part MI355X --freq-ghz 2.4 -o MI355X_peak.yaml
"""

import argparse

# ---------------------------------------------------------------------------
# Architectural constants (frequency-INDEPENDENT).
#
# Derivation for MI355X (CDNA4, gfx950), from AMD's published peak figures:
#   256 CU x 4 matrix cores = 1024 matrix cores.
#   BF16/FP16 dense: 2.5 PFLOPS @ 2.4 GHz
#     -> 2.5e15 / 2 (FLOP per MAC) / 2.4e9 = 520,833 MAC/cycle
#     -> rounds to the clean datapath number 1024 cores x 512 = 524,288
#     -> check: 524288 x 2 x 2.4e9 = 2.52 PFLOPS. Matches spec.
#   Each subsequent precision step doubles throughput:
#     FP8 (OCP e4m3/e5m2, MXFP8), INT8 : 2x BF16 -> 5.0 PFLOPS / POPS
#     MXFP4, MXFP6                     : 4x BF16 -> 10.1 PFLOPS
#   Vector FP32: 16384 stream processors x 1 FMA, packed (v_pk_fma_f32) x2
#     -> 32,768 MAC/cycle -> 157.3 TFLOPS @ 2.4 GHz. Matches spec.
#   FP64: 78.6 TFLOPS @ 2.4 GHz -> 16,384 MAC/cycle.
#
# !! VERIFY ON SILICON / IN THE CDNA4 ISA GUIDE BEFORE TRUSTING FOR SCORING:
#   V1  TF32 matrix support. CDNA3 had it (~653 TFLOPS on MI300X); CDNA4 is
#       reported to have dropped it in favour of FP16/BF16/FP8/FP4. If absent,
#       decide the fallback policy for any problem SOLAR tags as tf32
#       (recommendation: fall back to bf16 rate, and document it).
#   V2  Infinity Cache (LLC) bandwidth. The 256 MB capacity is published; the
#       bandwidth figure below is a PLACEHOLDER and drives the Orojenesis-style
#       buffer-aware memory bound for fused L2-category problems. Measure it.
#   V3  Whether the 4x MXFP4 rate is achievable dense or only with sparsity.
#       AMD quotes 10.1 PFLOPS for MXFP4/MXFP6 dense and 10.1 for FP8 *with*
#       sparsity -- do not conflate the two rows of the spec sheet.
# ---------------------------------------------------------------------------

PARTS = {
    "MI355X": {
        "name": "MI355X",
        "peak_freq_ghz": 2.4,
        # capacity in bytes
        "LLC_capacity": 256 * 2**20,          # 256 MB Infinity Cache
        "DRAM_capacity": 288 * 2**30,         # 288 GB HBM3E
        # absolute bandwidth in bytes/second (frequency-independent)
        "DRAM_bytes_per_sec": 8.0e12,         # 8 TB/s, published
        "LLC_bytes_per_sec": 17.0e12,         # PLACEHOLDER -- see V2
        # MAC/cycle, architectural
        "mac_per_cycle": {
            "fp64_tc": 16384,
            "fp32_sm": 32768,                 # vector, packed FP32
            "bf16_tc": 524288,
            "fp16_tc": 524288,
            "fp8_tc": 1048576,                # OCP e4m3/e5m2 + MXFP8
            "int8_tc": 1048576,
            "mxfp6_tc": 2097152,
            "mxfp4_tc": 2097152,
        },
        # SOLAR's perf model resolves precision via the literal key
        # f"MAC_per_cycle_{precision}_tc", and its dtype map sends
        # float4_e2m1fn_x2 -> "nvfp4". Emitting an nvfp4 alias pointing at the
        # MXFP4 rate lets the existing SOLAR run unmodified on AMD configs.
        # Long-term: patch the dtype map and drop the alias.
        "aliases": {"nvfp4_tc": "mxfp4_tc"},
    },
    "MI300X": {
        "name": "MI300X",
        "peak_freq_ghz": 2.1,
        "LLC_capacity": 256 * 2**20,
        "DRAM_capacity": 192 * 2**30,
        "DRAM_bytes_per_sec": 5.3e12,
        "LLC_bytes_per_sec": 17.0e12,
        "mac_per_cycle": {
            "fp64_tc": 40960,
            "fp32_sm": 40960,
            "bf16_tc": 311296,
            "fp16_tc": 311296,
            "fp8_tc": 622592,                 # fnuz variants on CDNA3
            "int8_tc": 622592,
        },
        "aliases": {},
    },
}


def gen(part_key: str, freq_ghz: float) -> str:
    p = PARTS[part_key]
    hz = freq_ghz * 1e9
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        f"# SOLAR arch config for AMD Instinct {p['name']}.",
        "# GENERATED by gen_arch_yaml.py -- do not hand-edit freq_GHz.",
        "# Re-run the generator to change the locked clock; bandwidth terms",
        "# are rescaled automatically.",
        "#",
        f"# Locked clock: {freq_ghz} GHz  ({freq_ghz / p['peak_freq_ghz'] * 100:.1f}% of "
        f"{p['peak_freq_ghz']} GHz peak)",
        "",
        f'name: "{p["name"]}"',
        f"freq_GHz: {freq_ghz}",
        "",
        "# --- capacity (frequency-independent) ---",
        f"SRAM_capacity: {p['LLC_capacity']}  # {p['LLC_capacity'] // 2**20} MB last-level (Infinity Cache)",
        f"DRAM_capacity: {p['DRAM_capacity']}  # {p['DRAM_capacity'] // 2**30} GB HBM",
        "",
        "# --- bandwidth (DERIVED: bytes_per_sec / freq) ---",
        f"SRAM_byte_per_cycle: {p['LLC_bytes_per_sec'] / hz:.1f}  "
        f"# {p['LLC_bytes_per_sec'] / 1e12:.1f} TB/s @ {freq_ghz} GHz  [PLACEHOLDER - verify]",
        f"DRAM_byte_per_cycle: {p['DRAM_bytes_per_sec'] / hz:.1f}  "
        f"# {p['DRAM_bytes_per_sec'] / 1e12:.1f} TB/s @ {freq_ghz} GHz",
        "",
        "# --- compute (ARCHITECTURAL: frequency-independent, do not rescale) ---",
    ]
    for k, v in p["mac_per_cycle"].items():
        pflops = v * 2 * hz / 1e15
        unit = "POPS" if "int" in k else "PFLOPS"
        lines.append(f"MAC_per_cycle_{k}: {v}  # @ {freq_ghz} GHz: {pflops:.3f} {unit}")
    for alias, target in p["aliases"].items():
        v = p["mac_per_cycle"][target]
        lines.append(
            f"MAC_per_cycle_{alias}: {v}  # alias -> {target}; see note in generator"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="MI355X", choices=sorted(PARTS))
    ap.add_argument("--freq-ghz", type=float, required=True)
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    out = gen(a.part, a.freq_ghz)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out)
        print(f"wrote {a.out}")
    else:
        print(out, end="")
