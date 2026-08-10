#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manifest v1.1: two corrections to T_SOL, neither of them a re-measurement.

    python scripts/rebuild_manifest_v11.py

v1 is frozen and stays frozen. This writes `artifacts/09/manifest-v1.1.json`
beside it, changing exactly two things and recording both in the file.

**D35 -- one F_LOCK was not enough.** `T_SOL_ms = t_sol_cycles / F_LOCK`, with
F_LOCK = 1300 MHz measured under a dense bf16 matrix-core load. Every
matrix-core path does hold that clock (bf16 1296, fp16 1299, fp8 1314) but the
fp32 vector path sustains 1441 MHz, because it draws far less power and
`--setperfdeterminism` caps the ceiling rather than pinning the clock. So 836
compute-bound fp32 workloads were divided by a frequency 10.8% below the one
their arithmetic actually runs at, and their bounds are 10.8% too large.

The cycle counts do not move. They are architectural, which is the property
`sol_bounds.py` was built around, and it is what makes this a division rather
than a re-derivation. Memory-bound bounds do not move either, and that is
algebra rather than luck: the traffic tier computes `cycles = bytes /
DRAM_byte_per_cycle` and then `ms = cycles / freq`, and `DRAM_byte_per_cycle`
is bytes-per-second over freq, so the frequency cancels.

**D18 -- a paged cache was priced at its allocation.** Six FlashInfer problems
declare a KV cache shaped `[num_pages, page_size, head_dim]` and, separately,
an axis `num_kv_indices` saying how many of those pages the workload gathers.
The declared-traffic tier multiplied out `num_pages`. On the first workload of
018 that is 989,669 pages against 8 actually touched, and the bound it produces
-- 185,274 cycles, 1.140 GB at 8 TB/s -- is the same for every workload of the
problem, because the allocation is the same and swamps everything that is not.

The correction is not a special case for those two problems: a tensor whose
shape carries an allocation axis, where the problem separately declares how
many rows of it are indexed, is gathered and not streamed. It is applied to all
six problems that declare that pair, and the recomputed traffic bound is
recombined with SOLAR's the same way v1 combined them -- `max` of the two, then
the same `T_SOL <= T_b` gate, which is what rejects a "bound" that is not one.

What this does NOT do: it does not touch the five problems whose bounds are
beaten by more than either correction explains (L2__045, L1__006, L1__054,
L1__005, L1__057). Those are undiagnosed, they stay marked, and inventing a
number for them here would be the exact failure this file exists to undo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402
from sol_cross_checks import DTYPE_BYTES, resolved_axes  # noqa: E402

V1 = ROOT / "artifacts" / "09" / "manifest-v1.json"
T_SOL = ROOT / "artifacts" / "03" / "t_sol.json"
CLOCKS = ROOT / "artifacts" / "01" / "f_lock_by_datapath.json"
DATA = ROOT / "data" / "SOL-ExecBench" / "benchmark"
ARCH = ROOT / "SOLAR" / "configs" / "arch" / "MI350X.yaml"
OUT = ROOT / "artifacts" / "09" / "manifest-v1.1.json"

#: Task 01's measured F_LOCK. Still the floor for every datapath; see clock_for.
F_LOCK_MHZ = 1300.0

#: `MAC_per_cycle` -> the datapath name whose clock applies. Keyed by the rate
#: because that is what is recoverable from the artifact: `macs /
#: t_sol_cycles_exact` is the rate SOLAR divided by, exactly, and no new field
#: has to be trusted or back-filled. bf16 and fp16 share a rate and a clock.
RATE_TO_DATAPATH = {
    16384: "fp64_tc",
    32768: "fp32_sm",
    524288: "bf16_tc",
    1048576: "fp8_tc",
    2097152: "mxfp4_tc",
}

#: The axis naming an allocation, and the axis naming how much of it is
#: touched. Both are declared by the problem; v1's traffic tier read only the
#: first.
ALLOC_AXIS = "num_pages"
GATHERED_AXIS = "num_kv_indices"

#: MAC/cycle by declared float dtype, for recomputing a paged problem's
#: ARITHMETIC term when its memory term has to be thrown away. Same table as
#: src/solexbench_rocm/parts.py; not imported, because that module is the
#: architectural source and this is a lookup keyed by a dataset dtype string.
DTYPE_MAC_PER_CYCLE = {
    "bfloat16": 524288, "float16": 524288, "float32": 32768,
    "float8_e4m3fn": 1048576, "float8_e5m2": 1048576, "float64": 16384,
}


def load_arch_bpc() -> float:
    """DRAM bytes per cycle, from the same arch config v1 used."""
    for line in ARCH.read_text().splitlines():
        if line.strip().startswith("DRAM_byte_per_cycle"):
            return float(line.split(":", 1)[1].split("#")[0].strip())
    raise SystemExit(f"{ARCH}: no DRAM_byte_per_cycle")


def datapath_of(w: dict) -> str | None:
    """Which arithmetic datapath SOLAR priced this workload at, or None.

    Recovered from the artifact rather than declared: `macs / cycles` is the
    MAC/cycle rate that produced the number, so it cannot disagree with it.
    A rate that does not land on a known entry returns None and the workload
    keeps its v1 millisecond value -- guessing a datapath would put a wrong
    clock under a bound, which is the defect being fixed.
    """
    macs = w.get("macs")
    cycles = w.get("t_sol_cycles_exact") or w.get("t_sol_cycles")
    if not macs or not cycles:
        return None
    rate = macs / cycles
    best = min(RATE_TO_DATAPATH, key=lambda r: abs(r - rate))
    return RATE_TO_DATAPATH[best] if abs(best - rate) / best < 0.02 else None


def gathered_traffic(definition: dict, axes: dict) -> int | None:
    """Declared bytes, with an allocation-shaped tensor priced at what it gathers.

    Identical to `sol_cross_checks.declared_traffic` except that a dimension
    named by ALLOC_AXIS is replaced by GATHERED_AXIS when the problem declares
    it. Returns None on any unresolved symbol or unknown dtype, exactly as the
    original does: a partial byte count is not a bound.
    """
    if GATHERED_AXIS not in axes:
        return None
    total = 0
    for group in ("inputs", "outputs"):
        for spec in (definition.get(group) or {}).values():
            shape = spec.get("shape")
            if not shape:
                continue
            n = 1
            for dim in shape:
                if dim == ALLOC_AXIS:
                    n *= axes[GATHERED_AXIS]
                elif isinstance(dim, int):
                    n *= dim
                elif dim in axes:
                    n *= axes[dim]
                elif str(dim).isdigit():
                    n *= int(dim)
                else:
                    return None
            width = DTYPE_BYTES.get(spec.get("dtype"))
            if width is None:
                return None
            total += n * width
    return total


def arithmetic_rate(definition: dict) -> int | None:
    """MAC/cycle for a problem's dominant declared float dtype."""
    from collections import Counter
    c: Counter = Counter()
    for group in ("inputs", "outputs"):
        for spec in (definition.get(group) or {}).values():
            dt = spec.get("dtype")
            if dt in DTYPE_MAC_PER_CYCLE:
                c[dt] += 1
    return DTYPE_MAC_PER_CYCLE[c.most_common(1)[0][0]] if c else None


def paged_problems() -> dict[str, dict]:
    """Every problem declaring both an allocation axis and a gathered axis."""
    out = {}
    for defn in sorted(DATA.glob("*/*/definition.json")):
        d = json.loads(defn.read_text())
        axes = d.get("axes") or {}
        if ALLOC_AXIS in axes and GATHERED_AXIS in axes:
            key = f"{defn.parent.parent.name}__{defn.parent.name}"
            out[key] = d
    return out


def workload_axes(key: str) -> dict[str, dict]:
    """{uuid -> resolved axes} straight from the dataset's workload list."""
    cat, name = key.split("__", 1)
    f = DATA / cat / name / "workload.jsonl"
    return {w["uuid"]: (w.get("axes") or {})
            for w in (json.loads(ln) for ln in f.read_text().splitlines() if ln.strip())}


def clock_for(datapath: str, measured: dict) -> float:
    """The frequency a bound for this datapath may be divided by.

    `max(F_LOCK, measured)` and not the measurement alone, for two reasons that
    happen to point the same way.

    T_SOL is a LOWER bound, so where the hardware could be faster the bound has
    to assume it is: dividing by the smaller of the two would produce a larger
    T_SOL and a bound a kernel can beat, which is the whole defect being fixed.

    And the matrix-core measurements sit within scatter of F_LOCK -- bf16 1296,
    fp16 1299, against 1300 and against 1303 on a separate 4 s run. Writing 1296
    into the manifest would assert a 0.3% downward correction that the
    measurement does not support, and would move 926 bf16 bounds for no reason.
    Only fp32 (1441) and fp8 (1314) clear F_LOCK, and only those move.
    """
    return max(F_LOCK_MHZ, float(measured.get(datapath, 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.loads(V1.read_text())
    solar = json.loads(T_SOL.read_text())["problems"]
    clocks = json.loads(CLOCKS.read_text())["datapaths"]
    mhz = {k: v["clk_steady_last5s_mhz"] for k, v in clocks.items()
           if v.get("clk_steady_last5s_mhz")}
    bpc = load_arch_bpc()
    paged = paged_problems()

    n_clock, n_paged, n_regate, by_dp = 0, 0, 0, {}
    changes = []

    for key, prob in man["problems"].items():
        s_prob = (solar.get(key) or {}).get("workloads", {})
        axes_by_uuid = workload_axes(key) if key in paged else {}
        defn = paged.get(key)

        for uuid, w in prob.get("workloads", {}).items():
            old_ms = w.get("t_sol_ms")
            if not old_ms:
                continue

            # --- D18 first: it can change which tier wins, and therefore
            # which correction the clock fix should be applied to.
            if defn is not None:
                axes = axes_by_uuid.get(uuid)
                bytes_gathered = (gathered_traffic(defn, resolved_axes(defn, axes))
                                  if axes else None)
                if bytes_gathered:
                    cyc = max(1, math.ceil(bytes_gathered / bpc))
                    sw = s_prob.get(uuid) or {}
                    solar_cyc = sw.get("t_sol_cycles")
                    # SOLAR's number cannot simply be maxed against the
                    # corrected traffic, because on these problems SOLAR has
                    # the SAME defect: for 018 its `memory_bytes` is
                    # 1,140,133,554 -- the whole allocation, to the byte -- and
                    # its bottleneck is `memory`, so its bound IS the
                    # allocation-streaming time and taking the larger of the
                    # two would keep exactly the number being corrected.
                    #
                    # A memory term computed over the allocation is the same
                    # error wherever it appears, so both tiers are recomputed
                    # from the gathered bytes and only SOLAR's ARITHMETIC term
                    # survives from it. On this family that term is small
                    # (0-4 cycles for 018's decode workloads) and the traffic
                    # term wins anyway -- but it is carried rather than
                    # assumed, because "it was negligible last time" is not a
                    # derivation.
                    rate = arithmetic_rate(defn)
                    macs = sw.get("macs")
                    comp = (max(1, math.ceil(macs / rate))
                            if (macs and rate) else 0)
                    if comp >= cyc:
                        new_cyc, src = comp, "solar_arithmetic_gathered"
                    else:
                        new_cyc, src = cyc, "declared_traffic_gathered"
                    w["t_sol_cycles_solar_v1"] = solar_cyc
                    w["t_sol_cycles_arithmetic"] = comp
                    w["t_sol_cycles_traffic_v1"] = w.get("t_sol_cycles_traffic")
                    w["t_sol_cycles_traffic"] = cyc
                    w["t_sol_cycles"] = new_cyc
                    w["t_sol_source_v1"] = w.get("t_sol_source")
                    w["t_sol_source"] = src
                    w["paged_bytes_v11"] = bytes_gathered
                    w["gathered_pages"] = axes.get(GATHERED_AXIS)
                    w["allocated_pages"] = axes.get(ALLOC_AXIS)
                    n_paged += 1

            # --- D35: divide by the clock of the datapath, not by one F_LOCK.
            f = 1300.0
            dp = None
            if w.get("sol_bottleneck") == "compute":
                dp = datapath_of(s_prob.get(uuid) or {})
                if dp:
                    f = clock_for(dp, mhz)
                    if f > F_LOCK_MHZ:
                        by_dp[dp] = by_dp.get(dp, 0) + 1
                        n_clock += 1
            w["f_lock_mhz_used"] = f
            w["datapath"] = dp
            w["t_sol_ms_v1"] = old_ms
            w["t_sol_ms"] = w["t_sol_cycles"] / (f * 1e3)

            # The gate v1 applied, re-applied: a lower bound above the measured
            # anchor is not a loose bound, it is a wrong one.
            t_b = w.get("t_b_ms")
            if t_b and w["t_sol_ms"] > t_b:
                w["t_sol_ms"] = old_ms
                w["t_sol_ms_v11_rejected"] = w["t_sol_cycles"] / (f * 1e3)
                w["t_sol_v11_note"] = ("recomputed bound landed above T_b and "
                                       "was rejected; v1 value kept")
                n_regate += 1
            elif abs(w["t_sol_ms"] - old_ms) / old_ms > 1e-9:
                changes.append((key, uuid, old_ms, w["t_sol_ms"]))

    man["manifest_version"] = "v1.1"
    v1_prov = man.get("_provenance") or {}
    prov = stamp("09-manifest-v1.1")
    # The part is carried over from v1, not re-detected. This script is pure
    # arithmetic over v1's own numbers and runs on the host python, which has
    # no torch -- so `stamp()` finds no devices and the part would silently go
    # missing from a manifest that describes exactly the same silicon v1 does.
    # `ingest.py` refuses a manifest that cannot name its part, which is how
    # the omission surfaced rather than shipping.
    prov["part"] = v1_prov.get("part")
    prov["torch"] = v1_prov.get("torch")
    prov["rocm"] = v1_prov.get("rocm")
    prov["f_lock_mhz"] = v1_prov.get("f_lock_mhz")
    prov["derived_from"] = {
        "manifest": "artifacts/09/manifest-v1.json",
        "provenance": v1_prov,
        "note": "no measurement was repeated; every cycle count is v1's",
    }
    man["_provenance"] = prov
    man["v1_1_changes"] = {
        "supersedes": "manifest-v1.json, which is unchanged and still valid for "
                      "every score published against it",
        "d35_clock": {
            "what": "T_SOL_ms now divides t_sol_cycles by the measured clock of "
                    "the datapath SOLAR priced the workload at, not by a single "
                    "F_LOCK of 1300 MHz.",
            "source": "artifacts/01/f_lock_by_datapath.json",
            "clocks_mhz": mhz,
            "workloads_moved": n_clock,
            "workloads_by_datapath": by_dp,
            "cycles_unchanged": True,
            "memory_tier_unchanged": "the frequency cancels in that tier; see "
                                     "the module docstring",
        },
        "d18_paged": {
            "what": "A KV cache declared [num_pages, ...] is priced at the "
                    "num_kv_indices pages the workload gathers, not at its "
                    "allocation.",
            "problems": sorted(paged),
            "workloads_moved": n_paged,
            "character_change": (
                "Worth stating plainly: on the smallest of these workloads the "
                "corrected bound is very small -- 018's first workload goes "
                "from 185,274 cycles to 8, because the 44 KB it actually "
                "touches takes 8 cycles to stream. The bound stops being "
                "informative there and the score degenerates towards T_b / T_k, "
                "a plain speedup against the PyTorch anchor. That is a worse "
                "bound to read and a correct one to use: T_SOL is a LOWER "
                "bound, so too small is loose and too large is wrong, and only "
                "one of the two lets a kernel score above 1. A tighter bound "
                "for small paged workloads would have to model launch overhead "
                "and achievable bandwidth at that size, which is a new "
                "derivation and not a correction."),
        },
        "regated": n_regate,
        "still_wrong": {
            "note": "Unfixed under v1.1 and still marked. The first five are "
                    "beaten by more than either correction above can explain. "
                    "The sixth is different and is listed with them because a "
                    "run scored against v1.1 flags six, not five, and a field "
                    "here saying five would be contradicted by the artifact "
                    "beside it.",
            "problems": ["L2__045_audio_encoder_to_language_model_multimodal_fusion",
                         "L1__006_hyena_depthwise_conv1d_split_gate",
                         "L1__054_audio_attention_qkv_projection_with_normalization",
                         "L1__005_conv_gated_projection_with_causal_conv",
                         "L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion",
                         "L2__073_feedforward_mlp_backward"],
            "l2_073_is_a_residue_not_a_defect": (
                "1.120 under v1, 1.010 under v1.1. Its kernel holds 1466 MHz "
                "where the saturating fp32 GEMM holds 1441, and "
                "1.010 x (1441/1466) = 0.993 -- the 1% left is exactly the gap "
                "between the clock the DATAPATH sustains and the clock that "
                "one kernel happened to reach. Closing it would mean pricing a "
                "bound at the clock a kernel achieved, which is circular: the "
                "bound would then depend on the submission being scored "
                "against it. 1441 is the defensible divisor and the residue "
                "is the honest cost of using it."),
        },
    }

    print(f"clock-corrected workloads : {n_clock}  {by_dp}")
    print(f"paged-corrected workloads : {n_paged}")
    print(f"re-gated (kept v1 value)  : {n_regate}")
    print(f"total ms values changed   : {len(changes)}")
    for key, uuid, o, n in changes[:5]:
        print(f"   {key[:44]:<44} {o:.6f} -> {n:.6f}  ({n/o:.4f}x)")
    if a.dry_run:
        return 0
    a.out.write_text(json.dumps(man, indent=1) + "\n")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
