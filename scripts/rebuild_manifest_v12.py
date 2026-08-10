#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manifest v1.2: SOLAR re-derived for the seven grouped-convolution problems.

    python scripts/rebuild_manifest_v12.py

v1 and v1.1 are frozen and stay frozen. This writes
`artifacts/09/manifest-v1.2.json` beside them.

**D37 -- SOLAR priced a grouped convolution as a dense one.** It reads `groups`
from `module_args`, which the graph processor fills in only for `nn.Module`
convolutions; every convolution in this benchmark is functional, so `groups`
never arrived and defaulted to 1. `src/solexbench_rocm/solar/conv_groups.py`
recovers it from the tensor shapes -- `groups = in_channels // weight.shape[1]`,
exact for every convolution however it was called -- and `sol_bounds.py` applies
that before running the pipeline.

This is NOT arithmetic over v1's numbers the way v1.1 was. The SOLAR pipeline
was re-run, on `device="meta"` and therefore with no GPU and no measurement, for
the seven problems that call a convolution with non-1 groups. Their new cycle
counts are in `artifacts/11/d37/`.

    L1__006   macs   x768.000 exactly  (the group count, to the digit)
    L1__029   macs   x4.999
    L2__058   macs   x4.663-4.754
    L2__035   macs   x6.698-7.069
    L2__051   macs   x3.173-3.256
    L1__005   macs   x1.999
    L2__036   macs   x1.000  -- UNCHANGED, see below

Every one of those makes T_SOL *smaller*, which lowers scores on those problems
and removes bound violations rather than creating them. A T_SOL that was too
large is the direction that lets a score exceed 1; correcting it cannot inflate
anything.

`L2__036` did not move and is not fixed. It is a backward problem, and SOLAR
routes backward graphs through `graph/backward_processor.py`, which assembles
its own `module_args` and never reaches the forward conv handler this correction
wraps. It stays on the open list; asserting it is fine because six of its
siblings moved would be exactly the reasoning this file exists to avoid.

**What is recombined and what is not.** For each affected workload the new
SOLAR cycle count is combined with the traffic tier the same way v1 combined
them -- `max` of the two -- and then divided by the clock of whichever datapath
the *new* bottleneck names, because six of the seven change bottleneck on at
least some workloads (L1__006 flips compute -> memory on all sixteen, which is
what a 768x arithmetic over-count was hiding). The `T_SOL <= T_b` gate is
re-applied last, unchanged.

Nothing outside those seven problems is touched, and no measurement is repeated.
"""

from __future__ import annotations

import argparse
import glob
import math
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import stamp  # noqa: E402

V11 = ROOT / "artifacts" / "09" / "manifest-v1.1.json"
D37 = ROOT / "artifacts" / "11" / "d37"
CLOCKS = ROOT / "artifacts" / "01" / "f_lock_by_datapath.json"
OUT = ROOT / "artifacts" / "09" / "manifest-v1.2.json"

F_LOCK_MHZ = 1300.0

#: Same table v1.1 used: MAC/cycle rate -> the datapath whose clock applies.
RATE_TO_DATAPATH = {
    16384: "fp64_tc",
    32768: "fp32_sm",
    524288: "bf16_tc",
    1048576: "fp8_tc",
    2097152: "mxfp4_tc",
}


def datapath_of(w: dict) -> str | None:
    """Recover the datapath from `macs / cycles`, as v1.1 does.

    The rate lands on a `MAC_per_cycle` entry when the workload is
    compute-bound, so no new field is needed to know which clock applies.
    """
    macs = w.get("macs")
    cycles = w.get("t_sol_cycles_exact") or w.get("t_sol_cycles")
    if not macs or not cycles:
        return None
    rate = macs / cycles
    best = min(RATE_TO_DATAPATH, key=lambda r: abs(r - rate))
    return RATE_TO_DATAPATH[best] if abs(best - rate) / best < 0.02 else None


def clock_for(datapath: str | None, measured: dict) -> float:
    """`max(F_LOCK, measured)`.

    T_SOL is a LOWER bound, so where the hardware can run faster the bound has
    to assume it does. 1296 MHz on bf16 is inside the scatter of 1300 and
    writing it in would assert a precision the measurement does not have.
    """
    if not datapath:
        return F_LOCK_MHZ
    return max(F_LOCK_MHZ, float(measured.get(datapath, 0.0)))


def load_new_solar() -> dict[str, dict]:
    """Re-derived SOLAR results, keyed problem -> uuid -> workload dict."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(D37 / "*.json"))):
        doc = json.loads(Path(f).read_text())
        if doc.get("status") != "ok":
            raise SystemExit(f"{f}: status {doc.get('status')!r}, refusing")
        out[doc["problem"]] = doc.get("workloads", {})
    if not out:
        raise SystemExit(f"no re-derived bounds in {D37}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.loads(V11.read_text())
    new_solar = load_new_solar()
    clocks = json.loads(CLOCKS.read_text())["datapaths"]
    mhz = {k: v["clk_steady_last5s_mhz"] for k, v in clocks.items()
           if v.get("clk_steady_last5s_mhz")}

    missing = sorted(set(new_solar) - set(man["problems"]))
    if missing:
        raise SystemExit(f"re-derived problems absent from v1.1: {missing}")

    n_moved = n_same = n_regate = n_flip = 0
    per_problem: dict[str, dict] = {}

    for key, sw_by_uuid in new_solar.items():
        prob = man["problems"][key]
        moved = 0
        for uuid, w in prob.get("workloads", {}).items():
            sw = sw_by_uuid.get(uuid)
            if not sw:
                continue
            old_cyc = w.get("t_sol_cycles")
            old_ms = w.get("t_sol_ms")
            if not old_ms:
                continue

            src_v11 = w.get("t_sol_source") or ""
            if src_v11 != "solar_fused" and "traffic" not in src_v11:
                # A workload whose v1 bound came from a rejection path, or from
                # v1.1's paged-gather recombination, is not something this
                # correction knows how to recombine. None of the seven is one;
                # refusing rather than skipping means that stops being true
                # loudly.
                raise SystemExit(
                    f"{key} {uuid}: unexpected v1.1 source {src_v11!r}; "
                    "the recombination below does not cover it"
                )

            solar_exact = sw.get("t_sol_cycles_exact") or sw.get("t_sol_cycles")
            if not solar_exact:
                continue
            # Ceil, because that is the convention v1 stored and comparing an
            # exact float against a ceiled integer reports every workload as
            # moved by ~0.003%. L2__036 "moved" 8 of 14 that way before this
            # line existed, on a problem whose MAC count did not change at all.
            solar_cyc = math.ceil(solar_exact)
            traffic_cyc = w.get("t_sol_cycles_traffic") or 0

            # v1's combination rule, re-applied to the corrected term.
            if solar_cyc >= traffic_cyc:
                new_cyc, src, bottleneck = solar_cyc, "solar_fused", sw.get("bottleneck")
            else:
                new_cyc, src, bottleneck = traffic_cyc, "declared_traffic", "memory"

            if bottleneck != w.get("sol_bottleneck"):
                n_flip += 1

            f = F_LOCK_MHZ
            dp = None
            if bottleneck == "compute":
                dp = datapath_of(sw)
                f = clock_for(dp, mhz)

            new_ms = new_cyc / (f * 1e3)

            # Self-check, and the reason it is worth the lines: this script
            # reimplements v1's `max(solar, traffic)` combination and v1.1's
            # per-datapath division. If either reimplementation differs from the
            # original, every affected bound moves for a reason that has nothing
            # to do with grouped convolutions -- and the movement would look
            # exactly like the correction working. So where SOLAR's own number
            # did not change, the result must reproduce v1.1 to the bit.
            if solar_cyc == w.get("t_sol_cycles_solar") and abs(new_ms - old_ms) > 1e-15:
                raise SystemExit(
                    f"{key} {uuid}: SOLAR unchanged at {solar_cyc} cycles but "
                    f"the bound moved {old_ms} -> {new_ms}. The recombination "
                    "here does not reproduce v1.1; fix that before trusting "
                    "anything this writes."
                )

            w["t_sol_cycles_v11"] = old_cyc
            w["t_sol_cycles_solar_v11"] = w.get("t_sol_cycles_solar")
            w["t_sol_cycles_solar"] = solar_cyc
            w["t_sol_cycles"] = new_cyc
            w["t_sol_source_v11"] = w.get("t_sol_source")
            w["t_sol_source"] = src
            w["sol_bottleneck_v11"] = w.get("sol_bottleneck")
            w["sol_bottleneck"] = bottleneck
            w["f_lock_mhz_used"] = f
            w["datapath"] = dp
            w["t_sol_ms_v11"] = old_ms

            t_b = w.get("t_b_ms")
            if t_b and new_ms > t_b:
                # Cannot happen in this direction -- every correction here makes
                # T_SOL smaller -- but the gate is re-applied rather than
                # assumed, because "it cannot happen" is how it happens.
                w["t_sol_ms"] = old_ms
                w["t_sol_ms_v12_rejected"] = new_ms
                w["t_sol_v12_note"] = ("recomputed bound landed above T_b and "
                                       "was rejected; v1.1 value kept")
                n_regate += 1
                continue

            w["t_sol_ms"] = new_ms
            if abs(new_ms - old_ms) / old_ms > 1e-9:
                moved += 1
                n_moved += 1
            else:
                n_same += 1

        per_problem[key] = {
            "workloads_moved": moved,
            "workloads": len(prob.get("workloads", {})),
        }

    man["manifest_version"] = "v1.2"
    v11_prov = man.get("_provenance") or {}
    prov = stamp("09-manifest-v1.2")
    # Carried forward for the same reason v1.1 carried it: this runs on the host
    # python, which has no torch, so `stamp()` cannot name the part -- and
    # `ingest.py` refuses a manifest that cannot.
    prov["part"] = v11_prov.get("part")
    prov["torch"] = v11_prov.get("torch")
    prov["f_lock_mhz"] = v11_prov.get("f_lock_mhz")
    man["_provenance"] = prov
    man["v1_2_changes"] = {
        "defect": "D37 -- SOLAR priced a grouped convolution as a dense one",
        "fix": "src/solexbench_rocm/solar/conv_groups.py, applied by sol_bounds.py",
        "re_derived_not_re_measured": (
            "the SOLAR pipeline was re-run on device=meta for the seven "
            "problems that call a convolution with non-1 groups. No GPU, no "
            "measurement repeated."
        ),
        "source_artifacts": sorted(p.name for p in D37.glob("*.json")),
        "problems": per_problem,
        "workloads_moved": n_moved,
        "workloads_unchanged": n_same,
        "bottleneck_flips": n_flip,
        "regated": n_regate,
        "still_open": {
            "L2__036_convnextv2_layer_with_nhwc_persistence_backward": (
                "unchanged at x1.000. A backward problem: SOLAR routes backward "
                "graphs through graph/backward_processor.py, which builds its "
                "own module_args and never reaches the forward conv handler "
                "this correction wraps. Not fixed, not asserted to be fine."
            ),
        },
    }

    print(f"problems re-derived        : {len(new_solar)}")
    for k, v in sorted(per_problem.items()):
        print(f"  {k[:58]:58} {v['workloads_moved']:3d}/{v['workloads']} moved")
    print(f"workloads moved            : {n_moved}")
    print(f"workloads unchanged        : {n_same}")
    print(f"bottleneck flips           : {n_flip}")
    print(f"re-gated (kept v1.1 value) : {n_regate}")

    if a.dry_run:
        print("dry run, nothing written")
        return 0
    a.out.write_text(json.dumps(man, indent=1))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
