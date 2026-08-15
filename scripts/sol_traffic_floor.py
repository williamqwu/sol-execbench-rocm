#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 03, second tier — the memory bound the problem's own definition implies.

This tier exists because SOLAR's graph is not always the whole kernel, and it
answers two different problems at once.

*Fifty problems have no SOLAR bound at all.* SOLAR models einsum layers, and a
kernel that is pure elementwise arithmetic and indexing produces
`einsum_graph has no layers`. Those problems are not scoreable without a second
derivation.

*Forty-eight more have a SOLAR bound that is below their own declared traffic*
(cross-check A). There the traced graph missed tensors the problem itself
declares, so the bound is real but loose, and a loose lower bound understates
every score measured against it.

Both are answered by the simplest bound in the roofline: every declared input
must be read at least once and every declared output written at least once, so

    T >= (declared input bytes + declared output bytes) / DRAM bandwidth

which is the same formula SOLAR applies to its own byte count, against the
same arch config, at the same locked clock.

**This is a second derivation and is labelled as one.** Every workload it
produces carries `t_sol_source: "declared_traffic"`; SOLAR's carry
`"solar_fused"`. Nothing merges them into an unmarked column, because the two
are not equally strong: SOLAR's accounts for the arithmetic and this does not,
so this tier is a bound only for kernels whose arithmetic is genuinely free.

**A tensor the kernel GATHERS is priced at what it gathers (D18).** Where the
problem's own reference reads a declared tensor only through an index vector --
a paged KV cache read through `kv_indices` -- pricing the allocation is not
loose, it is wrong: on `FlashInfer-Bench__015` it put the bound 43x above a
real kernel's measured time. `scripts/sol_gathered_traffic.py` derives that
pairing from the definition and the reference, per problem, and the dimension
is priced at the rows the workload names. Every record carries the uncorrected
`allocation_bytes` beside `memory_bytes` so the correction stays auditable.

**It is gated on being a bound at all.** Where a problem declares a tensor it
indexes rather than streams and no gather pairing is derivable -- a
131072-position KV cache, an embedding table -- the declared total is still
above the traffic any kernel performs, and the "bound" would land above the
measured time. Those are dropped: any workload
whose derived T_SOL exceeds its measured T_b is rejected, with the pair
recorded. A lower bound above a measured time is not a loose bound, it is a
wrong one, and it would push scores above 1.

    python scripts/sol_traffic_floor.py --t-b artifacts/06/authoritative \\
        --out artifacts/03/t_sol_traffic.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import write_artifact  # noqa: E402
from sol_cross_checks import declared_traffic, load_arch, resolved_axes  # noqa: E402
from sol_gathered_traffic import gathered_axes, gathered_traffic  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--arch", default="SOLAR/configs/arch/MI350X.yaml")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--t-b", default="artifacts/06/authoritative",
                    help="required: the gate. A derived bound above the "
                         "measured time is rejected, not shipped.")
    ap.add_argument("--out", default="artifacts/03/t_sol_traffic.json")
    a = ap.parse_args()

    arch = load_arch(Path(a.arch))
    freq_ghz = arch["freq_GHz"]
    dram_bpc = arch["DRAM_byte_per_cycle"]

    doc = json.loads(Path(a.t_sol).read_text())
    solar = doc.get("problems", doc)

    # Enumerate from the DATASET, not from SOLAR's output. Five problems failed
    # SOLAR at definition-load time and so have no workloads recorded at all;
    # keying off SOLAR's list would silently skip exactly the problems that
    # need this tier most.
    problems: dict[str, dict] = {}
    for cat_dir in sorted(Path(a.data).glob("*")):
        if not cat_dir.is_dir():
            continue
        for prob in sorted(cat_dir.glob("*")):
            wl_file = prob / "workload.jsonl"
            if not (prob / "definition.json").exists() or not wl_file.exists():
                continue
            key = f"{cat_dir.name}__{prob.name}"
            entry = dict(solar.get(key) or {})
            recorded = entry.get("workloads") or {}
            workloads = {}
            for line in wl_file.read_text().splitlines():
                if not line.strip():
                    continue
                w = json.loads(line)
                workloads[w["uuid"]] = {**(recorded.get(w["uuid"]) or {}),
                                        "axes": w.get("axes") or {}}
            entry["workloads"] = workloads
            problems[key] = entry

    # measured T_b, per workload
    t_b: dict[str, dict[str, float]] = {}
    tb_dir = Path(a.t_b)
    for f in sorted(tb_dir.glob("*.json")):
        d = json.loads(f.read_text())
        wins = d.get("winner_by_workload") or {}
        if wins:
            t_b[f.stem] = {u: w["t_b_ms"] for u, w in wins.items()}

    out: dict[str, dict] = {}
    n_workloads = n_rejected = n_unresolved = 0
    n_gathered_problems = n_gathered_workloads = 0
    rejected: list[dict] = []

    for key, entry in sorted(problems.items()):
        workloads = entry.get("workloads") or {}
        category, name = key.split("__", 1)
        defn_path = Path(a.data) / category / name / "definition.json"
        if not defn_path.exists():
            continue
        definition = json.loads(defn_path.read_text())
        # D18: which declared axes this problem's reference only ever GATHERS
        # from. Derived once per problem from the definition and its reference,
        # never tabulated per problem -- see scripts/sol_gathered_traffic.py.
        gather = gathered_axes(definition)
        if gather:
            n_gathered_problems += 1

        got: dict[str, dict] = {}
        for uuid, w in workloads.items():
            axes = resolved_axes(definition, w.get("axes") or {})
            allocation = declared_traffic(definition, axes)
            declared = (gathered_traffic(definition, axes, gather) if gather
                        else allocation)
            if gather and declared and declared != allocation:
                n_gathered_workloads += 1
            if not declared:
                n_unresolved += 1
                continue
            cycles = max(1, math.ceil(declared / dram_bpc))
            ms = cycles / (freq_ghz * 1e6)
            measured = (t_b.get(key) or {}).get(uuid)
            if measured is not None and ms > measured:
                n_rejected += 1
                rejected.append({"problem": key, "workload": uuid,
                                 "t_sol_ms": ms, "t_b_ms": measured,
                                 "declared_bytes": declared,
                                 "allocation_bytes": allocation})
                continue
            got[uuid] = {
                "solar_t_sol_cycles": w.get("t_sol_cycles"),
                "t_sol_cycles": cycles,
                "t_sol_cycles_exact": declared / dram_bpc,
                "t_sol_ms": ms,
                "bottleneck": "memory",
                "memory_bytes": declared,
                # The uncorrected number, kept so the correction stays auditable
                # rather than silently replacing the artifact's history: equal
                # to `memory_bytes` wherever no gather was found.
                "allocation_bytes": allocation,
                "gathered_axes": gather or None,
                "macs": None,
                # -- The fields `t_sol_at` needs to re-max at another clock
                # (docs/TODO-MI355X.md §4.2(b)). Without them this whole tier
                # raises `MissingBoundTerms`, which on an unlocked part means it
                # cannot be scored at all -- 328 workloads across 38 problems on
                # the MI350X manifest rest on this tier.
                #
                # `compute_cycles = 0.0` is the literal truth about this
                # derivation, not filler: the declared-traffic tier accounts for
                # ALL the traffic and NONE of the arithmetic. A pure traffic
                # bound is therefore clock-invariant in time, and that is exactly
                # what `t_sol_ms_at` returns from these numbers.
                #
                # `mac_per_cycle = None` for the same reason: no arithmetic term,
                # so no rate. Emitted rather than omitted so a consumer can tell
                # "this tier has no compute term" from "this record predates the
                # split".
                #
                # `dram_byte_per_sec` is `sol_bounds.py`'s own derivation
                # inverted: DRAM_byte_per_cycle is *defined* in the arch YAML as
                # bytes_per_sec / freq, so multiplying back is exact rather than
                # a re-estimate.
                "compute_cycles": 0.0,
                "memory_cycles_at_f_ref": declared / dram_bpc,
                "mac_per_cycle": None,
                "dram_byte_per_sec": dram_bpc * freq_ghz * 1e9,
                "axes": dict(w.get("axes") or {}),
                "t_sol_source": "declared_traffic",
                "gated_against_t_b": measured is not None,
            }
            n_workloads += 1
        if got:
            out[key] = {
                "definition": entry.get("definition"),
                "precision": entry.get("precision"),
                "solar_error": (entry.get("error")
                                or next(iter(workloads.values()), {}).get("error")),
                "workloads": got,
            }

    write_artifact(Path(a.out), "03-traffic-floor", {
        "_note": "Second-tier analytic bound for problems SOLAR cannot model. "
                 "Every workload here is labelled t_sol_source=declared_traffic "
                 "and must stay distinguishable from SOLAR's bounds: this tier "
                 "accounts for memory traffic only.",
        "arch": {"freq_GHz": freq_ghz, "DRAM_byte_per_cycle": dram_bpc},
        "problems": out,
        "stats": {
            "problems_recovered": len(out),
            "workloads_recovered": n_workloads,
            "workloads_rejected_above_measured": n_rejected,
            "workloads_unresolvable_shape": n_unresolved,
            "problems_with_gathered_axis": n_gathered_problems,
            "workloads_repriced_from_allocation_to_gather": n_gathered_workloads,
        },
        "rejected": rejected[:200],
    })

    print(f"traffic-floor tier -> {a.out}")
    print(f"  problems recovered            {len(out)}")
    print(f"  workloads recovered           {n_workloads}")
    print(f"  rejected (bound above T_b)    {n_rejected}")
    print(f"  unresolvable declared shape   {n_unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
