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

**A tensor the kernel STREAMS is priced at the rows the mask leaves alive.**
The same defect one step further out, and it is what four real kernels
falsified on `FlashInfer-Bench__014/__015`: the tier charged a full read of `q`
on causal paged-prefill workloads whose own `qo_indptr`/`kv_indptr` leave 25 of
15783, 2, 3 and 1 of their query rows live. `causal_masked_axis` derives that
from the reference's own empty-window skip and `masked_live_rows` counts the
live rows out of the workload's index vectors -- so this correction, unlike
D18's, reads the workload's safetensors blob at derivation time. It is still
CPU-only and still needs no GPU. Measured cost of the whole rule: 131 ms to
run the detector over all 235 references and 37 ms to read the 136 blobs the
68 workloads that take it name, taking the MI355X tier build from 1.80-1.85 s
to 1.91-2.07 s. Records carry `gathered_bytes` (after D18, before this) and
`masked_rows` beside `memory_bytes`, so the two corrections stay separable.
Outputs are not repriced: a correct kernel writes `(0, -inf)` into every dead
row.

**It is gated on being a bound at all.** Where a problem declares a tensor it
indexes rather than streams and no gather pairing is derivable -- a
131072-position KV cache, an embedding table -- the declared total is still
above the traffic any kernel performs, and the "bound" would land above the
measured time. Those are dropped: any workload
whose derived T_SOL exceeds its measured T_b is rejected, with the pair
recorded. A lower bound above a measured time is not a loose bound, it is a
wrong one, and it would push scores above 1.

**...and the gate must run against the anchors the release actually uses.**
`--t-b` names the tree the rejection is decided by, and if that is not the tree
the manifest is built from then this file's central claim is about a comparison
nobody downstream makes. That is not hypothetical: the MI355X tier was built
with `--t-b artifacts/06-MI355X/authoritative` while every MI355X manifest
declares `sources.t_b = artifacts/06-MI355X/authoritative-merged`, so 237
records shipped marked `gated_against_t_b` against anchors the release does not
use, and one -- `L1__057`/`650d87fb` -- shipped a tier bound 1.68x its own
published anchor, inside the artifact whose `--t-b` help says such a bound "is
rejected, not shipped". Nothing downstream was wrong, because `build_manifest`
re-applies the gate; but that makes the manifest the safety net and this tier's
gate decorative. So the mismatch is now REFUSED rather than defaulted away:
`declared_anchor_trees()` reads the `sources` block of every manifest built
from this `--out`, and a disagreement exits non-zero. A default could have been
pointed at the merged tree instead, but a default is overridden silently and
this is the failure mode where silence is the whole problem.

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
from sol_gathered_traffic import (causal_masked_axis, gathered_axes,  # noqa: E402
                                  gathered_traffic, masked_live_rows)


def declared_anchor_trees(out: Path,
                          manifest_root: Path | None = None) -> dict[str, str]:
    """``{manifest path: its sources.t_b}`` for manifests built from *out*.

    A manifest states the three artifacts it was built from. Where one of them
    is the tier file this run is about to write, the manifest's own
    ``sources.t_b`` is the anchor tree the release compares against -- and
    therefore the only tree whose verdicts this tier's `gated_against_t_b`
    column can honestly describe.

    Only `manifest-*.json` is read, not `candidate-*.json`: a candidate is an
    experiment and is allowed to disagree. Manifests that state no `sources`
    block at all (both frozen MI350X manifests) constrain nothing and are
    skipped, so this can never turn an MI350X tier build red.

    A manifest that cannot be parsed is REPORTED, not skipped quietly. The
    whole value of this function is that it refuses; a silent `except` around
    the read would make it stop refusing exactly when the artifacts are in the
    worst shape.
    """
    root = ROOT if manifest_root is None else Path(manifest_root)
    target = out.resolve()
    found: dict[str, str] = {}
    for man in sorted(root.glob("artifacts/09*/manifest-*.json")):
        try:
            sources = (json.loads(man.read_text()) or {}).get("sources") or {}
        except Exception as e:                             # noqa: BLE001
            print(f"WARNING: cannot read {man}: {type(e).__name__}: {e}; "
                  f"its declared anchor tree is NOT being checked",
                  file=sys.stderr)
            continue
        tier, t_b = sources.get("t_sol_traffic"), sources.get("t_b")
        if not tier or not t_b:
            continue
        if (root / tier).resolve() != target:
            continue
        try:
            name = str(man.relative_to(root))
        except ValueError:
            name = str(man)
        found[name] = t_b
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--arch", default="SOLAR/configs/arch/MI350X.yaml")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--t-b", default="artifacts/06/authoritative",
                    help="required: the gate. A derived bound above the "
                         "measured time is rejected, not shipped.")
    ap.add_argument("--out", default="artifacts/03/t_sol_traffic.json")
    # The part this tier is ABOUT, declared rather than inferred. `--arch`
    # defaults to MI350X.yaml, so a run on this node that forgets the flag
    # prices traffic at MI350X bandwidth and writes it into an artifact that
    # every downstream inference path -- device names, hostname -- would certify
    # as MI355X. Declaring it makes `provenance.stamp()` cross-check the
    # declaration against the visible cards and raise instead of shipping.
    # Deliberately NOT read out of the arch YAML: `load_arch` keeps only values
    # that parse as a float, so `arch["name"]` raises KeyError there.
    ap.add_argument("--part", default=None,
                    help="e.g. MI355X. Declares what this artifact is about; "
                         "refused if the visible cards say otherwise.")
    ap.add_argument("--allow-anchor-mismatch", action="store_true",
                    help="build against a --t-b tree that no manifest built "
                         "from this --out declares. Needed exactly once: when "
                         "a NEW anchor tree is being adopted and the manifest "
                         "that will name it does not exist yet. The mismatch "
                         "is then recorded in the artifact rather than lost.")
    a = ap.parse_args()

    # The gate has to be decided against the anchors the release uses; see the
    # module docstring. Checked BEFORE any derivation so a refusal costs
    # nothing and cannot half-write the artifact.
    out_path = Path(a.out)
    mismatch = {m: tb for m, tb in declared_anchor_trees(out_path).items()
                if (ROOT / tb).resolve() != Path(a.t_b).resolve()}
    if mismatch and not a.allow_anchor_mismatch:
        print(f"REFUSED: --t-b {a.t_b} is not the anchor tree the manifests "
              f"built from {a.out} compare against.", file=sys.stderr)
        for man, tb in sorted(mismatch.items()):
            print(f"  {man} declares sources.t_b = {tb}", file=sys.stderr)
        print("Every `gated_against_t_b` in this file would then be a verdict "
              "about anchors no published score uses -- which is how 237 "
              "records shipped ungated, one of them 1.68x its own published "
              "T_b. Re-run with the declared tree, or pass "
              "--allow-anchor-mismatch if adopting a new one on purpose.",
              file=sys.stderr)
        return 2

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
                # `inputs` is carried because the masked-stream correction needs
                # the workload's own index vectors, not just its axes.
                workloads[w["uuid"]] = {**(recorded.get(w["uuid"]) or {}),
                                        "axes": w.get("axes") or {},
                                        "inputs": w.get("inputs") or {}}
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
    n_masked_problems = n_masked_workloads = n_masked_unresolved = 0
    rejected: list[dict] = []
    f_ref_seen: set[float] = set()

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
        # The masked-stream correction: an axis whose rows the reference's own
        # empty causal window makes dead. Derived once per problem from the
        # reference; the row COUNT is per workload and comes out of its blobs.
        mask = causal_masked_axis(definition)
        if mask:
            n_masked_problems += 1

        got: dict[str, dict] = {}
        for uuid, w in workloads.items():
            axes = resolved_axes(definition, w.get("axes") or {})
            allocation = declared_traffic(definition, axes)
            gathered = (gathered_traffic(definition, axes, gather) if gather
                        else allocation)
            if gather and gathered and gathered != allocation:
                n_gathered_workloads += 1

            live_rows = None
            if mask:
                live_rows = masked_live_rows(mask, axes,
                                             w.get("inputs") or {}, ROOT)
                if live_rows is None:
                    n_masked_unresolved += 1
            live = {mask["axis"]: live_rows} if live_rows is not None else None
            declared = (gathered_traffic(definition, axes, gather, live)
                        if live else gathered)
            if declared and gathered and declared != gathered:
                n_masked_workloads += 1
            if not declared:
                n_unresolved += 1
                continue
            cycles = max(1, math.ceil(declared / dram_bpc))
            ms = cycles / (freq_ghz * 1e6)
            f_ref_mhz = float(freq_ghz * 1000.0)
            f_ref_seen.add(f_ref_mhz)
            measured = (t_b.get(key) or {}).get(uuid)
            if measured is not None and ms > measured:
                n_rejected += 1
                rejected.append({"problem": key, "workload": uuid,
                                 "t_sol_ms": ms,
                                 # D63 again, on the list nobody looked at: a
                                 # millisecond column with no clock beside it
                                 # is the exact shape `t_sol_at.bound_ms`
                                 # exists to refuse, and the header's "one
                                 # distinct value" invariant used to hold here
                                 # only because these 21 records were excluded
                                 # from it. They are computed at the body's
                                 # clock; they now say so.
                                 "f_ref_mhz": f_ref_mhz,
                                 "t_b_ms": measured,
                                 "declared_bytes": declared,
                                 "gathered_bytes": gathered,
                                 "masked_rows": live_rows,
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
                # After D18's gather correction, before the masked-stream one,
                # so the two are separable in the artifact rather than folded
                # into a single "corrected" number nobody can take apart.
                "gathered_bytes": gathered,
                "gathered_axes": gather or None,
                "masked_axis": mask["axis"] if mask else None,
                "masked_rows": live_rows,
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
                # D63: a cycle count is only meaningful next to the clock it
                # was expressed at. This is the clock THIS record's own
                # cycles -> t_sol_ms conversion used, stated in the record
                # rather than left to be inferred from a header or an arch
                # file a consumer may not be holding.
                "f_ref_mhz": f_ref_mhz,
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
        # The anchor tree every `gated_against_t_b` in this file is a verdict
        # about. Recorded because it was not, and a rejection gate that does
        # not name what it gated against cannot be audited: the shipped MI355X
        # tier was gated on `authoritative` while the manifest was built from
        # `authoritative-merged`, and the only way to find that out was to
        # rebuild the file four times.
        "t_b": str(a.t_b),
        # Non-null only when --allow-anchor-mismatch was used, i.e. when this
        # file was deliberately gated against a tree some manifest disagrees
        # with. Null is the normal state and means the refusal above passed.
        "t_b_manifest_mismatch": mismatch or None,
        # D63, the shared field contract: the ONE clock every record in this
        # file converted at. Null when the records disagree -- a header that
        # names a clock some of its records did not use is exactly the defect
        # this field exists to make impossible, so it fails loudly instead.
        "f_ref_mhz": f_ref_seen.pop() if len(f_ref_seen) == 1 else None,
        "problems": out,
        "stats": {
            "problems_recovered": len(out),
            "workloads_recovered": n_workloads,
            "workloads_rejected_above_measured": n_rejected,
            "workloads_unresolvable_shape": n_unresolved,
            "problems_with_gathered_axis": n_gathered_problems,
            "workloads_repriced_from_allocation_to_gather": n_gathered_workloads,
            "problems_with_masked_stream": n_masked_problems,
            "workloads_repriced_from_stream_to_live_rows": n_masked_workloads,
            "workloads_masked_rows_unresolved": n_masked_unresolved,
        },
        "rejected": rejected[:200],
    }, part=a.part)

    print(f"traffic-floor tier -> {a.out}")
    print(f"  problems recovered            {len(out)}")
    print(f"  workloads recovered           {n_workloads}")
    print(f"  rejected (bound above T_b)    {n_rejected}")
    print(f"  unresolvable declared shape   {n_unresolved}")
    print(f"  repriced: gather (D18)        {n_gathered_workloads}")
    print(f"  repriced: masked stream       {n_masked_workloads}"
          f"  ({n_masked_problems} problems, "
          f"{n_masked_unresolved} rows unresolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
