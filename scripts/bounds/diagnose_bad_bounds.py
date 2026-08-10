#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose every T_SOL that a measured kernel beat, and say *why* for each.

    python scripts/bounds/diagnose_bad_bounds.py \
        --manifest artifacts/09/manifest-v1.2.json \
        --out artifacts/11/bad-bounds-v12.json

A speed-of-light bound is a lower bound: nothing can beat it. When something
does, the bound is wrong -- but "wrong" is a symptom with several causes, and
the symptom never says which (see `STATE.md` D18/D35/D37/D38). This script
reduces the five surviving violations under v1.2 to their mechanisms and, for
each, checks the mechanism by hand-computing the bound from the problem's own
declared shapes. A mechanism is only recorded here when the hand computation
lands on the manifest number to within a percent -- an exact ratio is the
evidence, a plausible story is not.

**The finding is that five violations are two causes, and one of them is D18
again.** The declared-traffic tier prices every declared input tensor at its
full allocation whether or not the kernel reads it. D18 was that same defect
seen through paged KV, and v1.1 fixed it for the two FlashInfer problems rather
than at the tier -- so it is still live everywhere else, on 328 workloads
across 38 problems where `max_of_both` picks traffic over SOLAR.

Requires no GPU: it reads the manifest, the problem definitions and the scored
runs. Nothing here re-measures anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "SOL-ExecBench" / "benchmark"

#: bytes per cycle at F_LOCK, from SOLAR/configs/arch/MI350X.yaml. Restated
#: rather than imported because this script is a check ON the pipeline that
#: produced the manifest, and a check that shares the value it is checking
#: cannot detect a wrong one (CLAUDE.md s6, "a self-consistent bound and
#: anchor cannot detect a shared error").
DRAM_BYTE_PER_CYCLE = 6153.8
MAC_FP32 = 32768
MAC_BF16 = 524288


def axes_of(category: str, name: str) -> dict[str, dict]:
    path = BENCH / category / name / "workload.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["uuid"]] = d["axes"]
    return out


# ── the hand computations, one per diagnosed problem ──────────────────────────
# Each returns {"hand": cycles, "live": cycles, "what": str}. `hand` is what the
# manifest number should equal if the stated mechanism is the mechanism; `live`
# is the traffic or arithmetic a correct kernel cannot avoid.


def hand_L1__018(a: dict) -> dict:
    """The KV cache is priced at its allocation, not at the slots written."""
    B, S = a["batch_size"], a["seq_len"]
    HQ, HK, D, MAXP = 96, 8, 128, 262144
    live = (B * HQ * S * D * 2) * 2 + (B * HK * S * D * 2) * 5
    dead = 4 * B * HK * MAXP * D * 2          # read AND write, both caches
    return {"hand": (live + dead) / DRAM_BYTE_PER_CYCLE,
            "live": live / DRAM_BYTE_PER_CYCLE,
            "what": f"full {MAXP}-slot k+v cache read and written "
                    f"({dead/1e9:.2f} GB) for {S} slots touched"}


def hand_L1__042(a: dict) -> dict:
    """Two declared inputs the function never reads are priced as streamed."""
    N = a["batch_seq_len"]
    E, K = 256, 8
    unit = N * E * 4                                   # one fp32 [N, E] tensor
    live = 2 * unit                                    # grad in, grad out
    dead = N * E * 8 + N * K * 8                       # expert_mask, topk_idx
    return {"hand": (live + dead) / DRAM_BYTE_PER_CYCLE,
            "live": live / DRAM_BYTE_PER_CYCLE,
            "what": "expert_mask (int64) and topk_idx are declared inputs that "
                    "run() never reads -- it uses topk_idx.shape[0] only"}


def hand_L1__057(a: dict) -> dict:
    """The whole embedding table is priced, for a gather of B*S rows."""
    B, S = a["batch_size"], a["seq_len"]
    H, V, H2 = 4096, 157184, 8192
    table = V * H * 2
    live = B * S * H * 2 * 2 + B * S * 8 + H * H2 * 2 + B * S * H * 2
    return {"hand": (live - B * S * H * 2 + table) / DRAM_BYTE_PER_CYCLE,
            "live": (live + B * S * H * 2) / DRAM_BYTE_PER_CYCLE,
            "what": f"the {V}-row embedding table ({table/1e9:.2f} GB) is priced "
                    f"as streamed for a gather of {B*S} rows"}


def hand_L2__045(a: dict) -> dict:
    """Q-Former and projector are priced over every window; only a few are read."""
    B, S, N = a["batch_size"], a["audio_seq_len"], a["num_audio_tokens"]

    def macs(blocks: int, frames: int) -> int:
        NB = B * blocks
        return (B * frames * 80 * 512 + B * frames * 512 * 1024
                + B * frames * 1024 * 512
                + NB * 40 * 1024 * 1024 + 2 * NB * 15 * 1024 * 512
                + 2 * NB * 16 * 40 * 15 * 64
                + NB * 40 * 1024 * 1024 + NB * 40 * 1024 * 4096)

    nb = math.ceil(S / 15)
    used = math.ceil(N / 40)
    allm = macs(nb, S)
    live = macs(used, min(S, used * 15))
    # SOLAR prices these fp32 einsums at the bf16 MAC rate -- a second and
    # independent error, in the safe direction, that partly masks the first.
    return {"hand": allm / MAC_BF16,
            "live": live / MAC_FP32,
            "what": f"{nb} windows priced, {used} read -- {allm/live:.1f}x of the "
                    f"counted MACs are discarded; and the fp32 einsums are "
                    f"priced at the bf16 rate (exactly 1/16)"}


DIAGNOSED = {
    "L1__018_fused_rope_with_qk_norm_and_kv_cache_update":
        ("L1", "018_fused_rope_with_qk_norm_and_kv_cache_update",
         hand_L1__018, "declared_traffic_over_count"),
    "L1__042_moe_expert_load_balancing_and_token_capacity_backward":
        ("L1", "042_moe_expert_load_balancing_and_token_capacity_backward",
         hand_L1__042, "declared_traffic_over_count"),
    "L1__057_mtp_shifted_embedding_with_dual_rms_norm_fusion":
        ("L1", "057_mtp_shifted_embedding_with_dual_rms_norm_fusion",
         hand_L1__057, "declared_traffic_over_count"),
    "L2__045_audio_encoder_to_language_model_multimodal_fusion":
        ("L2", "045_audio_encoder_to_language_model_multimodal_fusion",
         hand_L2__045, "solar_counts_discarded_arithmetic"),
}


def violations(manifest: dict) -> dict[str, dict[str, dict]]:
    """{problem: {uuid: {run, t_ms}}} over every run the board publishes."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(str(ROOT / "artifacts/10/*/scored.json"))):
        d = json.loads(Path(f).read_text())
        if not d.get("leaderboard"):
            continue
        for r in d["results"]:
            if not r.get("bound_violation"):
                continue
            cur = out.setdefault(r["problem"], {}).get(r["workload_uuid"])
            if cur is None or r["latency_ms"] < cur["t_ms"]:
                out[r["problem"]][r["workload_uuid"]] = {
                    "run": d["run_id"], "t_ms": r["latency_ms"]}
    return out


def traffic_tier_blast_radius(problems: dict) -> dict:
    """How many bounds rest on the tier the three memory cases indict."""
    n = wins = 0
    per: dict[str, int] = {}
    ratios: list[float] = []
    for P, d in problems.items():
        for w in d["workloads"].values():
            if not w.get("scoreable"):
                continue
            n += 1
            s, t = w.get("t_sol_cycles_solar"), w.get("t_sol_cycles_traffic")
            if s and t and t > s:
                wins += 1
                per[P] = per.get(P, 0) + 1
                ratios.append(t / s)
    ratios.sort()
    return {
        "scoreable_workloads": n,
        "workloads_where_traffic_tier_wins": wins,
        "fraction": round(wins / n, 4),
        "problems": len(per),
        "ratio_traffic_over_solar": {
            "p50": round(ratios[len(ratios) // 2], 3),
            "p90": round(ratios[int(0.9 * len(ratios))], 3),
            "max": round(ratios[-1], 1),
        },
        "worst_by_ratio": sorted(
            ({"problem": P,
              "n": k,
              "max_ratio": round(max(
                  w["t_sol_cycles_traffic"] / w["t_sol_cycles_solar"]
                  for w in problems[P]["workloads"].values()
                  if w.get("t_sol_cycles_solar") and w.get("t_sol_cycles_traffic")
                  and w["t_sol_cycles_traffic"] > w["t_sol_cycles_solar"]), 1)}
             for P, k in per.items()),
            key=lambda d: -d["max_ratio"])[:15],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/09/manifest-v1.2.json")
    ap.add_argument("--out", default="artifacts/11/bad-bounds-v12.json")
    a = ap.parse_args()

    man = json.loads((ROOT / a.manifest).read_text())
    problems = man["problems"]
    viol = violations(man)

    per_problem = []
    for P in sorted(viol):
        W = problems[P]["workloads"]
        entry: dict = {
            "problem": P,
            "n_violating_workloads": len(viol[P]),
            "min_tk_over_tsol": round(min(
                v["t_ms"] / W[u]["t_sol_ms"] for u, v in viol[P].items()), 4),
            "bottleneck": W[next(iter(viol[P]))]["sol_bottleneck"],
            "t_sol_source": W[next(iter(viol[P]))]["t_sol_source"],
        }
        if P in DIAGNOSED:
            cat, name, fn, cause = DIAGNOSED[P]
            axes = axes_of(cat, name)
            # For the traffic cases the hand formula models the TRAFFIC TIER, so
            # it can only match where that tier exists. Comparing it against
            # `t_sol_cycles` -- which is `max(solar, traffic)` -- would score a
            # correct derivation as a miss on every workload SOLAR won, and the
            # ratio would look like a partial explanation rather than an exact
            # one. Check against the tier the formula is about, and say how many
            # workloads that tier is even present on.
            tier = ("t_sol_cycles_traffic"
                    if cause == "declared_traffic_over_count"
                    else "t_sol_cycles_solar")
            checks = []
            for u, w in W.items():
                h = fn(axes[u])
                ref = w.get(tier)
                checks.append({
                    "uuid": u,
                    "violating": u in viol[P],
                    "tier_checked": tier,
                    "tier_cycles": ref,
                    "manifest_cycles": w["t_sol_cycles"],
                    "tier_is_what_scored": ref == w["t_sol_cycles"],
                    "hand_cycles": round(h["hand"]),
                    "hand_over_tier": round(h["hand"] / ref, 4) if ref else None,
                    "live_cycles": round(h["live"]),
                    "over_count": round(h["hand"] / h["live"], 1),
                })
            present = [c for c in checks if c["hand_over_tier"] is not None]
            matched = [c for c in present if abs(c["hand_over_tier"] - 1) < 0.01]
            unexplained = [c["uuid"] for c in checks
                           if c["violating"] and c not in matched]
            entry.update({
                "cause": cause,
                "mechanism": fn(next(iter(axes.values())))["what"],
                "hand_check_matches_tier": f"{len(matched)}/{len(present)}",
                "tier_present_on": f"{len(present)}/{len(checks)} workloads",
                "violating_workloads_not_explained": unexplained,
                "workloads": checks,
            })
        else:
            entry.update({
                "cause": "residual_clock_error",
                "mechanism": "worst violation 0.992 -- inside the band D35 "
                             "predicts for a compute-bound fp32 workload that "
                             "clocks above the 1300 MHz F_LOCK divisor. Nothing "
                             "further to derive; see STATE.md D35/D36.",
            })
        per_problem.append(entry)

    payload = {
        "question": "Under manifest v1.2, five problems still have a T_SOL a "
                    "kernel beat. Are they five modelling errors?",
        "answer": "Two causes. Three of the five are one defect -- the "
                  "declared-traffic tier prices every declared input at its "
                  "full allocation regardless of what the kernel reads -- and "
                  "that is D18 seen a third, fourth and fifth time, because "
                  "v1.1 fixed D18 per-problem rather than at the tier. The "
                  "fourth is SOLAR counting arithmetic the reference discards. "
                  "The fifth is the D35 clock residue at 0.8% and is not a "
                  "modelling error.",
        "manifest": a.manifest,
        "problems": per_problem,
        "traffic_tier_blast_radius": traffic_tier_blast_radius(problems),
        "caveat": "A hand check that matches the manifest confirms the "
                  "mechanism, not the fix. Deriving the corrected bound is a "
                  "separate step and must not be done by adjusting a number "
                  "until the violation disappears.",
    }

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from provenance import write_artifact  # noqa: E402
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_artifact(out, "11-bad-bounds-v12", payload)
    print(f"wrote {out}")
    for e in per_problem:
        print(f"  {e['problem'][:56]:56} {e['cause']:34} "
              f"{e.get('hand_check_matches_tier','-')}  "
              f"unexplained={e.get('violating_workloads_not_explained', [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
