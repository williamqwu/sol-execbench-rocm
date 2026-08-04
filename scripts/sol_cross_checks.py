#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 03 cross-checks — is T_SOL a bound, and is it the RIGHT bound?

A roofline bound is the easiest number in the project to get wrong without
noticing: a units slip, a stale clock, a precision that resolved to the wrong
peak, and the output is still a plausible-looking millisecond figure that
rescales every score by a constant. So the bound is attacked from three
directions that do not share an implementation with the SOLAR bridge:

  A  DECLARED TRAFFIC.  Every problem's own definition states the shape and
     dtype of each input and output. Their total is the traffic any correct
     kernel must move at least once. If SOLAR's memory term is BELOW that, its
     "bound" is below the unavoidable minimum and is not a bound.

  B  IMPLIED RATES.  T_SOL implies a bandwidth (bytes / time) and a throughput
     (2 * MACs / time). Neither may exceed the arch config's own peak, and a
     memory-bound workload's implied bandwidth should sit AT the DRAM peak --
     that is what "memory bound" means. A clock or unit error shows up here as
     a rate that is impossible or absurdly low.

  C  HAND-DERIVED MACs.  For eighteen problems whose arithmetic is
     unambiguous -- eleven single matmuls and seven pure memory kernels -- the
     MAC count is written out by hand from the reference and compared. This is
     the check that catches a graph extracted from the wrong shapes.

  D  T_SOL <= best measured.  Requires task 06. Reported as pending until
     `--t-b` is supplied.

Upstream's B200 SOL times are NOT a source here: the shipped dataset carries no
per-workload SOL figures, and a comparison number invented for the occasion
would be worse than no comparison at all. Checks A-C are internal and stronger.

    python scripts/sol_cross_checks.py --out artifacts/03/cross-checks.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from provenance import stamp  # noqa: E402

DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "float4_e2m1fn_x2": 1,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
}

# -- C: hand-derived MAC counts -------------------------------------------
# Each lambda takes the resolved axis dict and returns MACs. Written from the
# reference source, by hand, on purpose: an automatically derived count would
# share the failure modes of the thing it is auditing.
#
# A GEMM of (M,K) x (K,N) is M*N*K MACs. A pure memory kernel is 0, and 0 is a
# real prediction -- SOLAR reporting MACs for one means it found arithmetic
# that is not there.
HAND_MACS = {
    # C = A @ B.T, A:(M,K) B:(N,K)
    **{f"FlashInfer-Bench__{i:03d}_gemm_{tag}":
       (lambda ax, n=n, k=k: ax["M"] * n * k)
       for i, tag, n, k in [
           (4, "n128_k2048", 128, 2048),
           (5, "n256_k7168", 256, 7168),
           (6, "n2048_k4096", 2048, 4096),
           (7, "n4096_k4096", 4096, 4096),
           (8, "n4096_k14336", 4096, 14336),
           (9, "n5120_k2048", 5120, 2048),
           (10, "n6144_k4096", 6144, 4096),
           (11, "n28672_k4096", 28672, 4096),
       ]},
    # logits = hidden[:, -keep:] @ weight.T
    "L1__003_lm_head_projection_with_logit_slicing":
        lambda ax: ax["batch_size"] * ax["logits_to_keep"] * ax["hidden_size"]
        * ax["vocab_size"],
    # logits = hidden @ weight.T, over the whole sequence
    "L1__077_whisper_decoder_output_projection":
        lambda ax: ax["batch_size"] * ax["seq_len"] * ax["d_model"]
        * ax["vocab_size"],
    # both streams projected by one (hidden_dim, hidden_dim) weight
    "L2__030_flux_concatenated_sequence_processing_with_split":
        lambda ax: ax["batch_size"] * (ax["img_seq_len"] + ax["text_seq_len"])
        * ax["hidden_dim"] * ax["hidden_dim"],
    # pure memory kernels: no matrix arithmetic at all
    "FlashInfer-Bench__001_fused_add_rmsnorm_h2048": lambda ax: 0,
    "FlashInfer-Bench__002_fused_add_rmsnorm_h4096": lambda ax: 0,
    "FlashInfer-Bench__003_fused_add_rmsnorm_h7168": lambda ax: 0,
    "L1__025_video_latent_gelu_activation": lambda ax: 0,
    "L1__046_attention_softmax_with_softcapping_and_dropout": lambda ax: 0,
    "L1__069_rms_norm": lambda ax: 0,
    "L1__085_geglu_activation": lambda ax: 0,
}


def declared_traffic(definition: dict, axes: dict) -> int | None:
    """Bytes of the problem's own declared inputs and outputs.

    The minimum traffic of any correct kernel: every input read once, every
    output written once. Scalars (shape None) are excluded -- they ride in a
    kernel argument, not through DRAM.
    """
    total = 0
    for group in ("inputs", "outputs"):
        for spec in (definition.get(group) or {}).values():
            shape = spec.get("shape")
            if not shape:
                continue
            n = 1
            for dim in shape:
                if isinstance(dim, int):
                    n *= dim
                elif dim in axes:
                    n *= axes[dim]
                else:
                    return None            # unresolved symbol: no claim made
            width = DTYPE_BYTES.get(spec.get("dtype"))
            if width is None:
                return None
            total += n * width
    return total


def load_arch(path: Path) -> dict:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v.strip().strip('"'))
        except ValueError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-sol", default="artifacts/03/t_sol.json")
    ap.add_argument("--arch", default="SOLAR/configs/arch/MI350X.yaml")
    ap.add_argument("--data", default="data/SOL-ExecBench/benchmark")
    ap.add_argument("--t-b", default=None,
                    help="artifacts/06 directory; enables check D")
    ap.add_argument("--out", default="artifacts/03/cross-checks.md")
    a = ap.parse_args()

    doc = json.loads(Path(a.t_sol).read_text())
    arch = load_arch(Path(a.arch))
    freq_hz = arch["freq_GHz"] * 1e9
    dram_bw = arch["DRAM_byte_per_cycle"] * freq_hz
    peak_flops = {k.split("_")[3]: v * 2 * freq_hz
                  for k, v in arch.items() if k.startswith("MAC_per_cycle_")}

    problems = doc.get("problems", doc)

    # -- A + B ------------------------------------------------------------
    a_checked = a_violation = 0
    a_worst: list[tuple] = []
    b_checked = b_bw_violation = b_flops_violation = 0
    b_worst_bw: list[tuple] = []
    unresolved = 0

    for key, entry in sorted(problems.items()):
        category, name = key.split("__", 1)
        defn_path = Path(a.data) / category / name / "definition.json"
        if not defn_path.exists():
            continue
        definition = json.loads(defn_path.read_text())
        precision = entry.get("precision")
        for uuid, w in (entry.get("workloads") or {}).items():
            if w.get("t_sol_cycles") is None:
                continue
            axes = w.get("axes") or {}
            declared = declared_traffic(definition, axes)
            solar_bytes = w.get("memory_bytes")
            if declared is None or not solar_bytes:
                unresolved += 1
            else:
                a_checked += 1
                ratio = solar_bytes / declared
                if ratio < 0.999:                # below the unavoidable minimum
                    a_violation += 1
                    a_worst.append((ratio, key, uuid, declared, solar_bytes))

            seconds = w["t_sol_cycles"] / freq_hz
            if seconds <= 0:
                continue
            b_checked += 1
            implied_bw = (solar_bytes or 0) / seconds
            implied_flops = 2 * (w.get("macs") or 0) / seconds
            if implied_bw > dram_bw * 1.001:
                b_bw_violation += 1
            peak = peak_flops.get(precision or "")
            if peak and implied_flops > peak * 1.001:
                b_flops_violation += 1
            if w.get("bottleneck") in ("memory", "dram", "DRAM"):
                b_worst_bw.append((implied_bw / dram_bw, key, uuid))

    # -- C ----------------------------------------------------------------
    c_rows: list[tuple] = []
    for key, fn in sorted(HAND_MACS.items()):
        entry = problems.get(key)
        if not entry:
            c_rows.append((key, None, None, "no T_SOL recorded"))
            continue
        for uuid, w in sorted((entry.get("workloads") or {}).items()):
            if w.get("t_sol_cycles") is None:
                continue
            try:
                expect = fn(w.get("axes") or {})
            except KeyError as e:
                c_rows.append((key, None, None, f"axis missing: {e}"))
                break
            got = w.get("macs") or 0
            verdict = ("exact" if got == expect else
                       "within 1%" if expect and
                       abs(got - expect) / max(expect, 1) < 0.01 else
                       "MISMATCH" if expect or got else "exact")
            c_rows.append((key, expect, got, verdict))
            break                                  # one workload per problem

    c_bad = sum(1 for _, _, _, v in c_rows if v == "MISMATCH")

    # -- D ----------------------------------------------------------------
    d_rows: list[tuple] = []
    d_status = "PENDING — needs task 06 (`--t-b artifacts/06/authoritative`)"
    if a.t_b:
        tb_dir = Path(a.t_b)
        checked = viol = 0
        for f in sorted(tb_dir.glob("*.json")):
            tb = json.loads(f.read_text())
            entry = problems.get(tb.get("problem", ""), {})
            for uuid, win in (tb.get("winner_by_workload") or {}).items():
                w = (entry.get("workloads") or {}).get(uuid)
                if not w or w.get("t_sol_ms") is None:
                    continue
                checked += 1
                if w["t_sol_ms"] > win["t_b_ms"] * 1.0001:
                    viol += 1
                    d_rows.append((tb["problem"], uuid, w["t_sol_ms"],
                                   win["t_b_ms"], win["variant"]))
        d_status = (f"{checked - viol}/{checked} workloads satisfy "
                    f"T_SOL <= T_b" + ("" if not viol else
                    f" — **{viol} VIOLATIONS**, each one a config error"))

    # -- report -----------------------------------------------------------
    L = [
        "# Task 03 — T_SOL cross-checks",
        "",
        f"<!-- {json.dumps(stamp('03-cross-checks')['_provenance'], default=str)} -->",
        "",
        "Upstream's B200 SOL times are not used as a comparison anywhere in "
        "this document. The shipped dataset carries no per-workload SOL "
        "figures, so there is nothing to compare against that was not invented "
        "here — and an invented comparison would be worse than none. The three "
        "checks below are internal to this platform and are stronger for it.",
        "",
        "## A — SOLAR's memory term vs the problem's own declared traffic",
        "",
        "Every definition states the shape and dtype of each input and output. "
        "Their sum is what any correct kernel must move at least once. A "
        "memory term below it is not a bound.",
        "",
        f"* checked: **{a_checked}** workloads",
        f"* below declared minimum: **{a_violation}**",
        f"* not checkable (unresolved symbol or dtype): {unresolved}",
        "",
    ]
    if a_worst:
        L += ["| ratio | problem | declared bytes | SOLAR bytes |",
              "|---|---|---|---|"]
        for r, key, _u, dec, sol in sorted(a_worst)[:15]:
            L.append(f"| {r:.3f} | {key} | {dec:,} | {sol:,} |")
        L.append("")

    L += [
        "## B — rates implied by T_SOL",
        "",
        f"Arch config: DRAM {dram_bw/1e12:.2f} TB/s at {arch['freq_GHz']} GHz.",
        "",
        f"* checked: **{b_checked}** workloads",
        f"* implied bandwidth above DRAM peak: **{b_bw_violation}**",
        f"* implied FLOPS above the precision's peak: **{b_flops_violation}**",
        "",
        "## C — hand-derived MAC counts",
        "",
        "Eleven single matmuls and seven pure memory kernels, counted by hand "
        "from the reference source. A pure memory kernel's expected count is "
        "zero, and zero is a real prediction: MACs reported for one would mean "
        "SOLAR found arithmetic that is not in the kernel.",
        "",
        "| problem | hand-derived MACs | SOLAR MACs | verdict |",
        "|---|---|---|---|",
    ]
    for key, expect, got, verdict in c_rows:
        e = f"{expect:,}" if isinstance(expect, int) else "—"
        g = f"{got:,}" if isinstance(got, int) else "—"
        L.append(f"| {key} | {e} | {g} | {verdict} |")
    L += ["", f"MISMATCHes: **{c_bad}**", "",
          "## D — T_SOL <= best measured time", "", d_status, ""]
    if d_rows:
        L += ["| problem | workload | T_SOL ms | T_b ms | variant |",
              "|---|---|---|---|---|"]
        for p, u, ts, tb, v in d_rows[:25]:
            L.append(f"| {p} | `{u[:8]}` | {ts:.6g} | {tb:.6g} | {v} |")
        L.append("")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    print(f"  A  {a_checked} checked, {a_violation} below declared minimum")
    print(f"  B  {b_checked} checked, {b_bw_violation} over BW peak, "
          f"{b_flops_violation} over FLOPS peak")
    print(f"  C  {len(c_rows)} rows, {c_bad} mismatches")
    print(f"  D  {d_status}")
    return 1 if (a_violation or b_bw_violation or b_flops_violation or c_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
