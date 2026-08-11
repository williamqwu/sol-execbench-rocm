#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two questions for L2__058_mamba2_selective_scan.

(1) MECHANISM. Stage 06 `(conv_out * sigmoid(conv_out))` is the first
    divergence, and conv_out itself is bit-identical. Is the compiled value
    exactly the fp32-intermediate SiLU rounded once to bf16, and the eager
    value exactly the bf16-intermediate SiLU? If so the difference is
    Inductor's pointwise-in-fp32 policy and nothing else.

(2) GROWTH. Does the chunked scan amplify that perturbation along the
    sequence? Reported per chunk index for the inter-chunk recurrence output
    and per sequence position for the final output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work/artifacts/11/compile-diag")

import torch  # noqa: E402
from _common import load_problem, prepare_inputs, exec_reference  # noqa: E402
import localise_058 as L  # noqa: E402


def maxabs(a, b):
    x, y = a.detach().to(torch.float64), b.detach().to(torch.float64)
    return float((x - y).abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    prob = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid == a.uuid][0]
    _, ns = exec_reference(definition)
    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ns, device="cuda:0")

    L.CAPS.clear(); L.ORDER.clear()
    out_e = L.run(*ins); torch.cuda.synchronize()
    eager = {k: v.detach().clone() for k, v in L.CAPS.items()}

    L.CAPS.clear(); L.ORDER.clear()
    out_c = torch.compile(L.run, dynamic=False)(*ins); torch.cuda.synchronize()
    comp = {k: v.detach().clone() for k, v in L.CAPS.items()}

    doc = {"uuid": a.uuid, "axes": dict(w.axes), "torch": torch.__version__}

    # ---- (1) mechanism -------------------------------------------------
    conv_out = eager["05_conv_out"]                      # bit-identical, checked
    assert maxabs(conv_out, comp["05_conv_out"]) == 0.0
    silu_bf16 = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)
    cf = conv_out.float()
    silu_fp32 = (cf * torch.sigmoid(cf)).transpose(1, 2).to(conv_out.dtype)
    doc["mechanism"] = {
        "eager06_vs_bf16_intermediate_silu": maxabs(eager["06_conv_silu"], silu_bf16),
        "compiled06_vs_fp32_intermediate_silu": maxabs(comp["06_conv_silu"], silu_fp32),
        "compiled06_vs_bf16_intermediate_silu": maxabs(comp["06_conv_silu"], silu_bf16),
        "eager06_vs_fp32_intermediate_silu": maxabs(eager["06_conv_silu"], silu_fp32),
        "bf16_vs_fp32_intermediate_silu": maxabs(silu_bf16, silu_fp32),
        "frac_elems_where_bf16_and_fp32_silu_differ": float(
            ((silu_bf16.to(torch.float64) - silu_fp32.to(torch.float64)).abs() > 0)
            .double().mean()),
    }
    print("MECHANISM", json.dumps(doc["mechanism"], indent=1), flush=True)

    # ---- (2) growth along the scan -------------------------------------
    # 24_new_states: (batch, heads, chunks+1, head_dim, state); axis 2 is the
    # inter-chunk recurrence index -- chunk 0 is the zero initial state.
    ns_e, ns_c = eager["24_new_states"], comp["24_new_states"]
    per_chunk = []
    for c in range(ns_e.shape[2]):
        e = ns_e[:, :, c].to(torch.float64)
        cc = ns_c[:, :, c].to(torch.float64)
        scale = float(e.abs().max())
        m = float((e - cc).abs().max())
        per_chunk.append({"chunk": c, "max_abs": m, "scale": scale,
                          "rel_to_scale": (m / scale) if scale else 0.0})
    doc["new_states_per_chunk"] = per_chunk
    print("\nchunk  max_abs      scale        rel_to_scale")
    for r in per_chunk:
        print(f"{r['chunk']:5d}  {r['max_abs']:.6e} {r['scale']:.6e} {r['rel_to_scale']:.6e}")

    # final output error per sequence position
    oe, oc = out_e.to(torch.float64), out_c.to(torch.float64)
    err = (oe - oc).abs().amax(dim=-1).amax(dim=0)      # (seq,)
    scale = oe.abs().amax(dim=-1).amax(dim=0)
    doc["output_per_position"] = {
        "seq_len": int(err.numel()),
        "max_abs_by_position_first16": [float(x) for x in err[:16]],
        "max_abs_by_position_last16": [float(x) for x in err[-16:]],
        "mean_abs_first_eighth": float(err[: err.numel() // 8].mean()),
        "mean_abs_last_eighth": float(err[-(err.numel() // 8):].mean()),
        "rel_mean_first_eighth": float((err / scale)[: err.numel() // 8].mean()),
        "rel_mean_last_eighth": float((err / scale)[-(err.numel() // 8):].mean()),
    }
    print("\nOUTPUT BY POSITION", json.dumps(doc["output_per_position"], indent=1))

    # per-chunk block of the output, mean error
    cs = 128
    nblk = err.numel() // cs
    blocks = []
    for i in range(nblk):
        sl = slice(i * cs, (i + 1) * cs)
        blocks.append({"chunk": i, "mean_abs": float(err[sl].mean()),
                       "max_abs": float(err[sl].max()),
                       "scale": float(scale[sl].max())})
    doc["output_per_chunk_block"] = blocks
    print("\nblk  mean_abs     max_abs      scale")
    for b in blocks:
        print(f"{b['chunk']:3d}  {b['mean_abs']:.6e} {b['max_abs']:.6e} {b['scale']:.6e}")

    Path(a.json_out).write_text(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()
