#!/usr/bin/env python3
"""Adversarial check: for L1__074_fused_gated_mlp_silu (all-float32, no bf16,
no fp8), is torch.compile closer to or further from an exact (float64 CPU)
evaluation than eager?  Also localise where eager and compiled first diverge.

Writes nothing to the repo outside artifacts/11/compile-diag/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "runners"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PROB = "data/SOL-ExecBench/benchmark/L1/074_fused_gated_mlp_silu"


def ref_impl(hidden_states, gate_up_weight, down_weight):
    up = F.linear(hidden_states, gate_up_weight)
    gate, up2 = up.chunk(2, dim=-1)
    silu_gate = gate * torch.sigmoid(gate)
    up2 = up2 * silu_gate
    return F.linear(up2, down_weight)


def cmp_vs(a, g):
    a64 = a.detach().to("cpu", torch.float64)
    d = (a64 - g).abs()
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rms": float(torch.sqrt((d * d).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[3]
    definition, workloads = load_problem(root / PROB)
    w = next(x for x in workloads if x.uuid.startswith(a.uuid))
    atol = float(w.tolerance.max_atol)
    rtol = float(w.tolerance.max_rtol)

    ref_run, ref_ns = exec_reference(definition)
    ns2: dict = {}
    exec(compile(definition.reference, "<reference>", "exec"), ns2)
    cmp_run = torch.compile(ns2["run"], dynamic=False)

    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ref_ns, device=a.device)
    out_e = ref_run(*ins)
    torch.cuda.synchronize()

    torch.manual_seed(0)
    ins_c = prepare_inputs(definition, w, ns2, device=a.device)
    for x, y in zip(ins, ins_c):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y), "inputs differ between namespaces"
    out_c = cmp_run(*ins_c)
    torch.cuda.synchronize()

    # exact float64 CPU golden of the same mathematics
    h64, gu64, dw64 = (t.detach().to("cpu", torch.float64) for t in ins[:3])
    g = ref_impl(h64, gu64, dw64)

    res = {
        "uuid": w.uuid,
        "axes": dict(w.axes),
        "atol": atol,
        "rtol": rtol,
        "out_dtype": str(out_e.dtype),
        "eager_vs_golden": cmp_vs(out_e, g),
        "compiled_vs_golden": cmp_vs(out_c, g),
        "compiled_vs_eager": {
            "max_abs": float((out_c - out_e).abs().max()),
        },
    }
    de = (out_e.to("cpu", torch.float64) - g).abs()
    dc = (out_c.to("cpu", torch.float64) - g).abs()
    res["frac_elements_compiled_closer"] = float((dc < de).float().mean())
    res["frac_elements_eager_closer"] = float((de < dc).float().mean())
    res["mean_abs_ratio_compiled_over_eager"] = (
        res["compiled_vs_golden"]["mean_abs"] / res["eager_vs_golden"]["mean_abs"]
    )

    # --- stage localisation: which op diverges? -------------------------
    hs, guw, dw = ins[0], ins[1], ins[2]
    lin1 = torch.compile(lambda x, w_: F.linear(x, w_), dynamic=False)
    e1 = F.linear(hs, guw)
    c1 = lin1(hs, guw)
    torch.cuda.synchronize()
    res["stage_linear1"] = {
        "n_diff": int((e1 != c1).sum()),
        "numel": int(e1.numel()),
        "max_abs": float((e1 - c1).abs().max()),
    }

    gate, up2 = e1.chunk(2, dim=-1)
    act = torch.compile(lambda gg, uu: uu * (gg * torch.sigmoid(gg)), dynamic=False)
    e2 = up2 * (gate * torch.sigmoid(gate))
    c2 = act(gate, up2)
    torch.cuda.synchronize()
    res["stage_silu_mul"] = {
        "n_diff": int((e2 != c2).sum()),
        "numel": int(e2.numel()),
        "max_abs": float((e2 - c2).abs().max()),
    }

    # exact golden for the isolated activation, fed identical fp32 inputs
    g_act = (up2.to("cpu", torch.float64)
             * (gate.to("cpu", torch.float64)
                * torch.sigmoid(gate.to("cpu", torch.float64))))
    res["stage_silu_mul_eager_vs_golden"] = cmp_vs(e2, g_act)
    res["stage_silu_mul_compiled_vs_golden"] = cmp_vs(c2, g_act)

    # exact golden for linear1 alone
    g_lin1 = F.linear(hs.to("cpu", torch.float64), guw.to("cpu", torch.float64))
    res["stage_linear1_eager_vs_golden"] = cmp_vs(e1, g_lin1)
    res["stage_linear1_compiled_vs_golden"] = cmp_vs(c1, g_lin1)

    print(json.dumps(res, indent=2))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
