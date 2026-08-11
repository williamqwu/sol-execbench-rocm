#!/usr/bin/env python3
"""ADVERSARIAL: is the first compiled-vs-eager divergence really the SiLU
after the depthwise conv, and is the mechanism really Inductor's fp32
intermediate?

Minimal instrumentation: only three extra returns, all of which are already
materialised in the graph (mm output, extern conv output, the silu result that
feeds three slices), so fusion is barely perturbed. The end-to-end output is
returned too, and compared against the uninstrumented run.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402
import torch  # noqa: E402

PATCH_ANCHOR = "    hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu"
PATCH = PATCH_ANCHOR + "\n    _dbg_silu = hidden_B_C\n    _dbg_proj = projected\n    _dbg_conv = conv_out\n"


def instrument(src: str) -> str:
    assert PATCH_ANCHOR in src, "anchor line not found"
    src = src.replace(PATCH_ANCHOR + "\n", PATCH, 1)
    assert src.rstrip().endswith("return output")
    src = src.rstrip()[: -len("return output")] + \
        "return output, _dbg_proj, _dbg_conv, _dbg_silu\n"
    return src


def d(a, b, label):
    x, y = a.detach().to(torch.float32), b.detach().to(torch.float32)
    ae = (x - y).abs()
    scale = float(y.abs().max().item())
    print(f"{label:18s} dtype={str(a.dtype):15s} max_abs={float(ae.max().item()):.6e} "
          f"absmax={scale:.4e} rel_to_scale={float(ae.max().item())/scale if scale else 0:.4e} "
          f"frac_diff={float((x != y).float().mean().item()):.4f}")
    return float(ae.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
    ap.add_argument("--uuid", default="4c88b9e7")
    a = ap.parse_args()

    definition, workloads = load_problem(Path(a.problem))
    w = [x for x in workloads if x.uuid.startswith(a.uuid)][0]
    print(f"uuid={w.uuid} axes={dict(w.axes)}")

    plain_run, ns = exec_reference(definition)
    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ns, device="cuda:0")

    src_i = instrument(definition.reference)
    nsE: dict = {}; exec(compile(src_i, "<inst_eager>", "exec"), nsE)
    nsC: dict = {}; exec(compile(src_i, "<inst_cmp>", "exec"), nsC)
    eager_i = nsE["run"]
    cmp_i = torch.compile(nsC["run"], dynamic=False)

    oE = eager_i(*ins); torch.cuda.synchronize()
    oC = cmp_i(*ins); torch.cuda.synchronize()

    # uninstrumented, for the "did instrumentation change the answer" check
    plain_out = plain_run(*ins); torch.cuda.synchronize()
    nsP: dict = {}; exec(compile(definition.reference, "<p>", "exec"), nsP)
    plain_cmp = torch.compile(nsP["run"], dynamic=False)(*ins); torch.cuda.synchronize()

    print("\n-- stage-by-stage, compiled vs eager (both instrumented) --")
    d(oC[1], oE[1], "01_projected(mm)")
    d(oC[2], oE[2], "05_conv_out")
    d(oC[3], oE[3], "06_conv_silu")
    d(oC[0], oE[0], "34_output(inst)")
    print()
    d(plain_cmp, plain_out, "34_output(plain)")
    d(oE[0], plain_out, "eager inst-vs-plain")

    # ---- mechanism: which rounding policy does each side match, bit-exactly?
    conv = oE[2]
    assert torch.equal(conv, oC[2]), "conv_out not bit-identical; mechanism test invalid"
    silu_bf16 = (conv * torch.sigmoid(conv))                      # eager policy
    silu_fp32 = (conv.float() * torch.sigmoid(conv.float())).to(conv.dtype)  # inductor policy
    # instrumented stages carry the .transpose(1,2); undo for comparison
    eS = oE[3].transpose(1, 2)
    cS = oC[3].transpose(1, 2)
    print("\n-- mechanism (bit-exact identity tests) --")
    print("eager06   == bf16-intermediate silu :", torch.equal(eS, silu_bf16))
    print("eager06   == fp32-intermediate silu :", torch.equal(eS, silu_fp32))
    print("compiled06== bf16-intermediate silu :", torch.equal(cS, silu_bf16))
    print("compiled06== fp32-intermediate silu :", torch.equal(cS, silu_fp32))
    diff = (silu_bf16.float() - silu_fp32.float()).abs()
    print(f"bf16-vs-fp32 policy: max_abs={float(diff.max()):.6e} "
          f"frac_diff={float((silu_bf16 != silu_fp32).float().mean()):.4f}")


if __name__ == "__main__":
    main()
