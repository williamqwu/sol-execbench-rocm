#!/usr/bin/env python3
"""Adversarial verification of the L2__009 torch.compile divergence claim.

Key design point vs the original diagnostic: eager and compiled are run on the
*same input tensor objects*, generated once. Input-generation RNG can therefore
not be the source of any measured difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/work")
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402

import torch  # noqa: E402


def thash(t):
    if not isinstance(t, torch.Tensor):
        return f"scalar:{t!r}"
    b = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def stats(got, ref, atol, rtol):
    x = got.detach().to(torch.float32)
    y = ref.detach().to(torch.float32)
    ae = (x - y).abs()
    bad = (ae > (atol + rtol * y.abs())) | ~torch.isfinite(ae)
    mr = 1.0 - float(bad.sum().item()) / ae.numel()
    finite = torch.isfinite(ae)
    return {
        "max_abs": float(ae[finite].max().item()) if finite.any() else float("nan"),
        "matched_ratio": mr,
        "bitwise_diff": int((x != y).sum().item()),
        "numel": int(ae.numel()),
        "verdict": "PASS" if mr >= 0.99 else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uuid", default="3de37835-4d55-5720-b5d4-993e4b89f98b")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    prob = ROOT / "data/SOL-ExecBench/benchmark/L2/009_decoder_layer_with_residual_connections"
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid == a.uuid][0]
    atol = float(w.tolerance.max_atol)
    rtol = float(w.tolerance.max_rtol)
    dev = "cuda:0"
    out = {"uuid": w.uuid, "axes": dict(w.axes), "atol": atol, "rtol": rtol,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    run_e, ns_e = exec_reference(definition)
    run_c_src, ns_c = exec_reference(definition)

    # ---- Test A: is input generation deterministic at all? -----------------
    torch.manual_seed(0)
    ins_a = prepare_inputs(definition, w, ns_e, device=dev)
    torch.manual_seed(0)
    ins_b = prepare_inputs(definition, w, ns_c, device=dev)
    hashes_a = [thash(t) for t in ins_a]
    hashes_b = [thash(t) for t in ins_b]
    out["input_gen_deterministic"] = hashes_a == hashes_b
    out["input_hashes_a"] = hashes_a
    out["input_hashes_b"] = hashes_b
    print("A. input gen deterministic across two manual_seed(0) calls:",
          out["input_gen_deterministic"], flush=True)

    # ---- Test B: eager vs compiled on the SAME tensors ---------------------
    o_e1 = run_e(*ins_a)
    torch.cuda.synchronize()
    o_e2 = run_e(*ins_a)   # same tensors, second eager call
    torch.cuda.synchronize()
    out["eager_vs_eager_same_tensors"] = stats(o_e2, o_e1, atol, rtol)
    print("B1. eager vs eager, SAME tensors:", out["eager_vs_eager_same_tensors"], flush=True)

    # inputs unmutated by run()?
    out["inputs_unmutated_after_eager"] = [thash(t) for t in ins_a] == hashes_a
    print("B2. inputs unmutated after two eager runs:",
          out["inputs_unmutated_after_eager"], flush=True)

    torch._dynamo.reset()
    cfn = torch.compile(ns_c["run"], dynamic=False)
    o_c = cfn(*ins_a)      # SAME tensors as eager
    torch.cuda.synchronize()
    out["compiled_vs_eager_same_tensors"] = stats(o_c, o_e1, atol, rtol)
    print("B3. compiled vs eager, SAME tensors:", out["compiled_vs_eager_same_tensors"], flush=True)

    out["inputs_unmutated_after_compiled"] = [thash(t) for t in ins_a] == hashes_a
    print("B4. inputs unmutated after compiled run:",
          out["inputs_unmutated_after_compiled"], flush=True)

    # ---- Test C: divergence site -------------------------------------------
    hs = ins_a[0]
    ilw = ins_a[4]
    eps = ins_a[-1]
    assert isinstance(eps, float), type(eps)

    def f_pow2(x, wgt, e):
        return x.to(torch.float32).pow(2)

    def f_mean(x, wgt, e):
        return x.to(torch.float32).pow(2).mean(-1, keepdim=True)

    def f_rsqrt(x, wgt, e):
        x = x.to(torch.float32)
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + e)

    def f_norm(x, wgt, e):
        return ns_e["rms_norm"](x, wgt, e)

    subops = {}
    for name, fn in [("pow2", f_pow2), ("mean", f_mean), ("rsqrt", f_rsqrt),
                     ("rms_norm_full", f_norm)]:
        torch._dynamo.reset()
        ce = fn(hs, ilw, eps)
        cc = torch.compile(fn, dynamic=False)(hs, ilw, eps)
        torch.cuda.synchronize()
        subops[name] = stats(cc, ce, atol, rtol)
        print(f"C. {name:14s}", subops[name], flush=True)
    out["subops"] = subops

    # ---- Test D: is the first-norm divergence SUFFICIENT? -------------------
    # Take the inductor-computed first rms_norm output, feed it into an
    # otherwise fully eager run, and see whether the final error matches what
    # the fully compiled run produced.
    torch._dynamo.reset()
    h1_compiled = torch.compile(f_norm, dynamic=False)(hs, ilw, eps)
    torch.cuda.synchronize()
    h1_eager = f_norm(hs, ilw, eps)
    out["h1_compiled_vs_eager"] = stats(h1_compiled, h1_eager, atol, rtol)

    real_rms = ns_e["rms_norm"]
    state = {"n": 0}

    def patched(x, wgt, e):
        state["n"] += 1
        if state["n"] == 1:
            return h1_compiled
        return real_rms(x, wgt, e)

    ns_e["rms_norm"] = patched
    o_perturbed = run_e(*ins_a)
    torch.cuda.synchronize()
    ns_e["rms_norm"] = real_rms
    out["rms_norm_calls_in_run"] = state["n"]
    out["perturbed_vs_eager"] = stats(o_perturbed, o_e1, atol, rtol)
    print("D. eager-with-inductor-first-norm vs pure eager:",
          out["perturbed_vs_eager"], flush=True)
    out["perturbed_vs_compiled"] = stats(o_perturbed, o_c, atol, rtol)
    print("D2. eager-with-inductor-first-norm vs fully compiled:",
          out["perturbed_vs_compiled"], flush=True)

    out["recompile_limit"] = torch._dynamo.config.recompile_limit
    print("recompile_limit =", out["recompile_limit"], flush=True)

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
