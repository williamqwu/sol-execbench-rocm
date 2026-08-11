#!/usr/bin/env python3
"""Independent float64 golden (computed on GPU, not CPU) for L2__058, to check
the claim that the compiled result is closer to the truth than eager."""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs  # noqa
import torch

PROB = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
UUID = os.environ.get("ADV_UUID", "4c88b9e7")


def make_run(src):
    ns = {}
    exec(compile(src, "<v>", "exec"), ns)
    return ns["run"], ns


def as_t(o):
    return o if isinstance(o, torch.Tensor) else o[0]


def stats(x, g, atol, rtol):
    """x: candidate (bf16), g: float64 golden."""
    xd = as_t(x).detach().to(torch.float64)
    d = (xd - g).abs()
    mr = 1.0 - (d > (atol + rtol * g.abs())).double().mean().item()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                rms=(d.pow(2).mean().sqrt()).item(), matched_ratio=mr)


def main():
    definition, workloads = load_problem(PROB)
    wl = [w for w in workloads if w.uuid.startswith(UUID)][0]
    atol, rtol = float(wl.tolerance.max_atol), float(wl.tolerance.max_rtol)
    print(f"# {wl.uuid[:8]} {dict(wl.axes)} atol={atol:.4e} rtol={rtol:.4e}")

    src = definition.reference
    run_eager, ns = exec_reference(definition)
    torch.manual_seed(0)
    ins = prepare_inputs(definition, wl, ns, device="cuda:0")
    out_e = run_eager(*ins); torch.cuda.synchronize()
    out_c = torch.compile(make_run(src)[0], dynamic=False)(*ins); torch.cuda.synchronize()

    # float64 version: every .float() becomes .to(torch.float64), inputs promoted.
    src64 = src.replace(".float()", ".to(torch.float64)")
    n = len(re.findall(r"\.to\(torch\.float64\)", src64))
    print(f"# .float() sites promoted to float64: {n}")
    run64, _ = make_run(src64)
    ins64 = [t.to(torch.float64) if isinstance(t, torch.Tensor) else t for t in ins]
    with torch.no_grad():
        g = as_t(run64(*ins64)).detach()
    torch.cuda.synchronize()
    print("# golden dtype", g.dtype, "absmax", float(g.abs().max()))
    del ins64
    torch.cuda.empty_cache()

    se = stats(out_e, g, atol, rtol)
    sc = stats(out_c, g, atol, rtol)
    print("eager    vs golden:", json.dumps({k: round(v, 8) for k, v in se.items()}))
    print("compiled vs golden:", json.dumps({k: round(v, 8) for k, v in sc.items()}))
    print("rms ratio eager/compiled = %.4f" % (se["rms"] / sc["rms"]))
    de = (as_t(out_e).to(torch.float64) - g).abs()
    dc = (as_t(out_c).to(torch.float64) - g).abs()
    print("elementwise: compiled strictly closer %.4f | eager strictly closer %.4f | tie %.4f"
          % ((dc < de).double().mean().item(), (de < dc).double().mean().item(),
             (de == dc).double().mean().item()))


if __name__ == "__main__":
    raise SystemExit(main())
