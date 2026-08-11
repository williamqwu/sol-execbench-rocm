#!/usr/bin/env python3
"""Adversarial-verifier probe: (a) did Inductor actually compile anything, and
(b) what do the FAILING elements look like -- are they one-ULP-of-themselves
rounding differences, or small-magnitude elements carrying a typical-element
absolute error?

Usage: adv_gen_failing_elements.py <problem_dir> <uuid>
"""
import sys
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")

import torch  # noqa: E402
from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402


def as_tuple(o):
    if isinstance(o, (list, tuple)):
        return tuple(o)
    if isinstance(o, dict):
        return tuple(o[k] for k in sorted(o))
    return (o,)


def main():
    prob, uid = Path(sys.argv[1]).resolve(), sys.argv[2]
    definition, workloads = load_problem(prob)
    w = [x for x in workloads if x.uuid.startswith(uid)][0]
    atol = float(w.tolerance.max_atol)
    rtol = float(w.tolerance.max_rtol)

    ref_run, ref_ns = exec_reference(definition)
    ns2: dict = {}
    exec(compile(definition.reference, "<r>", "exec"), ns2)
    cmp_run = torch.compile(ns2["run"], dynamic=False)

    torch.manual_seed(0)
    out_ref = as_tuple(ref_run(*prepare_inputs(definition, w, ref_ns, device="cuda:0")))
    torch.cuda.synchronize()

    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    metrics.reset()
    counters.clear()
    torch.manual_seed(0)
    out_cmp = as_tuple(cmp_run(*prepare_inputs(definition, w, ns2, device="cuda:0")))
    torch.cuda.synchronize()
    gb = sum(counters["graph_break"].values()) if "graph_break" in counters else 0
    print(f"problem={prob.name} uuid={uid} atol={atol:.4e} rtol={rtol:.4e}")
    print(f"  inductor generated_kernel_count={metrics.generated_kernel_count} "
          f"graph_breaks={gb}")

    for i, (g, r) in enumerate(zip(out_cmp, out_ref)):
        if not isinstance(g, torch.Tensor) or not g.is_floating_point():
            if isinstance(g, torch.Tensor):
                ne = int((g != r).sum())
                print(f"  out[{i}] dtype={g.dtype} INTEGER/BOOL mismatching="
                      f"{ne}/{g.numel()} ({ne / g.numel():.4%})")
            continue
        x, y = g.float(), r.float()
        d = (x - y).abs()
        bound = atol + rtol * y.abs()
        bad = d > bound
        nb = int(bad.sum())
        rms = float((y.double() ** 2).mean().sqrt())
        # one ULP of the reference value, in the reference's own dtype
        ulp = torch.where(y.abs() > 0,
                          torch.exp2(torch.floor(torch.log2(y.abs().clamp(min=1e-30)))),
                          torch.ones_like(y)) * float(torch.finfo(r.dtype).eps)
        print(f"  out[{i}] dtype={r.dtype} n={y.numel()} rms={rms:.4e} "
              f"absmax={float(y.abs().max()):.4e}")
        print(f"        max_abs_diff={float(d.max()):.4e} "
              f"frac_diff={float((d > 0).float().mean()):.4f} "
              f"matched_ratio={1 - nb / y.numel():.6f}")
        if nb:
            fy, fd, fu = y.abs()[bad], d[bad], ulp[bad]
            print(f"        FAILING elements: n={nb} "
                  f"median|ref|={float(fy.median()):.4e} "
                  f"(= {float(fy.median()) / rms:.4f} x rms)  "
                  f"median|d|={float(fd.median()):.4e}  "
                  f"median |d|/ulp(ref)={float((fd / fu).median()):.2f}  "
                  f"max |d|/ulp(ref)={float((fd / fu).max()):.2f}")
            allr = (d / ulp)
            print(f"        ALL elements: median |d|/ulp(ref)="
                  f"{float(allr.median()):.3f} p99={float(allr.flatten().kthvalue(max(1, int(0.99 * allr.numel())))[0]):.3f}")


main()
