#!/usr/bin/env python3
"""Adversarial alternative-cause probe for L1__062 workload 6c293638."""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs
import torch

UUID = "6c293638-c56b-593a-97e8-07d715d154ba"
PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/062_kv_cache_update_with_rope_backward")
NAMES = ["grad_key_states","grad_value_states","grad_cos","grad_sin",
         "grad_key_cache_input","grad_value_cache_input"]

definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
print(f"axes={dict(w.axes)} atol={atol:.6e} rtol={rtol:.6e} "
      f"req_mr={w.tolerance.required_matched_ratio}")

ref_run, ref_ns = exec_reference(definition)
ns2 = {}; exec(compile(definition.reference, "<r>", "exec"), ns2)
cmp_run = torch.compile(ns2["run"], dynamic=False)

def h(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

# ---- A. are the inputs the two paths see bit-identical?
torch.manual_seed(0); ia = prepare_inputs(definition, w, ref_ns, device="cuda:0")
torch.manual_seed(0); ib = prepare_inputs(definition, w, ns2,   device="cuda:0")
print("\n[A] input identity ref_ns vs cmp_ns:",
      all(torch.equal(x,y) for x,y in zip(ia,ib)), [h(x) for x in ia])

# ---- B. does the compiled fn mutate its inputs? (functionalisation bug)
torch.manual_seed(0); ic = prepare_inputs(definition, w, ns2, device="cuda:0")
before = [h(x) for x in ic]
out_c = cmp_run(*ic); torch.cuda.synchronize()
after = [h(x) for x in ic]
print("[B] compiled mutated any input:", before != after, "| eq:", before==after)
torch.manual_seed(0); id_ = prepare_inputs(definition, w, ref_ns, device="cuda:0")
b2=[h(x) for x in id_]; out_e = ref_run(*id_); torch.cuda.synchronize(); a2=[h(x) for x in id_]
print("[B] eager    mutated any input:", b2 != a2)

# ---- C. determinism of each path (2 runs, same inputs)
torch.manual_seed(0); i2 = prepare_inputs(definition, w, ns2, device="cuda:0")
out_c2 = cmp_run(*i2); torch.cuda.synchronize()
print("[C] compiled run-to-run bit-identical:",
      all(torch.equal(a,b) for a,b in zip(out_c, out_c2)))
torch.manual_seed(0); i3 = prepare_inputs(definition, w, ref_ns, device="cuda:0")
out_e2 = ref_run(*i3); torch.cuda.synchronize()
print("[C] eager    run-to-run bit-identical:",
      all(torch.equal(a,b) for a,b in zip(out_e, out_e2)))

# ---- D. per-output diff + dtype check
print("\n[D] per-output (compiled vs eager)")
for n, g, r in zip(NAMES, out_c, out_e):
    d = (g.float()-r.float()).abs()
    bad = (d > (atol + rtol*r.float().abs()))
    print(f"   {n:24s} dtype_c={str(g.dtype):16s} dtype_e={str(r.dtype):16s} "
          f"numel={r.numel():8d} ndiff={int((g!=r).sum())::>8} nbad={int(bad.sum()):5d} "
          f"max|d|={float(d.max()):.6e} mr={1-int(bad.sum())/r.numel():.6f}")
