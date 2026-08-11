#!/usr/bin/env python3
"""Spot-check of the claim's decisive number: eager's OWN matched_ratio against
a float64 CPU golden, under the workload's own tolerance. uuid b0c05812."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch, torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
definition, workloads = load_problem(PROB)
ref_run, ref_ns = exec_reference(definition)
w = [x for x in workloads if x.uuid.startswith("b0c05812")][0]
atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
torch.manual_seed(0)
ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")

eager = ref_run(*ins).double().cpu()
ns2: dict = {}
exec(compile(definition.reference, "<r>", "exec"), ns2)
comp = torch.compile(ns2["run"], dynamic=False)(*ins).double().cpu()

# float64 CPU golden: same graph, fp64 throughout
ins64 = [t.double().cpu() for t in ins]
golden = ref_run(*ins64)
print("golden dtype", golden.dtype)


def rep(tag, x):
    e = (x - golden).abs()
    bad = (e > (atol + rtol * golden.abs()))
    print(f"{tag:9s} max_abs={float(e.max()):.6e} mean_abs={float(e.mean()):.6e} "
          f"RMS={float((e**2).mean().sqrt()):.6e} "
          f"matched_ratio_vs_golden={1.0 - float(bad.float().mean()):.6f}", flush=True)


print(f"atol={atol:.6e} rtol={rtol:.6e} required_matched_ratio=0.99")
rep("eager", eager)
rep("compiled", comp)
ee = (eager - golden).abs(); ce = (comp - golden).abs()
print("compiled strictly closer: %.4f  tie: %.4f  eager closer: %.4f"
      % (float((ce < ee).float().mean()), float((ce == ee).float().mean()),
         float((ee < ce).float().mean())))
print("compiled/eager RMS ratio: %.6f" % (float((ce**2).mean().sqrt()) / float((ee**2).mean().sqrt())))
print("DONE")
