#!/usr/bin/env python3
"""ADVERSARIAL CHECK 2: does ONE torch.compile object across N distinct shapes
silently fall back to eager at the 9th shape (config.recompile_limit = 8)?

Runs the first 9 workloads (by workload.jsonl file order) of L1__067 through a
single compile object, exactly as reference/tb-candidates/variants.py::_compile
does, and reports compiled-vs-eager per workload in order.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
import torch._dynamo as dynamo
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 9

definition, workloads = load_problem(PROB)
ref_run, ref_ns = exec_reference(definition)
ns2: dict = {}
exec(compile(definition.reference, "<reference>", "exec"), ns2)
cmp_run = torch.compile(ns2["run"], dynamic=False)

print("recompile_limit =", dynamo.config.recompile_limit, flush=True)

for i, w in enumerate(workloads[:N]):
    atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
    r = ref_run(*ins); torch.cuda.synchronize()
    c = cmp_run(*ins); torch.cuda.synchronize()
    ae = (c.float() - r.float()).abs()
    bad = (ae > (atol + rtol * r.float().abs())) | ~torch.isfinite(ae)
    mr = 1.0 - float(bad.sum()) / ae.numel()
    print(f"#{i+1} {w.uuid[:8]} bs={w.axes['batch_size']} s={w.axes['seq_len']} "
          f"atol={atol:.3e} c/e={float(ae.max()):.3e} mr={mr:.6f} "
          f"-> {'FAIL' if mr < 0.99 else 'pass'}", flush=True)
    del ins, r, c, ae, bad
    torch.cuda.empty_cache()
print("DONE")
