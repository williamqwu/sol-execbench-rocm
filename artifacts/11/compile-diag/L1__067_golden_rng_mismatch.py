#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
import torch
from _common import exec_reference, load_problem, prepare_inputs
PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == "b0c05812-9ac0-5ecb-a7a9-73edaf552dde"][0]
_, ns = exec_reference(definition)
torch.manual_seed(0); cpu = prepare_inputs(definition, w, ns, device="cpu")
torch.manual_seed(0); gpu = prepare_inputs(definition, w, ns, device="cuda:0")
print("seed-0 CPU inputs vs seed-0 CUDA inputs (this is what gen_golden.py vs")
print("calibrate_tolerance.py each generate):")
for n, a, b in zip(definition.inputs, cpu, gpu):
    d = (a.double() - b.cpu().double()).abs().max().item()
    print(f"  {n:16s} shape={tuple(a.shape)!s:22s} max_abs_diff={d:.6e}")
g = torch.load("/work/artifacts/golden/L1__067_flash_attention_gqa_ultralong.pt",
               map_location="cpu", weights_only=False)
e = g["b0c05812-9ac0-5ecb-a7a9-73edaf552dde"]
print("stored golden mode:", e.get("mode"), "n_outputs:", len(e["outputs"]))
go = e["outputs"][0]
print("stored golden out dtype/shape:", go.dtype, tuple(go.shape),
      "absmax", go.abs().max().item())
