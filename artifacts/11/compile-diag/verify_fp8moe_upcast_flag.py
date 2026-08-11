import sys
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
from pathlib import Path
from _common import load_problem, prepare_inputs
import torch, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--upcast", default="1"); a = ap.parse_args()
from torch._inductor import config as icfg
icfg.triton.codegen_upcast_to_fp32 = (a.upcast == "1")
PROB = Path("/work/data/SOL-ExecBench/benchmark/Quant/004_fp8_moe_expert_linear")
UUID = "31856bae-d378-581b-9b0e-cdc02fbabe56"
definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
ns = {}; exec(compile(definition.reference, "<ref>", "exec"), ns)
torch.manual_seed(0); ins = prepare_inputs(definition, w, ns, device="cuda:0")
out_e = ns["run"](*ins); torch.cuda.synchronize()
ns2 = {}; exec(compile(definition.reference, "<ref>", "exec"), ns2)
comp = torch.compile(ns2["run"], dynamic=False)
torch.manual_seed(0); ins2 = prepare_inputs(definition, w, ns2, device="cuda:0")
out_c = comp(*ins2); torch.cuda.synchronize()
x = out_c.to(torch.float32); y = out_e.to(torch.float32)
d = (x - y).abs(); n = int((d > 0).sum())
atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
bad = d > (atol + rtol * y.abs())
mr = 1.0 - float(bad.sum()) / bad.numel()
print(f"codegen_upcast_to_fp32={icfg.triton.codegen_upcast_to_fp32} "
      f"n_diff={n}/{d.numel()} frac={n/d.numel():.6f} max_abs={float(d.max()):.6e} "
      f"mean_abs={float(d.mean()):.6e} matched_ratio={mr:.6f} "
      f"-> {'PASS' if mr>=0.99 else 'FAIL'}")
