import sys, re; from pathlib import Path
sys.path.insert(0,"/work/scripts/runners"); sys.path.insert(0,"/work/src")
from _common import exec_reference, load_problem, prepare_inputs
import torch
UUID="6c293638-c56b-593a-97e8-07d715d154ba"
PROB=Path("/work/data/SOL-ExecBench/benchmark/L1/062_kv_cache_update_with_rope_backward")
NAMES=["grad_key_states","grad_value_states","grad_cos","grad_sin","grad_key_cache_input","grad_value_cache_input"]
definition,wls=load_problem(PROB); w=[x for x in wls if x.uuid==UUID][0]
atol=float(w.tolerance.max_atol); rtol=float(w.tolerance.max_rtol)
ref_run,ref_ns=exec_reference(definition)
ns={}; exec(compile(definition.reference,"<r>","exec"),ns); f=torch.compile(ns["run"],dynamic=False)
torch.manual_seed(0); ie=prepare_inputs(definition,w,ref_ns,device="cuda:0"); oe=[t.clone() for t in ref_run(*ie)]
torch.manual_seed(0); ic=prepare_inputs(definition,w,ns,device="cuda:0"); oc=[t.clone() for t in f(*ic)]
torch.cuda.synchronize()
# golden: strip the final bf16 casts so fp64 stays fp64
src=definition.reference.replace(".to(torch.bfloat16)","")
gns={}; exec(compile(src,"<g>","exec"),gns)
g64=[t.detach().cpu().double() if t.is_floating_point() else t.detach().cpu() for t in ie]
og=gns["run"](*g64)
print("golden dtypes:",[str(t.dtype) for t in og])
print(f"{'output':24s} {'eager maxabs':>13s} {'cmp maxabs':>13s} {'eager RMS':>12s} {'cmp RMS':>12s}  closer(RMS)")
for n,e,c,g in zip(NAMES,oe,oc,og):
    g=g.double(); de=(e.detach().cpu().double()-g).abs(); dc=(c.detach().cpu().double()-g).abs()
    re_=de.pow(2).mean().sqrt().item(); rc=dc.pow(2).mean().sqrt().item()
    tag="tie" if re_==rc else ("compiled" if rc<re_ else "EAGER")
    print(f"{n:24s} {de.max().item():13.4e} {dc.max().item():13.4e} {re_:12.4e} {rc:12.4e}  {tag}")
e,c,g=oe[2].flatten(),oc[2].flatten(),og[2].double().flatten()
d=(c.float()-e.float()).abs(); bound=atol+rtol*e.float().abs()
for i in (d>bound).nonzero().flatten().tolist():
    print(f" idx {i:4d} eager={e[i].item():.8f} cmp={c[i].item():.8f} golden={g[i].item():.10f} "
          f"|e-g|={abs(e[i].item()-g[i].item()):.3e} |c-g|={abs(c[i].item()-g[i].item()):.3e}")
