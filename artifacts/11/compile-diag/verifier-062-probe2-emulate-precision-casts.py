#!/usr/bin/env python3
"""Discriminate: elided bf16 downcast vs reduction order vs miscompile."""
import sys, os
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs
import torch, torch._inductor.config as icfg, torch._dynamo as dyn

UUID = "6c293638-c56b-593a-97e8-07d715d154ba"
PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/062_kv_cache_update_with_rope_backward")
NAMES = ["grad_key_states","grad_value_states","grad_cos","grad_sin",
         "grad_key_cache_input","grad_value_cache_input"]
definition, workloads = load_problem(PROB)
w = [x for x in workloads if x.uuid == UUID][0]
atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)
ref_run, ref_ns = exec_reference(definition)

def build():
    ns = {}; exec(compile(definition.reference, "<r>", "exec"), ns); return ns

def run_compiled(emulate):
    dyn.reset()
    icfg.emulate_precision_casts = emulate
    ns = build(); f = torch.compile(ns["run"], dynamic=False)
    torch.manual_seed(0); ins = prepare_inputs(definition, w, ns, device="cuda:0")
    o = f(*ins); torch.cuda.synchronize(); return [t.clone() for t in o]

torch.manual_seed(0); ie = prepare_inputs(definition, w, ref_ns, device="cuda:0")
out_e = [t.clone() for t in ref_run(*ie)]; torch.cuda.synchronize()

for emu in (False, True):
    oc = run_compiled(emu)
    print(f"\n=== emulate_precision_casts={emu} ===")
    for n,g,r in zip(NAMES,oc,out_e):
        d=(g.float()-r.float()).abs(); bad=(d>(atol+rtol*r.float().abs()))
        print(f"   {n:24s} ndiff={int((g!=r).sum()):7d} nbad={int(bad.sum()):4d} "
              f"max|d|={float(d.max()):.4e} bit_identical_to_eager={torch.equal(g,r)}")
    if emu: oc_emu = oc
    else:   oc_raw = oc

# ---- emulation models on grad_cos / grad_k1
torch.manual_seed(0); ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
gkc, gvc, ks, cos, sin, cp = ins
gksr = gkc[:, :, cp]
half = ks.shape[-1]//2
# grad_cos
prod_bf = gksr * ks                              # bf16 materialised (eager)
m_eager = prod_bf.float().sum(dim=1).to(torch.bfloat16)
m_fp32  = (gksr.float()*ks.float()).sum(dim=1).to(torch.bfloat16)
m_bf16acc = prod_bf.sum(dim=1, dtype=torch.bfloat16).to(torch.bfloat16)
gc_e, gc_c = out_e[2], oc_raw[2]
print("\n[E] grad_cos models")
print("   eager    == bf16-product-then-fp32-sum :", torch.equal(gc_e, m_eager))
print("   compiled == fp32-product-then-fp32-sum :", torch.equal(gc_c, m_fp32))
print("   compiled == bf16-product, bf16 accum   :", torch.equal(gc_c, m_bf16acc))
# reduction-order alternative: same bf16 products, reversed order
m_rev = torch.flip(prod_bf,[1]).float().sum(dim=1).to(torch.bfloat16)
print("   compiled == bf16-product, REVERSED order:", torch.equal(gc_c, m_rev),
      " (order-only diff vs eager:", int((m_rev!=m_eager).sum()), "elems )")
# pairwise-tree order on bf16 products
p = prod_bf.float()
tree = (((p[:,0]+p[:,1])+(p[:,2]+p[:,3])) + ((p[:,4]+p[:,5])+(p[:,6]+p[:,7]))).to(torch.bfloat16)
print("   compiled == bf16-product, TREE order    :", torch.equal(gc_c, tree))
print("   emu-cast compiled == eager grad_cos     :", torch.equal(oc_emu[2], gc_e))
