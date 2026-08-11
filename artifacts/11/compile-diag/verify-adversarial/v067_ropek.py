#!/usr/bin/env python3
"""Follow-up: rope(k) measured 0.0 in the combined isolation run but the claim
reports 4.7684e-07. Fresh process, k compiled FIRST, and both q and k shapes
compiled in isolation from each other."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch, torch.nn.functional as F
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
H, KVH, D = 32, 8, 128
ORDER = sys.argv[1] if len(sys.argv) > 1 else "k_first"

definition, workloads = load_problem(PROB)
ref_run, ref_ns = exec_reference(definition)
w = [x for x in workloads if x.uuid.startswith("b0c05812")][0]
torch.manual_seed(0)
hs, cos, sin, qw, kw, vw, ow = prepare_inputs(definition, w, ref_ns, device="cuda:0")
B, S, _ = hs.shape
q = F.linear(hs, qw).view(B, S, H, D).transpose(1, 2)
k = F.linear(hs, kw).view(B, S, KVH, D).transpose(1, 2)
ce, se = cos.unsqueeze(1), sin.unsqueeze(1)


def rope(x, c, s):
    x1 = x[..., : D // 2]; x2 = x[..., D // 2:]
    return (x * c) + (torch.cat((-x2, x1), dim=-1) * s)


def report(tag, x):
    # separate compile object AND separate code object per call site
    src = "def rope2(x, c, s, D=128):\n    return (x*c) + (torch.cat((-x[...,D//2:], x[...,:D//2]), dim=-1)*s)\n"
    ns = {"torch": torch}
    exec(src, ns)
    f = ns["rope2"]
    e = f(x, ce, se)
    c = torch.compile(f, dynamic=False)(x, ce, se)
    torch.cuda.synchronize()
    ae = (c - e).abs()
    print(f"{tag:12s} shape={tuple(x.shape)} max_abs={float(ae.max()):.4e} "
          f"frac_diff={float((ae>0).float().mean()):.4f}", flush=True)


if ORDER == "k_first":
    report("rope(k)", k); report("rope(q)", q)
else:
    report("rope(q)", q); report("rope(k)", k)

# and via the shared-closure form used in the combined run
print("-- shared code object, q then k --", flush=True)
cr = torch.compile(rope, dynamic=False)
for tag, x in (("rope(q)", q), ("rope(k)", k)):
    e = rope(x, ce, se); c = cr(x, ce, se); torch.cuda.synchronize()
    ae = (c - e).abs()
    print(f"{tag:12s} max_abs={float(ae.max()):.4e} frac_diff={float((ae>0).float().mean()):.4f}",
          flush=True)
print("DONE")
