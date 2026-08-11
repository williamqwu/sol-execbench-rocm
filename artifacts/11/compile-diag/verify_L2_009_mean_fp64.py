#!/usr/bin/env python3
"""Is the first-op divergence 'reassociation' (both equally right) or is one
side actually less accurate? fp64 adjudication of x.pow(2).mean(-1) alone."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/work")
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402

import torch  # noqa: E402

prob = ROOT / "data/SOL-ExecBench/benchmark/L2/009_decoder_layer_with_residual_connections"
definition, workloads = load_problem(prob)
w = [x for x in workloads if x.uuid.startswith("3de37835")][0]
_, ns = exec_reference(definition)
torch.manual_seed(0)
ins = prepare_inputs(definition, w, ns, device="cuda:0")
hs = ins[0]


def f_mean(x):
    return x.to(torch.float32).pow(2).mean(-1, keepdim=True)


torch._dynamo.reset()
m_e = f_mean(hs)
m_c = torch.compile(f_mean, dynamic=False)(hs)
torch.cuda.synchronize()
m_g = hs.double().pow(2).mean(-1, keepdim=True)          # fp64 on GPU
m_g_cpu = hs.cpu().double().pow(2).mean(-1, keepdim=True)  # fp64 on CPU, cross-check
print("fp64 gpu vs fp64 cpu max abs:", float((m_g.cpu() - m_g_cpu).abs().max()))

de = (m_e.double() - m_g).abs()
dc = (m_c.double() - m_g).abs()
res = {
    "n_rows": int(m_e.numel()),
    "bitwise_differing_eager_vs_compiled": int((m_e != m_c).sum().item()),
    "max_abs_eager_vs_compiled": float((m_e - m_c).abs().max().item()),
    "rms_err_eager_vs_fp64": float(de.pow(2).mean().sqrt().item()),
    "rms_err_compiled_vs_fp64": float(dc.pow(2).mean().sqrt().item()),
    "max_err_eager_vs_fp64": float(de.max().item()),
    "max_err_compiled_vs_fp64": float(dc.max().item()),
    "rows_closer_eager": int((de < dc).sum().item()),
    "rows_closer_compiled": int((dc < de).sum().item()),
    "rows_tied": int((dc == de).sum().item()),
    "mean_magnitude": float(m_g.mean().item()),
}
print(json.dumps(res, indent=2))
Path("/work/artifacts/11/compile-diag/verifier.L2_009.mean_fp64.3de37835.json").write_text(
    json.dumps(res, indent=2))
