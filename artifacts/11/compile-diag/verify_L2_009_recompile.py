#!/usr/bin/env python3
"""Causal test of the 'dynamo stops compiling after recompile_limit' claim.

Runs N distinct shapes through ONE torch.compile object, with
`recompile_limit` set to a value we choose. If the claim is right, the
workloads past the limit come back bit-identical to eager (dynamo ran the
plain Python function) while the ones inside the limit diverge.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path("/work")
sys.path.insert(0, str(ROOT / "scripts" / "runners"))
sys.path.insert(0, str(ROOT / "src"))

from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402

import torch  # noqa: E402


def stats(got, ref, atol, rtol):
    x = got.detach().to(torch.float32)
    y = ref.detach().to(torch.float32)
    ae = (x - y).abs()
    bad = (ae > (atol + rtol * y.abs())) | ~torch.isfinite(ae)
    mr = 1.0 - float(bad.sum().item()) / ae.numel()
    return {"max_abs": float(ae.max().item()), "matched_ratio": mr,
            "bitwise_diff": int((x != y).sum().item()),
            "verdict": "PASS" if mr >= 0.99 else "FAIL"}


class Catch(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        try:
            self.msgs.append(record.getMessage())
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2, help="recompile_limit")
    ap.add_argument("--uuid", action="append", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    torch._dynamo.config.recompile_limit = a.limit
    cap = Catch()
    for name in ("torch._dynamo", "torch._dynamo.convert_frame",
                 "torch._dynamo.guards"):
        logging.getLogger(name).addHandler(cap)
        logging.getLogger(name).setLevel(logging.WARNING)

    prob = ROOT / "data/SOL-ExecBench/benchmark/L2/009_decoder_layer_with_residual_connections"
    definition, workloads = load_problem(prob)
    order = {u: i for i, u in enumerate(a.uuid)}
    sel = sorted([w for w in workloads if w.uuid in order], key=lambda w: order[w.uuid])

    run_e, ns_e = exec_reference(definition)
    _, ns_c = exec_reference(definition)
    torch._dynamo.reset()
    cfn = torch.compile(ns_c["run"], dynamic=False)

    rows = []
    for i, w in enumerate(sel):
        atol = float(w.tolerance.max_atol)
        rtol = float(w.tolerance.max_rtol)
        torch.manual_seed(0)
        ins = prepare_inputs(definition, w, ns_e, device="cuda:0")
        oe = run_e(*ins)
        torch.cuda.synchronize()
        oc = cfn(*ins)          # SAME tensors
        torch.cuda.synchronize()
        s = stats(oc, oe, atol, rtol)
        s.update(pos=i, uuid=w.uuid, axes=dict(w.axes))
        rows.append(s)
        print(f"pos={i} {w.uuid[:8]} {dict(w.axes)} -> max_abs={s['max_abs']:.3e} "
              f"mr={s['matched_ratio']:.6f} bitdiff={s['bitwise_diff']} {s['verdict']}",
              flush=True)
        del ins, oe, oc
        torch.cuda.empty_cache()

    warn = [m for m in cap.msgs if "recompile" in m.lower() or "cache_size" in m.lower()]
    print("dynamo warnings:", json.dumps(warn[:5], indent=1), flush=True)
    doc = {"recompile_limit": a.limit, "rows": rows, "dynamo_warnings": warn[:10],
           "torch": torch.__version__}
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
