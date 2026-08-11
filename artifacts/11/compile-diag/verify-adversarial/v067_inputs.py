#!/usr/bin/env python3
"""ADVERSARIAL CHECK 1: is input generation deterministic, and is the
compiled-vs-eager difference measured on THE SAME tensors?

The repo diag script regenerates inputs three times (manual_seed(0) each).
If gen_inputs were RNG-order-sensitive or nondeterministic, the whole
measurement would be an artefact. Here we:
  (a) generate twice and compare BITWISE (torch.equal + bit pattern hash)
  (b) run eager and compiled on the IDENTICAL tensor objects (no regen at all)
  (c) run eager twice on identical objects, to isolate kernel nondeterminism
"""
from __future__ import annotations
import sys, hashlib, json
from pathlib import Path

sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
import torch
from _common import exec_reference, load_problem, prepare_inputs

PROB = Path("/work/data/SOL-ExecBench/benchmark/L1/067_flash_attention_gqa_ultralong")
UUIDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["b0c05812-9ac0-5ecb-a7a9-73edaf552dde"]
MODE = sys.argv[2] if len(sys.argv) > 2 else "default"


def bhash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def stats(g, r, atol, rtol):
    x, y = g.float(), r.float()
    ae = (x - y).abs()
    bad = (ae > (atol + rtol * y.abs())) | ~torch.isfinite(ae)
    return dict(max_abs=float(ae.max()), mr=1.0 - float(bad.sum()) / ae.numel(),
                nbad=int(bad.sum()), n=int(ae.numel()))


definition, workloads = load_problem(PROB)
ref_run, ref_ns = exec_reference(definition)

ns2: dict = {}
exec(compile(definition.reference, "<reference>", "exec"), ns2)
fn = ns2["run"]
if MODE == "eager":
    cmp_run = fn
elif MODE == "default":
    cmp_run = torch.compile(fn, dynamic=False)
else:
    cmp_run = torch.compile(fn, dynamic=False, mode="max-autotune-no-cudagraphs")

out = []
for w in workloads:
    if w.uuid[:8] not in {u[:8] for u in UUIDS}:
        continue
    atol = float(w.tolerance.max_atol); rtol = float(w.tolerance.max_rtol)

    torch.manual_seed(0)
    ins_a = prepare_inputs(definition, w, ref_ns, device="cuda:0")
    torch.manual_seed(0)
    ins_b = prepare_inputs(definition, w, ns2, device="cuda:0")   # different ns!
    # also generate a THIRD time after burning RNG state, seeded again
    _ = torch.randn(12345, device="cuda:0")
    torch.manual_seed(0)
    ins_c = prepare_inputs(definition, w, ref_ns, device="cuda:0")

    eq_ab = all(torch.equal(a, b) for a, b in zip(ins_a, ins_b))
    eq_ac = all(torch.equal(a, c) for a, c in zip(ins_a, ins_c))
    hashes = [bhash(t) for t in ins_a]
    hashes_b = [bhash(t) for t in ins_b]

    # (b) SAME tensor objects for both runs
    o_eager = ref_run(*ins_a); torch.cuda.synchronize()
    o_eager2 = ref_run(*ins_a); torch.cuda.synchronize()   # eager determinism on same input
    o_cmp = cmp_run(*ins_a); torch.cuda.synchronize()

    ee = stats(o_eager2, o_eager, atol, rtol)
    ce = stats(o_cmp, o_eager, atol, rtol)
    # cross-check: compiled on ins_b (identical values, different alloc)
    o_cmp_b = cmp_run(*ins_b); torch.cuda.synchronize()
    ce_b = stats(o_cmp_b, o_eager, atol, rtol)

    row = dict(uuid=w.uuid[:8], axes=dict(w.axes), atol=atol, rtol=rtol,
               inputs_bitwise_equal_across_namespaces=eq_ab,
               inputs_bitwise_equal_after_rng_burn=eq_ac,
               input_hashes=hashes, input_hashes_b=hashes_b,
               eager_vs_eager_SAME_INPUT_OBJECTS=ee,
               compiled_vs_eager_SAME_INPUT_OBJECTS=ce,
               compiled_on_regen_inputs=ce_b,
               compiled_bitwise_same_across_input_copies=bhash(o_cmp) == bhash(o_cmp_b))
    print(json.dumps(row, indent=1), flush=True)
    out.append(row)
print("DONE")
