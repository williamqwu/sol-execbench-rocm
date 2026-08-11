#!/usr/bin/env python3
"""Decompose the compiled-vs-eager gap into (a) bf16 pointwise upcast and
(b) Inductor's fp32 transcendental implementations, by injecting each into an
otherwise-eager reference."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs  # noqa
import torch
import torch._dynamo as dynamo
dynamo.config.recompile_limit = 64
dynamo.config.accumulated_recompile_limit = 512

PROB = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
UUID = os.environ.get("ADV_UUID", "4c88b9e7")

SILU_ORIG = "hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu"
SILU_F32_KEEP = ("_c32 = conv_out.float()\n    hidden_B_C = "
                 "(_c32 * torch.sigmoid(_c32)).transpose(1, 2)  # silu fp32 kept")
SP_ORIG = "dt = F.softplus(dt + dt_bias)"
SP_F32_KEEP = "dt = F.softplus(dt.float() + dt_bias.float())"
NORMW_ORIG = "y_normed = y_normed * norm_weight"
NORMW_F32 = "y_normed = y_normed.float() * norm_weight.float()"
GATE_ORIG = "y = y_normed * (gate * torch.sigmoid(gate))  # silu gate"
GATE_F32 = ("_g32 = gate.float()\n    y = (y_normed * (_g32 * torch.sigmoid(_g32)))"
            ".to(gate.dtype)  # silu gate")
TAIL = [(NORMW_ORIG, NORMW_F32), (GATE_ORIG, GATE_F32)]
UPCAST = [(SILU_ORIG, SILU_F32_KEEP), (SP_ORIG, SP_F32_KEEP)] + TAIL
TRANS = [("torch.exp(", "_IND_EXP("), ("torch.rsqrt(", "_IND_RSQRT("),
         ("torch.sigmoid(", "_IND_SIGMOID(")]


def variant(src, *subs):
    out = src
    for a, b in subs:
        assert a in out, a[:50]
        out = out.replace(a, b)
    return out


def make_run(src, extra=None):
    ns = dict(extra or {})
    exec(compile(src, "<v>", "exec"), ns)
    return ns["run"], ns


def as_t(o):
    return o if isinstance(o, torch.Tensor) else o[0]


def cmp(a, b, atol, rtol):
    x, y = as_t(a).detach(), as_t(b).detach()
    d = (x.float() - y.float()).abs()
    mr = 1.0 - (d > (atol + rtol * y.float().abs())).float().mean().item()
    return dict(max_abs=round(d.max().item(), 8), matched_ratio=round(mr, 6),
                frac_exact=round((x == y).float().mean().item(), 6),
                verdict="PASS" if mr >= 0.99 else "FAIL")


def main():
    definition, workloads = load_problem(PROB)
    wl = [w for w in workloads if w.uuid.startswith(UUID)][0]
    atol, rtol = float(wl.tolerance.max_atol), float(wl.tolerance.max_rtol)
    import torch._inductor.config as ic
    print(f"# {wl.uuid[:8]} {dict(wl.axes)} atol={atol:.4e} rtol={rtol:.4e} "
          f"emulate_precision_casts={ic.emulate_precision_casts}")

    src = definition.reference
    run_eager, ns = exec_reference(definition)
    torch.manual_seed(0)
    ins = prepare_inputs(definition, wl, ns, device="cuda:0")
    out_e = run_eager(*ins); torch.cuda.synchronize()
    out_c = torch.compile(make_run(src)[0], dynamic=False)(*ins); torch.cuda.synchronize()
    print("BASELINE compiled vs eager:", json.dumps(cmp(out_c, out_e, atol, rtol)))

    _e = torch.compile(lambda t: torch.exp(t), dynamic=False)
    _r = torch.compile(lambda t: torch.rsqrt(t), dynamic=False)
    _s = torch.compile(lambda t: torch.sigmoid(t), dynamic=False)
    inj = {"_IND_EXP": _e, "_IND_RSQRT": _r, "_IND_SIGMOID": _s}
    # self-check: the injected ops really are different from aten
    t = torch.randn(1024, 1024, device="cuda:0") * 3
    for nm, f, ref in (("exp", _e, torch.exp), ("rsqrt", _r, torch.rsqrt),
                       ("sigmoid", _s, torch.sigmoid)):
        g, h = f(t.abs() if nm == "rsqrt" else t), ref(t.abs() if nm == "rsqrt" else t)
        print(f"# injected {nm}: frac_bit_exact_vs_aten="
              f"{(g == h).float().mean().item():.6f}")
    del t
    torch.cuda.empty_cache()

    cases = {
        "V11 full bf16-upcast emulation": UPCAST,
        "V12 inductor transcendentals only": TRANS,
        "V13 upcast + transcendentals": UPCAST + TRANS,
    }
    for name, subs in cases.items():
        vr, _ = make_run(variant(src, *subs), inj)
        ov = vr(*ins); torch.cuda.synchronize()
        print("%-36s vs_eager=%s vs_compiled=%s"
              % (name, json.dumps(cmp(ov, out_e, atol, rtol)),
                 json.dumps(cmp(ov, out_c, atol, rtol))))
        del ov
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
