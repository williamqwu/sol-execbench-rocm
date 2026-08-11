#!/usr/bin/env python3
"""Controls: emulate_precision_casts, tf32-under-compile, and the end-to-end
contribution of Inductor's fp32 cumsum (a cause independent of bf16 upcast)."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners")
sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs  # noqa
import torch

PROB = Path("/work/data/SOL-ExecBench/benchmark/L2/058_mamba2_selective_scan")
UUID = os.environ.get("ADV_UUID", "4c88b9e7")

SILU_ORIG = "hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu"
SILU_F32_KEEP = ("_c32 = conv_out.float()\n    hidden_B_C = "
                 "(_c32 * torch.sigmoid(_c32)).transpose(1, 2)  # silu fp32 kept")
SP_ORIG = "dt = F.softplus(dt + dt_bias)"
SP_F32_KEEP = "dt = F.softplus(dt.float() + dt_bias.float())"
GATE_ORIG = "y = y_normed * (gate * torch.sigmoid(gate))  # silu gate"
GATE_F32 = ("_g32 = gate.float()\n    y = (y_normed.float() * (_g32 * torch.sigmoid(_g32)))"
            ".to(y_normed.dtype)  # silu gate")
CS1_ORIG = "A_cumsum = torch.cumsum(A_perm, dim=-1)"
CS1_IND = "A_cumsum = _IND_CUMSUM(A_perm, -1)"
CS2_ORIG = "tensor_segsum = torch.cumsum(x_masked, dim=-2)"
CS2_IND = "tensor_segsum = _IND_CUMSUM(x_masked, -2)"


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
    return dict(max_abs=d.max().item(), matched_ratio=mr,
                frac_exact=(x == y).float().mean().item(),
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

    cfn = torch.compile(make_run(src)[0], dynamic=False)
    out_c = cfn(*ins); torch.cuda.synchronize()
    print("BASELINE compiled vs eager:", json.dumps(cmp(out_c, out_e, atol, rtol)))

    # compiled with tf32 on -- does the compiled path route fp32 gemms to xf32?
    torch.backends.cuda.matmul.allow_tf32 = True
    out_c_tf32 = torch.compile(make_run(src)[0], dynamic=False)(*ins)
    torch.cuda.synchronize()
    print("compiled(tf32=True) vs compiled(tf32=False):",
          json.dumps(cmp(out_c_tf32, out_c, atol, rtol)))
    torch.backends.cuda.matmul.allow_tf32 = False
    del out_c_tf32
    torch.cuda.empty_cache()

    # Inductor's own fp32 cumsum, injected into an otherwise-eager reference.
    _ind_cs = torch.compile(lambda t, d: torch.cumsum(t, dim=d), dynamic=False)
    def IND_CUMSUM(t, d):
        return _ind_cs(t, d)

    cases = {
        "V4 bf16-upcast emulation only":
            [(SILU_ORIG, SILU_F32_KEEP), (SP_ORIG, SP_F32_KEEP), (GATE_ORIG, GATE_F32)],
        "V7 inductor-cumsum only":
            [(CS1_ORIG, CS1_IND), (CS2_ORIG, CS2_IND)],
        "V8 upcast + inductor-cumsum":
            [(SILU_ORIG, SILU_F32_KEEP), (SP_ORIG, SP_F32_KEEP), (GATE_ORIG, GATE_F32),
             (CS1_ORIG, CS1_IND), (CS2_ORIG, CS2_IND)],
    }
    for name, subs in cases.items():
        vr, _ = make_run(variant(src, *subs), {"_IND_CUMSUM": IND_CUMSUM})
        ov = vr(*ins); torch.cuda.synchronize()
        print("%-32s vs_eager=%s  vs_compiled=%s"
              % (name, json.dumps(cmp(ov, out_e, atol, rtol)),
                 json.dumps(cmp(ov, out_c, atol, rtol))))
        del ov
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
