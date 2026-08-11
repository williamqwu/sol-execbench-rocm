#!/usr/bin/env python3
"""L1__062: localise the eager/compiled divergence and answer the accuracy
question against a float64 CPU golden."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, "/work/scripts/runners"); sys.path.insert(0, "/work/src")
from _common import exec_reference, load_problem, prepare_inputs  # noqa: E402
import torch  # noqa: E402

UUID = "6c293638-c56b-593a-97e8-07d715d154ba"
PROB = "/work/data/SOL-ExecBench/benchmark/L1/062_kv_cache_update_with_rope_backward"
NAMES = ["grad_key_states", "grad_value_states", "grad_cos", "grad_sin",
         "grad_key_cache_input", "grad_value_cache_input"]


def maxabs(a, b):
    return float((a.double() - b.double()).abs().max().item())


def bitexact(a, b):
    return bool(torch.equal(a, b))


def golden_f64(gkc, gvc, ks, cos, sin, cp):
    """float64 CPU golden, no intermediate rounding below fp64."""
    gkc = gkc.cpu().double(); gvc = gvc.cpu().double(); ks = ks.cpu().double()
    cos = cos.cpu().double(); sin = sin.cpu().double(); cp = cp.cpu()
    h = ks.shape[-1] // 2
    k1, k2 = ks[..., :h], ks[..., h:]
    krh = torch.cat((-k2, k1), dim=-1)
    ce, se = cos.unsqueeze(1), sin.unsqueeze(1)
    gksr = gkc[:, :, cp]; gvs = gvc[:, :, cp]
    a = gksr * ce; b = gksr * se
    gks = torch.cat([a[..., :h] + b[..., h:], a[..., h:] - b[..., :h]], dim=-1)
    gcos = (gksr * ks).sum(dim=1)
    gsin = (gksr * krh).sum(dim=1)
    gkci = gkc.clone(); gkci[:, :, cp] = 0
    gvci = gvc.clone(); gvci[:, :, cp] = 0
    return (gks, gvs, gcos, gsin, gkci, gvci)


def main():
    definition, workloads = load_problem(Path(PROB))
    w = [x for x in workloads if x.uuid == UUID][0]
    atol, rtol = float(w.tolerance.max_atol), float(w.tolerance.max_rtol)
    ref_run, ref_ns = exec_reference(definition)

    torch.manual_seed(0)
    ins = prepare_inputs(definition, w, ref_ns, device="cuda:0")
    gkc, gvc, ks, cos, sin, cp = ins
    out_eager = ref_run(*ins)
    torch.cuda.synchronize()

    torch._dynamo.reset()
    ns2 = {}; exec(compile(definition.reference, "<r>", "exec"), ns2)
    cfn = torch.compile(ns2["run"], dynamic=False)
    torch.manual_seed(0)
    ins2 = prepare_inputs(definition, w, ns2, device="cuda:0")
    assert all(torch.equal(a, b) for a, b in zip(ins, ins2)), "inputs differ!"
    out_cmp = cfn(*ins2)
    torch.cuda.synchronize()

    print("=" * 78)
    print("STEP-BY-STEP LOCALISATION  (workload %s, seed 0)" % UUID[:8])
    print("=" * 78)
    h = ks.shape[-1] // 2
    ce, se = cos.unsqueeze(1), sin.unsqueeze(1)

    # ---- intermediate 1: the gather --------------------------------------
    gksr_e = gkc[:, :, cp]
    torch._dynamo.reset()
    f1 = torch.compile(lambda g, c: g[:, :, c], dynamic=False)
    gksr_c = f1(gkc, cp)
    print("1  grad_key_states_rotated = grad_key_cache[:,:,cache_position]")
    print("     dtype=%s  eager-vs-compiled max|d| = %.6e  bit-exact=%s"
          % (gksr_e.dtype, maxabs(gksr_c, gksr_e), bitexact(gksr_c, gksr_e)))

    # ---- intermediate 2: the bf16 elementwise products --------------------
    a_e = gksr_e * ce
    torch._dynamo.reset()
    f2 = torch.compile(lambda g, c, cp_: g[:, :, cp_] * c, dynamic=False)
    a_c = f2(gkc, ce, cp)
    print("2  grad_from_cos_term = grad_key_states_rotated * cos_expanded")
    print("     dtype=%s  eager-vs-compiled max|d| = %.6e  bit-exact=%s"
          % (a_e.dtype, maxabs(a_c, a_e), bitexact(a_c, a_e)))

    b_e = gksr_e * se
    # ---- intermediate 3: the ADD of two bf16 products ---------------------
    g1_e = a_e[..., :h] + b_e[..., h:]

    def _add(g, c, s, cp_, h_):
        r = g[:, :, cp_]
        return (r * c)[..., :h_] + (r * s)[..., h_:]
    torch._dynamo.reset()
    f3 = torch.compile(_add, dynamic=False)
    g1_c = f3(gkc, ce, se, cp, h)
    # emulation A: bf16 intermediates rounded (what eager does)
    g1_bf16 = ((gksr_e.float() * ce.float()).to(torch.bfloat16).float()[..., :h]
               + (gksr_e.float() * se.float()).to(torch.bfloat16).float()[..., h:]
               ).to(torch.bfloat16)
    # emulation B: fp32 throughout, single rounding at the end
    g1_fp32 = ((gksr_e.float() * ce.float())[..., :h]
               + (gksr_e.float() * se.float())[..., h:]).to(torch.bfloat16)
    print("3  grad_k1_total = grad_from_cos_term[...,:h] + grad_k_rotated_half[...,h:]")
    print("     eager-vs-compiled max|d| = %.6e   n_differ=%d/%d"
          % (maxabs(g1_c, g1_e), int((g1_c != g1_e).sum().item()), g1_e.numel()))
    print("     eager    == bf16-rounded-intermediate emulation : %s"
          % bitexact(g1_e, g1_bf16))
    print("     compiled == fp32-throughout emulation           : %s"
          % bitexact(g1_c, g1_fp32))

    # ---- intermediate 4: the reduction ------------------------------------
    gcos_e = (gksr_e * ks).sum(dim=1)

    def _gcos(g, k, cp_):
        return (g[:, :, cp_] * k).sum(dim=1)
    torch._dynamo.reset()
    f4 = torch.compile(_gcos, dynamic=False)
    gcos_c = f4(gkc, ks, cp)
    gcos_bf16 = ((gksr_e.float() * ks.float()).to(torch.bfloat16)
                 .float().sum(dim=1)).to(torch.bfloat16)
    gcos_fp32 = ((gksr_e.float() * ks.float()).sum(dim=1)).to(torch.bfloat16)
    dif = (gcos_c != gcos_e)
    print("4  grad_cos = (grad_key_states_rotated * key_states).sum(dim=1)   "
          "[reduction over num_kv_heads=8]")
    print("     eager-vs-compiled max|d| = %.6e   n_differ=%d/%d"
          % (maxabs(gcos_c, gcos_e), int(dif.sum().item()), gcos_e.numel()))
    print("     eager    == bf16-product-then-sum emulation : %s"
          % bitexact(gcos_e, gcos_bf16))
    print("     compiled == fp32-product-then-sum emulation : %s"
          % bitexact(gcos_c, gcos_fp32))

    # ---- full-graph outputs -----------------------------------------------
    print()
    print("=" * 78)
    print("FULL GRAPH: which outputs differ (eager vs compiled)")
    print("=" * 78)
    for n, e, c in zip(NAMES, out_eager, out_cmp):
        nd = int((e != c).sum().item())
        print("   %-22s n=%-8d n_differ=%-6d max|d|=%.6e"
              % (n, e.numel(), nd, maxabs(c, e)))

    # ---- float64 CPU golden ------------------------------------------------
    print()
    print("=" * 78)
    print("ACCURACY vs float64 CPU GOLDEN")
    print("=" * 78)
    gold = golden_f64(gkc, gvc, ks, cos, sin, cp)
    hdr = ("   %-22s %-12s %-12s %-12s %-12s %s"
           % ("output", "eager_maxabs", "cmp_maxabs", "eager_rms", "cmp_rms", "closer"))
    print(hdr)
    summary = []
    for n, e, c, g in zip(NAMES, out_eager, out_cmp, gold):
        ge = e.detach().cpu().double(); gc = c.detach().cpu().double()
        de = (ge - g).abs(); dc = (gc - g).abs()
        rmse = float((de ** 2).mean().sqrt().item())
        rmsc = float((dc ** 2).mean().sqrt().item())
        closer = ("compiled" if rmsc < rmse else
                  ("eager" if rmse < rmsc else "tie"))
        print("   %-22s %-12.4e %-12.4e %-12.4e %-12.4e %s"
              % (n, de.max().item(), dc.max().item(), rmse, rmsc, closer))
        summary.append({"output": n, "eager_max_abs_vs_golden": de.max().item(),
                        "compiled_max_abs_vs_golden": dc.max().item(),
                        "eager_rms_vs_golden": rmse,
                        "compiled_rms_vs_golden": rmsc, "closer": closer})

    # the 3 elements of grad_cos that break the check
    print()
    print("=" * 78)
    print("THE 3 grad_cos ELEMENTS THAT FAIL THE HARNESS CHECK")
    print("=" * 78)
    e = out_eager[2].detach().float().flatten()
    c = out_cmp[2].detach().float().flatten()
    g = gold[2].flatten()
    bad = ((c - e).abs() > (atol + rtol * e.abs())).nonzero().flatten()
    print("   atol=%.6e  rtol=%.6e  n_bad=%d/%d  matched_ratio=%.6f"
          % (atol, rtol, bad.numel(), e.numel(), 1 - bad.numel() / e.numel()))
    print("   %-6s %-14s %-14s %-16s %-12s %-12s"
          % ("idx", "eager(bf16)", "compiled(bf16)", "golden(fp64)",
             "|cmp-eag|", "tol_bound"))
    for i in bad.tolist():
        print("   %-6d %-14.8f %-14.8f %-16.10f %-12.4e %-12.4e"
              % (i, e[i].item(), c[i].item(), g[i].item(),
                 abs(c[i].item() - e[i].item()), atol + rtol * abs(e[i].item())))
        # show the 8 head terms so cancellation is visible
        terms = (gkc[:, :, cp] * ks).float().flatten(0, 1)  # (heads, seq, dim)
    print()
    print("   the 8 per-head terms summed at each failing index (fp32 view of "
          "the bf16 products):")
    prod_bf16 = (gkc[:, :, cp] * ks)          # eager's rounded product
    prod_f32 = gkc[:, :, cp].float() * ks.float()
    for i in bad.tolist():
        d = i % ks.shape[-1]
        t_bf = prod_bf16[0, :, 0, d].float().tolist()
        t_f32 = prod_f32[0, :, 0, d].tolist()
        print("     idx=%d  sum_bf16_terms=%+0.6f  sum_fp32_terms=%+0.6f  "
              "sum|terms|=%0.6f" % (i, sum(t_bf), sum(t_f32),
                                    sum(abs(x) for x in t_f32)))
        print("        bf16 terms: %s" % ["%+0.4f" % x for x in t_bf])
        print("        fp32 terms: %s" % ["%+0.4f" % x for x in t_f32])

    Path("/work/artifacts/11/compile-diag/"
         "L1__062_localisation.json").write_text(json.dumps(
             {"uuid": UUID, "seed": 0, "torch": torch.__version__,
              "atol": atol, "rtol": rtol, "golden": summary}, indent=2))


main()
