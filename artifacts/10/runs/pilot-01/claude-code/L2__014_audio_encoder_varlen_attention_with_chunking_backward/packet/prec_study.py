"""Emulate precision choices in torch to find the cheapest one that passes."""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import reference

CONST = {"d_model": 1280, "num_heads": 20, "head_dim": 64}
NAMES = ["grad_hidden_states", "grad_q_weight", "grad_q_bias", "grad_k_weight",
         "grad_k_bias", "grad_v_weight", "grad_v_bias", "grad_out_weight", "grad_out_bias"]
ORDER = ["grad_output", "hidden_states", "query_states", "key_states", "value_states",
         "cu_seqlens", "q_weight", "k_weight", "v_weight", "out_weight"]
dev = torch.device("cuda:0")


def rb(x, on):
    return x.to(torch.bfloat16).to(torch.float32) if on else x


def emulate(args, cfg):
    (go, hs, qs, ks, vs, cu, qw, kw, vw, ow) = args
    N, D = hs.shape
    H, HD = qs.shape[1], qs.shape[3]
    QD = H * HD
    scale = HD ** -0.5
    f = torch.float32
    go32, hs32 = go.to(f), hs.to(f)
    q, k, v = qs.to(f), ks.to(f), vs.to(f)

    cuc = cu.cpu()
    lens = (cuc[1:] - cuc[:-1]).tolist()

    GA = go32 @ ow.to(f)
    GA = rb(GA, cfg.get("round_ga"))
    O = torch.empty(1, H, N, HD, device=dev, dtype=f)
    dq = torch.empty_like(O); dk = torch.empty_like(O); dv = torch.empty_like(O)
    gac = GA.reshape(N, H, HD).transpose(0, 1).unsqueeze(0).contiguous()

    s = 0
    for L in lens:
        e = s + L
        qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
        doc = gac[:, :, s:e]
        w = (qc @ kc.transpose(-2, -1)) * scale
        p = F.softmax(w, dim=-1, dtype=f)
        O[:, :, s:e] = p @ vc
        pr = rb(p, cfg.get("round_p"))
        dor = rb(doc, cfg.get("round_do"))
        dv[:, :, s:e] = pr.transpose(-2, -1) @ dor
        dp = dor @ rb(vc, False).transpose(-2, -1)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        dsr = rb(ds, cfg.get("round_ds"))
        dq[:, :, s:e] = (dsr @ kc) * scale
        dk[:, :, s:e] = (dsr.transpose(-2, -1) @ qc) * scale
        s = e

    Of = O.squeeze(0).transpose(0, 1).reshape(N, QD).contiguous()
    gow = rb(go32, False).t() @ rb(Of, cfg.get("round_o"))
    gob = go32.sum(0)

    def flat(x):
        return x.squeeze(0).transpose(0, 1).reshape(N, QD).contiguous()
    DQ, DK, DV = flat(dq), flat(dk), flat(dv)
    G = torch.cat([DQ, DK, DV], dim=1)
    Gs = rb(G, cfg.get("round_store"))
    gw = Gs.t() @ hs32
    gb = (Gs if cfg.get("bias_from_store") else G).sum(0)
    W = torch.cat([qw, kw, vw], dim=0).to(f)
    ghs = Gs @ W if not cfg.get("split_hidden") else None
    if cfg.get("split_hidden"):
        hi = G.to(torch.bfloat16).to(f)
        lo = (G - hi).to(torch.bfloat16).to(f)
        ghs = hi @ W + lo @ W
    b = torch.bfloat16
    return (ghs.to(b), gw[:QD].to(b), gb[:QD].to(b), gw[QD:2*QD].to(b), gb[QD:2*QD].to(b),
            gw[2*QD:].to(b), gb[2*QD:].to(b), gow.to(b), gob.to(b))


def check(ref, got, tol):
    worst = 1.0
    bad = []
    for n, r, g in zip(NAMES, ref, got):
        rf, gf = r.float(), g.float()
        d = (rf - gf).abs()
        t = tol["max_atol"] + tol["max_rtol"] * rf.abs()
        m = (d <= t).float().mean().item()
        if m < worst:
            worst = m
        if m < tol["required_matched_ratio"]:
            bad.append((n, m))
    return worst, bad


CFGS = {
    "all_bf16": dict(round_ga=1, round_p=1, round_do=1, round_ds=1, round_store=1, round_o=1, bias_from_store=1),
    "store_fp32_bias+split": dict(round_ga=1, round_p=1, round_do=1, round_ds=1, round_o=1, split_hidden=1, round_store=1),
    "+fp32_p_ds": dict(round_ga=1, round_do=1, round_o=1, split_hidden=1, round_store=1),
    "+fp32_do": dict(round_ga=1, round_o=1, split_hidden=1, round_store=1),
    "+fp32_ga": dict(round_o=1, split_hidden=1, round_store=1),
}

wls = [json.loads(l) for l in open("workload.jsonl")]
sel = [int(x) for x in sys.argv[1:]] or [13, 11, 8, 0]
res = {c: [] for c in CFGS}
for i in sel:
    w = wls[i]
    ax = dict(CONST); ax.update(w["axes"])
    torch.manual_seed(999 + i)
    inp = reference.get_inputs(ax, dev)
    args = [inp[kk] for kk in ORDER]
    ref = reference.run(*args)
    for cname, cfg in CFGS.items():
        got = emulate(args, cfg)
        worst, bad = check(ref, got, w["tolerance"])
        res[cname].append((i, worst, bad))
        print(f"wl{i:2d} N={ax['total_seq_len']:5d} {cname:24s} worst={worst:.5f} "
              f"{'OK ' if not bad else 'FAIL ' + str([(n, round(m,4)) for n,m in bad])}")
    print()

print("=== summary ===")
for c, rows in res.items():
    print(f"{c:24s} minworst={min(r[1] for r in rows):.5f} nfail={sum(1 for r in rows if r[2])}")
