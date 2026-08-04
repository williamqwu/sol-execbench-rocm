"""Finer ablation: with GA in fp32, what else actually needs precision?

Models the *implementable* options:
  ga:  'f32' (exact)  | 'split' (hi+lo bf16 dots) | 'bf16'
  do:  precision of dO entering dv/dp dots
  p:   precision of P entering dv dot
  ds:  precision of dS entering dq/dk dots
  st:  precision of DQKV as stored/consumed by the 3 big GEMMs
  hid: 'bf16' | 'split' for grad_hidden_states
"""
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
f = torch.float32
b = torch.bfloat16


def split2(x):
    hi = x.to(b).to(f)
    lo = (x - hi).to(b).to(f)
    return hi, lo


def mm(x, y, mode):
    """emulate x@y where x is carried at precision `mode`."""
    if mode == "f32":
        return x @ y
    if mode == "bf16":
        return x.to(b).to(f) @ y
    if mode == "split":
        hi, lo = split2(x)
        return hi @ y + lo @ y
    raise ValueError(mode)


def emulate(args, cfg):
    (go, hs, qs, ks, vs, cu, qw, kw, vw, ow) = args
    N, D = hs.shape
    H, HD = qs.shape[1], qs.shape[3]
    QD = H * HD
    scale = HD ** -0.5
    go32, hs32 = go.to(f), hs.to(f)
    q, k, v = qs.to(f), ks.to(f), vs.to(f)
    cuc = cu.cpu()
    lens = (cuc[1:] - cuc[:-1]).tolist()

    GA = go32 @ ow.to(f)
    if cfg["ga"] != "f32":
        GA = mm(GA, torch.eye(QD, device=dev, dtype=f), cfg["ga"])

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
        # dv = p^T @ do
        pt = p.transpose(-2, -1)
        if cfg["dv"] == "bf16":
            dv[:, :, s:e] = pt.to(b).to(f) @ doc.to(b).to(f)
        elif cfg["dv"] == "do_split":
            hi, lo = split2(doc)
            pb = pt.to(b).to(f)
            dv[:, :, s:e] = pb @ hi + pb @ lo
        else:
            dv[:, :, s:e] = pt @ doc
        # dp = do @ v^T
        if cfg["dp"] == "bf16":
            dp = doc.to(b).to(f) @ vc.to(b).to(f).transpose(-2, -1)
        elif cfg["dp"] == "do_split":
            hi, lo = split2(doc)
            vb = vc.to(b).to(f).transpose(-2, -1)
            dp = hi @ vb + lo @ vb
        else:
            dp = doc @ vc.transpose(-2, -1)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        if cfg["ds"] == "bf16":
            dsx = ds.to(b).to(f)
            dq[:, :, s:e] = (dsx @ kc) * scale
            dk[:, :, s:e] = (dsx.transpose(-2, -1) @ qc) * scale
        elif cfg["ds"] == "split":
            hi = ds.to(b).to(f); lo = (ds - hi).to(b).to(f)
            dq[:, :, s:e] = ((hi @ kc) + (lo @ kc)) * scale
            dk[:, :, s:e] = ((hi.transpose(-2,-1) @ qc) + (lo.transpose(-2,-1) @ qc)) * scale
        else:
            dq[:, :, s:e] = (ds @ kc) * scale
            dk[:, :, s:e] = (ds.transpose(-2, -1) @ qc) * scale
        s = e

    def flat(x):
        return x.squeeze(0).transpose(0, 1).reshape(N, QD).contiguous()
    Of = O.squeeze(0).transpose(0, 1).reshape(N, QD).contiguous()
    gow = go32.t() @ Of.to(b).to(f)
    gob = go32.sum(0)

    G = torch.cat([flat(dq), flat(dk), flat(dv)], dim=1)
    gw = mm(G.t(), hs32, cfg["gw"])
    gb = G.to(b).to(f).sum(0) if cfg["gb"] == "bf16" else G.sum(0)
    W = torch.cat([qw, kw, vw], dim=0).to(f)
    ghs = mm(G, W, cfg["hid"])
    return (ghs.to(b), gw[:QD].to(b), gb[:QD].to(b), gw[QD:2*QD].to(b), gb[QD:2*QD].to(b),
            gw[2*QD:].to(b), gb[2*QD:].to(b), gow.to(b), gob.to(b))


def check(ref, got, tol):
    worst = 1.0; bad = []
    for n, r, g in zip(NAMES, ref, got):
        rf, gf = r.float(), g.float()
        d = (rf - gf).abs()
        t = tol["max_atol"] + tol["max_rtol"] * rf.abs()
        m = (d <= t).float().mean().item()
        worst = min(worst, m)
        if m < tol["required_matched_ratio"]:
            bad.append((n, round(m, 4)))
    return worst, bad


BASE = dict(ga="f32", dv="do_split", dp="do_split", ds="split", gw="bf16", gb="f32", hid="split")
CFGS = {
    "P ds=split,dp=split":  dict(BASE),
    "Q ds=split,dp=bf16":   dict(BASE, dp="bf16"),
    "R ds=f32,dp=split":    dict(BASE, ds="f32"),
    "S ds=split gw=split":  dict(BASE, gw="split"),
}

wls = [json.loads(l) for l in open("workload.jsonl")]
sel = [int(x) for x in sys.argv[1:]] or [13, 11, 8, 0]
agg = {c: [] for c in CFGS}
for i in sel:
    w = wls[i]
    ax = dict(CONST); ax.update(w["axes"])
    torch.manual_seed(999 + i)
    inp = reference.get_inputs(ax, dev)
    args = [inp[kk] for kk in ORDER]
    ref = reference.run(*args)
    for cname, cfg in CFGS.items():
        worst, bad = check(ref, emulate(args, cfg), w["tolerance"])
        agg[cname].append((worst, bad))
        print(f"wl{i:2d} N={ax['total_seq_len']:5d} {cname:30s} worst={worst:.5f} {'OK' if not bad else 'FAIL '+str(bad)}")
    print()

print("=== summary ===")
for c, rows in agg.items():
    print(f"{c:30s} minworst={min(r[0] for r in rows):.5f} nfail={sum(1 for r in rows if r[1])}")
