import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, ".")
import reference

CONST = {"d_model": 1280, "num_heads": 20, "head_dim": 64}
ORDER = ["grad_output", "hidden_states", "query_states", "key_states", "value_states",
         "cu_seqlens", "q_weight", "k_weight", "v_weight", "out_weight"]
dev = torch.device("cuda:0"); f = torch.float32; b = torch.bfloat16
wls = [json.loads(l) for l in open("workload.jsonl")]
i = int(sys.argv[1]) if len(sys.argv) > 1 else 13
w = wls[i]
ax = dict(CONST); ax.update(w["axes"])
torch.manual_seed(999 + i)
inp = reference.get_inputs(ax, dev)
args = [inp[k] for k in ORDER]
ref = reference.run(*args)

(go, hs, qs, ks, vs, cu, qw, kw, vw, ow) = args
N = hs.shape[0]; H = qs.shape[1]; HD = qs.shape[3]; QD = H * HD
scale = HD ** -0.5
go32 = go.to(f)
q, k, v = qs.to(f), ks.to(f), vs.to(f)
cuc = cu.cpu(); lens = (cuc[1:] - cuc[:-1]).tolist()

GA = go32 @ ow.to(f)
gac = GA.reshape(N, H, HD).transpose(0, 1).unsqueeze(0).contiguous()
O = torch.empty(1, H, N, HD, device=dev, dtype=f)
dq = torch.empty_like(O); dk = torch.empty_like(O); dv = torch.empty_like(O)
s = 0
for L in lens:
    e = s + L
    qc, kc, vc, doc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], gac[:, :, s:e]
    p = F.softmax((qc @ kc.transpose(-2, -1)) * scale, dim=-1, dtype=f)
    O[:, :, s:e] = p @ vc
    dv[:, :, s:e] = p.transpose(-2, -1) @ doc
    dp = doc @ vc.transpose(-2, -1)
    ds = p * (dp - (p * dp).sum(-1, keepdim=True))
    dq[:, :, s:e] = (ds @ kc) * scale
    dk[:, :, s:e] = (ds.transpose(-2, -1) @ qc) * scale
    s = e

def flat(x):
    return x.squeeze(0).transpose(0, 1).reshape(N, QD).contiguous()
DQ, DK, DV = flat(dq), flat(dk), flat(dv)

print(f"wl{i} N={N} lens[0]={lens[0]}")
for n, t in [("GA", GA), ("DQ", DQ), ("DK", DK), ("DV", DV)]:
    print(f"  {n:4s} absmean={t.abs().mean():.4f} absmax={t.abs().max():.3f} rms={t.pow(2).mean().sqrt():.4f}")

cq = DQ @ qw.to(f); ck = DK @ kw.to(f); cvv = DV @ vw.to(f)
tot = cq + ck + cvv
print("  contributions rms: dq@qw=%.3f dk@kw=%.3f dv@vw=%.3f total=%.3f" %
      (cq.pow(2).mean().sqrt(), ck.pow(2).mean().sqrt(), cvv.pow(2).mean().sqrt(), tot.pow(2).mean().sqrt()))

g = ref[0].float()
tol = w["tolerance"]["max_atol"] + w["tolerance"]["max_rtol"] * g.abs()
print(f"  ref grad_hidden rms={g.pow(2).mean().sqrt():.3f} absmax={g.abs().max():.2f}")
print(f"  tol: atol={w['tolerance']['max_atol']:.3f} median_tol={tol.median():.3f}")

def rep(name, val):
    d = (val.float() - g).abs()
    m = (d <= tol).float().mean().item()
    print(f"  {name:34s} matched={m:.5f} err rms={d.pow(2).mean().sqrt():.4f} p99={d.flatten().kthvalue(int(d.numel()*0.99))[0]:.4f} max={d.max():.4f}")

rep("fp32 gold ->bf16", tot.to(b))
rep("G->bf16 @ W", torch.cat([DQ, DK, DV], 1).to(b).to(f) @ torch.cat([qw, kw, vw], 0).to(f))
hi = torch.cat([DQ, DK, DV], 1).to(b).to(f)
lo = (torch.cat([DQ, DK, DV], 1) - hi).to(b).to(f)
W = torch.cat([qw, kw, vw], 0).to(f)
rep("G split2 @ W", (hi @ W + lo @ W).to(b))
rep("only DV->bf16", (DQ @ qw.to(f) + DK @ kw.to(f) + DV.to(b).to(f) @ vw.to(f)).to(b))
rep("only DQ,DK->bf16", (DQ.to(b).to(f) @ qw.to(f) + DK.to(b).to(f) @ kw.to(f) + cvv).to(b))

# how much does the reference's own recompute differ (i.e. is ref itself noisy)?
ref2 = reference.run(*args)
rep("reference rerun", ref2[0])
