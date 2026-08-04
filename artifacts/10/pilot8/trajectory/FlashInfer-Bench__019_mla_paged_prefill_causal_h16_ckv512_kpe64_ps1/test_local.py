import math
import sys
import torch

sys.path.insert(0, "/var/tmp/solbench/agent/pilot8/FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1")
import reference
import importlib as _il
import sys as _sys
kernel=_il.import_module(_sys.argv[1] if len(_sys.argv)>1 else "kernel")
import importlib



torch.manual_seed(0)
dev = "cuda"
NP = 200000


def make(total_q, batch, kv_extra, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # split total_q into batch parts
    base = total_q // batch
    qlens = [base] * batch
    for i in range(total_q - base * batch):
        qlens[i] += 1
    qo = [0]
    for l in qlens:
        qo.append(qo[-1] + l)
    kvlens = [l + kv_extra for l in qlens]
    kvp = [0]
    for l in kvlens:
        kvp.append(kvp[-1] + l)
    qo_indptr = torch.tensor(qo, dtype=torch.int32, device=dev)
    kv_indptr = torch.tensor(kvp, dtype=torch.int32, device=dev)
    kv_indices = torch.randint(0, NP, (kvp[-1],), generator=g, dtype=torch.int32).to(dev)
    H, D, DP = 16, 512, 64
    q_nope = torch.randn(total_q, H, D, generator=g, dtype=torch.float32).to(dev).bfloat16()
    q_pe = torch.randn(total_q, H, DP, generator=g, dtype=torch.float32).to(dev).bfloat16()
    ckv = torch.randn(NP, 1, D, generator=g, dtype=torch.float32).to(dev).bfloat16()
    kpe = torch.randn(NP, 1, DP, generator=g, dtype=torch.float32).to(dev).bfloat16()
    return q_nope, q_pe, ckv, kpe, qo_indptr, kv_indptr, kv_indices, 1 / math.sqrt(192)


cases = [
    (33, 1, 1), (1, 1, 33), (17, 1, 2), (52, 4, 3), (376, 1, 5),
    (5, 1, 2), (10, 1, 2), (3, 1, 2), (26, 2, 3), (1028, 1, 10),
    (22, 22, 800), (69, 3, 7), (2, 2, 25), (1187, 3, 6),
]

for c in cases:
    args = make(*c)
    o_ref, l_ref = reference.run(*args)
    o, l = kernel.run(*args)
    do = (o.float() - o_ref.float()).abs().max().item()
    dl = (l - l_ref).abs()
    dl = dl[torch.isfinite(dl)].max().item() if torch.isfinite(dl).any() else 0.0
    ok = do < 0.05 and dl < 0.02
    print(f"{c}: out_maxdiff={do:.5f} lse_maxdiff={dl:.6f} {'OK' if ok else 'FAIL'}")
