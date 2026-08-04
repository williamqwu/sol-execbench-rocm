import math
import sys
import torch

sys.path.insert(0, "/var/tmp/solbench/agent/pilot8/FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1")
import kernel
import importlib

importlib.reload(kernel)

dev = "cuda"
NP = 989669


def make(total_q, batch, kv_extra, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
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


def bench(args, iters=20):
    for _ in range(5):
        kernel.run(*args)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        kernel.run(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


# (total_q, batch, kv_extra) approximating real workloads
cases = [
    ("tiny_1", 1, 1, 33),
    ("small_33", 33, 1, 1),
    ("med_376", 376, 1, 5),
    ("med_1028", 1028, 1, 10),
    ("big_3024", 3024, 3, 2),
    ("big_8987", 8987, 56, 60),
    ("dec_22", 22, 22, 800),
    ("big_1187", 1187, 3, 6),
]
for name, tq, b, ke in cases:
    args = make(tq, b, ke)
    t = bench(args)
    print(f"{name:12s} total_q={tq:6d} batch={b:3d} : {t:8.4f} ms")
