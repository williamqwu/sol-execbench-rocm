import math
import sys
import importlib
import torch

ROOT = "/var/tmp/solbench/agent/pilot8/FlashInfer-Bench__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1"
sys.path.insert(0, ROOT)

MOD = sys.argv[1] if len(sys.argv) > 1 else "kernel_v2"
mod = importlib.import_module(MOD)

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
        mod.run(*args)
    torch.cuda.synchronize()
    st = torch.cuda.Event(True); en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        mod.run(*args)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


import os

_SET = os.environ.get("CASESET", "small")
if _SET == "small":
    cases = [
        ("tiny_1", 1, 1, 33), ("small_33", 33, 1, 1), ("med_376", 376, 1, 5),
        ("med_1028", 1028, 1, 10), ("big_3024", 3024, 3, 2), ("big_8987", 8987, 56, 60),
        ("dec_22", 22, 22, 800), ("big_1187", 1187, 3, 6),
    ]
else:
    cases = [
        ("h_8k_1b", 8192, 1, 0), ("h_16k_1b", 16384, 1, 0),
        ("h_4k_4b", 4096, 4, 2000), ("h_2k_long", 2048, 1, 30000),
        ("h_8987", 8987, 56, 60), ("h_dec_64", 64, 64, 4000),
    ]
ARGS = {n: make(tq, b, ke) for n, tq, b, ke in cases}

configs = []
for bq in [1, 2, 4, 8]:
    for bn in [32, 64, 128]:
        for nw in [4, 8]:
            configs.append((bq, bn, nw))

results = {}
for bq, bn, nw in configs:
    mod.BLOCK_Q = bq
    mod.BLOCK_N = bn
    mod.NUM_WARPS = nw
    try:
        row = []
        for n, *_ in cases:
            row.append(bench(ARGS[n]))
    except Exception as e:
        print(f"BQ={bq} BN={bn} NW={nw}: FAIL {type(e).__name__} {str(e)[:80]}")
        continue
    tot = math.exp(sum(math.log(x) for x in row) / len(row))
    results[(bq, bn, nw)] = row
    print(f"BQ={bq:2d} BN={bn:3d} NW={nw}: geo={tot:7.4f} " + " ".join(f"{n}={t:.4f}" for (n, *_), t in zip(cases, row)))

best = min(results.items(), key=lambda kv: math.exp(sum(math.log(x) for x in kv[1]) / len(kv[1])))
print("BEST", best[0])
