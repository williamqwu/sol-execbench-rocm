import sys, torch, time, triton
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import tk5 as tk

dev = 'cuda'
H, I = 3584, 2048


def bench(fn, n=50, w=20):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(n)]
    for a, b in ev:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    ts = sorted(a.elapsed_time(b) for a, b in ev)
    return ts[len(ts) // 2]


guw = torch.randn(2 * I, H, device=dev, dtype=torch.bfloat16) * H ** -0.5
dw = torch.randn(H, I, device=dev, dtype=torch.bfloat16) * I ** -0.5
for M in [384, 1024, 2048, 3072, 4096]:
    hs = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    rw = torch.randn(M, 1, device=dev, dtype=torch.bfloat16)
    aq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    asc = torch.empty((M, H // 128), device=dev)
    wq = torch.empty((2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
    wsc = torch.empty((2 * I // 128, H // 128), device=dev)
    dq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
    dsc = torch.empty((H // 128, I // 128), device=dev)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    KB1, KB2 = H // 128, I // 128
    NP1 = (2 * I // 128) * KB1
    NP2 = (H // 128) * KB2
    NPA = triton.cdiv(M, 64) * KB1

    def q():
        tk._quant_all[(NP1 + NP2 + NPA,)](
            guw, wq, wsc, dw, dq, dsc, hs, aq, asc, M, guw.stride(0),
            wsc.stride(0), dw.stride(0), dsc.stride(0), hs.stride(0),
            aq.stride(0), asc.stride(0), IB=KB2, KB1=KB1, KB2=KB2, NP1=NP1,
            NP2=NP2, BM=64, num_warps=4)

    bm1, nw1, ns1, w1 = tk._cfg1(M, I)

    def g1():
        tk._gemm1[(triton.cdiv(M, bm1), I // 128)](
            aq, asc, wq, wsc, gq, gs, M, aq.stride(0), asc.stride(0),
            wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
            BLOCK_M=bm1, NUM_K=KB1, num_warps=nw1, num_stages=ns1,
            waves_per_eu=w1)

    bm2, bn2, nw2, ns2, w2 = tk._cfg2(M, H)

    def g2():
        tk._gemm2[(triton.cdiv(M, bm2), H // bn2)](
            gq, gs, dq, dsc, rw, out, M, gq.stride(0), gs.stride(0),
            dq.stride(0), dsc.stride(0), out.stride(0), BLOCK_M=bm2,
            BLOCK_N=bn2, NUM_K=KB2, num_warps=nw2, num_stages=ns2,
            waves_per_eu=w2)

    tq, t1, t2 = bench(q), bench(g1), bench(g2)
    tot = bench(lambda: tk.moe(hs, rw, guw, dw))
    f1 = 2 * M * 2 * I * H / 1e12
    f2 = 2 * M * H * I / 1e12
    print(f"M={M:5d} q {tq*1e3:6.1f}us  g1 {t1*1e3:6.1f}us({f1/(t1/1e3):5.0f}TF) "
          f"g2 {t2*1e3:6.1f}us({f2/(t2/1e3):5.0f}TF)  sum {(tq+t1+t2)*1e3:6.1f} "
          f"tot {tot*1e3:6.1f}us  overhead {(tot-tq-t1-t2)*1e3:5.1f}us")
