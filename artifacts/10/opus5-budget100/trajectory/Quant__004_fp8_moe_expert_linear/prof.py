import sys, torch, time, triton
D = "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear"
sys.path.insert(0, D)
import reference as R
import tk

dev = 'cuda'
torch.manual_seed(0)


def bench(fn, n=50, w=10):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


for nt in [int(x) for x in sys.argv[1:]] or [384, 1024, 2048, 4096]:
    inp = R.get_inputs({"num_tokens": nt}, dev)
    hs, rw, guw, dw = (inp["hidden_states"], inp["routing_weight"],
                       inp["gate_up_weight"], inp["down_weight"])
    M, H = hs.shape
    I = 2048
    t_qa = bench(lambda: tk.quant_act(hs))
    t_qw = bench(lambda: (tk.quant_weight(guw), tk.quant_weight(dw)))
    aq, asc = tk.quant_act(hs)
    wq, wsc = tk.quant_weight(guw)
    dq, dsc = tk.quant_weight(dw)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, I // 128), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)

    def g1():
        tk._gemm1_silu_quant[(triton.cdiv(M, 128), I // 128)](
            aq, asc, wq, wsc, gq, gs, M, H, I, aq.stride(0), asc.stride(0),
            wq.stride(0), wsc.stride(0), gq.stride(0), gs.stride(0),
            BLOCK_M=128, NUM_K=H // 128, num_warps=8, num_stages=2)

    def g2():
        tk._gemm2[(triton.cdiv(M, 128), H // 128)](
            gq, gs, dq, dsc, rw, out, M, I, H, gq.stride(0), gs.stride(0),
            dq.stride(0), dsc.stride(0), out.stride(0),
            BLOCK_M=128, BLOCK_N=128, NUM_K=I // 128, num_warps=8, num_stages=2)

    t1 = bench(g1)
    t2 = bench(g2)
    tot = bench(lambda: tk.moe(hs, rw, guw, dw))
    fl1 = 2 * M * 4096 * H / 1e12
    fl2 = 2 * M * H * I / 1e12
    print(f"M={M}: qa {t_qa:.3f} qw {t_qw:.3f} g1 {t1:.3f}({fl1/(t1/1e3):.0f}TF) "
          f"g2 {t2:.3f}({fl2/(t2/1e3):.0f}TF) sum {t_qa+t_qw+t1+t2:.3f} tot {tot:.3f}")
