import enum, torch, triton, sys
if not hasattr(enum, "StrEnum"):
    class S(str, enum.Enum):
        def __str__(self): return self.value
    enum.StrEnum = S
import reference as ref
import kernel as K

dev = torch.device("cuda:0")
H = 3584; I = 2048; hk = H // 128; ik = I // 128; NGU = 2 * I


def t(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(True); b = torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000  # us


for M in [384, 1024, 1920, 2048, 4096]:
    torch.manual_seed(M)
    inp = ref.get_inputs({"num_tokens": M}, dev)
    hs_, rw_, gu_, dn_ = inp["hidden_states"], inp["routing_weight"], inp["gate_up_weight"], inp["down_weight"]
    hq = torch.empty((M, H), dtype=torch.float8_e4m3fn, device=dev)
    hsq = torch.empty((M, hk), dtype=torch.float32, device=dev)
    guq = torch.empty((NGU, H), dtype=torch.float8_e4m3fn, device=dev)
    gus = torch.empty((NGU // 128, hk), dtype=torch.float32, device=dev)
    dnq = torch.empty((H, I), dtype=torch.float8_e4m3fn, device=dev)
    dns = torch.empty((H // 128, ik), dtype=torch.float32, device=dev)
    gq = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=dev)
    gs = torch.empty((M, ik), dtype=torch.float32, device=dev)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    nt0 = (NGU // 128) * hk; nt1 = (H // 128) * ik

    qa = lambda: K._quant_act_kernel[(triton.cdiv(M, 32), hk)](
        hs_, hq, hsq, M, hs_.stride(0), hq.stride(0), hsq.stride(0),
        BLOCK_M=32, num_warps=4, num_stages=2)
    qw = lambda: K._quant_w2_kernel[(nt0 + nt1,)](
        gu_, guq, gus, gu_.stride(0), guq.stride(0), gus.stride(0),
        dn_, dnq, dns, dn_.stride(0), dnq.stride(0), dns.stride(0),
        NK0=hk, NTILES0=nt0, NK1=ik, num_warps=8, num_stages=2)
    qa(); qw()

    print(f"M={M}  quant_act={t(qa):7.1f}us  quant_w={t(qw):7.1f}us")
    for bm in (32, 64, 128, 256):
        nmt = triton.cdiv(M, bm)
        for nw in (4, 8):
            for ns in (1, 2, 3):
                try:
                    g1 = lambda: K._gemm1_kernel[(nmt * 16,)](
                        hq, hsq, guq, gus, gq, gs, M,
                        hq.stride(0), hsq.stride(0), guq.stride(0), gus.stride(0),
                        gq.stride(0), gs.stride(0),
                        KBLK=hk, NTILE=16, IHALF=I, BLOCK_M=bm, GROUP_M=8, NUM_MT=nmt,
                        num_warps=nw, num_stages=ns)
                    g1()
                    nmt2 = triton.cdiv(M, bm)
                    g2 = lambda: K._gemm2_kernel[(nmt2 * 28,)](
                        gq, gs, dnq, dns, rw_, out, M,
                        gq.stride(0), gs.stride(0), dnq.stride(0), dns.stride(0), out.stride(0),
                        KBLK=ik, NTILE=28, BLOCK_M=bm, GROUP_M=8, NUM_MT=nmt2,
                        num_warps=nw, num_stages=ns)
                    g2()
                    print(f"   bm={bm:3d} nw={nw} ns={ns}  gemm1={t(g1):7.1f}us  gemm2={t(g2):7.1f}us")
                except Exception as e:
                    print(f"   bm={bm:3d} nw={nw} ns={ns}  ERR {type(e).__name__} {str(e)[:60]}")
    sys.stdout.flush()
