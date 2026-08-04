import torch, triton, time
import kernel as K

dev = "cuda:0"
torch.manual_seed(0)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


for M in [128, 512, 1536, 3840]:
    hs = torch.randn(M, 1536, dtype=torch.bfloat16, device=dev)
    w = torch.randn(4608, 1536, dtype=torch.bfloat16, device=dev) * 0.05
    b = torch.randn(4608, dtype=torch.bfloat16, device=dev)
    num_kb = 12
    qx = torch.empty((M, 1536), dtype=torch.float8_e4m3fn, device=dev)
    sx = torch.empty((M, num_kb), dtype=torch.float32, device=dev)
    qw = torch.empty((4608, 1536), dtype=torch.float8_e4m3fn, device=dev)
    sw = torch.empty((36, num_kb), dtype=torch.float32, device=dev)
    out = torch.empty((3, M, 16, 96), dtype=torch.bfloat16, device=dev)
    cfg = K._cfg(M)
    BM, BN = cfg["BLOCK_M"], cfg["BLOCK_N"]

    ta = bench(lambda: K._quant_act[(triton.cdiv(M, 32), num_kb)](
        hs, qx, sx, M, hs.stride(0), qx.stride(0), sx.stride(0),
        BLOCK_M=32, num_warps=4, num_stages=1))
    tw = bench(lambda: K._quant_w[(36, num_kb)](
        w, qw, sw, w.stride(0), qw.stride(0), sw.stride(0),
        num_warps=8, num_stages=1))
    tg = bench(lambda: K._gemm[(triton.cdiv(M, BM) * (4608 // BN),)](
        qx, qw, sx, sw, b, out, M,
        qx.stride(0), qw.stride(0), sx.stride(0), sw.stride(0), 1536,
        NB_PER_OUT=1536 // BN, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=cfg["GROUP_M"],
        NUM_KB=num_kb, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]))
    tt = bench(lambda: K.run(hs, w, b))

    # rough SOL
    bytes_moved = M * 1536 * 2 + 4608 * 1536 * 2 + 3 * M * 1536 * 2
    flops = 2 * M * 4608 * 1536
    sol_mem = bytes_moved / 8e12 * 1e6
    sol_cmp = flops / 5e15 * 1e6
    print(f"M={M:5d} act={ta:6.1f} w={tw:6.1f} gemm={tg:6.1f} sum={ta+tw+tg:6.1f} "
          f"total={tt:6.1f}us   SOL~{max(sol_mem, sol_cmp):5.1f}us (mem {sol_mem:.1f} cmp {sol_cmp:.1f})")
