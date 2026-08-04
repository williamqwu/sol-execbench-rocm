import torch, triton, time
import kernel as K
from triton.testing import do_bench

dev = "cuda:0"
torch.manual_seed(0)

print(f"{'M':>5} {'act':>7} {'wq':>7} {'gemm':>7} {'sum':>7} {'e2e':>7} {'SOLmem':>7} {'SOLcmp':>7}")
for M in [128, 384, 512, 1024, 1536, 2688, 3840]:
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

    ta = do_bench(lambda: K._quant_act[(triton.cdiv(M, 32), num_kb)](
        hs, qx, sx, M, hs.stride(0), qx.stride(0), sx.stride(0),
        BLOCK_M=32, num_warps=4, num_stages=1), warmup=50, rep=200) * 1e3
    tw = do_bench(lambda: K._quant_w[(36, num_kb)](
        w, qw, sw, w.stride(0), qw.stride(0), sw.stride(0),
        num_warps=8, num_stages=1), warmup=50, rep=200) * 1e3
    tg = do_bench(lambda: K._gemm[(triton.cdiv(M, BM) * (4608 // BN),)](
        qx, qw, sx, sw, b, out, M,
        qx.stride(0), qw.stride(0), sx.stride(0), sw.stride(0), 1536,
        NB_PER_OUT=1536 // BN, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=cfg["GROUP_M"],
        NUM_KB=num_kb, num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]),
        warmup=50, rep=200) * 1e3
    tt = do_bench(lambda: K.run(hs, w, b), warmup=50, rep=200) * 1e3

    bytes_min = M * 1536 * 2 + 4608 * 1536 * 2 + 3 * M * 1536 * 2
    flops = 2 * M * 4608 * 1536
    print(f"{M:5d} {ta:7.1f} {tw:7.1f} {tg:7.1f} {ta+tw+tg:7.1f} {tt:7.1f} "
          f"{bytes_min/8e12*1e6:7.1f} {flops/5e15*1e6:7.1f}")
