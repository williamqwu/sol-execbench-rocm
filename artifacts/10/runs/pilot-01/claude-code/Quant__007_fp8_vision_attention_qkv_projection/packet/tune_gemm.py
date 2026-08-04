import torch, triton, itertools, json
from triton.testing import do_bench
import kernel as K

dev = "cuda:0"
torch.manual_seed(0)
SEQS = [128, 384, 512, 768, 896, 1024, 1152, 1536, 1664, 2176,
        2688, 2944, 3200, 3456, 3712, 3840]

w = torch.randn(4608, 1536, dtype=torch.bfloat16, device=dev) * 0.05
bi = torch.randn(4608, dtype=torch.bfloat16, device=dev)
qw = torch.empty((4608, 1536), dtype=torch.float8_e4m3fn, device=dev)
sw = torch.empty((36, 12), dtype=torch.float32, device=dev)
K._quant_w[(36, 12)](w, qw, sw, w.stride(0), qw.stride(0), sw.stride(0),
                     num_warps=8, num_stages=1)

best_all = {}
for M in SEQS:
    qx = torch.randn(M, 1536, device=dev).to(torch.float8_e4m3fn)
    sx = torch.rand(M, 12, device=dev) + 0.01
    out = torch.empty((3, M, 16, 96), dtype=torch.bfloat16, device=dev)
    best = None
    for BM, BN, GM, nw, ns in itertools.product(
            [32, 64, 128, 256], [128, 256, 512], [1, 2, 4, 8], [4, 8], [1, 2]):
        if 4608 % BN:
            continue
        if BM * BN > 128 * 512:
            continue
        grid = (triton.cdiv(M, BM) * (4608 // BN),)
        try:
            f = lambda: K._gemm[grid](
                qx, qw, sx, sw, bi, out, M,
                qx.stride(0), qw.stride(0), sx.stride(0), sw.stride(0), 1536,
                NB_PER_OUT=1536 // BN, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=GM,
                NUM_KB=12, num_warps=nw, num_stages=ns)
            f()
            torch.cuda.synchronize()
            t = do_bench(f, warmup=25, rep=100) * 1e3
            if best is None or t < best[0]:
                best = (t, BM, BN, GM, nw, ns)
        except Exception:
            pass
    cur = K._cfg(M)
    grid = (triton.cdiv(M, cur["BLOCK_M"]) * (4608 // cur["BLOCK_N"]),)
    tc = do_bench(lambda: K._gemm[grid](
        qx, qw, sx, sw, bi, out, M, qx.stride(0), qw.stride(0), sx.stride(0),
        sw.stride(0), 1536, NB_PER_OUT=1536 // cur["BLOCK_N"],
        BLOCK_M=cur["BLOCK_M"], BLOCK_N=cur["BLOCK_N"], GROUP_M=cur["GROUP_M"],
        NUM_KB=12, num_warps=cur["num_warps"], num_stages=cur["num_stages"]),
        warmup=25, rep=100) * 1e3
    best_all[M] = best
    print(f"M={M:5d} cur={tc:7.2f} best={best[0]:7.2f} "
          f"BM={best[1]} BN={best[2]} GM={best[3]} nw={best[4]} ns={best[5]}", flush=True)

json.dump({str(k): v for k, v in best_all.items()}, open("gemm_best.json", "w"), indent=1)
