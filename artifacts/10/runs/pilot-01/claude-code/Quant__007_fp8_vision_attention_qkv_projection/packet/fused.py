import torch, triton, triton.language as tl
from triton.testing import do_bench

RECIP = tl.constexpr(1.0 / 448.0)
EMAX = tl.constexpr(448.0)
FP8 = tl.constexpr(tl.float8e4nv)


# Fully fused: quantize A tile and W tile on the fly, no intermediates.
@triton.jit
def _fused(
    x_ptr, w_ptr, bias_ptr, o_ptr,
    M,
    sx_m, sw_n, so_m,
    NB_PER_OUT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr, NUM_KB: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = 4608 // BLOCK_N
    nig = GROUP_M * num_pid_n
    gid = pid // nig
    fm = gid * GROUP_M
    gsz = min(num_pid_m - fm, GROUP_M)
    pid_m = fm + ((pid % nig) % gsz)
    pid_n = (pid % nig) // gsz

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    am = tl.where(mask_m, offs_m, 0)
    offs_k = tl.arange(0, 128)

    a_ptrs = x_ptr + am[:, None] * sx_m + offs_k[None, :]
    b_ptrs = w_ptr + offs_n[:, None] * sw_n + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in tl.range(0, NUM_KB):
        af = tl.load(a_ptrs).to(tl.float32)
        sa = tl.maximum(tl.max(tl.abs(af), axis=1) * RECIP, 1e-12)
        qa = tl.minimum(tl.maximum(af / sa[:, None], -EMAX), EMAX).to(FP8)

        bfv = tl.load(b_ptrs).to(tl.float32)
        # amax per 128-row subtile of N
        bg = tl.reshape(bfv, (BLOCK_N // 128, 128 * 128))
        sb = tl.maximum(tl.max(tl.abs(bg), axis=1) * RECIP, 1e-12)
        sbf = tl.reshape(
            tl.broadcast_to(sb[:, None], (BLOCK_N // 128, 128)), (BLOCK_N,))
        qb = tl.minimum(tl.maximum(bfv / sbf[:, None], -EMAX), EMAX).to(FP8)

        p = tl.dot(qa, tl.trans(qb), out_dtype=tl.float32)
        acc += p * (sa[:, None] * sbf[None, :])

        a_ptrs += 128
        b_ptrs += 128

    acc += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    out = acc.to(tl.bfloat16)
    sel = pid_n // NB_PER_OUT
    tl.store(o_ptr + sel * M * 1536 + offs_m[:, None] * so_m
             + (offs_n - sel * 1536)[None, :], out, mask=mask_m[:, None])


def run_fused(hs, w, b, BM, BN, GM, nw, ns):
    M = hs.shape[0]
    out = torch.empty((3, M, 16, 96), dtype=torch.bfloat16, device=hs.device)
    grid = (triton.cdiv(M, BM) * (4608 // BN),)
    _fused[grid](hs, w, b, out, M, hs.stride(0), w.stride(0), 1536,
                 NB_PER_OUT=1536 // BN, BLOCK_M=BM, BLOCK_N=BN, GROUP_M=GM,
                 NUM_KB=12, num_warps=nw, num_stages=ns)
    return out


if __name__ == "__main__":
    dev = "cuda:0"
    torch.manual_seed(0)
    import kernel as K
    for M in [128, 384, 512, 1024, 1536, 2688, 3840]:
        hs = torch.randn(M, 1536, dtype=torch.bfloat16, device=dev)
        w = torch.randn(4608, 1536, dtype=torch.bfloat16, device=dev) * 0.05
        bi = torch.randn(4608, dtype=torch.bfloat16, device=dev)
        base = do_bench(lambda: K.run(hs, w, bi), warmup=50, rep=200) * 1e3
        best = None
        ref = K.run(hs, w, bi)
        for BM in [64, 128, 256]:
            for BN in [128, 256]:
                for GM in [1, 4, 8]:
                    for nw in [4, 8]:
                        for ns in [1, 2]:
                            try:
                                o = run_fused(hs, w, bi, BM, BN, GM, nw, ns)
                                d = max((o[i].float() - ref[i].float()).abs().max().item()
                                        for i in range(3))
                                if d > 0:
                                    continue
                                t = do_bench(lambda: run_fused(hs, w, bi, BM, BN, GM, nw, ns),
                                             warmup=25, rep=100) * 1e3
                                if best is None or t < best[0]:
                                    best = (t, BM, BN, GM, nw, ns)
                            except Exception:
                                pass
        print(f"M={M:5d} 3kernel={base:7.2f}us  fused_best={best[0]:7.2f}us "
              f"cfg BM={best[1]} BN={best[2]} GM={best[3]} nw={best[4]} ns={best[5]}",
              flush=True)
