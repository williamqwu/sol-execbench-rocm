"""Targeted sweep: small/medium M, where the GEMM is weight-bandwidth bound.

Crucial constraint: total W traffic == ceil(M/BLOCK_M) * K * N * 2 bytes.  With
K*N*2 == 235 MB, any config that splits M into 2 blocks reads the weight twice
and cannot beat a config that reads it once.  So BLOCK_M should cover M.
Also sweeps the AMD-specific knobs (matrix_instr_nonkdim, waves_per_eu, kpack).
"""
import sys, time, json, torch, triton
import triton.language as tl

dev = "cuda:0"
H, D, N = 128, 128, 7168
K = H * D
WBYTES = K * N * 2


@triton.jit
def _k(A_ptr, W_ptr, C_ptr, M, N, S,
       stride_ab, stride_ah, stride_as, stride_wn, stride_cm,
       K: tl.constexpr, D: tl.constexpr,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
       SPLIT_K: tl.constexpr, ONE_BATCH: tl.constexpr, EVEN_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EVEN_M:
        m_mask = tl.full((BLOCK_M,), 1, tl.int1)
        rm = offs_m
    else:
        m_mask = offs_m < M
        rm = tl.where(m_mask, offs_m, 0)
    if ONE_BATCH:
        a_row = rm * stride_as
    else:
        a_row = (rm // S) * stride_ab + (rm % S) * stride_as

    K_SPLIT: tl.constexpr = K // SPLIT_K
    offs_k = pid_k * K_SPLIT + tl.arange(0, BLOCK_K)
    a_col = (offs_k // D) * stride_ah + (offs_k % D)
    a_ptrs = A_ptr + a_row[:, None] + a_col[None, :]
    w_ptrs = W_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for _ in tl.range(0, K_SPLIT // BLOCK_K):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, b, acc)
        offs_k += BLOCK_K
        a_col = (offs_k // D) * stride_ah + (offs_k % D)
        a_ptrs = A_ptr + a_row[:, None] + a_col[None, :]
        w_ptrs += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=m_mask[:, None])
    else:
        tl.atomic_add(c_ptrs, acc, mask=m_mask[:, None], sem="relaxed")


def launch(a, w, cfg):
    B, _, S, _ = a.shape
    M = B * S
    BM, BN, BK, SK, nw, ns, nkd, wpe = cfg
    if SK == 1:
        c = torch.empty((B, S, N), device=dev, dtype=torch.bfloat16)
    else:
        c = torch.zeros((B, S, N), device=dev, dtype=torch.float32)
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM), SK)
    kw = {}
    if nkd:
        kw["matrix_instr_nonkdim"] = nkd
    if wpe:
        kw["waves_per_eu"] = wpe
    _k[grid](a, w, c, M, N, S,
             a.stride(0), a.stride(1), a.stride(2), w.stride(0), N,
             K=K, D=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
             ONE_BATCH=(B == 1), EVEN_M=(M % BM == 0),
             num_warps=nw, num_stages=ns, **kw)
    return c if SK == 1 else c.to(torch.bfloat16)


def bench(fn, iters=25, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def ref(a, w):
    b, h, s, d = a.shape
    return torch.matmul(a.transpose(1, 2).reshape(b, s, h * d), w.t())


torch.manual_seed(0)
w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

targets = [tuple(int(x) for x in t.split(",")) for t in sys.argv[1].split(";")]

out = {}
for (B, S) in targets:
    M = B * S
    a = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    r = ref(a, w)
    t_ref = bench(lambda: ref(a, w))
    cands = []
    for BM in (16, 32, 64, 128, 256):
        if BM < min(M, 16):
            continue
        nblk_m = -(-M // BM)
        if nblk_m > max(2, -(-M // 256)) * 2:
            continue
        for BN in (32, 64, 128, 256):
            for BK in (32, 64, 128, 256):
                for SK in (1, 2, 4, 8, 16):
                    if (K // SK) % BK:
                        continue
                    if BM * BK * 2 + BK * BN * 2 > 132 * 1024:
                        continue
                    if BM * BN > 128 * 256:
                        continue
                    nprog = (-(-N // BN)) * nblk_m * SK
                    if nprog < 128 or nprog > 8192:
                        continue
                    for nw in (4, 8):
                        for ns in (2,):
                            for nkd in (0, 16):
                                cands.append((BM, BN, BK, SK, nw, ns, nkd, 0))
    res = []
    for cfg in cands:
        try:
            o = launch(a, w, cfg)
            if o.shape != r.shape:
                continue
            d = (r.float() - o.float()).abs().max().item()
            if not (d < 5.0):
                continue
            t = bench(lambda: launch(a, w, cfg))
        except Exception:
            continue
        res.append((t, cfg))
    res.sort()
    wtraffic = (-(-M // res[0][1][0])) * WBYTES if res else 0
    print(f"\n== B={B} S={S} M={M}  ref={t_ref:.4f} ms   ({len(cands)} cfgs)", flush=True)
    for t, cfg in res[:8]:
        bw = ((-(-M // cfg[0])) * WBYTES + M * K * 2 + M * N * 2) / t * 1e-9
        print(f"   {t:8.4f} ms {t_ref/t:5.2f}x  BM{cfg[0]:<4}BN{cfg[1]:<4}BK{cfg[2]:<4}"
              f"SK{cfg[3]:<3}w{cfg[4]}s{cfg[5]}nkd{cfg[6]}  {bw:6.0f} GB/s", flush=True)
    out[f"{B}_{S}"] = res[:8]
    del a, r
    torch.cuda.empty_cache()

json.dump(out, open(f"sweep2_{sys.argv[2] if len(sys.argv)>2 else 'x'}.json", "w"), indent=1)
