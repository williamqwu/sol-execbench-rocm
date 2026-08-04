import torch, triton, time, itertools, sys, json
import triton.language as tl

DEV = "cuda:0"
HEAD_DIM = 128
NUM_KV = 8


@triton.jit
def gemm(
    a_ptr, w_ptr, o_ptr,
    S, M,
    K: tl.constexpr, N: tl.constexpr, HD: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = N // BN

    num_in_group = GROUP_M * num_pid_n
    gid = pid // num_in_group
    first_m = gid * GROUP_M
    gsz = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % num_in_group) % gsz)
    pid_n = (pid % num_in_group) // gsz

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * (K // SPLIT_K) + tl.arange(0, BK)

    if EVEN_M:
        am = offs_m
    else:
        am = tl.where(offs_m < M, offs_m, 0)

    a_ptrs = a_ptr + am[:, None].to(tl.int64) * K + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in tl.range(0, K // (BK * SPLIT_K)):
        a = tl.load(a_ptrs)
        w = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(w), acc)
        a_ptrs += BK
        w_ptrs += BK

    # out index: m -> (b, s);  n -> (h, d)
    b = offs_m // S
    s = offs_m % S
    h = offs_n // HD
    d = offs_n % HD
    o_ptrs = (o_ptr
              + b[:, None].to(tl.int64) * (N * S)
              + h[None, :] * (HD * S)
              + s[:, None] * HD
              + d[None, :])
    mask = offs_m[:, None] < M
    if SPLIT_K == 1:
        if EVEN_M:
            tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty))
        else:
            tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask)
    else:
        if EVEN_M:
            tl.atomic_add(o_ptrs, acc, sem="relaxed")
        else:
            tl.atomic_add(o_ptrs, acc, mask=mask, sem="relaxed")


@triton.jit
def castk(src, dst, n, BLK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLK + tl.arange(0, BLK)
    m = o < n
    tl.store(dst + o, tl.load(src + o, mask=m, other=0.).to(dst.dtype.element_ty), mask=m)


def gt(fn, iters=200):
    try:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    except Exception as e:
        return None
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
    n = max(1, iters // 20)
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / (n * 20) * 1e6


torch.manual_seed(0)
W = torch.randn(1024, 5120, device=DEV, dtype=torch.bfloat16)
K_, N_ = 5120, 1024

SHAPES = [(1, 128), (8, 128), (16, 128), (4, 541), (64, 128), (1, 8192)]

CFGS = []
for BM in [16, 32, 64, 128, 256]:
    for BN in [64, 128, 256]:
        for BK in [64, 128, 256]:
            for SK in [1, 2, 4, 5, 8, 10, 16, 20]:
                if K_ % (BK * SK):
                    continue
                if BM * BN > 256 * 256:
                    continue
                if BM * BN * 4 // 256 > 512 and BM >= 128:
                    pass
                for nw in ([4, 8] if BM * BN >= 128 * 128 else [2, 4]):
                    CFGS.append((BM, BN, BK, SK, nw))

results = {}
for (b, s) in SHAPES:
    M = b * s
    h = torch.randn(b, s, K_, device=DEV, dtype=torch.bfloat16)
    ha = h.view(M, K_)
    ref = torch.nn.functional.linear(h, W).view(b, s, 8, 128).transpose(1, 2).contiguous()
    best = []
    for (BM, BN, BK, SK, nw) in CFGS:
        if BM > 64 and M < BM // 2:
            continue
        acc_bytes = BM * BN * 4
        if acc_bytes // nw > 64 * 256 * 4:
            continue
        try:
            if SK == 1:
                out = torch.empty(b, 8, s, 128, device=DEV, dtype=torch.bfloat16)
                grid = (triton.cdiv(M, BM) * (N_ // BN), 1)

                def f(out=out, grid=grid, BM=BM, BN=BN, BK=BK, SK=SK, nw=nw):
                    gemm[grid](ha, W, out, s, M, K_, N_, 128, BM, BN, BK, 8, SK,
                               (M % BM == 0), num_warps=nw, num_stages=2)
            else:
                out32 = torch.zeros(b, 8, s, 128, device=DEV, dtype=torch.float32)
                outb = torch.empty(b, 8, s, 128, device=DEV, dtype=torch.bfloat16)
                grid = (triton.cdiv(M, BM) * (N_ // BN), SK)
                nel = M * N_

                def f(out32=out32, outb=outb, grid=grid, BM=BM, BN=BN, BK=BK, SK=SK, nw=nw, nel=nel):
                    out32.zero_()
                    gemm[grid](ha, W, out32, s, M, K_, N_, 128, BM, BN, BK, 8, SK,
                               (M % BM == 0), num_warps=nw, num_stages=2)
                    castk[(triton.cdiv(nel, 1024),)](out32, outb, nel, 1024, num_warps=4)
            f(); torch.cuda.synchronize()
            got = out if SK == 1 else outb
            err = (got.float() - ref.float()).abs()
            thr = 0.0078 + 0.0078 * ref.float().abs()
            ok = (err <= thr).float().mean().item()
            if ok < 0.995:
                continue
            t = gt(f)
            if t is None:
                continue
            best.append((t, (BM, BN, BK, SK, nw), ok))
        except Exception as e:
            continue
    best.sort()
    results[(b, s)] = best[:6]
    print(f"=== B={b} S={s} M={M}")
    for t, c, ok in best[:6]:
        print(f"    {t:8.2f}us  BM={c[0]:3d} BN={c[1]:3d} BK={c[2]:3d} SK={c[3]:2d} nw={c[4]}  match={ok:.4f}")
    sys.stdout.flush()
    del h, ha, ref
    torch.cuda.empty_cache()
