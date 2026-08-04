"""Offline config sweep for the fused kernel. No verify attempts consumed."""
import itertools, json, sys, time, torch, triton
import triton.language as tl

dev = "cuda:0"
H, D, N = 128, 128, 7168
K = H * D


@triton.jit
def _k(
    A_ptr, W_ptr, C_ptr,
    M, N, S,
    stride_ab, stride_ah, stride_as,
    stride_wn, stride_cm,
    K: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, GROUP_M: tl.constexpr,
    ONE_BATCH: tl.constexpr, EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K_SPLIT // BLOCK_K):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
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


def launch(a, w, cfg, out=None):
    B, _, S, _ = a.shape
    M = B * S
    BM, BN, BK, SK, GM, nw, ns = cfg
    if SK == 1:
        c = torch.empty((B, S, N), device=dev, dtype=torch.bfloat16)
    else:
        c = torch.zeros((B, S, N), device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), SK)
    _k[grid](
        a, w, c, M, N, S,
        a.stride(0), a.stride(1), a.stride(2), w.stride(0), N,
        K=K, D=D, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK, GROUP_M=GM,
        ONE_BATCH=(B == 1), EVEN_M=(M % BM == 0),
        num_warps=nw, num_stages=ns,
    )
    return c if SK == 1 else c.to(torch.bfloat16)


def bench(fn, iters=20, warmup=5):
    try:
        for _ in range(warmup):
            fn()
    except Exception:
        return float("inf")
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

targets = [(1, 131), (1, 2048), (4, 512), (8, 512), (32, 128), (2, 512), (4, 128)]
if len(sys.argv) > 1:
    targets = [tuple(int(x) for x in t.split(",")) for t in sys.argv[1].split(";")]

cands = []
for BM in (16, 32, 64, 128, 256):
    for BN in (64, 128, 256):
        for BK in (64, 128, 256):
            for SK in (1, 2, 4, 8):
                for nw in (4, 8):
                    for ns in (1, 2):
                        if BM * BN > 256 * 256:
                            continue
                        if (K // SK) % BK:
                            continue
                        if BM * BK * 2 + BK * BN * 2 > 160 * 1024:
                            continue
                        cands.append((BM, BN, BK, SK, 8, nw, ns))

print(f"{len(cands)} candidate configs", flush=True)

best = {}
for (B, S) in targets:
    a = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
    r = ref(a, w)
    t_ref = bench(lambda: ref(a, w))
    res = []
    for cfg in cands:
        try:
            o = launch(a, w, cfg)
            if o.shape != r.shape:
                continue
            d = (r.float() - o.float()).abs().max().item()
            if not (d < 5.0):
                continue
        except Exception:
            continue
        t = bench(lambda: launch(a, w, cfg))
        res.append((t, cfg))
    res.sort()
    best[(B, S)] = res[:6]
    print(f"\n== B={B} S={S} M={B*S}  ref={t_ref:.3f} ms", flush=True)
    for t, cfg in res[:6]:
        print(f"   {t:8.3f} ms  {t_ref/t:5.2f}x  {cfg}", flush=True)
    del a, r
    torch.cuda.empty_cache()

json.dump({f"{b}_{s}": v for (b, s), v in best.items()}, open("sweep_out.json", "w"), indent=1)
