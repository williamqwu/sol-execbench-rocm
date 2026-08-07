import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_swiglu(
    X, G, U, Out,
    M, N, K,
    stride_xm, stride_xk,
    stride_gn, stride_gk,
    stride_un, stride_uk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped ordering along M for better L2 reuse of the weight tiles
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_M:
        offs_am = offs_m
    else:
        offs_am = tl.where(offs_m < M, offs_m, 0)
    offs_bn = offs_n

    x_ptrs = X + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    g_ptrs = G + (offs_bn[:, None] * stride_gn + offs_k[None, :] * stride_gk)
    u_ptrs = U + (offs_bn[:, None] * stride_un + offs_k[None, :] * stride_uk)

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in tl.range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(x_ptrs)
            bg = tl.load(g_ptrs)
            bu = tl.load(u_ptrs)
        else:
            kmask = offs_k[None, :] < K - k0 * BLOCK_K
            a = tl.load(x_ptrs, mask=kmask, other=0.0)
            bg = tl.load(g_ptrs, mask=kmask, other=0.0)
            bu = tl.load(u_ptrs, mask=kmask, other=0.0)
        acc_g = tl.dot(a, tl.trans(bg), acc_g)
        acc_u = tl.dot(a, tl.trans(bu), acc_u)
        x_ptrs += BLOCK_K * stride_xk
        g_ptrs += BLOCK_K * stride_gk
        u_ptrs += BLOCK_K * stride_uk

    # --- epilogue: reproduce the reference's rounding exactly ---
    # torch.matmul(bf16, bf16) -> bf16 : round the fp32 accumulator to bf16
    g_bf = acc_g.to(tl.bfloat16)
    u_bf = acc_u.to(tl.bfloat16)

    gf = g_bf.to(tl.float32)
    inner = 0.7978845608028654 * (gf + 0.044715 * gf * gf * gf)
    # 0.5 * gf * (1 + tanh(inner))  ==  gf * sigmoid(2 * inner)
    act = gf * tl.sigmoid(2.0 * inner)
    act_bf = act.to(tl.bfloat16)

    out = (act_bf.to(tl.float32) * u_bf.to(tl.float32)).to(tl.bfloat16)

    out_ptrs = Out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    if EVEN_M:
        tl.store(out_ptrs, out)
    else:
        tl.store(out_ptrs, out, mask=offs_m[:, None] < M)


def _pick_config(M, N, K):
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
    if M <= 160:
        return (64, 64, 64, 1, 4, 2)
    if M <= 640:
        return (128, 64, 64, 1, 4, 2)
    return (128, 128, 64, 1, 8, 1)


def run(x: torch.Tensor, gate_proj: torch.Tensor, up_proj: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    K = orig_shape[-1]
    N = gate_proj.shape[0]
    x2 = x.reshape(-1, K)
    M = x2.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BM, BN, BK, GM, nw, ns = _pick_config(M, N, K)

    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _fused_gate_up_swiglu[grid](
        x2, gate_proj, up_proj, out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        gate_proj.stride(0), gate_proj.stride(1),
        up_proj.stride(0), up_proj.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
        EVEN_M=(M % BM == 0), EVEN_K=(K % BK == 0),
        num_warps=nw, num_stages=ns,
    )
    return out.reshape(*orig_shape[:-1], N)
