import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_residual_kernel(
    A, W, R, O,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_rm, stride_rn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
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
    if not EVEN_M:
        offs_am = tl.where(offs_m < M, offs_m, 0)
    else:
        offs_am = offs_m
    if not EVEN_N:
        offs_wn = tl.where(offs_n < N, offs_n, 0)
    else:
        offs_wn = offs_n
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_wn = tl.max_contiguous(tl.multiple_of(offs_wn, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = W + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            kmask = offs_k[None, :] < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=kmask, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_wk

    # Match the reference's rounding: torch.matmul emits bfloat16, then the
    # elementwise add runs in float32 opmath and rounds once more.
    proj = acc.to(tl.bfloat16).to(tl.float32)

    r_ptrs = R + (offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    if EVEN_M and EVEN_N:
        r = tl.load(r_ptrs).to(tl.float32)
        tl.store(o_ptrs, (proj + r).to(tl.bfloat16))
    else:
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        r = tl.load(r_ptrs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptrs, (proj + r).to(tl.bfloat16), mask=mask)


_CONFIG_CACHE = {}


def _pick_config(M, N, K):
    key = (M, N, K)
    cfg = _CONFIG_CACHE.get(key)
    if cfg is not None:
        return cfg
    # Measured on MI355X (gfx950) by config sweep; see sweep.py.
    if M <= 512:
        bm, bn, bk, gm, w, s = 64, 128, 64, 4, 4, 2
    elif M <= 2048:
        bm, bn, bk, gm, w, s = 128, 128, 64, 4, 8, 2
    else:
        bm, bn, bk, gm, w, s = 128, 128, 64, 4, 8, 2
    cfg = (bm, bn, bk, gm, w, s)
    _CONFIG_CACHE[key] = cfg
    return cfg


@torch.no_grad()
def run(
    attn_output: torch.Tensor,
    residual: torch.Tensor,
    o_proj_weight: torch.Tensor,
) -> torch.Tensor:
    out_shape = attn_output.shape
    K = out_shape[-1]
    a = attn_output.reshape(-1, K)
    r = residual.reshape(-1, K)
    if not a.is_contiguous():
        a = a.contiguous()
    if not r.is_contiguous():
        r = r.contiguous()
    N = o_proj_weight.shape[0]
    M = a.shape[0]

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    bm, bn, bk, gm, nw, ns = _pick_config(M, N, K)

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _gemm_residual_kernel[grid](
        a, o_proj_weight, r, out,
        M, N, K,
        a.stride(0), a.stride(1),
        o_proj_weight.stride(0), o_proj_weight.stride(1),
        r.stride(0), r.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
        EVEN_M=(M % bm == 0), EVEN_N=(N % bn == 0), EVEN_K=(K % bk == 0),
        num_warps=nw, num_stages=ns,
        matrix_instr_nonkdim=16,
    )
    return out.view(out_shape)
