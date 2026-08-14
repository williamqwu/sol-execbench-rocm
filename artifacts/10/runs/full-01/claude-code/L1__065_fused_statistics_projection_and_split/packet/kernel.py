import torch
import triton
import triton.language as tl


@triton.jit
def _conv1x1_kernel(
    X, W, Bs, O,
    T,
    stride_xb, stride_xc,
    stride_wm, stride_wk,
    stride_ob, stride_oc,
    M: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    n_mask = offs_n < T

    if EVEN_M:
        w_ptrs = W + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    else:
        w_ptrs = W + tl.minimum(offs_m, M - 1)[:, None] * stride_wm + offs_k[None, :] * stride_wk

    x_ptrs = X + pid_b * stride_xb + offs_k[:, None] * stride_xc + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(w_ptrs)
        b = tl.load(x_ptrs, mask=n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        w_ptrs += BLOCK_K * stride_wk
        x_ptrs += BLOCK_K * stride_xc

    bs = tl.load(Bs + offs_m, mask=offs_m < M, other=0.0)
    acc = acc + bs[:, None]

    o_ptrs = O + pid_b * stride_ob + offs_m[:, None] * stride_oc + offs_n[None, :]
    if EVEN_M:
        tl.store(o_ptrs, acc, mask=n_mask[None, :])
    else:
        tl.store(o_ptrs, acc, mask=n_mask[None, :] & (offs_m < M)[:, None])


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_DEFAULT = (128, 64, 64, 4, 2)


def _pick_cfg(B, M, T):
    return _DEFAULT


def _launch(x, weight, bias, cfg=None):
    B, K, T = x.shape
    M = weight.shape[0]

    if cfg is None:
        cfg = _pick_cfg(B, M, T)
    BM, BN, BK, nw, ns = cfg

    out = torch.empty((B, M, T), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(T, BN), triton.cdiv(M, BM), B)
    _conv1x1_kernel[grid](
        x, weight, bias, out,
        T,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        M=M, K=K,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        EVEN_M=(M % BM == 0),
        num_warps=nw, num_stages=ns,
    )
    return out


@torch.no_grad()
def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    if not x.is_contiguous():
        x = x.contiguous()
    out = _launch(x, weight.view(weight.shape[0], weight.shape[1]), bias)
    oc = weight.shape[0] // 2
    return out[:, :oc], out[:, oc:]
