import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_bwd_scale(GO, AW, DS, M, N, sm,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    ptr = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :]
    go = tl.load(GO + ptr, mask=mask, other=0.0).to(tl.float32)
    aw = tl.load(AW + ptr, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(go * aw, axis=1)
    ds = aw * (go - s[:, None])
    dsb = ds.to(tl.bfloat16).to(tl.float32) * sm
    tl.store(DS + ptr, dsb.to(tl.bfloat16), mask=mask)


def _ds(grad_output, attn_weights, scaling):
    B, H, Sq, Sk = grad_output.shape
    M = B * H * Sq
    ds = torch.empty_like(grad_output)
    BLOCK_N = triton.next_power_of_2(Sk)
    BLOCK_M = max(1, min(16, 8192 // BLOCK_N))
    nw = 8 if BLOCK_N >= 2048 else 4
    grid = (triton.cdiv(M, BLOCK_M),)
    _softmax_bwd_scale[grid](grad_output, attn_weights, ds, M, Sk, float(scaling),
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                             num_warps=nw, num_stages=1)
    return ds


def run(grad_output, query, key, attn_weights, scaling):
    grad_output = grad_output.contiguous()
    attn_weights = attn_weights.contiguous()
    ds = _ds(grad_output, attn_weights, scaling)
    grad_query = torch.matmul(ds, key)
    grad_key = torch.matmul(ds.transpose(-2, -1), query)
    return grad_query, grad_key
