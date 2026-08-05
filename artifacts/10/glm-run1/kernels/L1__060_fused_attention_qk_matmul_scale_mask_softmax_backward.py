import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_bw_1d(
    grad_out_ptr,
    attn_w_ptr,
    out_ptr,
    scaling,
    SK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, SK, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < SK
        go = tl.load(grad_out_ptr + row * SK + cols, mask=mask, other=0.0).to(tl.float32)
        aw = tl.load(attn_w_ptr + row * SK + cols, mask=mask, other=0.0).to(tl.float32)
        acc += go * aw
    sum_grad = tl.sum(acc, axis=0)
    for off in range(0, SK, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < SK
        go = tl.load(grad_out_ptr + row * SK + cols, mask=mask, other=0.0).to(tl.float32)
        aw = tl.load(attn_w_ptr + row * SK + cols, mask=mask, other=0.0).to(tl.float32)
        out = aw * (go - sum_grad) * scaling
        tl.store(out_ptr + row * SK + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _softmax_bw_2d(
    grad_out_ptr,
    attn_w_ptr,
    out_ptr,
    N,
    scaling,
    SK: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROW_TILE
    rows = row_start + tl.arange(0, ROW_TILE)
    acc = tl.zeros([ROW_TILE, BLOCK], dtype=tl.float32)
    for off in range(0, SK, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = (rows[:, None] < N) & (cols[None, :] < SK)
        go = tl.load(grad_out_ptr + rows[:, None] * SK + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        aw = tl.load(attn_w_ptr + rows[:, None] * SK + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += go * aw
    sum_grad = tl.sum(acc, axis=1)
    for off in range(0, SK, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = (rows[:, None] < N) & (cols[None, :] < SK)
        go = tl.load(grad_out_ptr + rows[:, None] * SK + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        aw = tl.load(attn_w_ptr + rows[:, None] * SK + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        out = aw * (go - sum_grad[:, None]) * scaling
        tl.store(out_ptr + rows[:, None] * SK + cols[None, :], out.to(tl.bfloat16), mask=mask)


def _softmax_backward_scaled(grad_output, attn_weights, scaling):
    B, H, Sq, Sk = grad_output.shape
    N = B * H * Sq
    go = grad_output.reshape(N, Sk)
    aw = attn_weights.reshape(N, Sk)
    out = torch.empty_like(grad_output).reshape(N, Sk)
    if Sk >= 1024:
        BLOCK = min(2048, triton.next_power_of_2(Sk))
        grid = (N,)
        _softmax_bw_1d[grid](go, aw, out, scaling, SK=Sk, BLOCK=BLOCK, num_warps=4, num_stages=2)
    else:
        BLOCK = min(triton.next_power_of_2(Sk), 1024)
        rt = max(1, 1024 // Sk)
        rt = triton.next_power_of_2(rt)
        rt = max(rt, 1)
        max_rt = max(1, N // 256)
        rt = min(rt, triton.next_power_of_2(max_rt))
        rt = max(rt, 1)
        while rt * BLOCK > 8192 and rt > 1:
            rt //= 2
        grid = (triton.cdiv(N, rt),)
        _softmax_bw_2d[grid](go, aw, out, N, scaling, SK=Sk, BLOCK=BLOCK, ROW_TILE=rt, num_warps=4, num_stages=2)
    return out.reshape(B, H, Sq, Sk)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    attn_weights: torch.Tensor,
    scaling: float,
):
    grad_scaled_logits = _softmax_backward_scaled(grad_output, attn_weights, scaling)
    grad_query = torch.matmul(grad_scaled_logits, key)
    grad_key = torch.matmul(grad_scaled_logits.transpose(-2, -1), query)
    return grad_query, grad_key
