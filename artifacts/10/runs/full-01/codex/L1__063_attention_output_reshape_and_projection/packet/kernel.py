import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_major(
    attn_ptr,
    reshaped_ptr,
    SEQ: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program owns complete 128-element head fragments from one logical
    # output row.  Both the loads and stores are therefore fully coalesced.
    m = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    batch = m // SEQ
    seq = m - batch * SEQ
    head = k // 128
    dim = k - head * 128
    source = (((batch * 128 + head) * SEQ + seq) * 128 + dim)
    tl.store(reshaped_ptr + m * K + k, tl.load(attn_ptr + source))


@triton.jit
def _copy_column_major(
    attn_ptr,
    reshaped_t_ptr,
    M: tl.constexpr,
    LD: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # For aligned large M, rocBLAS is faster when its left operand has unit
    # stride in M.  Transpose 2-D tiles while performing the attention-layout
    # permutation so the following GEMM can use that path.
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    batch = m // SEQ
    seq = m - batch * SEQ
    head = k // 128
    dim = k - head * 128
    source = (((batch[:, None] * 128 + head[None, :]) * SEQ + seq[:, None]) * 128 + dim[None, :])
    mask = m[:, None] < M
    values = tl.load(attn_ptr + source, mask=mask, other=0.0)
    destination = k[:, None] * LD + m[None, :]
    tl.store(reshaped_t_ptr + destination, tl.trans(values), mask=tl.trans(mask))


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    batch, _, seq_len, _ = attn_output.shape
    m = batch * seq_len
    n, k = o_proj_weight.shape

    # Padding a nearly aligned M selects rocBLAS's substantially faster
    # power-of-two column-major kernel.  The padded result rows are discarded.
    aligned_m = 1 << (m - 1).bit_length()
    if m >= 1536 and aligned_m - m <= 256:
        reshaped_t = torch.empty((k, aligned_m), device=attn_output.device, dtype=attn_output.dtype)
        _copy_column_major[(triton.cdiv(m, 128), k // 32)](
            attn_output,
            reshaped_t,
            M=m,
            LD=aligned_m,
            SEQ=seq_len,
            BLOCK_M=128,
            BLOCK_K=32,
            num_warps=4,
        )
        projected = torch.mm(reshaped_t.t(), o_proj_weight.t())
        return projected[:m].view(batch, seq_len, n)

    reshaped = torch.empty((m, k), device=attn_output.device, dtype=attn_output.dtype)
    _copy_row_major[(m, k // 2048)](
        attn_output,
        reshaped,
        SEQ=seq_len,
        K=k,
        BLOCK=2048,
        num_warps=4,
    )
    return torch.mm(reshaped, o_proj_weight.t()).view(batch, seq_len, n)
