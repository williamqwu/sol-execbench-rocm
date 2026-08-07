import torch
import triton
import triton.language as tl


@triton.jit
def _value_projection_kernel(
    hidden_states,
    weight,
    output,
    seq_len: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    seq_block = tl.program_id(0)
    batch = tl.program_id(1)
    column_block = tl.program_id(2)

    rows_in_batch = seq_block * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)
    n_offsets = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    input_rows = batch * seq_len + rows_in_batch
    row_mask = rows_in_batch < seq_len

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 5120, BLOCK_K):
        ks = k_start + k_offsets
        lhs = tl.load(
            hidden_states + input_rows[:, None] * 5120 + ks[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        rhs = tl.load(weight + n_offsets[None, :] * 5120 + ks[:, None])
        accumulator += tl.dot(lhs, rhs)

    heads = n_offsets // 128
    dims = n_offsets % 128
    output_offsets = (
        ((batch * 8 + heads[None, :]) * seq_len + rows_in_batch[:, None]) * 128
        + dims[None, :]
    )
    tl.store(output + output_offsets, accumulator, mask=row_mask[:, None])


@torch.no_grad()
def run(hidden_states: torch.Tensor, v_proj_weight: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    total_rows = batch_size * seq_len
    use_fused_kernel = total_rows >= 3072 or 1920 <= total_rows <= 2080
    if use_fused_kernel:
        if total_rows >= 6144:
            block_m, block_n = 128, 256
        elif total_rows >= 3072:
            block_m, block_n = 128, 128
        else:
            block_m, block_n = 64, 128

        output = torch.empty(
            (batch_size, 8, seq_len, 128),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        grid = (
            triton.cdiv(seq_len, block_m),
            batch_size,
            triton.cdiv(1024, block_n),
        )
        _value_projection_kernel[grid](
            hidden_states,
            v_proj_weight,
            output,
            seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64,
            num_warps=8,
            num_stages=3,
        )
        return output

    projected = torch.nn.functional.linear(hidden_states, v_proj_weight)
    return projected.view(batch_size, seq_len, 8, 128).transpose(1, 2)
