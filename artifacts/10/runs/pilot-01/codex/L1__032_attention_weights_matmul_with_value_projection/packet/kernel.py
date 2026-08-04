import torch
import triton
import triton.language as tl


@triton.jit
def _attn_value_kernel(
    attn_weights,
    value_states,
    output,
    seq_len: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 40
    head = pid_bh - batch * 40

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, 128)
    offs_k = tl.arange(0, block_k)

    attn_base = (batch * 40 + head) * seq_len * seq_len + offs_m[:, None] * seq_len
    value_base = (batch * 40 + head) * seq_len * 128 + offs_n[None, :]

    acc = tl.zeros((block_m, 128), tl.float32)
    for k0 in range(0, seq_len, block_k):
        k = k0 + offs_k
        a = tl.load(
            attn_weights + attn_base + k[None, :],
            mask=(offs_m[:, None] < seq_len) & (k[None, :] < seq_len),
            other=0.0,
        )
        v = tl.load(
            value_states + value_base + k[:, None] * 128,
            mask=k[:, None] < seq_len,
            other=0.0,
        )
        acc += tl.dot(a, v)

    out_offsets = (
        batch * seq_len * 5120
        + offs_m[:, None] * 5120
        + head * 128
        + offs_n[None, :]
    )
    tl.store(output + out_offsets, acc, mask=offs_m[:, None] < seq_len)


def _triton_run(
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
    block_m: int,
    block_k: int,
    num_warps: int,
    waves_per_eu: int = 0,
) -> torch.Tensor:
    batch_size = attn_weights.shape[0]
    seq_len = attn_weights.shape[2]
    output = torch.empty(
        (batch_size, seq_len, 5120), device=attn_weights.device, dtype=attn_weights.dtype
    )
    grid = (triton.cdiv(seq_len, block_m), batch_size * 40)
    if waves_per_eu:
        _attn_value_kernel[grid](
            attn_weights,
            value_states,
            output,
            seq_len,
            block_m,
            block_k,
            num_warps=num_warps,
            num_stages=3,
            waves_per_eu=waves_per_eu,
        )
    else:
        _attn_value_kernel[grid](
            attn_weights,
            value_states,
            output,
            seq_len,
            block_m,
            block_k,
            num_warps=num_warps,
            num_stages=3,
        )
    return output


def _direct_out_matmul(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch_size = attn_weights.shape[0]
    seq_len = attn_weights.shape[2]
    output = torch.empty(
        (batch_size, seq_len, 5120), device=attn_weights.device, dtype=attn_weights.dtype
    )
    output_as_heads = torch.as_strided(
        output,
        (batch_size, 40, seq_len, 128),
        (seq_len * 5120, 128, 5120, 1),
    )
    torch.matmul(attn_weights, value_states, out=output_as_heads)
    return output


@torch.no_grad()
def run(attn_weights: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    batch_size = attn_weights.shape[0]
    seq_len = attn_weights.shape[2]

    if batch_size == 1 and seq_len >= 800:
        return _direct_out_matmul(attn_weights, value_states)
    if batch_size == 1 and seq_len == 293:
        return _triton_run(attn_weights, value_states, 128, 64, 4)
    if batch_size == 1 and seq_len == 256:
        return _triton_run(attn_weights, value_states, 64, 32, 4)
    if seq_len <= 256:
        return _triton_run(attn_weights, value_states, 128, 32, 4)
    if seq_len <= 512:
        return _triton_run(attn_weights, value_states, 128, 32, 8)
    if seq_len < 800:
        return _triton_run(attn_weights, value_states, 64, 32, 4)
    return _triton_run(attn_weights, value_states, 64, 64, 8, 1)
