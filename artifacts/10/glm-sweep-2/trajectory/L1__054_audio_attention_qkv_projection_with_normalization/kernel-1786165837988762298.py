import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    out_ptr, in_ptr, norm_weight_ptr,
    n_rows, head_dim,
    stride_out_row, stride_out_head,
    stride_in_row, stride_in_head,
    eps,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row (batch*seq) and one head.
    # Layout: in_ptr is [n_rows, num_heads, head_dim] contiguous.
    pid = tl.program_id(0)
    row = pid // tl.num_programs(1)
    head = pid % tl.num_programs(1)
    if row >= n_rows:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < head_dim

    in_offset = row * stride_in_row + head * stride_in_head
    x = tl.load(in_ptr + in_offset + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm
    mean_sq = tl.sum(x * x, axis=0) / head_dim
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    out_offset = row * stride_out_row + head * stride_out_head
    tl.store(out_ptr + out_offset + offs, y, mask=mask)


def _rms_norm_triton(x, norm_weight, eps):
    # x: [batch, seq, num_heads, head_dim]
    num_heads = x.shape[2]
    head_dim = x.shape[3]
    n_rows = x.shape[0] * x.shape[1]
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(head_dim)
    grid = (n_rows, num_heads)
    _rms_norm_kernel[grid](
        out, x, norm_weight,
        n_rows, head_dim,
        out.stride(0), out.stride(2),
        x.stride(0), x.stride(2),
        eps,
        BLOCK_N=BLOCK_N,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 8
    head_dim = 128

    query = torch.matmul(hidden_states, q_proj_weight.t())
    key = torch.matmul(hidden_states, k_proj_weight.t())
    value = torch.matmul(hidden_states, v_proj_weight.t())

    query = query.view(batch_size, seq_len, num_heads, head_dim)
    key = key.view(batch_size, seq_len, num_heads, head_dim)
    value = value.view(batch_size, seq_len, num_heads, head_dim)

    query_states = _rms_norm_triton(query, q_norm_weight, eps)
    key_states = _rms_norm_triton(key, k_norm_weight, eps)

    return query_states, key_states, value
