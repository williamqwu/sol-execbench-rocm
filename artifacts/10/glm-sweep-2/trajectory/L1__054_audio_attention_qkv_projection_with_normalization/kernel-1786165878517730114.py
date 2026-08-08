import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    out_ptr, in_ptr, norm_weight_ptr,
    n_rows, head_dim, eps,
    BLOCK_N: tl.constexpr,
):
    # in_ptr/out_ptr: [n_rows * num_heads, head_dim] flattened
    # Each program handles one (row, head) -> one flattened row.
    row = tl.program_id(0)
    n_total = tl.num_programs(0)
    if row >= n_total:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < head_dim

    x = tl.load(in_ptr + row * head_dim + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / head_dim
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(norm_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w

    tl.store(out_ptr + row * head_dim + offs, y, mask=mask)


def _rms_norm_triton(x, norm_weight, eps):
    # x: [batch, seq, num_heads, head_dim]
    num_heads = x.shape[2]
    head_dim = x.shape[3]
    n_rows = x.shape[0] * x.shape[1] * num_heads
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(head_dim)
    grid = (n_rows,)
    _rms_norm_kernel[grid](
        out, x, norm_weight,
        n_rows, head_dim, eps,
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

    query_states = _rms_norm_triton(query.contiguous(), q_norm_weight, eps)
    key_states = _rms_norm_triton(key.contiguous(), k_norm_weight, eps)

    return query_states, key_states, value
