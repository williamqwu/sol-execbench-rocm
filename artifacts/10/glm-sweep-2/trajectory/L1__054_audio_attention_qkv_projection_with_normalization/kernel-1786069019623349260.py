import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_ob, stride_os, stride_oh, stride_od,
    HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
):
    # One program per (batch, seq, head) row.
    pid = tl.program_id(0)
    b = pid // (stride_xb and 1 or 1)  # placeholder; use 1D pid
    # Compute linear offset for this row.
    # We flatten (batch, seq, head) into a single index.
    row = pid
    # Offsets for the HEAD_DIM elements of this row.
    cols = tl.arange(0, HEAD_DIM)
    x = tl.load(x_ptr + row * stride_xh + cols * stride_xd)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    rms_inv = 1.0 / tl.sqrt(var + EPS)
    w = tl.load(w_ptr + cols)
    y = (x * rms_inv) * w
    tl.store(out_ptr + row * stride_oh + cols * stride_od, y)


def _triton_rms_norm(x, norm_weight, eps):
    # x: [batch, seq, num_heads, head_dim] contiguous in last dim
    b, s, h, d = x.shape
    x_row = x.reshape(b * s * h, d)
    out = torch.empty_like(x_row)
    total = b * s * h
    _rms_norm_kernel[(total,)](
        x_row, norm_weight, out,
        x_row.stride(0), 0, x_row.stride(0), 1,
        out.stride(0), 0, out.stride(0), 1,
        HEAD_DIM=d, EPS=eps,
    )
    return out.view(b, s, h, d)


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

    query = torch.matmul(hidden_states, q_proj_weight.t()).view(batch_size, seq_len, num_heads, head_dim)
    key = torch.matmul(hidden_states, k_proj_weight.t()).view(batch_size, seq_len, num_heads, head_dim)
    value = torch.matmul(hidden_states, v_proj_weight.t()).view(batch_size, seq_len, num_heads, head_dim)

    query_states = _triton_rms_norm(query, q_norm_weight, eps)
    key_states = _triton_rms_norm(key, k_norm_weight, eps)

    return query_states, key_states, value_states
