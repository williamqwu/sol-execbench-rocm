import torch
import triton
import triton.language as tl


@triton.jit
def _rms_head_kernel(
    X,  # [n_rows, D] fp32
    W,  # [D] fp32
    Y,  # [n_rows, D] fp32
    n_rows,
    eps,
    D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < n_rows
    cols = tl.arange(0, D)
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]

    x = tl.load(X + offs, mask=mask[:, None], other=0.0)
    w = tl.load(W + cols)

    ss = tl.sum(x * x, axis=1) * (1.0 / D)
    denom = tl.sqrt(ss + eps)
    y = tl.fdiv(x, denom[:, None], ieee_rounding=True) * w[None, :]

    tl.store(Y + offs, y, mask=mask[:, None])


def _rms_head(x2d, w, eps):
    n_rows, D = x2d.shape
    y = torch.empty_like(x2d)
    BLOCK_R = 8
    grid = (triton.cdiv(n_rows, BLOCK_R),)
    _rms_head_kernel[grid](
        x2d, w, y, n_rows, eps,
        D=D, BLOCK_R=BLOCK_R, num_warps=4, num_stages=1,
    )
    return y


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

    n_rows = batch_size * seq_len * num_heads
    q2 = query.view(n_rows, head_dim)
    k2 = key.view(n_rows, head_dim)

    query_states = _rms_head(q2, q_norm_weight, eps).view(
        batch_size, seq_len, num_heads, head_dim
    )
    key_states = _rms_head(k2, k_norm_weight, eps).view(
        batch_size, seq_len, num_heads, head_dim
    )
    value_states = value.view(batch_size, seq_len, num_heads, head_dim)

    return query_states, key_states, value_states
