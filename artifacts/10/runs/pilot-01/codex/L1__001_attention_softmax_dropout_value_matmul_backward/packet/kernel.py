import torch
import triton
import triton.language as tl


@triton.jit
def _scores_from_dropped_grad_kernel(
    grad_dropped,
    attn_weights,
    dropout_mask,
    out,
    n_cols: tl.constexpr,
    scale: tl.constexpr,
    apply_dropout: tl.constexpr,
    block_cols: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, block_cols)
    mask = offs < n_cols
    ptrs = row * n_cols + offs

    grad = tl.load(grad_dropped + ptrs, mask=mask, other=0.0)
    weights = tl.load(attn_weights + ptrs, mask=mask, other=0.0).to(tl.float32)
    if apply_dropout:
        keep = tl.load(dropout_mask + ptrs, mask=mask, other=0)
        grad = tl.where(keep, grad * scale, 0.0)

    sum_term = tl.sum(grad * weights, axis=0)
    scores = weights * (grad - sum_term)
    tl.store(out + ptrs, scores, mask=mask)


@triton.jit
def _direct_scores_kernel(
    grad_attn_output,
    value_states,
    attn_weights,
    dropout_mask,
    out,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    scale: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch = pid_bh // 80
    head = pid_bh - batch * 80
    kv_head = head // 10

    offs_m = pid_q * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    grad_ptrs = ((batch * q_len + offs_m[:, None]) * 80 + head) * 128 + offs_d[None, :]
    val_ptrs = ((batch * 8 + kv_head) * k_len + offs_n[None, :]) * 128 + offs_d[:, None]

    grad = tl.load(grad_attn_output + grad_ptrs, mask=offs_m[:, None] < q_len, other=0.0)
    vals = tl.load(value_states + val_ptrs, mask=offs_n[None, :] < k_len, other=0.0)
    dots = tl.dot(grad, vals, out_dtype=tl.float32)

    row_base = ((batch * 80 + head) * q_len + offs_m[:, None]) * k_len + offs_n[None, :]
    valid = (offs_m[:, None] < q_len) & (offs_n[None, :] < k_len)
    weights = tl.load(attn_weights + row_base, mask=valid, other=0.0).to(tl.float32)
    keep = tl.load(dropout_mask + row_base, mask=valid, other=0)
    dropped_grad = tl.where(keep, dots * scale, 0.0)
    sum_term = tl.sum(dropped_grad * weights, axis=1)
    scores = weights * (dropped_grad - sum_term[:, None])
    tl.store(out + row_base, scores, mask=valid)


@triton.jit
def _value_grad_kernel(
    attn_weights_dropped,
    grad_attn_output,
    out,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
    block_d: tl.constexpr,
    block_r: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bkv = tl.program_id(2)

    batch = pid_bkv // 8
    kv_head = pid_bkv - batch * 8

    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_d = pid_d * block_d + tl.arange(0, block_d)
    offs_r_base = tl.arange(0, block_r)
    acc = tl.zeros((block_k, block_d), tl.float32)

    for r_start in range(0, q_len * 10, block_r):
        offs_r = r_start + offs_r_base
        group = offs_r // q_len
        q = offs_r - group * q_len
        head = kv_head * 10 + group
        valid_r = offs_r < q_len * 10

        attn_ptrs = ((batch * 80 + head[None, :]) * q_len + q[None, :]) * k_len + offs_k[:, None]
        grad_ptrs = ((batch * q_len + q[:, None]) * 80 + head[:, None]) * 128 + offs_d[None, :]

        attn = tl.load(
            attn_weights_dropped + attn_ptrs,
            mask=(offs_k[:, None] < k_len) & valid_r[None, :],
            other=0.0,
        )
        grad = tl.load(
            grad_attn_output + grad_ptrs,
            mask=valid_r[:, None] & (offs_d[None, :] < 128),
            other=0.0,
        )
        acc += tl.dot(attn, grad, out_dtype=tl.float32)

    out_ptrs = ((batch * 8 + kv_head) * k_len + offs_k[:, None]) * 128 + offs_d[None, :]
    tl.store(out + out_ptrs, acc, mask=(offs_k[:, None] < k_len) & (offs_d[None, :] < 128))


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    batch_size = grad_attn_output.shape[0]
    seq_len_kv = value_states.shape[2]
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads

    grad_attn_scores = torch.empty_like(attn_weights)
    seq_len_kv = attn_weights.shape[-1]
    seq_len_q = grad_attn_output.shape[1]
    if seq_len_kv <= 256 and attention_dropout > 0.0:
        block_cols = _next_power_of_2(seq_len_kv)
        if block_cols <= 128:
            block_m = 128
        else:
            block_m = 64
        _direct_scores_kernel[(triton.cdiv(seq_len_q, block_m), batch_size * 80)](
            grad_attn_output,
            value_states,
            attn_weights,
            dropout_mask,
            grad_attn_scores,
            seq_len_q,
            seq_len_kv,
            1.0 / (1.0 - float(attention_dropout)),
            block_m,
            block_cols,
            128,
            num_warps=4,
        )
    else:
        grad_attn_output_t = grad_attn_output.transpose(1, 2).to(torch.float32)
        value_states_expanded = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, 128
        ).reshape(batch_size, num_attention_heads, seq_len_kv, 128)

        grad_attn_weights_dropped = torch.matmul(
            grad_attn_output_t,
            value_states_expanded.to(torch.float32).transpose(-2, -1),
        )
        rows = attn_weights.numel() // seq_len_kv
        block_cols = _next_power_of_2(seq_len_kv)
        num_warps = 8 if block_cols >= 2048 else 4
        _scores_from_dropped_grad_kernel[(rows,)](
            grad_attn_weights_dropped,
            attn_weights,
            dropout_mask,
            grad_attn_scores,
            seq_len_kv,
            1.0 / (1.0 - float(attention_dropout)),
            bool(attention_dropout > 0.0),
            block_cols,
            num_warps=num_warps,
        )

    grad_value_states = torch.empty(
        (batch_size, num_key_value_heads, seq_len_kv, 128),
        device=grad_attn_output.device,
        dtype=torch.bfloat16,
    )
    if seq_len_kv <= 512:
        if seq_len_kv == 512 and batch_size >= 16:
            block_k = 128
            block_r = 32 if batch_size == 16 else 64
        else:
            block_k = 64
            block_r = 64
    elif seq_len_kv <= 900:
        block_k = 128
        block_r = 32
    elif seq_len_kv <= 1024:
        block_k = 64
        block_r = 64
    elif seq_len_kv <= 2048:
        block_k = 256
        block_r = 32
    else:
        block_k = 128
        block_r = 64
    _value_grad_kernel[(triton.cdiv(seq_len_kv, block_k), 2, batch_size * 8)](
        attn_weights_dropped,
        grad_attn_output,
        grad_value_states,
        seq_len_q,
        seq_len_kv,
        block_k,
        64,
        block_r,
        num_warps=4,
    )

    return grad_attn_scores, grad_value_states
