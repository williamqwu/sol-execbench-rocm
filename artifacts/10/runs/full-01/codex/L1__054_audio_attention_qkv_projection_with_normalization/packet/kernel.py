import torch
import triton
import triton.language as tl


@triton.jit
def _cat3_kernel(a_ptr, b_ptr, c_ptr, out_ptr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    which = tl.program_id(1)
    src = tl.where(which == 0, a_ptr, tl.where(which == 1, b_ptr, c_ptr))
    x = tl.load(src + offset, mask=offset < 1024 * 1024)
    tl.store(out_ptr + which * (1024 * 1024) + offset, x, mask=offset < 1024 * 1024)


@triton.jit
def _qk_rms_kernel(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_out_ptr,
    k_out_ptr,
    rows: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    IN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    row = block * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, 128)
    in_offsets = row[:, None] * IN_STRIDE + head * 128 + col[None, :]
    out_offsets = row[:, None] * OUT_STRIDE + head * 128 + col[None, :]
    mask = row[:, None] < rows

    q = tl.load(q_ptr + in_offsets, mask=mask, other=0.0)
    q_var = tl.sum(q * q, axis=1) * (1.0 / 128.0)
    q_scale = tl.load(q_weight_ptr + col)
    q_out = (q / tl.sqrt(q_var[:, None] + eps)) * q_scale[None, :]
    tl.store(q_out_ptr + out_offsets, q_out, mask=mask)

    k = tl.load(k_ptr + in_offsets, mask=mask, other=0.0)
    k_var = tl.sum(k * k, axis=1) * (1.0 / 128.0)
    k_scale = tl.load(k_weight_ptr + col)
    k_out = (k / tl.sqrt(k_var[:, None] + eps)) * k_scale[None, :]
    tl.store(k_out_ptr + out_offsets, k_out, mask=mask)


@torch.no_grad()
def run(
    hidden_states,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    q_norm_weight,
    k_norm_weight,
    eps,
):
    batch_size, seq_len, _ = hidden_states.shape

    rows = batch_size * seq_len
    if rows == 128:
        query = torch.matmul(hidden_states, q_proj_weight.t())
        key = torch.matmul(hidden_states, k_proj_weight.t())
        value = torch.matmul(hidden_states, v_proj_weight.t())
        in_stride = 1024
    else:
        qkv_weight = torch.empty(
            (3072, 1024), device=hidden_states.device, dtype=hidden_states.dtype
        )
        _cat3_kernel[(256, 3)](
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            qkv_weight,
            BLOCK=4096,
            num_warps=8,
        )
        qkv = torch.matmul(hidden_states, qkv_weight.t())
        query = qkv[..., :1024]
        key = qkv[..., 1024:2048]
        value = qkv[..., 2048:]
        in_stride = 3072

    query = query.view(batch_size, seq_len, 8, 128)
    key = key.view(batch_size, seq_len, 8, 128)
    value = value.view(batch_size, seq_len, 8, 128)

    if rows >= 8192:
        block_m = 32
    elif rows >= 4096:
        block_m = 16
    elif rows >= 1024:
        block_m = 8
    else:
        block_m = 4

    _qk_rms_kernel[(triton.cdiv(rows, block_m), 8)](
        query,
        key,
        q_norm_weight,
        k_norm_weight,
        query,
        key,
        rows,
        eps,
        BLOCK_M=block_m,
        IN_STRIDE=in_stride,
        OUT_STRIDE=in_stride,
        num_warps=4,
    )

    return query, key, value
