import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_bf16_inplace_kernel(x_ptr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * n_cols + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = (numerator / denominator).to(tl.bfloat16)
    tl.store(x_ptr + row * n_cols + cols, result, mask=cols < n_cols)


@triton.jit
def _scale_add_mask_kernel(
    scores_ptr,
    mask_ptr,
    scale,
    n_elements,
    plane,
    BLOCK: tl.constexpr,
):
    block_start = tl.program_id(0).to(tl.int64) * BLOCK
    stride = tl.num_programs(0).to(tl.int64) * BLOCK
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK)
        valid = offsets < n_elements
        score = tl.load(scores_ptr + offsets, mask=valid)
        # Preserve the eager reference's two distinct bfloat16 rounding points.
        scaled = (score.to(tl.float32) * scale).to(tl.bfloat16)
        batch = offsets // (32 * plane)
        within_plane = offsets % plane
        mask_value = tl.load(mask_ptr + batch * plane + within_plane, mask=valid)
        result = (scaled + mask_value).to(tl.bfloat16)
        tl.store(scores_ptr + offsets, result, mask=valid)
        block_start += stride


@triton.jit
def _qk_norm_rope_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    k_out_ptr,
    v_out_ptr,
    cos_ptr,
    sin_ptr,
    qw_ptr,
    kw_ptr,
    eps,
    n_q,
    seq_len,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    is_q = row < n_q
    q_row = row
    k_row = row - n_q

    # The two masked loads let Q (32 heads/token) and K (8 heads/token)
    # share one launch while retaining their contiguous projection layouts.
    qx = tl.load(q_ptr + q_row * 128 + d, mask=is_q)
    kx = tl.load(k_ptr + k_row * 128 + d, mask=~is_q)
    x = tl.where(is_q, qx, kx).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 128.0)
    normalized = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)

    qw = tl.load(qw_ptr + d)
    kw = tl.load(kw_ptr + d)
    w = tl.where(is_q, qw, kw)
    y = (normalized * w).to(tl.bfloat16)

    partner = tl.where(d < 64, d + 64, d - 64)
    # Recompute the partner's normalized value; all 128 source values are
    # already live in this program, so this becomes a lane shuffle in codegen.
    x_partner = tl.load(q_ptr + q_row * 128 + partner, mask=is_q)
    x_partner_k = tl.load(k_ptr + k_row * 128 + partner, mask=~is_q)
    x_partner = tl.where(is_q, x_partner, x_partner_k).to(tl.float32)
    wp = tl.where(is_q, tl.load(qw_ptr + partner), tl.load(kw_ptr + partner))
    y_partner = ((x_partner * tl.rsqrt(variance + eps)).to(tl.bfloat16) * wp).to(tl.bfloat16)
    rotated = tl.where(d < 64, -y_partner, y_partner)

    token = tl.where(is_q, q_row // 32, k_row // 8)
    c = tl.load(cos_ptr + token * 128 + d)
    s = tl.load(sin_ptr + token * 128 + d)
    yc = (y * c).to(tl.bfloat16)
    ys = (rotated * s).to(tl.bfloat16)
    out = (yc + ys).to(tl.bfloat16)
    tl.store(q_ptr + q_row * 128 + d, out, mask=is_q)

    # Materialize grouped K and V directly in (B, 32, S, 128) layout.
    # This replaces both expand/reshape copies from the eager formulation.
    token_k = k_row // 8
    kv_head = k_row % 8
    batch = token_k // seq_len
    pos = token_k % seq_len
    v = tl.load(v_ptr + k_row * 128 + d, mask=~is_q)
    for group in tl.static_range(0, 4):
        out_head = kv_head * 4 + group
        out_offset = ((batch * 32 + out_head) * seq_len + pos) * 128 + d
        tl.store(k_out_ptr + out_offset, out, mask=~is_q)
        tl.store(v_out_ptr + out_offset, v, mask=~is_q)


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    rms_norm_eps,
    scaling,
):
    batch_size, seq_len, _ = hidden_states.shape

    q = F.linear(hidden_states, q_proj_weight).view(batch_size, seq_len, 32, 128)
    k = F.linear(hidden_states, k_proj_weight).view(batch_size, seq_len, 8, 128)
    v = F.linear(hidden_states, v_proj_weight).view(batch_size, seq_len, 8, 128)

    n_q = batch_size * seq_len * 32
    n_k = batch_size * seq_len * 8
    k_expanded = torch.empty(
        (batch_size, 32, seq_len, 128), device=k.device, dtype=k.dtype
    )
    v_expanded = torch.empty_like(k_expanded)
    _qk_norm_rope_kernel[(n_q + n_k,)](
        q,
        k,
        v,
        k_expanded,
        v_expanded,
        cos,
        sin,
        q_norm_weight,
        k_norm_weight,
        rms_norm_eps,
        n_q,
        seq_len,
        BLOCK=128,
        num_warps=1,
    )

    q = q.transpose(1, 2)
    p = torch.matmul(q, k_expanded.transpose(2, 3))
    if seq_len >= 8192:
        # PyTorch's vector kernel uses a grid-stride loop that is marginally
        # better at this exceptionally large (2^31 element) boundary.
        p = p * scaling + attention_mask
    else:
        n_scores = batch_size * 32 * seq_len * seq_len
        score_grid = min(triton.cdiv(n_scores, 1024), 65536)
        _scale_add_mask_kernel[(score_grid,)](
            p,
            attention_mask,
            scaling,
            n_scores,
            seq_len * seq_len,
            BLOCK=1024,
            num_warps=8,
        )
    softmax_block = triton.next_power_of_2(seq_len)
    # A single AMD wave gives the best occupancy through 4096 columns; at
    # 8192, two waves reduce register pressure.
    softmax_warps = 1 if softmax_block <= 4096 else 2
    _softmax_bf16_inplace_kernel[(batch_size * 32 * seq_len,)](
        p,
        n_cols=seq_len,
        BLOCK=softmax_block,
        num_warps=softmax_warps,
    )
    x = torch.matmul(p, v_expanded).transpose(1, 2).contiguous().reshape(batch_size, seq_len, 4096)
    return F.linear(x, o_proj_weight)
