import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fp32_mul(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32_add(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rope_exact_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    total,
    n_ctx: tl.constexpr,
    WIDTH: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for block_start in range(pid * BLOCK, total, GRID_SIZE * BLOCK):
        offsets = block_start + tl.arange(0, BLOCK).to(tl.int64)
        mask = offsets < total
        d = offsets % 128
        pair_offsets = tl.where(d < 64, offsets + 64, offsets - 64)
        token_linear = offsets // WIDTH
        rope_offsets = token_linear * 128 + d
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        pair = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0)
        rotated = tl.where(d < 64, -pair, pair)
        c = tl.load(cos_ptr + rope_offsets, mask=mask, other=0.0)
        s = tl.load(sin_ptr + rope_offsets, mask=mask, other=0.0)
        out = _fp32_add(_fp32_mul(x, c), _fp32_mul(rotated, s))
        tl.store(out_ptr + offsets, out, mask=mask)


def _rope_exact(x, cos, sin, heads):
    out = torch.empty_like(x)
    total = x.numel()
    block = 1024
    grid_size = min(triton.cdiv(total, block), 4096)
    _rope_exact_kernel[(grid_size,)](
        x,
        cos,
        sin,
        out,
        total,
        n_ctx=x.shape[1],
        WIDTH=heads * 128,
        GRID_SIZE=grid_size,
        BLOCK=block,
        num_warps=8,
        num_stages=1,
    )
    return out


@triton.jit
def _rope_qk_exact_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    total,
    GRID_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    for block_start in range(pid * BLOCK, total, GRID_SIZE * BLOCK):
        q_offsets = block_start + tl.arange(0, BLOCK).to(tl.int64)
        q_mask = q_offsets < total
        within_token = q_offsets % 4096
        d = within_token % 128
        q_pair_offsets = tl.where(d < 64, q_offsets + 64, q_offsets - 64)
        token_linear = q_offsets // 4096
        rope_offsets = token_linear * 128 + d

        c = tl.load(cos_ptr + rope_offsets, mask=q_mask, other=0.0)
        s = tl.load(sin_ptr + rope_offsets, mask=q_mask, other=0.0)
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        q_pair = tl.load(q_ptr + q_pair_offsets, mask=q_mask, other=0.0)
        q_rotated = tl.where(d < 64, -q_pair, q_pair)
        q_out = _fp32_add(_fp32_mul(q, c), _fp32_mul(q_rotated, s))
        tl.store(q_out_ptr + q_offsets, q_out, mask=q_mask)

        k_mask = q_mask & (within_token < 1024)
        k_offsets = token_linear * 1024 + within_token
        k_pair_offsets = tl.where(d < 64, k_offsets + 64, k_offsets - 64)
        k = tl.load(k_ptr + k_offsets, mask=k_mask, other=0.0)
        k_pair = tl.load(k_ptr + k_pair_offsets, mask=k_mask, other=0.0)
        k_rotated = tl.where(d < 64, -k_pair, k_pair)
        k_out = _fp32_add(_fp32_mul(k, c), _fp32_mul(k_rotated, s))
        tl.store(k_out_ptr + k_offsets, k_out, mask=k_mask)


def _rope_qk_exact(q, k, cos, sin):
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    total = q.numel()
    block = 1024
    grid_size = min(triton.cdiv(total, block), 4096)
    _rope_qk_exact_kernel[(grid_size,)](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        total,
        GRID_SIZE=grid_size,
        BLOCK=block,
        num_warps=8,
        num_stages=1,
    )
    return q_out, k_out


@triton.jit
def _scale_and_causal_mask(
    scores_ptr,
    total,
    n_ctx: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    matrix_size = n_ctx * n_ctx
    for block_start in range(pid * BLOCK, total, GRID_SIZE * BLOCK):
        # The 16K workload has more than 2**32 score elements.
        offsets = block_start + tl.arange(0, BLOCK).to(tl.int64)
        in_bounds = offsets < total
        within_matrix = offsets % matrix_size
        row = within_matrix // n_ctx
        col = within_matrix - row * n_ctx
        values = tl.load(scores_ptr + offsets, mask=in_bounds, other=0.0)
        values = tl.where(
            col > row,
            -float("inf"),
            values * 0.08838834764831845,
        )
        tl.store(scores_ptr + offsets, values, mask=in_bounds)


def _scale_mask_inplace(scores, seq_len):
    total = scores.numel()
    if seq_len >= 512:
        block = 4096
        grid_cap = 32768
        num_warps = 16
    else:
        block = 1024
        grid_cap = 4096
        num_warps = 8
    grid_size = min(triton.cdiv(total, block), grid_cap)
    _scale_and_causal_mask[(grid_size,)](
        scores,
        total,
        n_ctx=seq_len,
        GRID_SIZE=grid_size,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _gqa_flash_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_ctx: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_batch = tl.program_id(1)
    batch = head_batch // 32
    q_head = head_batch % 32
    kv_head = q_head // 4

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    pair_d = tl.where(offs_d < 64, offs_d + 64, offs_d - 64)

    q_base = batch * n_ctx * 4096 + q_head * 128
    q_offsets = q_base + offs_m[:, None] * 4096 + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=offs_m[:, None] < n_ctx, other=0.0)
    if APPLY_ROPE:
        qp_offsets = q_base + offs_m[:, None] * 4096 + pair_d[None, :]
        qp = tl.load(q_ptr + qp_offsets, mask=offs_m[:, None] < n_ctx, other=0.0)
        q_rot = tl.where(offs_d[None, :] < 64, -qp, qp)
        rope_base = batch * n_ctx * 128 + offs_m[:, None] * 128
        rope_mask = offs_m[:, None] < n_ctx
        q_cos = tl.load(cos_ptr + rope_base + offs_d[None, :], mask=rope_mask, other=0.0)
        q_sin = tl.load(sin_ptr + rope_base + offs_d[None, :], mask=rope_mask, other=0.0)
        q = q * q_cos + q_rot * q_sin

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    # Pass one finds a single maximum for each row.  Keeping this maximum
    # fixed in pass two avoids the repeated accumulator rescaling of the
    # usual one-pass online algorithm.
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_base = batch * n_ctx * 1024 + kv_head * 128
        k_offsets = kv_base + offs_n[:, None] * 1024 + offs_d[None, :]
        kv_mask = offs_n[:, None] < n_ctx
        k = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
        if APPLY_ROPE:
            kp_offsets = kv_base + offs_n[:, None] * 1024 + pair_d[None, :]
            kp = tl.load(k_ptr + kp_offsets, mask=kv_mask, other=0.0)
            k_rot = tl.where(offs_d[None, :] < 64, -kp, kp)
            k_rope_base = batch * n_ctx * 128 + offs_n[:, None] * 128
            k_cos = tl.load(cos_ptr + k_rope_base + offs_d[None, :], mask=kv_mask, other=0.0)
            k_sin = tl.load(sin_ptr + k_rope_base + offs_d[None, :], mask=kv_mask, other=0.0)
            k = k * k_cos + k_rot * k_sin

        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores *= 0.08838834764831845
        causal = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < n_ctx)
        scores = tl.where(causal, scores, -float("inf"))

        row_max = tl.maximum(row_max, tl.max(scores, axis=1))

    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_base = batch * n_ctx * 1024 + kv_head * 128
        k_offsets = kv_base + offs_n[:, None] * 1024 + offs_d[None, :]
        kv_mask = offs_n[:, None] < n_ctx
        k = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
        if APPLY_ROPE:
            kp_offsets = kv_base + offs_n[:, None] * 1024 + pair_d[None, :]
            kp = tl.load(k_ptr + kp_offsets, mask=kv_mask, other=0.0)
            k_rot = tl.where(offs_d[None, :] < 64, -kp, kp)
            k_rope_base = batch * n_ctx * 128 + offs_n[:, None] * 128
            k_cos = tl.load(cos_ptr + k_rope_base + offs_d[None, :], mask=kv_mask, other=0.0)
            k_sin = tl.load(sin_ptr + k_rope_base + offs_d[None, :], mask=kv_mask, other=0.0)
            k = k * k_cos + k_rot * k_sin

        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores *= 0.08838834764831845
        causal = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < n_ctx)
        scores = tl.where(causal, scores, -float("inf"))
        probs = tl.exp(scores - row_max[:, None])
        row_sum += tl.sum(probs, axis=1)

        v_offsets = kv_base + offs_n[:, None] * 1024 + offs_d[None, :]
        v = tl.load(v_ptr + v_offsets, mask=kv_mask, other=0.0)
        acc = tl.dot(probs, v, acc=acc, input_precision="ieee")

    acc /= row_sum[:, None]
    out_base = batch * n_ctx * 4096 + q_head * 128
    out_offsets = out_base + offs_m[:, None] * 4096 + offs_d[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=offs_m[:, None] < n_ctx)


def _flash_attention(q, k, v, cos, sin, apply_rope=True):
    if q.ndim == 4:
        batch_size, _, seq_len, _ = q.shape
        out = torch.empty(
            (batch_size, seq_len, 4096), device=q.device, dtype=q.dtype
        )
    else:
        batch_size, seq_len, _ = q.shape
        out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, 64), batch_size * 32)
    _gqa_flash_fwd[grid](
        q,
        k,
        v,
        cos,
        sin,
        out,
        n_ctx=seq_len,
        APPLY_ROPE=apply_rope,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=8,
        num_stages=1,
    )
    return out


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
):
    batch_size, seq_len, _ = hidden_states.shape

    query_states = F.linear(hidden_states, q_proj_weight)
    key_states = F.linear(hidden_states, k_proj_weight)
    value_states = F.linear(hidden_states, v_proj_weight)

    if query_states.numel() <= 8_500_000:
        query_states, key_states = _rope_qk_exact(
            query_states, key_states, cos, sin
        )
    else:
        query_states = _rope_exact(query_states, cos, sin, 32)
        key_states = _rope_exact(key_states, cos, sin, 8)

    query_states = query_states.view(batch_size, seq_len, 32, 128).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, 8, 128).transpose(1, 2)

    key_states = key_states[:, :, None, :, :].expand(
        batch_size, 8, 4, seq_len, 128
    ).reshape(batch_size, 32, seq_len, 128)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, 8, 4, seq_len, 128
    ).reshape(batch_size, 32, seq_len, 128)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    _scale_mask_inplace(attn_weights, seq_len)
    if seq_len >= 1024:
        torch.ops.aten._softmax.out(
            attn_weights, -1, False, out=attn_weights
        )
    else:
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        batch_size, seq_len, 4096
    )
    return F.linear(attn_output, o_proj_weight)
