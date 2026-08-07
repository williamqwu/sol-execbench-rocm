import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _rope_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    q_size,
    k_size,
    tokens,
    PACKED_QKV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_q = offs < q_size
    k_offs = offs - q_size
    is_k = (k_offs >= 0) & (k_offs < k_size)

    if PACKED_QKV:
        q_input_offs = (offs // 3584) * 4608 + offs % 3584
        k_input_offs = (k_offs // 512) * 4608 + k_offs % 512
    else:
        q_input_offs = offs
        k_input_offs = k_offs
    q_x = tl.load(q_ptr + q_input_offs, mask=is_q, other=0.0)
    k_x = tl.load(k_ptr + k_input_offs, mask=is_k, other=0.0)
    local = tl.where(is_q, offs, k_offs)
    x = tl.where(is_q, q_x, k_x)
    d = local % 128
    rot_delta = tl.where(d < 64, 64, -64)
    q_rot = tl.load(q_ptr + q_input_offs + rot_delta, mask=is_q, other=0.0)
    k_rot = tl.load(k_ptr + k_input_offs + rot_delta, mask=is_k, other=0.0)
    x_rot = tl.where(is_q, q_rot, k_rot)
    x_rot = tl.where(d < 64, -x_rot, x_rot)

    token = tl.where(is_q, offs // 3584, k_offs // 512)
    modality = tl.where(d < 32, 0, tl.where(d < 80, 1, 2))
    rope_offs = modality * tokens * 128 + token * 128 + d
    valid = is_q | is_k
    c = tl.load(cos_ptr + rope_offs, mask=valid, other=0.0)
    s = tl.load(sin_ptr + rope_offs, mask=valid, other=0.0)
    a = (x * c).to(tl.bfloat16)
    b = (x_rot * s).to(tl.bfloat16)
    y = (a + b).to(tl.bfloat16)
    tl.store(q_out_ptr + offs, y, mask=is_q)
    tl.store(k_out_ptr + k_offs, y, mask=is_k)


@triton.jit
def _gqa_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROUND_SCORES: tl.constexpr,
    V_STRIDE: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    bh = tl.program_id(1)
    batch = bh // 28
    q_head = bh % 28
    kv_head = q_head // 7

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)
    q_base = batch * N_CTX * 3584 + q_head * 128
    kv_base = batch * N_CTX * 512 + kv_head * 128
    v_base = batch * N_CTX * V_STRIDE + kv_head * 128
    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * 3584 + offs_d[None, :],
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(
            k_ptr + kv_base + cols[None, :] * 512 + offs_d[:, None],
            mask=cols[None, :] < N_CTX,
            other=0.0,
        )
        scores = tl.dot(q, k)
        if ROUND_SCORES:
            scores = scores.to(tl.bfloat16)
            scores = (scores * 0.08838834764831845).to(tl.bfloat16)
        else:
            scores *= 0.08838834764831845
        bias = tl.load(
            mask_ptr
            + batch * N_CTX * N_CTX
            + offs_m[:, None] * N_CTX
            + cols[None, :],
            mask=(offs_m[:, None] < N_CTX) & (cols[None, :] < N_CTX),
            other=-float("inf"),
        )
        scores += bias
        if ROUND_SCORES:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = tl.where(cols[None, :] < N_CTX, scores, -float("inf"))

        block_max = tl.maximum(row_max, tl.max(scores, axis=1))
        p = tl.exp2((scores - block_max[:, None]) * 1.4426950408889634)
        alpha = tl.exp2((row_max - block_max) * 1.4426950408889634)
        block_sum = tl.sum(p, axis=1)
        acc *= alpha[:, None]
        v = tl.load(
            v_ptr + v_base + cols[:, None] * V_STRIDE + offs_d[None, :],
            mask=cols[:, None] < N_CTX,
            other=0.0,
        )
        acc += tl.dot(p.to(tl.bfloat16), v)
        row_sum = row_sum * alpha + block_sum
        row_max = block_max

    acc /= row_sum[:, None]
    out_base = batch * N_CTX * 3584 + q_head * 128
    tl.store(
        out_ptr + out_base + offs_m[:, None] * 3584 + offs_d[None, :],
        acc,
        mask=offs_m[:, None] < N_CTX,
    )


@torch.no_grad()
def run(
    hidden_states,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    v_weight,
    v_bias,
    o_weight,
    cos,
    sin,
    attention_mask,
):
    bsz, q_len, _ = hidden_states.shape

    tokens = bsz * q_len
    packed_qkv = (
        600 <= tokens <= 2048
        or 6000 <= tokens <= 12000
        or (tokens >= 17000 and q_len >= 512)
    )
    if packed_qkv:
        qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
        qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
        q = qkv[..., :3584]
        k = qkv[..., 3584:4096]
        v = qkv[..., 4096:]
        v_stride = 4608
    else:
        q = F.linear(hidden_states, q_weight, q_bias)
        k = F.linear(hidden_states, k_weight, k_bias)
        v = F.linear(hidden_states, v_weight, v_bias)
        v_stride = 512
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    q_size = tokens * 3584
    k_size = tokens * 512
    if tokens == 128:
        rope_block = 128
    elif tokens <= 512:
        rope_block = 1024
    else:
        rope_block = 512
    _rope_qk_kernel[(triton.cdiv(q_size + k_size, rope_block),)](
        q,
        k,
        cos,
        sin,
        q_rope,
        k_rope,
        q_size,
        k_size,
        tokens,
        PACKED_QKV=packed_qkv,
        BLOCK=rope_block,
        num_warps=4,
    )
    out = torch.empty_like(q_rope)
    round_scores = True
    use_mfma16 = True
    block_n = 64
    if q_len == 541:
        block_m = 32
        block_n = 16
        num_warps = 2
        round_scores = False
    elif q_len == 128 and bsz == 1:
        block_m = 32
        num_warps = 4
        use_mfma16 = False
    elif q_len in (128, 256, 512):
        block_m = 64
        num_warps = 4
    elif q_len == 613:
        block_m = 32
        num_warps = 2
    elif q_len == 1024 and bsz != 2:
        block_m = 64
        num_warps = 4
    elif q_len >= 4096:
        block_m = 64
        num_warps = 4
    else:
        block_m = 128
        num_warps = 4
        use_mfma16 = False
    grid = (triton.cdiv(q_len, block_m), bsz * 28)
    if use_mfma16:
        _gqa_attention_kernel[grid](
            q_rope,
            k_rope,
            v,
            attention_mask,
            out,
            N_CTX=q_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ROUND_SCORES=round_scores,
            V_STRIDE=v_stride,
            num_warps=num_warps,
            num_stages=1,
            matrix_instr_nonkdim=16,
        )
    else:
        _gqa_attention_kernel[grid](
            q_rope,
            k_rope,
            v,
            attention_mask,
            out,
            N_CTX=q_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            ROUND_SCORES=round_scores,
            V_STRIDE=v_stride,
            num_warps=num_warps,
            num_stages=1,
        )
    return F.linear(out, o_weight)
