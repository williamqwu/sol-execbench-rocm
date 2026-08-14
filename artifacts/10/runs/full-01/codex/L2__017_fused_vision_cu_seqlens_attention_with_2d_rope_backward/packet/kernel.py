import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_sub_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softmax_backward_finish_kernel(
    grad_attn_ptr,
    weights_ptr,
    row_sum_ptr,
    output_ptr,
    scaling,
    n_elements,
    ROW_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    grad_attn = tl.load(grad_attn_ptr + offsets, mask=mask, other=0.0)
    weights = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    row_sum = tl.load(row_sum_ptr + offsets // ROW_SIZE, mask=mask, other=0.0)
    centered = _sub_rn(grad_attn, row_sum)
    result = _mul_rn(weights, centered)
    result = _mul_rn(result, scaling)
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _forward_rope_kernel(
    qkv_ptr,
    qk_out_ptr,
    cos_ptr,
    sin_ptr,
    n_tokens,
    HIDDEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    token = tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    mask = x < HIDDEN
    d = x & (HEAD_DIM - 1)
    rotated_x = x ^ (HEAD_DIM // 2)
    base = token * (3 * HIDDEN)
    emb_off = token * HEAD_DIM + d

    c = tl.load(cos_ptr + emb_off, mask=mask, other=0.0)
    s = tl.load(sin_ptr + emb_off, mask=mask, other=0.0)

    q = tl.load(qkv_ptr + base + x, mask=mask, other=0.0)
    qr = tl.load(qkv_ptr + base + rotated_x, mask=mask, other=0.0)
    qr = tl.where(d < HEAD_DIM // 2, -qr, qr)
    q_out = _add_rn(_mul_rn(q, c), _mul_rn(qr, s))

    k = tl.load(qkv_ptr + base + HIDDEN + x, mask=mask, other=0.0)
    kr = tl.load(qkv_ptr + base + HIDDEN + rotated_x, mask=mask, other=0.0)
    kr = tl.where(d < HEAD_DIM // 2, -kr, kr)
    k_out = _add_rn(_mul_rn(k, c), _mul_rn(kr, s))

    out_off = token * HIDDEN + x
    tl.store(qk_out_ptr + out_off, q_out, mask=mask)
    tl.store(qk_out_ptr + n_tokens * HIDDEN + out_off, k_out, mask=mask)


@triton.jit
def _attention_output_pack_kernel(
    attn_ptr,
    output_ptr,
    token_start,
    HIDDEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local_token = tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    mask = x < HIDDEN
    batch = local_token // LENGTH
    position = local_token - batch * LENGTH
    head = x // HEAD_DIM
    d = x & (HEAD_DIM - 1)
    input_off = ((batch * (HIDDEN // HEAD_DIM) + head) * LENGTH + position) * HEAD_DIM + d
    values = tl.load(attn_ptr + input_off, mask=mask, other=0.0)
    output_off = (token_start + local_token) * HIDDEN + x
    tl.store(output_ptr + output_off, values, mask=mask)


@triton.jit
def _backward_rope_pack_group_kernel(
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    grad_qkv_ptr,
    cos_ptr,
    sin_ptr,
    token_start,
    HIDDEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    local_token = tl.program_id(1)
    x = block * BLOCK + tl.arange(0, BLOCK)
    mask = x < HIDDEN
    batch = local_token // LENGTH
    position = local_token - batch * LENGTH
    h = x // HEAD_DIM
    d = x & (HEAD_DIM - 1)
    rotated_d = d ^ (HEAD_DIM // 2)
    input_base = (
        (batch * (HIDDEN // HEAD_DIM) + h) * LENGTH + position
    ) * HEAD_DIM
    global_token = token_start + local_token
    emb_base = global_token * HEAD_DIM

    c = tl.load(cos_ptr + emb_base + d, mask=mask, other=0.0)
    sr = tl.load(sin_ptr + emb_base + rotated_d, mask=mask, other=0.0)

    gq = tl.load(grad_q_ptr + input_base + d, mask=mask, other=0.0)
    gqr = tl.load(grad_q_ptr + input_base + rotated_d, mask=mask, other=0.0)
    q_rot_term = _mul_rn(gqr, sr)
    q_rot_term = tl.where(d < HEAD_DIM // 2, q_rot_term, -q_rot_term)
    q_out = _add_rn(_mul_rn(gq, c), q_rot_term)

    gk = tl.load(grad_k_ptr + input_base + d, mask=mask, other=0.0)
    gkr = tl.load(grad_k_ptr + input_base + rotated_d, mask=mask, other=0.0)
    k_rot_term = _mul_rn(gkr, sr)
    k_rot_term = tl.where(d < HEAD_DIM // 2, k_rot_term, -k_rot_term)
    k_out = _add_rn(_mul_rn(gk, c), k_rot_term)
    gv = tl.load(grad_v_ptr + input_base + d, mask=mask, other=0.0)

    output_base = global_token * (3 * HIDDEN)
    tl.store(grad_qkv_ptr + output_base + x, q_out, mask=mask)
    tl.store(grad_qkv_ptr + output_base + HIDDEN + x, k_out, mask=mask)
    tl.store(grad_qkv_ptr + output_base + 2 * HIDDEN + x, gv, mask=mask)


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    cos,
    sin,
    cu_seqlens,
    attention_dropout,
    scaling,
):
    seq_length = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_heads = 28
    head_dim = 128

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    value_states = qkv[:, 2]

    qk_states_rope = torch.empty(
        (2, seq_length, num_heads, head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    _forward_rope_kernel[
        (triton.cdiv(hidden_size, 256), seq_length)
    ](
        qkv,
        qk_states_rope,
        cos,
        sin,
        seq_length,
        HIDDEN=hidden_size,
        HEAD_DIM=head_dim,
        BLOCK=256,
        num_warps=4,
    )
    query_states_rope, key_states_rope = qk_states_rope.unbind(0)

    boundaries = cu_seqlens.tolist()
    lengths_list = [b - a for a, b in zip(boundaries, boundaries[1:])]

    # Adjacent sequences of an equal length are one strided-batched GEMM.  The
    # input generator makes almost every batch uniform, while this grouping
    # still handles arbitrary cu_seqlens correctly.
    groups = []
    seq_idx = 0
    token_idx = 0
    while seq_idx < len(lengths_list):
        length = lengths_list[seq_idx]
        run_count = 1
        while (
            seq_idx + run_count < len(lengths_list)
            and lengths_list[seq_idx + run_count] == length
        ):
            run_count += 1
        # rocBLAS changes SGEMM's reduction schedule above these batch sizes.
        # Keeping the same schedule as the per-sequence reference is required
        # by the tight fp32 tolerance.
        max_group = 4 if length >= 224 else 15
        remaining = run_count
        while remaining:
            count = min(remaining, max_group)
            groups.append((token_idx, count, length))
            token_idx += count * length
            remaining -= count
        seq_idx += run_count

    attn_output_reshaped = torch.empty(
        (seq_length, hidden_size), dtype=qkv.dtype, device=qkv.device
    )
    attn_weights_list = []
    qkv_groups = []
    for token_idx, count, length in groups:
        tokens = count * length
        q = (
            query_states_rope.narrow(0, token_idx, tokens)
            .reshape(count, length, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            key_states_rope.narrow(0, token_idx, tokens)
            .reshape(count, length, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            value_states.narrow(0, token_idx, tokens)
            .reshape(count, length, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        attn_weights.mul_(scaling)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output_group = torch.matmul(attn_weights, v)
        _attention_output_pack_kernel[
            (triton.cdiv(hidden_size, 256), tokens)
        ](
            attn_output_group,
            attn_output_reshaped,
            token_idx,
            HIDDEN=hidden_size,
            HEAD_DIM=head_dim,
            LENGTH=length,
            BLOCK=256,
            num_warps=4,
        )
        attn_weights_list.append(attn_weights)
        qkv_groups.append((q, k, v))

    grad_proj_bias = grad_output.sum(dim=0)
    grad_proj_weight = grad_output.t() @ attn_output_reshaped
    grad_attn_output = grad_output @ proj_weight
    grad_qkv = torch.empty(
        (seq_length, 3 * hidden_size),
        dtype=grad_attn_output.dtype,
        device=grad_attn_output.device,
    )
    for (token_idx, count, length), attn_weights, (q, k, v) in zip(
        groups, attn_weights_list, qkv_groups
    ):
        tokens = count * length
        grad_attn_out = (
            grad_attn_output.narrow(0, token_idx, tokens)
            .reshape(count, length, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        grad_attn_weights = torch.matmul(grad_attn_out, v.transpose(2, 3))
        grad_v = torch.matmul(attn_weights.transpose(2, 3), grad_attn_out)
        sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
        grad_attn_scores = grad_attn_weights
        n_attn = attn_weights.numel()
        _softmax_backward_finish_kernel[(triton.cdiv(n_attn, 256),)](
            grad_attn_weights,
            attn_weights,
            sum_grad,
            grad_attn_scores,
            scaling,
            n_attn,
            ROW_SIZE=length,
            BLOCK=256,
            num_warps=4,
        )
        grad_q = torch.matmul(grad_attn_scores, k)
        grad_k = torch.matmul(grad_attn_scores.transpose(2, 3), q)
        _backward_rope_pack_group_kernel[
            (triton.cdiv(hidden_size, 256), tokens)
        ](
            grad_q,
            grad_k,
            grad_v,
            grad_qkv,
            cos,
            sin,
            token_idx,
            HIDDEN=hidden_size,
            HEAD_DIM=head_dim,
            LENGTH=length,
            BLOCK=256,
            num_warps=4,
        )
    grad_hidden_states = grad_qkv @ qkv_weight
    grad_qkv_weight = grad_qkv.t() @ hidden_states
    grad_qkv_bias = grad_qkv.sum(dim=0)
    return (
        grad_hidden_states,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_proj_weight,
        grad_proj_bias,
    )
