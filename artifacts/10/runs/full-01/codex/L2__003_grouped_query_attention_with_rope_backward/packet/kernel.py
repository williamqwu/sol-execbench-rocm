import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_backward_scale(
    grad_attn_weights,
    attn_weights,
    grad_attn_scores,
    N_COLS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    offsets = row * N_COLS + cols

    grad = tl.load(grad_attn_weights + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weights = tl.load(attn_weights + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    correction = tl.sum(grad * weights, axis=0)
    score = weights * (grad - correction)

    # The reference explicitly rounds to bfloat16 before applying scaling.
    score = score.to(tl.bfloat16).to(tl.float32) * SCALE
    tl.store(grad_attn_scores + offsets, score, mask=mask)


@triton.jit
def _fused_scores_query(
    grad_attn_output,
    value_states,
    attn_weights,
    key_states,
    grad_attn_scores,
    grad_query_states,
    N_CTX: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 32
    head = batch_head % 32

    offs_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    # grad_attn_output is a transposed [B, S, H, D] view. Its sequence
    # stride is H*D and its head stride is D.
    do_base = batch * N_CTX * 4096 + head * HEAD_DIM
    do_ptrs = (
        grad_attn_output
        + do_base
        + offs_m[:, None] * 4096
        + offs_d[None, :]
    )
    grad_out = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    bh_base = batch_head * N_CTX * HEAD_DIM
    attn_base = batch_head * N_CTX * N_CTX
    correction = tl.zeros((BLOCK_M,), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        value_ptrs = (
            value_states
            + bh_base
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        values = tl.load(value_ptrs, mask=mask_n[:, None], other=0.0)
        grad_attn = tl.dot(grad_out, tl.trans(values), out_dtype=tl.float32)
        grad_attn = grad_attn.to(tl.bfloat16).to(tl.float32)
        weight_ptrs = (
            attn_weights
            + attn_base
            + offs_m[:, None] * N_CTX
            + offs_n[None, :]
        )
        weights = tl.load(
            weight_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        correction += tl.sum(grad_attn * weights, axis=1)

    query_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        value_ptrs = (
            value_states
            + bh_base
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        values = tl.load(value_ptrs, mask=mask_n[:, None], other=0.0)
        grad_attn = tl.dot(grad_out, tl.trans(values), out_dtype=tl.float32)
        grad_attn = grad_attn.to(tl.bfloat16).to(tl.float32)
        weight_ptrs = (
            attn_weights
            + attn_base
            + offs_m[:, None] * N_CTX
            + offs_n[None, :]
        )
        weights = tl.load(
            weight_ptrs,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = (weights * (grad_attn - correction[:, None])).to(
            tl.bfloat16
        )
        scores = (scores.to(tl.float32) * SCALE).to(tl.bfloat16)
        score_ptrs = (
            grad_attn_scores
            + attn_base
            + offs_m[:, None] * N_CTX
            + offs_n[None, :]
        )
        tl.store(
            score_ptrs,
            scores,
            mask=mask_m[:, None] & mask_n[None, :],
        )

        key_ptrs = (
            key_states
            + bh_base
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        keys = tl.load(key_ptrs, mask=mask_n[:, None], other=0.0)
        query_acc += tl.dot(scores, keys, out_dtype=tl.float32)

    query_ptrs = (
        grad_query_states
        + bh_base
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    tl.store(query_ptrs, query_acc, mask=mask_m[:, None])


@triton.jit
def _fused_key_value_grads(
    grad_attn_scores,
    query_states,
    attn_weights,
    grad_attn_output,
    grad_key_states,
    grad_value_states,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 32
    head = batch_head % 32
    offs_m = key_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    bh_vector_base = batch_head * N_CTX * HEAD_DIM
    bh_attn_base = batch_head * N_CTX * N_CTX
    do_base = batch * N_CTX * 4096 + head * HEAD_DIM
    key_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    value_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        matrix_ptrs = (
            bh_attn_base
            + offs_n[:, None] * N_CTX
            + offs_m[None, :]
        )
        scores = tl.load(
            grad_attn_scores + matrix_ptrs,
            mask=mask_n[:, None] & mask_m[None, :],
            other=0.0,
        )
        weights = tl.load(
            attn_weights + matrix_ptrs,
            mask=mask_n[:, None] & mask_m[None, :],
            other=0.0,
        )

        query_ptrs = (
            query_states
            + bh_vector_base
            + offs_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )
        queries = tl.load(query_ptrs, mask=mask_n[:, None], other=0.0)
        grad_out_ptrs = (
            grad_attn_output
            + do_base
            + offs_n[:, None] * 4096
            + offs_d[None, :]
        )
        grad_out = tl.load(grad_out_ptrs, mask=mask_n[:, None], other=0.0)
        key_acc += tl.dot(tl.trans(scores), queries, out_dtype=tl.float32)
        value_acc += tl.dot(tl.trans(weights), grad_out, out_dtype=tl.float32)

    output_ptrs = (
        bh_vector_base
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )
    tl.store(grad_key_states + output_ptrs, key_acc, mask=mask_m[:, None])
    tl.store(grad_value_states + output_ptrs, value_acc, mask=mask_m[:, None])


@triton.jit
def _pack_query_rope(
    grad_query,
    cos,
    sin,
    packed,
    SEQ_LEN: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // 32
    head = pid % 32
    batch = token // SEQ_LEN
    seq = token % SEQ_LEN
    d = tl.arange(0, 64)

    q_base = ((batch * 32 + head) * SEQ_LEN + seq) * 128
    trig_base = (batch * SEQ_LEN + seq) * 128
    q0 = tl.load(grad_query + q_base + d).to(tl.float32)
    q1 = tl.load(grad_query + q_base + 64 + d).to(tl.float32)
    c0 = tl.load(cos + trig_base + d).to(tl.float32)
    c1 = tl.load(cos + trig_base + 64 + d).to(tl.float32)
    s0 = tl.load(sin + trig_base + d).to(tl.float32)
    s1 = tl.load(sin + trig_base + 64 + d).to(tl.float32)

    q0c = (q0 * c0).to(tl.bfloat16).to(tl.float32)
    q1c = (q1 * c1).to(tl.bfloat16).to(tl.float32)
    q0s = (q0 * s0).to(tl.bfloat16).to(tl.float32)
    q1s = (q1 * s1).to(tl.bfloat16).to(tl.float32)

    out_base = token * 6144 + head * 128
    tl.store(packed + out_base + d, q0c + q1s)
    tl.store(packed + out_base + 64 + d, q1c - q0s)


@triton.jit
def _pack_key_value(
    grad_key,
    grad_value,
    cos,
    sin,
    packed,
    SEQ_LEN: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // 8
    kv_head = pid % 8
    batch = token // SEQ_LEN
    seq = token % SEQ_LEN
    d = tl.arange(0, 64)
    full_d0 = d
    full_d1 = d + 64

    head0 = kv_head * 4
    base0 = ((batch * 32 + head0) * SEQ_LEN + seq) * 128
    head_stride = SEQ_LEN * 128

    k00 = tl.load(grad_key + base0 + full_d0).to(tl.float32)
    k01 = tl.load(grad_key + base0 + full_d1).to(tl.float32)
    k10 = tl.load(grad_key + base0 + head_stride + full_d0).to(tl.float32)
    k11 = tl.load(grad_key + base0 + head_stride + full_d1).to(tl.float32)
    k20 = tl.load(grad_key + base0 + 2 * head_stride + full_d0).to(tl.float32)
    k21 = tl.load(grad_key + base0 + 2 * head_stride + full_d1).to(tl.float32)
    k30 = tl.load(grad_key + base0 + 3 * head_stride + full_d0).to(tl.float32)
    k31 = tl.load(grad_key + base0 + 3 * head_stride + full_d1).to(tl.float32)
    key0 = ((k00 + k10) + (k20 + k30)).to(tl.bfloat16).to(tl.float32)
    key1 = ((k01 + k11) + (k21 + k31)).to(tl.bfloat16).to(tl.float32)

    v00 = tl.load(grad_value + base0 + full_d0).to(tl.float32)
    v01 = tl.load(grad_value + base0 + full_d1).to(tl.float32)
    v10 = tl.load(grad_value + base0 + head_stride + full_d0).to(tl.float32)
    v11 = tl.load(grad_value + base0 + head_stride + full_d1).to(tl.float32)
    v20 = tl.load(grad_value + base0 + 2 * head_stride + full_d0).to(tl.float32)
    v21 = tl.load(grad_value + base0 + 2 * head_stride + full_d1).to(tl.float32)
    v30 = tl.load(grad_value + base0 + 3 * head_stride + full_d0).to(tl.float32)
    v31 = tl.load(grad_value + base0 + 3 * head_stride + full_d1).to(tl.float32)
    value0 = ((v00 + v10) + (v20 + v30)).to(tl.bfloat16).to(tl.float32)
    value1 = ((v01 + v11) + (v21 + v31)).to(tl.bfloat16).to(tl.float32)

    trig_base = (batch * SEQ_LEN + seq) * 128
    c0 = tl.load(cos + trig_base + full_d0).to(tl.float32)
    c1 = tl.load(cos + trig_base + full_d1).to(tl.float32)
    s0 = tl.load(sin + trig_base + full_d0).to(tl.float32)
    s1 = tl.load(sin + trig_base + full_d1).to(tl.float32)
    k0c = (key0 * c0).to(tl.bfloat16).to(tl.float32)
    k1c = (key1 * c1).to(tl.bfloat16).to(tl.float32)
    k0s = (key0 * s0).to(tl.bfloat16).to(tl.float32)
    k1s = (key1 * s1).to(tl.bfloat16).to(tl.float32)

    out_base = token * 6144 + kv_head * 128
    tl.store(packed + out_base + 4096 + full_d0, k0c + k1s)
    tl.store(packed + out_base + 4096 + full_d1, k1c - k0s)
    tl.store(packed + out_base + 5120 + full_d0, value0)
    tl.store(packed + out_base + 5120 + full_d1, value1)


@triton.jit
def _sum_hidden_grads(
    grad_q,
    grad_k,
    grad_v,
    output,
    N_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS
    q = tl.load(grad_q + offsets, mask=mask).to(tl.float32)
    k = tl.load(grad_k + offsets, mask=mask).to(tl.float32)
    v = tl.load(grad_v + offsets, mask=mask).to(tl.float32)
    # Match the two left-associative bfloat16 additions in the reference.
    qk = (q + k).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, qk + v, mask=mask)


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    cos,
    sin,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    query_states,
    key_states,
    value_states,
    attn_weights,
    attn_output,
    scaling,
):
    batch_size, seq_len, _ = hidden_states.shape
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128

    grad_attn_output = torch.matmul(grad_output, o_weight)
    grad_o_weight = torch.matmul(
        grad_output.reshape(-1, grad_output.shape[-1]).t(),
        attn_output.reshape(-1, attn_output.shape[-1]),
    )
    grad_attn_output = grad_attn_output.reshape(
        batch_size, seq_len, num_attention_heads, head_dim
    ).transpose(1, 2)

    use_fused_query = seq_len <= 160 or (batch_size == 1 and seq_len <= 384)
    if use_fused_query:
        grad_attn_scores = torch.empty_like(attn_weights)
        grad_query_states = torch.empty_like(query_states)
        if seq_len <= 160:
            block_m, block_n, num_warps = 32, 32, 4
        elif seq_len <= 256:
            block_m, block_n, num_warps = 32, 64, 4
        else:
            block_m, block_n, num_warps = 16, 32, 2
        _fused_scores_query[
            (triton.cdiv(seq_len, block_m), batch_size * num_attention_heads)
        ](
            grad_attn_output,
            value_states,
            attn_weights,
            key_states,
            grad_attn_scores,
            grad_query_states,
            N_CTX=seq_len,
            SCALE=scaling,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HEAD_DIM=head_dim,
            num_warps=num_warps,
        )
    else:
        grad_attn_weights = torch.matmul(
            grad_attn_output, value_states.transpose(2, 3)
        )
        grad_attn_scores = torch.empty_like(grad_attn_weights)
        block = triton.next_power_of_2(seq_len)
        rows = batch_size * num_attention_heads * seq_len
        if block <= 128:
            num_warps = 1
        elif block <= 1024:
            num_warps = 2
        else:
            num_warps = 4
        _softmax_backward_scale[(rows,)](
            grad_attn_weights,
            attn_weights,
            grad_attn_scores,
            N_COLS=seq_len,
            SCALE=scaling,
            BLOCK=block,
            num_warps=num_warps,
        )
        grad_query_states = torch.matmul(grad_attn_scores, key_states)

    if seq_len <= 160:
        grad_key_states = torch.empty_like(key_states)
        grad_value_states = torch.empty_like(value_states)
        _fused_key_value_grads[
            (triton.cdiv(seq_len, 32), batch_size * num_attention_heads)
        ](
            grad_attn_scores,
            query_states,
            attn_weights,
            grad_attn_output,
            grad_key_states,
            grad_value_states,
            N_CTX=seq_len,
            BLOCK_M=32,
            BLOCK_N=32,
            HEAD_DIM=head_dim,
            num_warps=4,
        )
    else:
        grad_value_states = torch.matmul(
            attn_weights.transpose(2, 3), grad_attn_output
        )
        grad_key_states = torch.matmul(
            grad_attn_scores.transpose(2, 3), query_states
        )

    tokens = batch_size * seq_len
    packed = torch.empty((tokens, 6144), device=hidden_states.device, dtype=torch.bfloat16)
    _pack_query_rope[(tokens * num_attention_heads,)](
        grad_query_states, cos, sin, packed, SEQ_LEN=seq_len, num_warps=1
    )
    _pack_key_value[(tokens * num_key_value_heads,)](
        grad_key_states,
        grad_value_states,
        cos,
        sin,
        packed,
        SEQ_LEN=seq_len,
        num_warps=1,
    )

    grad_query_proj = packed[:, :4096]
    grad_key_proj = packed[:, 4096:5120]
    grad_value_proj = packed[:, 5120:]

    grad_hidden_states_q = torch.mm(grad_query_proj, q_weight)
    grad_hidden_states_k = torch.mm(grad_key_proj, k_weight)
    grad_hidden_states_v = torch.mm(grad_value_proj, v_weight)

    grad_all_weight = torch.mm(packed.t(), hidden_states.reshape(tokens, 5120))
    grad_q_weight = grad_all_weight[:4096]
    grad_k_weight = grad_all_weight[4096:5120]
    grad_v_weight = grad_all_weight[5120:]

    grad_hidden_states = torch.empty_like(grad_hidden_states_q)
    hidden_elements = tokens * 5120
    _sum_hidden_grads[(triton.cdiv(hidden_elements, 256),)](
        grad_hidden_states_q,
        grad_hidden_states_k,
        grad_hidden_states_v,
        grad_hidden_states,
        N_ELEMENTS=hidden_elements,
        BLOCK=256,
        num_warps=1,
    )
    grad_hidden_states = grad_hidden_states.reshape(batch_size, seq_len, 5120)
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_o_weight.to(torch.bfloat16),
    )
