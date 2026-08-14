import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, N_ROWS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    x = tl.load(x_ptr + row * 1024 + cols).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / 1024.0)
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) * (1.0 / 1024.0)
    normalized = centered * tl.rsqrt(variance + 1.0e-5)
    weight = tl.load(weight_ptr + cols)
    bias = tl.load(bias_ptr + cols)
    tl.store(out_ptr + row * 1024 + cols, normalized * weight + bias)


@triton.jit
def _relative_flash_attention(
    q_ptr,
    kv_ptr,
    rel_ptr,
    out_ptr,
    scale,
    SEQ_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program computes BLOCK_M queries for one batch item and one head.
    # Query tiles never cross the 512-token context boundary.
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 8
    head = batch_head % 8

    tiles_per_context: tl.constexpr = 512 // BLOCK_M
    context = query_tile // tiles_per_context
    tile_in_context = query_tile % tiles_per_context
    context_start = context * 512
    m_start = tile_in_context * BLOCK_M
    context_len = tl.minimum(512, SEQ_LEN - context_start)

    m = m_start + tl.arange(0, BLOCK_M)
    d = tl.arange(0, 128)
    q_offsets = (
        (batch * SEQ_LEN + context_start + m[:, None]) * 1024
        + head * 128
        + d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=m[:, None] < context_len, other=0.0)

    neg_inf = float("-inf")
    running_max = tl.full((BLOCK_M,), neg_inf, tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 128), tl.float32)

    for n_start in range(0, 512, BLOCK_N):
        n = n_start + tl.arange(0, BLOCK_N)
        kv_base = (batch * SEQ_LEN + context_start + n[None, :]) * 2048
        k_offsets = kv_base + head * 128 + d[:, None]
        k = tl.load(
            kv_ptr + k_offsets,
            mask=n[None, :] < context_len,
            other=0.0,
        )
        qk = tl.dot(q, k)

        # For this (query,key) tile all relative distances form diagonals in a
        # BLOCK_M x BLOCK_N rectangle.  Compute the enclosing contiguous set
        # of relative projections with one MFMA, then gather its diagonals.
        rel_width: tl.constexpr = triton.next_power_of_2(BLOCK_M + BLOCK_N - 1)
        rel_col = tl.arange(0, rel_width)
        rel_base = 512 + m_start - n_start - (BLOCK_N - 1)
        rel_index = rel_base + rel_col
        rel_offsets = d[:, None] + rel_index[None, :] * 128
        rel_vectors = tl.load(
            rel_ptr + rel_offsets,
            mask=(rel_index[None, :] >= 0) & (rel_index[None, :] < 1025),
            other=0.0,
        )
        all_pos = tl.dot(q, rel_vectors)
        # The reference scales in fp32 and then rounds the positional bias to
        # bf16 before adding it to the content score.
        all_pos = (all_pos * scale).to(tl.bfloat16)
        diagonal = (
            tl.arange(0, BLOCK_M)[:, None]
            + (BLOCK_N - 1 - tl.arange(0, BLOCK_N))[None, :]
        )
        pos = tl.gather(all_pos, diagonal, axis=1)

        valid = (m[:, None] < context_len) & (n[None, :] < context_len)
        scores = tl.where(valid, qk * scale + pos, neg_inf)
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        probabilities = tl.exp(scores - new_max[:, None])
        correction = tl.exp(running_max - new_max)
        tile_sum = tl.sum(probabilities, axis=1)

        v_offsets = (
            (batch * SEQ_LEN + context_start + n[:, None]) * 2048
            + 1024
            + head * 128
            + d[None, :]
        )
        v = tl.load(
            kv_ptr + v_offsets,
            mask=n[:, None] < context_len,
            other=0.0,
        )
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.bfloat16), v)
        running_sum = running_sum * correction + tile_sum
        running_max = new_max

    result = accumulator / running_sum[:, None]
    out_offsets = (
        (batch * SEQ_LEN + context_start + m[:, None]) * 1024
        + head * 128
        + d[None, :]
    )
    tl.store(out_ptr + out_offsets, result, mask=m[:, None] < context_len)


@triton.jit
def _projected_relative_attention(
    q_ptr,
    kv_ptr,
    projected_rel_ptr,
    out_ptr,
    scale,
    SEQ_LEN: tl.constexpr,
    REL_WIDTH: tl.constexpr,
    REL_RADIUS: tl.constexpr,
    CONTEXT_CAP: tl.constexpr,
    CONTEXT_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // 8
    head = batch_head % 8
    tiles_per_context: tl.constexpr = 512 // BLOCK_M
    context = CONTEXT_BASE + query_tile // tiles_per_context
    tile_in_context = query_tile % tiles_per_context
    context_start = context * 512
    m_start = tile_in_context * BLOCK_M
    context_len = tl.minimum(512, SEQ_LEN - context_start)
    m = m_start + tl.arange(0, BLOCK_M)
    d = tl.arange(0, 128)
    q_offsets = (
        (batch * SEQ_LEN + context_start + m[:, None]) * 1024
        + head * 128
        + d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=m[:, None] < context_len, other=0.0)
    neg_inf = float("-inf")
    running_max = tl.full((BLOCK_M,), neg_inf, tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, 128), tl.float32)
    for n_start in range(0, CONTEXT_CAP, BLOCK_N):
        n = n_start + tl.arange(0, BLOCK_N)
        k_offsets = (
            (batch * SEQ_LEN + context_start + n[None, :]) * 2048
            + head * 128
            + d[:, None]
        )
        k = tl.load(kv_ptr + k_offsets, mask=n[None, :] < context_len, other=0.0)
        scores = tl.dot(q, k) * scale
        pos_offsets = (
            ((batch * SEQ_LEN + context_start + m[:, None]) * 8 + head) * REL_WIDTH
            + REL_RADIUS
            + m[:, None]
            - n[None, :]
        )
        pos = tl.load(
            projected_rel_ptr + pos_offsets,
            mask=(m[:, None] < context_len) & (n[None, :] < context_len),
            other=0.0,
        )
        valid = (m[:, None] < context_len) & (n[None, :] < context_len)
        scores = tl.where(valid, scores + pos, neg_inf)
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        probabilities = tl.exp(scores - new_max[:, None])
        correction = tl.exp(running_max - new_max)
        v_offsets = (
            (batch * SEQ_LEN + context_start + n[:, None]) * 2048
            + 1024
            + head * 128
            + d[None, :]
        )
        v = tl.load(kv_ptr + v_offsets, mask=n[:, None] < context_len, other=0.0)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(tl.bfloat16), v)
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        running_max = new_max
    result = accumulator / running_sum[:, None]
    out_offsets = (
        (batch * SEQ_LEN + context_start + m[:, None]) * 1024
        + head * 128
        + d[None, :]
    )
    tl.store(out_ptr + out_offsets, result, mask=m[:, None] < context_len)


@torch.no_grad()
def run(
    hidden_states,
    pre_norm_weight,
    pre_norm_bias,
    to_q_weight,
    to_kv_weight,
    to_out_weight,
    to_out_bias,
    rel_pos_emb_weight,
    scale,
):
    batch, seq_len, _ = hidden_states.shape
    rows = batch * seq_len
    if rows < 256:
        # At very small row counts the native launch has lower end-to-end
        # dispatch overhead; the Triton implementation wins thereafter.
        x = F.layer_norm(hidden_states, (1024,), pre_norm_weight, pre_norm_bias)
    else:
        x = torch.empty_like(hidden_states)
        _layer_norm_kernel[(rows,)](
            hidden_states,
            pre_norm_weight,
            pre_norm_bias,
            x,
            N_ROWS=rows,
            num_warps=1,
        )
    q = F.linear(x, to_q_weight)
    kv = F.linear(x, to_kv_weight)

    max_context = min(seq_len, 512)
    rel_radius = max_context - 1
    needed_rel_width = 2 * max_context - 1
    # Power-of-two N selects the fast hipBLASLt GEMM path.  The extra columns
    # are never indexed by attention (1023 -> 1024 for a full context).
    rel_width = (
        needed_rel_width
        if rows < 256
        else triton.next_power_of_2(needed_rel_width)
    )
    rel_start = 512 - rel_radius
    rel_slice = rel_pos_emb_weight[
        rel_start : rel_start + rel_width
    ]
    # addmm's alpha is applied to the fp32 GEMM accumulator before its bf16
    # output conversion, matching the positional-bias rounding in reference.py.
    projected_rel = torch.addmm(
        rel_pos_emb_weight[:1, :1],
        q.view(-1, 128),
        rel_slice.T,
        beta=0,
        alpha=scale,
    )

    y = torch.empty_like(q)
    full_contexts = seq_len // 512
    remainder = seq_len % 512
    if full_contexts and 0 < remainder < 64:
        # A tiny final context is cheaper as a second specialization than as
        # masked work inside every 512-key loop.
        parallel_contexts = batch * full_contexts * 8
        if parallel_contexts >= 64:
            block_m, block_n = 64, 16
        elif parallel_contexts >= 32:
            block_m, block_n = 32, 64
        else:
            block_m, block_n = 16, 64
        grid = (full_contexts * (512 // block_m), batch * 8)
        _projected_relative_attention[grid](
            q, kv, projected_rel, y, scale,
            SEQ_LEN=seq_len,
            REL_WIDTH=rel_width,
            REL_RADIUS=rel_radius,
            CONTEXT_CAP=512,
            CONTEXT_BASE=0,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
        block_m, block_n = 16, 32
        grid = (triton.cdiv(remainder, block_m), batch * 8)
        _projected_relative_attention[grid](
            q, kv, projected_rel, y, scale,
            SEQ_LEN=seq_len,
            REL_WIDTH=rel_width,
            REL_RADIUS=rel_radius,
            CONTEXT_CAP=triton.next_power_of_2(remainder),
            CONTEXT_BASE=full_contexts,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    else:
        contexts = math.ceil(seq_len / 512)
        parallel_contexts = batch * contexts * 8
        if max_context < 64:
            block_m, block_n = 16, 32
        elif max_context < 512:
            if batch <= 2:
                block_m, block_n = 16, 64
            elif batch < 32:
                block_m, block_n = 32, 64
            else:
                block_m, block_n = 64, 16
        elif parallel_contexts >= 64:
            block_m, block_n = 64, 16
        elif parallel_contexts >= 32:
            block_m, block_n = 32, 64
        else:
            block_m, block_n = 16, 64
        grid = (triton.cdiv(seq_len, block_m), batch * 8)
        if block_m == 64 and grid[0] * grid[1] >= 1024:
            _projected_relative_attention[grid](
                q, kv, projected_rel, y, scale,
                SEQ_LEN=seq_len,
                REL_WIDTH=rel_width,
                REL_RADIUS=rel_radius,
                CONTEXT_CAP=triton.next_power_of_2(max_context),
                CONTEXT_BASE=0,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
                waves_per_eu=4,
            )
        else:
            _projected_relative_attention[grid](
                q, kv, projected_rel, y, scale,
                SEQ_LEN=seq_len,
                REL_WIDTH=rel_width,
                REL_RADIUS=rel_radius,
                CONTEXT_CAP=triton.next_power_of_2(max_context),
                CONTEXT_BASE=0,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=4,
            )
    return F.linear(y, to_out_weight, to_out_bias)
