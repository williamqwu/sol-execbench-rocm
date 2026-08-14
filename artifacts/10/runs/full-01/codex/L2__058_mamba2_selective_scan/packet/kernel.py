import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _causal_conv_silu(
    projected_ptr, weight_ptr, bias_ptr, out_ptr,
    seq_len, n_t_blocks: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    bt_and_b = tl.program_id(1)
    batch = bt_and_b // n_t_blocks
    t_block = bt_and_b - batch * n_t_blocks
    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (t[:, None] < seq_len) & (c[None, :] < 20480)

    bias = tl.load(bias_ptr + c, mask=c < 20480, other=0.0)
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)
    for k in range(4):
        source_t = t + k - 3
        value = tl.load(
            projected_ptr
            + (batch * seq_len + source_t[:, None]) * 37120
            + 16384 + c[None, :],
            mask=mask & (source_t[:, None] >= 0),
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + c * 4 + k,
            mask=c < 20480,
            other=0.0,
        ).to(tl.float32)
        acc += value * weight[None, :]

    # MIOpen rounds the convolution accumulator to BF16 before its bias
    # epilogue, and the BF16 bias addition rounds once more.
    conv = (
        acc.to(tl.bfloat16).to(tl.float32) + bias.to(tl.float32)[None, :]
    ).to(tl.bfloat16)
    sigmoid = (
        1.0 / (1.0 + libdevice.exp(-conv.to(tl.float32)))
    ).to(tl.bfloat16)
    result = (conv.to(tl.float32) * sigmoid.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        out_ptr + (batch * seq_len + t[:, None]) * 20480 + c[None, :],
        result,
        mask=mask,
    )


@triton.jit
def _build_chunk_decay(a_ptr, out_ptr, BLOCK: tl.constexpr):
    outer = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    row = offs[:, None]
    col = offs[None, :]
    a = tl.load(a_ptr + outer * BLOCK + offs)
    expanded = tl.zeros((BLOCK, BLOCK), tl.float32) + a[:, None]
    masked = tl.where(row > col, expanded, 0.0)
    segment = tl.cumsum(masked, axis=0)
    value = libdevice.exp(segment)
    value = tl.where(col <= row, value, 0.0)
    tl.store(out_ptr + outer * BLOCK * BLOCK + row * BLOCK + col, value)


@triton.jit
def _scan_parts(
    x_ptr, B_ptr, C_ptr, dt_ptr, A_log_ptr, parts_ptr,
    seq_len,
    stride_xb: tl.constexpr, stride_xt: tl.constexpr, stride_xc: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # A program owns four dimensions for all 32 heads which share one of the
    # eight B/C groups, and one 64-wide slice of the state dimension.
    pid = tl.program_id(0)
    n_block = pid % 4
    pid = pid // 4
    d_block = pid % 16
    pid = pid // 16
    group = pid % 8
    batch = pid // 8

    r = tl.arange(0, 32)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    heads = r * 8 + group

    a_scale = -tl.exp(tl.load(A_log_ptr + heads).to(tl.float32))
    state = tl.zeros((32, BLOCK_D, BLOCK_N), tl.float32)

    for t in range(0, seq_len):
        dt = tl.load(dt_ptr + (batch * seq_len + t) * 256 + heads).to(tl.float32)
        decay = tl.exp(a_scale * dt)
        x = tl.load(
            x_ptr + batch * stride_xb + t * stride_xt
            + heads[:, None] * 64 * stride_xc + d[None, :] * stride_xc
        ).to(tl.float32)
        b = tl.load(
            B_ptr + batch * stride_xb + t * stride_xt
            + (group * 256 + n) * stride_xc
        ).to(tl.float32)
        state = state * decay[:, None, None] + (x * dt[:, None])[:, :, None] * b[None, None, :]
        c = tl.load(
            C_ptr + batch * stride_xb + t * stride_xt
            + (group * 256 + n) * stride_xc
        ).to(tl.float32)
        value = tl.sum(state * c[None, None, :], axis=2)
        out_idx = (
            (((batch * seq_len + t) * 256 + heads[:, None]) * 64 + d[None, :]) * 4
            + n_block
        )
        tl.store(parts_ptr + out_idx, value)


@triton.jit
def _finish_norm_gate(
    parts_ptr, x_ptr, D_ptr, gate_ptr, norm_ptr, out_ptr,
    seq_len, epsilon,
    stride_xb: tl.constexpr, stride_xt: tl.constexpr, stride_xc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    norm_group = pid % 8
    token = pid // 8
    batch = token // seq_len
    t = token - batch * seq_len

    offs = tl.arange(0, BLOCK)
    channel = norm_group * BLOCK + offs
    head = channel // 64
    d = channel % 64
    nb = tl.arange(0, 4)
    part_idx = (((token * 256 + head) * 64 + d) * 4)[:, None] + nb[None, :]
    y = tl.sum(tl.load(parts_ptr + part_idx), axis=1)
    raw_x = tl.load(
        x_ptr + batch * stride_xb + t * stride_xt + channel * stride_xc
    ).to(tl.float32)
    y += tl.load(D_ptr + head).to(tl.float32) * raw_x

    # These casts reproduce the explicit dtype boundaries in the reference.
    y_bf = y.to(tl.bfloat16)
    y_f = y_bf.to(tl.float32)
    variance = tl.sum(y_f * y_f, axis=0) * (1.0 / BLOCK)
    normalized = (y_f * tl.rsqrt(variance + epsilon)).to(tl.bfloat16)
    weighted = (
        normalized.to(tl.float32) * tl.load(norm_ptr + channel).to(tl.float32)
    ).to(tl.bfloat16)
    gate = tl.load(gate_ptr + token * 37120 + channel).to(tl.float32)
    gate_sigmoid = tl.sigmoid(gate).to(tl.bfloat16)
    gate_silu = (gate * gate_sigmoid.to(tl.float32)).to(tl.bfloat16)
    result = (
        weighted.to(tl.float32) * gate_silu.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(out_ptr + token * 16384 + channel, result)


@triton.jit
def _finish_grouped_scan(
    y_diag_ptr, y_off_ptr, raw_x_ptr, D_ptr, gate_ptr, norm_ptr, out_ptr,
    epsilon,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    norm_group = pid % 8
    token = pid // 8
    offs = tl.arange(0, BLOCK)
    channel = norm_group * BLOCK + offs
    idx = token * 16384 + channel
    head = channel // 64

    y = tl.load(y_diag_ptr + idx) + tl.load(y_off_ptr + idx)
    y += tl.load(D_ptr + head).to(tl.float32) * tl.load(raw_x_ptr + idx)
    y_bf = y.to(tl.bfloat16)
    y_f = y_bf.to(tl.float32)
    variance = tl.sum(y_f * y_f, axis=0) * (1.0 / BLOCK)
    normalized = (y_f * tl.rsqrt(variance + epsilon)).to(tl.bfloat16)
    weighted = (
        normalized.to(tl.float32) * tl.load(norm_ptr + channel).to(tl.float32)
    ).to(tl.bfloat16)
    gate = tl.load(gate_ptr + token * 37120 + channel).to(tl.float32)
    gate_sigmoid = tl.sigmoid(gate).to(tl.bfloat16)
    gate_silu = (gate * gate_sigmoid.to(tl.float32)).to(tl.bfloat16)
    result = (
        weighted.to(tl.float32) * gate_silu.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(out_ptr + idx, result)


@torch.no_grad()
def run(
    hidden_states,
    in_proj_weight,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A_log,
    D,
    norm_weight,
    out_proj_weight,
    time_step_limit_min,
    time_step_limit_max,
    layer_norm_epsilon,
):
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = 256
    head_dim = 64
    intermediate_size = 16384
    state_size = 256
    n_groups = 8
    chunk_size = 128
    group_state_size = n_groups * state_size
    conv_dim = intermediate_size + 2 * group_state_size

    projected = torch.matmul(hidden_states, in_proj_weight.t())
    gate = projected[..., :intermediate_size]
    hidden_B_C = projected[..., intermediate_size:intermediate_size + conv_dim]
    dt = projected[..., -num_heads:]

    hidden_B_C = torch.empty(
        (batch_size, seq_len, conv_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    n_t_blocks = triton.cdiv(seq_len, 16)
    _causal_conv_silu[(triton.cdiv(conv_dim, 256), batch_size * n_t_blocks)](
        projected, conv1d_weight, conv1d_bias, hidden_B_C,
        seq_len, n_t_blocks=n_t_blocks,
        BLOCK_T=16, BLOCK_C=256,
        num_warps=8,
    )
    x = hidden_B_C[..., :intermediate_size]
    B = hidden_B_C[..., intermediate_size:intermediate_size + group_state_size]
    C = hidden_B_C[..., intermediate_size + group_state_size:]

    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)
    # Keep the 32-way head repetition as a logical axis.  B and C are shared
    # by heads (repeat_index, group), so their inner product only needs to be
    # formed for the eight groups.
    repeats = num_heads // n_groups
    x = x.view(batch_size, seq_len, repeats, n_groups, head_dim).float()
    B = B.view(batch_size, seq_len, n_groups, state_size).float()
    C = C.view(batch_size, seq_len, n_groups, state_size).float()
    raw_x = x
    x = x * dt.view(batch_size, seq_len, repeats, n_groups, 1)
    A = -torch.exp(A_log.float()) * dt
    num_chunks = seq_len // chunk_size

    x = x.reshape(batch_size, num_chunks, chunk_size, repeats, n_groups, head_dim)
    B = B.reshape(batch_size, num_chunks, chunk_size, n_groups, state_size)
    C = C.reshape(batch_size, num_chunks, chunk_size, n_groups, state_size)
    A_perm = A.reshape(
        batch_size, num_chunks, chunk_size, repeats, n_groups
    ).permute(0, 3, 4, 1, 2).contiguous()
    A_cumsum = torch.cumsum(A_perm, dim=-1)

    L = torch.empty(
        (*A_cumsum.shape, chunk_size),
        device=x.device,
        dtype=torch.float32,
    )
    _build_chunk_decay[(A_perm.numel() // chunk_size,)](
        A_perm, L, BLOCK=chunk_size, num_warps=8
    )

    G = torch.einsum('bclgn,bcsgn->bclsg', C, B)
    M = G.unsqueeze(-2) * L.permute(0, 3, 4, 5, 1, 2)
    y_diag = torch.einsum('bclshg,bcshgd->bclhgd', M, x)

    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum).permute(0, 3, 4, 1, 2)
    B_decay = B.unsqueeze(3) * decay_states[..., None]
    states = torch.einsum('bcshgd,bcshgn->bchgdn', x, B_decay)
    states_flat = states.reshape(
        batch_size, num_chunks, num_heads, head_dim, state_size
    )
    states_with_prev = torch.cat((torch.zeros_like(states_flat[:, :1]), states_flat), dim=1)

    A_cumsum_flat = A_cumsum.reshape(batch_size, num_heads, num_chunks, chunk_size)
    chunk_ends = F.pad(A_cumsum_flat[..., -1], (1, 0))
    nc = num_chunks + 1
    expanded = chunk_ends[..., None].expand(*chunk_ends.size(), nc)
    lower = torch.tril(torch.ones(nc, nc, device=x.device, dtype=torch.bool), diagonal=-1)
    seg = torch.cumsum(expanded.masked_fill(~lower, 0), dim=-2)
    lower_diag = torch.tril(torch.ones(nc, nc, device=x.device, dtype=torch.bool))
    decay_chunk = torch.exp(seg.masked_fill(~lower_diag, -torch.inf))
    new_states = torch.einsum(
        'bhcd,bhdin->bhcin', decay_chunk, states_with_prev.permute(0, 2, 1, 3, 4)
    )[:, :, :-1]
    states_final = new_states.permute(0, 2, 1, 3, 4).reshape(
        batch_size, num_chunks, repeats, n_groups, head_dim, state_size
    )
    state_decay = torch.exp(A_cumsum).permute(0, 3, 4, 1, 2)
    y_off = torch.einsum('bcsgn,bchgdn,bcshg->bcshgd', C, states_final, state_decay)

    y_diag = y_diag.contiguous()
    y_off = y_off.contiguous()
    raw_x = raw_x.contiguous()
    y = torch.empty(
        (batch_size, seq_len, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    _finish_grouped_scan[(batch_size * seq_len * n_groups,)](
        y_diag, y_off, raw_x, D, projected, norm_weight, y,
        layer_norm_epsilon,
        BLOCK=2048,
        num_warps=8,
    )
    return torch.matmul(y, out_proj_weight.t())
