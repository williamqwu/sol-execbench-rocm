import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _matmul_bias_kernel(
    a,
    weight,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_m)
    pid_n = (pid % num_pid_in_group) // group_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a + rows[:, None] * K + ks[None, :]
    # weight is physically [N, K], i.e. the transposed right operand.
    w_ptrs = weight + cols[None, :] * K + ks[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        av = tl.load(
            a_ptrs + k_block * BLOCK_K,
            mask=rows[:, None] < M,
            other=0.0,
        )
        wv = tl.load(w_ptrs + k_block * BLOCK_K)
        acc += tl.dot(av, wv)
    bv = tl.load(bias + cols)
    acc += bv[None, :].to(tl.float32)
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        mask=rows[:, None] < M,
    )


@triton.jit
def _causal_gate_kernel(
    bcx,
    conv_weight,
    conv_bias,
    y,
    M: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fuse B*x, the four-tap depthwise convolution, and the C gate."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    chans = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < M
    chan_mask = chans < H
    mask = row_mask[:, None] & chan_mask[None, :]
    stride = 3 * H

    c = tl.load(
        bcx + rows[:, None] * stride + (H + chans[None, :]),
        mask=mask,
        other=0.0,
    )
    acc = tl.load(conv_bias + chans, mask=chan_mask, other=0.0).to(tl.float32)
    acc = tl.broadcast_to(acc[None, :], (BLOCK_M, BLOCK_H))

    # PyTorch's causal cross-correlation uses weight[3-lag] at row-lag.
    for lag in tl.static_range(0, 4):
        source_rows = rows - lag
        source_ok = row_mask & ((rows % S) >= lag)
        source_mask = source_ok[:, None] & chan_mask[None, :]
        b = tl.load(
            bcx + source_rows[:, None] * stride + chans[None, :],
            mask=source_mask,
            other=0.0,
        )
        x_proj = tl.load(
            bcx + source_rows[:, None] * stride + (2 * H + chans[None, :]),
            mask=source_mask,
            other=0.0,
        )
        # B*x is a materialized BF16 tensor in the reference.
        bx = (b.to(tl.float32) * x_proj.to(tl.float32)).to(tl.bfloat16)
        w = tl.load(
            conv_weight + chans * 4 + (3 - lag),
            mask=chan_mask,
            other=0.0,
        )
        acc += bx.to(tl.float32) * w[None, :].to(tl.float32)

    # conv1d writes BF16 before the second elementwise gate.
    conv_out = acc.to(tl.bfloat16)
    result = (c.to(tl.float32) * conv_out.to(tl.float32)).to(tl.bfloat16)
    tl.store(y + rows[:, None] * OUT_STRIDE + chans[None, :], result, mask=mask)


@triton.jit
def _bx_kernel(bcx, bx_out, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    rows = offsets // H
    chans = offsets % H
    base = rows * (3 * H) + chans
    b = tl.load(bcx + base, mask=mask, other=0.0)
    x_proj = tl.load(bcx + base + 2 * H, mask=mask, other=0.0)
    bx = (b.to(tl.float32) * x_proj.to(tl.float32)).to(tl.bfloat16)
    tl.store(bx_out + offsets, bx, mask=mask)


@triton.jit
def _conv_gate_kernel(
    bcx,
    bx,
    conv_weight,
    conv_bias,
    y,
    M: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    chans = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    row_mask = rows < M
    chan_mask = chans < H
    mask = row_mask[:, None] & chan_mask[None, :]
    c = tl.load(
        bcx + rows[:, None] * (3 * H) + H + chans[None, :],
        mask=mask,
        other=0.0,
    )
    acc = tl.load(conv_bias + chans, mask=chan_mask, other=0.0).to(tl.float32)
    acc = tl.broadcast_to(acc[None, :], (BLOCK_M, BLOCK_H))
    for lag in tl.static_range(0, 4):
        source_rows = rows - lag
        source_ok = row_mask & ((rows % S) >= lag)
        source_mask = source_ok[:, None] & chan_mask[None, :]
        bx_value = tl.load(
            bx + source_rows[:, None] * H + chans[None, :],
            mask=source_mask,
            other=0.0,
        )
        w = tl.load(
            conv_weight + chans * 4 + (3 - lag),
            mask=chan_mask,
            other=0.0,
        )
        acc += bx_value.to(tl.float32) * w[None, :].to(tl.float32)
    conv_out = acc.to(tl.bfloat16)
    result = (c.to(tl.float32) * conv_out.to(tl.float32)).to(tl.bfloat16)
    tl.store(y + rows[:, None] * H + chans[None, :], result, mask=mask)


@triton.jit
def _causal_gate_gather_kernel(
    bcx,
    conv_weight,
    conv_bias,
    y,
    M: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SOURCE_M: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_M
    rows = block_start + tl.arange(0, BLOCK_M)
    chans = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    source_rows = block_start - 3 + tl.arange(0, SOURCE_M)
    row_mask = rows < M
    chan_mask = chans < H
    mask = row_mask[:, None] & chan_mask[None, :]
    source_mask = (
        (source_rows[:, None] >= 0)
        & (source_rows[:, None] < M)
        & (tl.arange(0, SOURCE_M)[:, None] < BLOCK_M + 3)
        & chan_mask[None, :]
    )
    stride = 3 * H
    b = tl.load(
        bcx + source_rows[:, None] * stride + chans[None, :],
        mask=source_mask,
        other=0.0,
    )
    x_proj = tl.load(
        bcx + source_rows[:, None] * stride + 2 * H + chans[None, :],
        mask=source_mask,
        other=0.0,
    )
    bx = (b.to(tl.float32) * x_proj.to(tl.float32)).to(tl.bfloat16)
    c = tl.load(
        bcx + rows[:, None] * stride + H + chans[None, :],
        mask=mask,
        other=0.0,
    )
    acc = tl.load(conv_bias + chans, mask=chan_mask, other=0.0).to(tl.float32)
    acc = tl.broadcast_to(acc[None, :], (BLOCK_M, BLOCK_H))
    row_indices = tl.arange(0, BLOCK_M)[:, None]
    for lag in tl.static_range(0, 4):
        gather_indices = tl.broadcast_to(
            row_indices + (3 - lag), (BLOCK_M, BLOCK_H)
        )
        bx_value = tl.gather(bx, gather_indices, axis=0)
        w = tl.load(
            conv_weight + chans * 4 + (3 - lag),
            mask=chan_mask,
            other=0.0,
        )
        valid = ((rows % S) >= lag)[:, None]
        acc += tl.where(valid, bx_value.to(tl.float32) * w[None, :], 0.0)
    conv_out = acc.to(tl.bfloat16)
    result = (c.to(tl.float32) * conv_out.to(tl.float32)).to(tl.bfloat16)
    tl.store(y + rows[:, None] * H + chans[None, :], result, mask=mask)


@torch.no_grad()
def run(
    x,
    in_proj_weight,
    in_proj_bias,
    conv_weight,
    conv_bias,
    out_proj_weight,
    out_proj_bias,
):
    batch_size, seq_len, hidden_size = x.shape
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    rows = batch_size * seq_len
    y = torch.empty_like(x)
    block_m = 8
    if rows > 4096:
        block_h, num_warps = 256, 4
    else:
        block_h, num_warps = 64, 1
    _causal_gate_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(hidden_size, block_h))
    ](
        BCx,
        conv_weight,
        conv_bias,
        y,
        M=rows,
        S=seq_len,
        H=hidden_size,
        OUT_STRIDE=hidden_size,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return F.linear(y, out_proj_weight, out_proj_bias)
