import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_SIDE_STREAM = None


def _get_side_stream(device):
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream(device=device)
    return _SIDE_STREAM


@triton.jit
def _pack_qkv_kernel(
    grad_q,
    grad_k,
    grad_v,
    grad_qkv,
    scale,
    N_ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    EMBED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program copies one 1024-wide Q/K/V slice while changing the
    # [B,H,S,D] layout into [B,S,H*D].  Folding Q's scale into this copy
    # avoids three materialized transposes and the subsequent cat.
    pid = tl.program_id(0)
    row = pid // 3
    part = pid - row * 3
    e = tl.arange(0, BLOCK)
    mask = (row < N_ROWS) & (e < EMBED)
    b = row // SEQ
    s = row - b * SEQ
    h = e // 64
    d = e - h * 64
    src = ((b * 16 + h) * SEQ + s) * 64 + d
    qv = tl.load(grad_q + src, mask=mask & (part == 0), other=0.0) * scale
    kv = tl.load(grad_k + src, mask=mask & (part == 1), other=0.0) * scale
    vv = tl.load(grad_v + src, mask=mask & (part == 2), other=0.0)
    value = qv + kv + vv
    tl.store(grad_qkv + row * (3 * EMBED) + part * EMBED + e, value, mask=mask)


@triton.jit
def _softmax_backward_kernel(
    grad_attn_weights,
    attn_weights,
    sum_grad,
    grad_attn_scores,
    n_elements: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gaw = tl.load(grad_attn_weights + offsets, mask=mask)
    aw = tl.load(attn_weights + offsets, mask=mask)
    row_sum = tl.load(sum_grad + offsets // SEQ, mask=mask)
    centered = gaw - row_sum
    tl.store(grad_attn_scores + offsets, aw * centered, mask=mask)


@triton.jit
def _layer_norm_prepare_kernel(
    x,
    x_mean,
    x_var,
    grad_x_norm,
    ln_weight,
    x_centered,
    std,
    ln_product,
    grad_x_from_norm,
    centered_product,
    norm_eps,
    N_ROWS: tl.constexpr,
    EMBED: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, EMBED)
    row_mask = rows < N_ROWS
    offsets = rows[:, None] * EMBED + columns[None, :]
    mask = row_mask[:, None]

    xv = tl.load(x + offsets, mask=mask)
    mean = tl.load(x_mean + rows, mask=row_mask)[:, None]
    var = tl.load(x_var + rows, mask=row_mask)
    stdev_rows = tl.extra.libdevice.sqrt(var + norm_eps)
    stdev = stdev_rows[:, None]
    gxn_in = tl.load(grad_x_norm + offsets, mask=mask)
    weight = tl.load(ln_weight + columns)[None, :]

    centered = xv - mean
    normalized = centered / stdev
    ln_prod = gxn_in * normalized
    grad_normalized = gxn_in * weight
    grad_from_norm = grad_normalized / stdev
    center_prod = grad_normalized * centered

    tl.store(x_centered + offsets, centered, mask=mask)
    tl.store(std + rows, stdev_rows, mask=row_mask)
    tl.store(ln_product + offsets, ln_prod, mask=mask)
    tl.store(grad_x_from_norm + offsets, grad_from_norm, mask=mask)
    tl.store(centered_product + offsets, center_prod, mask=mask)


@triton.jit
def _layer_norm_finish_kernel(
    grad_output,
    grad_x_from_norm,
    x_centered,
    mean_grad,
    mean_centered_product,
    std,
    grad_x,
    n_elements: tl.constexpr,
    EMBED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // EMBED
    go = tl.load(grad_output + offsets, mask=mask)
    gx = tl.load(grad_x_from_norm + offsets, mask=mask)
    xc = tl.load(x_centered + offsets, mask=mask)
    mg = tl.load(mean_grad + row, mask=mask)
    mcp = tl.load(mean_centered_product + row, mask=mask)
    stdev = tl.load(std + row, mask=mask)
    denominator = stdev * stdev
    mgxc = mcp / denominator
    correction = xc * mgxc
    gx = gx - mg
    gx = gx - correction
    gx = go + gx
    tl.store(grad_x + offsets, gx, mask=mask)


@torch.no_grad()
def run(
    grad_output,
    x,
    x_mean,
    x_var,
    x_norm,
    ln_weight,
    qkv_weight,
    q,
    k,
    v,
    attn_weights,
    attn_output,
    out_weight,
    scale,
    norm_eps,
):
    batch_size, seq_len, embed_dim = x.shape
    num_heads = 16
    head_dim = 64

    use_out_overlap = 2 <= batch_size <= 5
    use_full_overlap = 2 <= batch_size <= 3
    current_stream = torch.cuda.current_stream(x.device)
    side_stream = _get_side_stream(x.device) if use_out_overlap else None
    if use_out_overlap:
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            grad_out_weight = torch.matmul(
                grad_output.reshape(-1, embed_dim).t(),
                attn_output.reshape(-1, embed_dim),
            )
            grad_out_bias = grad_output.sum(dim=(0, 1))
    else:
        grad_out_weight = torch.matmul(
            grad_output.reshape(-1, embed_dim).t(),
            attn_output.reshape(-1, embed_dim),
        )
        grad_out_bias = grad_output.sum(dim=(0, 1))

    grad_attn_output = F.linear(grad_output, out_weight.t())

    grad_attn_output_heads = grad_attn_output.view(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_attn_output_heads)
    grad_attn_weights = torch.matmul(
        grad_attn_output_heads, v.transpose(-2, -1)
    )
    sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_attn_scores = torch.empty_like(grad_attn_weights)
    n_scores = grad_attn_weights.numel()
    _softmax_backward_kernel[(triton.cdiv(n_scores, 4096),)](
        grad_attn_weights,
        attn_weights,
        sum_grad,
        grad_attn_scores,
        n_elements=n_scores,
        SEQ=seq_len,
        BLOCK=4096,
        num_warps=8,
        enable_fp_fusion=False,
    )
    grad_q = torch.matmul(grad_attn_scores, k)
    grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), q)

    grad_qkv = torch.empty(
        (batch_size, seq_len, 3 * embed_dim),
        device=x.device,
        dtype=x.dtype,
    )
    n_rows = batch_size * seq_len
    _pack_qkv_kernel[(n_rows * 3,)](
        grad_q,
        grad_k,
        grad_v,
        grad_qkv,
        scale,
        N_ROWS=n_rows,
        SEQ=seq_len,
        EMBED=embed_dim,
        BLOCK=1024,
        num_warps=8,
    )

    if use_full_overlap:
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            grad_qkv_weight = torch.matmul(
                grad_qkv.reshape(-1, 3 * embed_dim).t(),
                x_norm.reshape(-1, embed_dim),
            )
            grad_qkv_bias = grad_qkv.sum(dim=(0, 1))
        grad_x_norm = F.linear(grad_qkv, qkv_weight.t())
    else:
        grad_x_norm = F.linear(grad_qkv, qkv_weight.t())
        grad_qkv_weight = torch.matmul(
            grad_qkv.reshape(-1, 3 * embed_dim).t(),
            x_norm.reshape(-1, embed_dim),
        )
        grad_qkv_bias = grad_qkv.sum(dim=(0, 1))

    std = torch.empty_like(x_var)
    x_centered = torch.empty_like(x)
    ln_product = torch.empty_like(x)
    grad_x_from_norm = torch.empty_like(x)
    centered_product = torch.empty_like(x)
    n_x = x.numel()
    _layer_norm_prepare_kernel[(triton.cdiv(n_rows, 4),)](
        x,
        x_mean,
        x_var,
        grad_x_norm,
        ln_weight,
        x_centered,
        std,
        ln_product,
        grad_x_from_norm,
        centered_product,
        norm_eps,
        N_ROWS=n_rows,
        EMBED=embed_dim,
        ROWS=4,
        num_warps=8,
        enable_fp_fusion=False,
    )
    grad_ln_weight = ln_product.sum(dim=(0, 1))
    grad_ln_bias = grad_x_norm.sum(dim=(0, 1))
    mean_grad = grad_x_from_norm.mean(dim=-1, keepdim=True)
    mean_centered_product = centered_product.mean(dim=-1, keepdim=True)
    grad_x = torch.empty_like(x)
    _layer_norm_finish_kernel[(triton.cdiv(n_x, 4096),)](
        grad_output,
        grad_x_from_norm,
        x_centered,
        mean_grad,
        mean_centered_product,
        std,
        grad_x,
        n_elements=n_x,
        EMBED=embed_dim,
        BLOCK=4096,
        num_warps=8,
        enable_fp_fusion=False,
    )
    if use_out_overlap:
        current_stream.wait_stream(side_stream)

    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
