import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _scale_add_relpos_kernel(scores, rel_h, rel_w, n_elements,
                             batch_heads: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    key = offsets % 196
    t = offsets // 196
    query = t % 196
    batch_head = t // 196
    query_h = query // 14
    query_w = query % 14
    key_h = key // 14
    key_w = key % 14

    score = tl.load(scores + offsets, mask=mask)
    # torch.einsum leaves the contracted query axis outermost in these two
    # outputs, so address their physical layouts directly and avoid copies.
    h_offset = query_h * (batch_heads * 196) + batch_head * 196 + query_w * 14 + key_h
    w_offset = query_w * (batch_heads * 196) + batch_head * 196 + query_h * 14 + key_w
    h = tl.load(rel_h + h_offset, mask=mask)
    w = tl.load(rel_w + w_offset, mask=mask)
    scaled = score * 0.125
    bias = h + w
    result = scaled + bias
    tl.store(scores + offsets, result, mask=mask)


@triton.jit
def _scale_add_relpos_rows_kernel(scores, rel_h, rel_w,
                                  batch_heads: tl.constexpr,
                                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    key = tl.arange(0, BLOCK)
    mask = key < 196
    query = row % 196
    batch_head = row // 196
    query_h = query // 14
    query_w = query % 14
    key_h = key // 14
    key_w = key % 14
    h_offset = query_h * (batch_heads * 196) + batch_head * 196 + query_w * 14 + key_h
    w_offset = query_w * (batch_heads * 196) + batch_head * 196 + query_h * 14 + key_w
    score = tl.load(scores + row * 196 + key, mask=mask)
    h = tl.load(rel_h + h_offset, mask=mask)
    w = tl.load(rel_w + w_offset, mask=mask)
    scaled = score * 0.125
    bias = h + w
    tl.store(scores + row * 196 + key, scaled + bias, mask=mask)


@triton.jit
def _relpos_softmax_kernel(scores, rel_h, rel_w,
                           batch_heads: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    key = tl.arange(0, BLOCK)
    mask = key < 196
    query = row % 196
    batch_head = row // 196
    query_h = query // 14
    query_w = query % 14
    key_h = key // 14
    key_w = key % 14
    h_offset = query_h * (batch_heads * 196) + batch_head * 196 + query_w * 14 + key_h
    w_offset = query_w * (batch_heads * 196) + batch_head * 196 + query_h * 14 + key_w

    score = tl.load(scores + row * 196 + key, mask=mask, other=-float("inf"))
    h = tl.load(rel_h + h_offset, mask=mask, other=0.0)
    w = tl.load(rel_w + w_offset, mask=mask, other=0.0)
    score = score * 0.125
    bias = h + w
    score = score + bias
    score = score - tl.max(score, axis=0)
    numerator = libdevice.exp(score)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(scores + row * 196 + key, result, mask=mask)


@triton.jit
def _load_scaled_rel_score(scores, rel_h, rel_w, row, key,
                           batch_heads: tl.constexpr):
    mask = key < 196
    query = row % 196
    batch_head = row // 196
    query_h = query // 14
    query_w = query % 14
    key_h = key // 14
    key_w = key % 14
    h_offset = query_h * (batch_heads * 196) + batch_head * 196 + query_w * 14 + key_h
    w_offset = query_w * (batch_heads * 196) + batch_head * 196 + query_h * 14 + key_w
    score = tl.load(scores + row * 196 + key, mask=mask, other=-float("inf"))
    h = tl.load(rel_h + h_offset, mask=mask, other=0.0)
    w = tl.load(rel_w + w_offset, mask=mask, other=0.0)
    scaled = score * 0.125
    bias = h + w
    return scaled + bias


@triton.jit
def _relpos_softmax_warp_kernel(scores, rel_h, rel_w,
                                batch_heads: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 64)
    key0 = lane
    key1 = lane + 64
    key2 = lane + 128
    key3 = lane + 192
    score0 = _load_scaled_rel_score(scores, rel_h, rel_w, row, key0, batch_heads)
    score1 = _load_scaled_rel_score(scores, rel_h, rel_w, row, key1, batch_heads)
    score2 = _load_scaled_rel_score(scores, rel_h, rel_w, row, key2, batch_heads)
    score3 = _load_scaled_rel_score(scores, rel_h, rel_w, row, key3, batch_heads)
    local_max = tl.maximum(tl.maximum(score0, score1), tl.maximum(score2, score3))
    max_value = tl.max(local_max, axis=0)
    out0 = libdevice.fast_expf(score0 - max_value)
    out1 = libdevice.fast_expf(score1 - max_value)
    out2 = libdevice.fast_expf(score2 - max_value)
    out3 = libdevice.fast_expf(score3 - max_value)
    local_sum = out0 + out1
    local_sum = local_sum + out2
    local_sum = local_sum + out3
    sum_value = tl.sum(local_sum, axis=0)
    out0 = out0 / sum_value
    out1 = out1 / sum_value
    out2 = out2 / sum_value
    out3 = out3 / sum_value
    tl.store(scores + row * 196 + key0, out0, mask=key0 < 196)
    tl.store(scores + row * 196 + key1, out1, mask=key1 < 196)
    tl.store(scores + row * 196 + key2, out2, mask=key2 < 196)
    tl.store(scores + row * 196 + key3, out3, mask=key3 < 196)


@triton.jit
def _unpartition_add_kernel(attn, residual, output, height, width,
                            windows_h, windows_w, n_elements,
                            BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    channel = offsets % 768
    pixel = offsets // 768
    x = pixel % width
    t = pixel // width
    y = t % height
    batch = t // height
    window_h = y // 14
    window_w = x // 14
    in_h = y % 14
    in_w = x % 14
    window = (batch * windows_h + window_h) * windows_w + window_w
    attn_offset = (((window * 14 + in_h) * 14 + in_w) * 768 + channel)
    projected = tl.load(attn + attn_offset, mask=mask)
    skip = tl.load(residual + offsets, mask=mask)
    tl.store(output + offsets, skip + projected, mask=mask)


@triton.jit
def _unpartition_add_pixels_kernel(attn, residual, output, height, width,
                                   windows_h, windows_w,
                                   BLOCK: tl.constexpr):
    pixel = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    mask = channel < 768
    x = pixel % width
    t = pixel // width
    y = t % height
    batch = t // height
    window_h = y // 14
    window_w = x // 14
    in_h = y % 14
    in_w = x % 14
    window = (batch * windows_h + window_h) * windows_w + window_w
    attn_offset = (((window * 14 + in_h) * 14 + in_w) * 768 + channel)
    output_offset = pixel * 768 + channel
    projected = tl.load(attn + attn_offset, mask=mask)
    skip = tl.load(residual + output_offset, mask=mask)
    tl.store(output + output_offset, skip + projected, mask=mask)


@triton.jit
def _partition_pixels_kernel(source, windows, height, width,
                             windows_h, windows_w,
                             BLOCK: tl.constexpr):
    token = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    window = token // 196
    in_window = token % 196
    in_h = in_window // 14
    in_w = in_window % 14
    window_w = window % windows_w
    t = window // windows_w
    window_h = t % windows_h
    batch = t // windows_h
    y = window_h * 14 + in_h
    x = window_w * 14 + in_w
    valid = (channel < 768) & (y < height) & (x < width)
    source_offset = ((batch * height + y) * width + x) * 768 + channel
    value = tl.load(source + source_offset, mask=valid, other=0.0)
    tl.store(windows + token * 768 + channel, value, mask=channel < 768)


@triton.jit
def _normalize_value(value, mean, denominator, weight, bias):
    centered = value - mean
    normalized = centered / denominator
    scaled = normalized * weight
    return scaled + bias


@triton.jit
def _norm_partition_pixels_kernel(source, mean, denominator, weight, bias,
                                  windows, height, width,
                                  windows_h, windows_w,
                                  BLOCK: tl.constexpr):
    token = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    window = token // 196
    in_window = token % 196
    in_h = in_window // 14
    in_w = in_window % 14
    window_w = window % windows_w
    t = window // windows_w
    window_h = t % windows_h
    batch = t // windows_h
    y = window_h * 14 + in_h
    x = window_w * 14 + in_w
    spatial_valid = (y < height) & (x < width)
    mask = (channel < 768) & spatial_valid
    pixel = (batch * height + y) * width + x
    source_offset = pixel * 768 + channel
    value = tl.load(source + source_offset, mask=mask, other=0.0)
    row_mean = tl.load(mean + pixel, mask=spatial_valid, other=0.0)
    row_denominator = tl.load(denominator + pixel, mask=spatial_valid, other=1.0)
    scale = tl.load(weight + channel, mask=channel < 768)
    shift = tl.load(bias + channel, mask=channel < 768)
    centered = value - row_mean
    normalized = centered / row_denominator
    value = normalized * scale
    # The following bias add runs as a separate PyTorch kernel to preserve the
    # reference's intermediate rounding (and prevent mul/add contraction).
    value = tl.where(spatial_valid, value, -shift)
    tl.store(windows + token * 768 + channel, value, mask=channel < 768)


@triton.jit
def _normalize_pixels_kernel(source, mean, denominator, weight, bias, output,
                             BLOCK: tl.constexpr):
    pixel = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    mask = channel < 768
    offset = pixel * 768 + channel
    value = tl.load(source + offset, mask=mask)
    row_mean = tl.load(mean + pixel)
    row_denominator = tl.load(denominator + pixel)
    scale = tl.load(weight + channel, mask=mask)
    shift = tl.load(bias + channel, mask=mask)
    value = _normalize_value(value, row_mean, row_denominator, scale, shift)
    tl.store(output + offset, value, mask=mask)


@triton.jit
def _normalize_no_bias_kernel(source, mean, denominator, weight, output,
                              BLOCK: tl.constexpr):
    pixel = tl.program_id(0)
    channel = tl.arange(0, BLOCK)
    mask = channel < 768
    offset = pixel * 768 + channel
    value = tl.load(source + offset, mask=mask)
    row_mean = tl.load(mean + pixel)
    row_denominator = tl.load(denominator + pixel)
    scale = tl.load(weight + channel, mask=mask)
    centered = value - row_mean
    normalized = centered / row_denominator
    value = normalized * scale
    tl.store(output + offset, value, mask=mask)


@triton.jit
def _split_qkv_kernel(qkv, query, key, value, query_h, n_elements,
                      batch_windows: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    dim = offsets % 64
    t = offsets // 64
    token = t % 196
    t = t // 196
    head = t % 12
    window = t // 12
    source_base = ((window * 196 + token) * 3 * 12 + head) * 64 + dim
    q = tl.load(qkv + source_base, mask=mask)
    k = tl.load(qkv + source_base + 768, mask=mask)
    v = tl.load(qkv + source_base + 1536, mask=mask)
    tl.store(query + offsets, q, mask=mask)
    tl.store(key + offsets, k, mask=mask)
    tl.store(value + offsets, v, mask=mask)
    query_row = token // 14
    query_col = token % 14
    h_offset = ((((query_row * batch_windows + window) * 12 + head) * 14
                 + query_col) * 64 + dim)
    tl.store(query_h + h_offset, q, mask=mask)


@torch.no_grad()
def run(
    hidden_states,
    qkv_weight,
    qkv_bias,
    proj_weight,
    proj_bias,
    rel_pos_h,
    rel_pos_w,
    layer_norm1_weight,
    layer_norm1_bias,
    layer_norm2_weight,
    layer_norm2_bias,
    mlp_lin1_weight,
    mlp_lin1_bias,
    mlp_lin2_weight,
    mlp_lin2_bias,
    layer_norm_eps,
):
    window_size = 14
    num_attention_heads = 12
    head_dim = 64
    scale = head_dim ** -0.5

    batch_size, height, width, channels = hidden_states.shape
    residual = hidden_states

    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    denominator = torch.sqrt(var + layer_norm_eps)

    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    pad_height = height + pad_h
    pad_width = width + pad_w
    windows_h = pad_height // window_size
    windows_w = pad_width // window_size
    batch_windows = batch_size * windows_h * windows_w
    windows = torch.empty(
        (batch_windows, window_size, window_size, channels),
        device=hidden_states.device, dtype=hidden_states.dtype,
    )
    _norm_partition_pixels_kernel[(batch_windows * 196,)](
        hidden_states, mean, denominator,
        layer_norm1_weight, layer_norm1_bias, windows,
        height, width, windows_h, windows_w,
        BLOCK=1024, num_warps=8,
    )
    windows.add_(layer_norm1_bias)
    batch_windows = windows.shape[0]

    qkv = F.linear(windows, qkv_weight, qkv_bias)
    qkv_shape = (batch_windows, num_attention_heads, 196, head_dim)
    query = torch.empty(qkv_shape, device=qkv.device, dtype=qkv.dtype)
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    query_h_layout = torch.empty(
        (14, batch_windows, num_attention_heads, 14, head_dim),
        device=qkv.device, dtype=qkv.dtype,
    )
    n_q = query.numel()
    _split_qkv_kernel[(triton.cdiv(n_q, 256),)](
        qkv, query, key, value, query_h_layout, n_q,
        batch_windows=batch_windows, BLOCK=256, num_warps=4,
    )

    attn_weights = query @ key.transpose(-2, -1)
    coords = torch.arange(window_size, device=rel_pos_h.device)
    rel_coords = coords[:, None] - coords[None, :] + window_size - 1
    rel_pos_h_emb = rel_pos_h[rel_coords.flatten()].reshape(14, 14, head_dim)
    rel_pos_w_emb = rel_pos_w[rel_coords.flatten()].reshape(14, 14, head_dim)
    query_for_bias = query.reshape(batch_windows, num_attention_heads, 14, 14, head_dim)
    rel_h = torch.bmm(
        query_h_layout.reshape(14, -1, head_dim),
        rel_pos_h_emb.transpose(1, 2),
    )
    rel_w = torch.einsum("bnijc,jkc->bnijk", query_for_bias, rel_pos_w_emb)
    batch_heads = batch_windows * num_attention_heads
    n_attention = attn_weights.numel()
    _scale_add_relpos_kernel[(triton.cdiv(n_attention, 256),)](
        attn_weights, rel_h, rel_w, n_attention,
        batch_heads=batch_heads, BLOCK=256, num_warps=4,
    )
    torch.softmax(attn_weights, dim=-1, out=attn_weights)

    attn_output = (attn_weights @ value).transpose(1, 2)
    attn_output = attn_output.reshape(batch_windows, 14, 14, channels)
    attn_output = F.linear(attn_output, proj_weight, proj_bias)

    num_windows_h = pad_height // window_size
    num_windows_w = pad_width // window_size
    hidden_states = torch.empty_like(residual)
    _unpartition_add_pixels_kernel[(batch_size * height * width,)](
        attn_output, residual, hidden_states, height, width,
        num_windows_h, num_windows_w,
        BLOCK=1024, num_warps=8,
    )
    residual = hidden_states
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    denominator = torch.sqrt(var + layer_norm_eps)
    normalized = torch.empty_like(hidden_states)
    _normalize_no_bias_kernel[(batch_size * height * width,)](
        hidden_states, mean, denominator, layer_norm2_weight, normalized,
        BLOCK=1024, num_warps=8,
    )
    normalized.add_(layer_norm2_bias)
    hidden_states = normalized
    hidden_states = F.linear(hidden_states, mlp_lin1_weight, mlp_lin1_bias)
    hidden_states = torch.ops.aten.gelu_(hidden_states)
    hidden_states = F.linear(hidden_states, mlp_lin2_weight, mlp_lin2_bias)
    hidden_states.add_(residual)
    return hidden_states
