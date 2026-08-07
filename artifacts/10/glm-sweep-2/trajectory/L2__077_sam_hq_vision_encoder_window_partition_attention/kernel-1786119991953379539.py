import torch
import torch.nn.functional as F

_window_size = 14
_num_attention_heads = 12
_head_dim = 64
_scale = _head_dim ** -0.5

@torch.no_grad()
def _forward(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    mlp_lin1_weight: torch.Tensor,
    mlp_lin1_bias: torch.Tensor,
    mlp_lin2_weight: torch.Tensor,
    mlp_lin2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    window_size = _window_size
    num_attention_heads = _num_attention_heads
    head_dim = _head_dim
    scale = _scale

    batch_size, height, width, channels = hidden_states.shape

    residual = hidden_states

    # Layer norm 1
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm1_weight + layer_norm1_bias

    # Window partition with padding
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))

    pad_height = height + pad_h
    pad_width = width + pad_w

    hidden_states = hidden_states.reshape(
        batch_size,
        pad_height // window_size,
        window_size,
        pad_width // window_size,
        window_size,
        channels
    )
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.reshape(-1, window_size, window_size, channels)

    batch_windows = windows.shape[0]
    window_h = window_size
    window_w = window_size

    qkv = F.linear(windows, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_windows, window_h * window_w, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    query, key, value = qkv[0], qkv[1], qkv[2]

    attn_weights = (query @ key.transpose(-2, -1)) * scale

    coords_h = torch.arange(window_h, device=rel_pos_h.device)
    coords_w = torch.arange(window_w, device=rel_pos_w.device)

    rel_coords_h = coords_h[:, None] - coords_h[None, :]
    rel_coords_w = coords_w[:, None] - coords_w[None, :]

    rel_coords_h = rel_coords_h + window_h - 1
    rel_coords_w = rel_coords_w + window_w - 1

    rel_pos_h_emb = rel_pos_h[rel_coords_h.flatten()].reshape(
        window_h, window_h, head_dim
    )
    rel_pos_w_emb = rel_pos_w[rel_coords_w.flatten()].reshape(
        window_w, window_w, head_dim
    )

    query_for_bias = query.reshape(
        batch_windows, num_attention_heads, window_h, window_w, head_dim
    )

    rel_h = torch.einsum('bnijc,ikc->bnijk', query_for_bias, rel_pos_h_emb)
    rel_w = torch.einsum('bnijc,jkc->bnijk', query_for_bias, rel_pos_w_emb)

    rel_pos_bias = rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]
    rel_pos_bias = rel_pos_bias.reshape(
        batch_windows, num_attention_heads, window_h * window_w, window_h * window_w
    )

    attn_weights = attn_weights + rel_pos_bias

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = attn_weights @ value
    attn_output = attn_output.transpose(1, 2).reshape(batch_windows, window_h, window_w, channels)

    attn_output = F.linear(attn_output, proj_weight, proj_bias)

    num_windows_h = pad_height // window_size
    num_windows_w = pad_width // window_size

    attn_output = attn_output.reshape(
        batch_size,
        num_windows_h,
        num_windows_w,
        window_size,
        window_size,
        -1
    )
    attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).contiguous()
    attn_output = attn_output.reshape(batch_size, pad_height, pad_width, -1)

    attn_output = attn_output[:, :height, :width, :].contiguous()

    hidden_states = residual + attn_output

    residual = hidden_states

    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
    hidden_states = hidden_states * layer_norm2_weight + layer_norm2_bias

    hidden_states = F.linear(hidden_states, mlp_lin1_weight, mlp_lin1_bias)
    hidden_states = F.gelu(hidden_states)
    hidden_states = F.linear(hidden_states, mlp_lin2_weight, mlp_lin2_bias)

    output = residual + hidden_states

    return output


_compiled = torch.compile(_forward, mode="max-autotune", dynamic=True)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    mlp_lin1_weight: torch.Tensor,
    mlp_lin1_bias: torch.Tensor,
    mlp_lin2_weight: torch.Tensor,
    mlp_lin2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    return _compiled(
        hidden_states, qkv_weight, qkv_bias, proj_weight, proj_bias,
        rel_pos_h, rel_pos_w, layer_norm1_weight, layer_norm1_bias,
        layer_norm2_weight, layer_norm2_bias, mlp_lin1_weight, mlp_lin1_bias,
        mlp_lin2_weight, mlp_lin2_bias, layer_norm_eps,
    )
