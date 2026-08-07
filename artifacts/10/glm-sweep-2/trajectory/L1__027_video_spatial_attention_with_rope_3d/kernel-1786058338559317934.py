import torch
import torch.nn.functional as F
import math

_HEAD_DIM = 64
_NUM_HEADS = 16
_TD = _HEAD_DIM // 3          # 21
_SD = (_HEAD_DIM - _TD) // 2  # 21
_HALF = _TD // 2              # 10  (half_dim for each 21-dim part)


def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    temporal_freqs: torch.Tensor,
    spatial_freqs: torch.Tensor,
    scale: float,
):
    batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
    seq_len = num_frames * num_patches
    num_attention_heads = _NUM_HEADS
    head_dim = _HEAD_DIM

    device = hidden_states.device

    # --- position indices ---
    frame_positions = torch.arange(num_frames, device=device, dtype=torch.float32)
    frame_positions = frame_positions.unsqueeze(1).expand(num_frames, num_patches).reshape(-1)

    patches_per_side = int(math.sqrt(num_patches))
    if patches_per_side * patches_per_side != num_patches:
        patches_per_side = int(math.ceil(math.sqrt(num_patches)))
    idx = torch.arange(num_patches, device=device)
    height_idx = idx // patches_per_side
    width_idx = idx % patches_per_side
    height_positions = height_idx.unsqueeze(0).expand(num_frames, num_patches).reshape(-1).float()
    width_positions = width_idx.unsqueeze(0).expand(num_frames, num_patches).reshape(-1).float()

    # --- QKV projection ---
    hidden_states_flat = hidden_states.reshape(batch_size, seq_len, hidden_size)
    qkv = F.linear(hidden_states_flat, qkv_weight, qkv_bias)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # --- precompute RoPE cos/sin (depends only on positions + freqs) ---
    def make_cos_sin(positions, freqs):
        angles = positions.unsqueeze(-1) * freqs[:_HALF]   # [sl, 10]
        return torch.cos(angles), torch.sin(angles)

    cos_t, sin_t = make_cos_sin(frame_positions, temporal_freqs)
    cos_h, sin_h = make_cos_sin(height_positions, spatial_freqs)
    cos_w, sin_w = make_cos_sin(width_positions, spatial_freqs)

    def apply_rope_3d(x):
        x_temporal = x[..., :_TD]
        x_height = x[..., _TD:_TD + _SD]
        x_width = x[..., _TD + _SD:_TD + 2 * _SD]
        x_remaining = x[..., _TD + 2 * _SD:]

        def rotate(xp, cos_vals, sin_vals):
            x1 = xp[..., 0:2 * _HALF:2]   # even indices 0..18  -> 10
            x2 = xp[..., 1:2 * _HALF:2]   # odd  indices 1..19  -> 10
            cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
            sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)
            r1 = x1 * cos_vals - x2 * sin_vals
            r2 = x1 * sin_vals + x2 * cos_vals
            out = torch.stack([r1, r2], dim=-1).flatten(-2)
            out = torch.cat([out, x_temporal[..., -1:]], dim=-1) if False else out
            # _TD is odd (21): append the last element unchanged
            out = torch.cat([out, xp[..., 2 * _HALF:]], dim=-1)
            return out

        x_temporal = rotate(x_temporal, cos_t, sin_t)
        x_height = rotate(x_height, cos_h, sin_h)
        x_width = rotate(x_width, cos_w, sin_w)
        return torch.cat([x_temporal, x_height, x_width, x_remaining], dim=-1)

    q = apply_rope_3d(q)
    k = apply_rope_3d(k)

    # --- attention (exact arithmetic preserved) ---
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    output = F.linear(attn_output, out_weight, out_bias)
    output = output.reshape(batch_size, num_frames, num_patches, hidden_size)
    return output


run = torch.no_grad()(run)
