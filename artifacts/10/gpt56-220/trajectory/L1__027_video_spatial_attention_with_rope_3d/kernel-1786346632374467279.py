import torch
import torch.nn.functional as F
import math

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars['batch_size']
    num_frames = axes_and_scalars['num_frames']
    num_patches_per_frame = axes_and_scalars['num_patches_per_frame']
    hidden_size = 1024
    head_dim = 64
    
    hidden_states = torch.randn(batch_size, num_frames, num_patches_per_frame, hidden_size, device=device, dtype=torch.float32)
    qkv_weight = torch.randn(3 * hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    qkv_bias = torch.zeros(3 * hidden_size, device=device, dtype=torch.float32)
    out_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    out_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    
    rope_theta = 10000.0
    temporal_freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    spatial_freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    scale = 1.0 / math.sqrt(head_dim)
    
    return {
        'hidden_states': hidden_states,
        'qkv_weight': qkv_weight,
        'qkv_bias': qkv_bias,
        'out_weight': out_weight,
        'out_bias': out_bias,
        'temporal_freqs': temporal_freqs,
        'spatial_freqs': spatial_freqs,
        'scale': scale,
    }

@torch.no_grad()
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
    num_attention_heads = 16
    head_dim = 64
    
    # Generate position indices
    device = hidden_states.device
    # Assume square patches for height/width positions
    patches_per_side = int(math.sqrt(num_patches))
    if patches_per_side * patches_per_side != num_patches:
        patches_per_side = int(math.ceil(math.sqrt(num_patches)))

    token_idx = torch.arange(seq_len, device=device)
    patch_idx = token_idx % num_patches
    frame_positions = (token_idx // num_patches).float()
    height_positions = (patch_idx // patches_per_side).float()
    width_positions = (patch_idx % patches_per_side).float()
    
    # Reshape to (batch, seq_len, hidden_size)
    hidden_states_flat = hidden_states.reshape(batch_size, seq_len, hidden_size)
    
    # QKV projection: (batch, seq_len, 3 * hidden_size)
    qkv = F.linear(hidden_states_flat, qkv_weight, qkv_bias)
    
    # Reshape and split into Q, K, V
    qkv = qkv.reshape(batch_size, seq_len, 3, num_attention_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply 3D RoPE to Q and K
    # Q and K use identical position-dependent rotations.  Build the three
    # trigonometric tables once instead of evaluating sin/cos twice.
    temporal_angles = frame_positions[:, None] * temporal_freqs[:10]
    height_angles = height_positions[:, None] * spatial_freqs[:10]
    width_angles = width_positions[:, None] * spatial_freqs[:10]
    rope_tables = (
        (torch.cos(temporal_angles), torch.sin(temporal_angles)),
        (torch.cos(height_angles), torch.sin(height_angles)),
        (torch.cos(width_angles), torch.sin(width_angles)),
    )

    def apply_rope_3d(x, rope_tables):
        # The three 21-wide sections each rotate ten pairs and retain their
        # final odd element. Pack all pairs into one elementwise operation.
        x1 = torch.cat((x[..., 0:20:2], x[..., 21:41:2], x[..., 42:62:2]), dim=-1)
        x2 = torch.cat((x[..., 1:20:2], x[..., 22:41:2], x[..., 43:62:2]), dim=-1)
        cos_vals = torch.cat((rope_tables[0][0], rope_tables[1][0], rope_tables[2][0]), dim=-1)[None, None]
        sin_vals = torch.cat((rope_tables[0][1], rope_tables[1][1], rope_tables[2][1]), dim=-1)[None, None]
        r1 = x1 * cos_vals - x2 * sin_vals
        r2 = x1 * sin_vals + x2 * cos_vals
        rotated = torch.stack((r1, r2), dim=-1).flatten(-2)
        return torch.cat((rotated[..., :20], x[..., 20:21],
                          rotated[..., 20:40], x[..., 41:42],
                          rotated[..., 40:60], x[..., 62:]), dim=-1)

    qk = apply_rope_3d(torch.cat((q, k), dim=0), rope_tables)
    q, k = qk.split(batch_size, dim=0)
    
    # Compute attention scores
    # Scale the linear-sized Q tensor rather than launching a multiply over
    # the quadratic attention-score tensor.
    q.mul_(scale)
    attn_scores = torch.matmul(q, k.transpose(-2, -1))

    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)

    # Apply attention to values
    attn_output = torch.matmul(attn_probs, v)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    output = F.linear(attn_output, out_weight, out_bias)
    
    # Reshape back to original shape
    output = output.reshape(batch_size, num_frames, num_patches, hidden_size)
    
    return output
