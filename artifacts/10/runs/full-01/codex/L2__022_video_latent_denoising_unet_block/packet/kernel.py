import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    video_latents,
    text_embeddings,
    temporal_norm_weight,
    temporal_norm_bias,
    temporal_qkv_weight,
    temporal_qkv_bias,
    temporal_out_proj_weight,
    temporal_out_proj_bias,
    spatial_norm_weight,
    spatial_norm_bias,
    spatial_qkv_weight,
    spatial_qkv_bias,
    spatial_out_proj_weight,
    spatial_out_proj_bias,
    cross_attn_norm_weight,
    cross_attn_norm_bias,
    cross_attn_q_weight,
    cross_attn_q_bias,
    cross_attn_kv_weight,
    cross_attn_kv_bias,
    cross_attn_out_proj_weight,
    cross_attn_out_proj_bias,
    ffn_norm_weight,
    ffn_norm_bias,
    ffn_fc1_weight,
    ffn_fc1_bias,
    ffn_fc2_weight,
    ffn_fc2_bias,
    num_frames_scalar,
    num_spatial_tokens_scalar,
):
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-5

    batch_size, video_seq_len, _ = video_latents.shape
    text_seq_len = text_embeddings.shape[1]
    num_frames = int(num_frames_scalar)
    num_spatial_tokens = int(num_spatial_tokens_scalar)

    x = video_latents

    residual = x
    x_norm = F.layer_norm(
        x, (hidden_size,), temporal_norm_weight, temporal_norm_bias, eps
    )
    qkv = F.linear(x_norm, temporal_qkv_weight, temporal_qkv_bias)
    qkv = qkv.view(
        batch_size,
        num_frames,
        num_spatial_tokens,
        3,
        num_heads,
        head_dim,
    )
    # Pack directly into the head-major layout consumed by batched GEMM.  This
    # replaces both the three temporal transposes and matmul's three hidden
    # clones with one value-preserving copy.
    qkv = qkv.permute(3, 0, 2, 4, 1, 5).contiguous()
    q, k, v = qkv.unbind(0)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    attn = torch.matmul(probs, v)
    # Compose head interleaving and the spatial/temporal transpose into one copy.
    attn = attn.view(
        batch_size, num_spatial_tokens, num_heads, num_frames, head_dim
    )
    attn = attn.permute(0, 3, 1, 2, 4).contiguous()
    attn = attn.view(batch_size, video_seq_len, hidden_size)
    x = F.linear(attn, temporal_out_proj_weight, temporal_out_proj_bias) + residual

    residual = x
    x_norm = F.layer_norm(
        x, (hidden_size,), spatial_norm_weight, spatial_norm_bias, eps
    )
    x_reshaped = x_norm.view(
        batch_size, num_frames, num_spatial_tokens, hidden_size
    )
    spatial_batch = batch_size * num_frames
    qkv = F.linear(
        x_reshaped.view(spatial_batch, num_spatial_tokens, hidden_size),
        spatial_qkv_weight,
        spatial_qkv_bias,
    )
    qkv = qkv.view(
        spatial_batch, num_spatial_tokens, 3, num_heads, head_dim
    )
    # As above, produce the exact matrices matmul would otherwise clone one at
    # a time from the token-major projection output.
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
    q, k, v = qkv.unbind(0)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    attn = torch.matmul(probs, v)
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.view(spatial_batch, num_spatial_tokens, hidden_size)
    attn = attn.view(batch_size, video_seq_len, hidden_size)
    x = F.linear(attn, spatial_out_proj_weight, spatial_out_proj_bias) + residual

    residual = x
    x_norm = F.layer_norm(
        x, (hidden_size,), cross_attn_norm_weight, cross_attn_norm_bias, eps
    )
    q = F.linear(x_norm, cross_attn_q_weight, cross_attn_q_bias)
    kv = F.linear(text_embeddings, cross_attn_kv_weight, cross_attn_kv_bias)
    if batch_size > 1:
        q = q.view(
            batch_size, video_seq_len, num_heads, head_dim
        ).permute(0, 2, 1, 3).contiguous()
        kv = kv.view(
            batch_size, text_seq_len, 2, num_heads, head_dim
        ).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv.unbind(0)
    else:
        kv = kv.view(batch_size, text_seq_len, 2, hidden_size)
        k, v = kv.chunk(2, dim=2)
        k, v = k.squeeze(2), v.squeeze(2)
        q = q.view(batch_size, video_seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, text_seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, text_seq_len, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    attn = torch.matmul(probs, v)
    attn = attn.transpose(1, 2).contiguous()
    attn = attn.view(batch_size, video_seq_len, hidden_size)
    x = F.linear(attn, cross_attn_out_proj_weight, cross_attn_out_proj_bias) + residual

    residual = x
    x_norm = F.layer_norm(x, (hidden_size,), ffn_norm_weight, ffn_norm_bias, eps)
    x_ffn = F.linear(x_norm, ffn_fc1_weight, ffn_fc1_bias)
    x_ffn = F.gelu(x_ffn)
    x_ffn = F.linear(x_ffn, ffn_fc2_weight, ffn_fc2_bias)
    return x_ffn + residual
