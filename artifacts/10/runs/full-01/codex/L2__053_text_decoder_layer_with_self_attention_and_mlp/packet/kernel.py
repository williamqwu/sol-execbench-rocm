import torch
import torch.nn.functional as F


@torch.compile(fullgraph=True, dynamic=True)
def _rms_norm(x, weight, eps):
    xf = x.float()
    variance = xf.square().mean(-1, keepdim=True)
    return weight * (xf * torch.rsqrt(variance + eps)).to(x.dtype)


@torch.compile(fullgraph=True, dynamic=True)
def _rope_and_repeat(q, k, v, rope_theta):
    bsz, _, seq_len, _ = q.shape
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, 128, 2, dtype=torch.float32, device=q.device) / 128)
    )
    positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    freqs = (inv_freq[:, None] @ positions[None, :]).T
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None]
    sin = emb.sin()[None, None]

    q_rot = torch.cat((-q[..., 64:], q[..., :64]), dim=-1)
    k_rot = torch.cat((-k[..., 64:], k[..., :64]), dim=-1)
    q = (q * cos + q_rot * sin).to(q.dtype)
    k = (k * cos + k_rot * sin).to(k.dtype)

    k = k[:, :, None].expand(bsz, 8, 4, seq_len, 128).reshape(
        bsz, 32, seq_len, 128
    )
    v = v[:, :, None].expand(bsz, 8, 4, seq_len, 128).reshape(
        bsz, 32, seq_len, 128
    )
    return q, k, v


@torch.compile(fullgraph=True, dynamic=True)
def _add_rms_norm(projected, residual, weight, eps):
    hidden = residual + projected
    xf = hidden.float()
    variance = xf.square().mean(-1, keepdim=True)
    normed = weight * (xf * torch.rsqrt(variance + eps)).to(hidden.dtype)
    return hidden, normed


@torch.compile(fullgraph=True, dynamic=True)
def _swiglu(gate, up):
    return F.silu(gate) * up


@torch.no_grad()
def run(
    hidden_states,
    attention_mask,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    gate_proj_weight,
    up_proj_weight,
    down_proj_weight,
    input_layernorm_weight,
    post_attention_layernorm_weight,
    rms_norm_eps,
    rope_theta,
):
    bsz, seq_len, hidden_size = hidden_states.shape

    residual = hidden_states
    x = _rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)

    q = F.linear(x, q_proj_weight).view(bsz, seq_len, 32, 128).transpose(1, 2)
    k = F.linear(x, k_proj_weight).view(bsz, seq_len, 8, 128).transpose(1, 2)
    v = F.linear(x, v_proj_weight).view(bsz, seq_len, 8, 128).transpose(1, 2)

    q, k, v = _rope_and_repeat(q, k, v, rope_theta)

    attn = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask[:, :, :seq_len, :seq_len],
        dropout_p=0.0,
        scale=128 ** -0.5,
    )
    attn = attn.transpose(1, 2).contiguous().reshape(bsz, seq_len, hidden_size)

    projected = F.linear(attn, o_proj_weight)
    residual, x = _add_rms_norm(
        projected, residual, post_attention_layernorm_weight, rms_norm_eps
    )
    gate = F.linear(x, gate_proj_weight)
    up = F.linear(x, up_proj_weight)
    hidden = F.linear(_swiglu(gate, up), down_proj_weight)
    return residual + hidden
