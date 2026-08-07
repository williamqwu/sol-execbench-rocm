import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    text_seq_len = axes_and_scalars["text_seq_len"]
    hidden_size = 3072
    joint_attention_dim = 4096
    modulation_dim = hidden_size * 6
    rope_axis0_dim = 16
    rope_axis1_dim = 56
    rope_axis2_dim = 56
    head_dim = 128
    rope_dim_per_axis = head_dim // 3  # 42
    is_joint_block = axes_and_scalars["is_joint_block"]

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    timestep_embedding = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, joint_attention_dim, device=device, dtype=torch.float32)

    adaln_linear_weight = torch.randn(modulation_dim, hidden_size, device=device, dtype=torch.float32) * 0.02
    adaln_linear_bias = torch.zeros(modulation_dim, device=device, dtype=torch.float32)

    to_q_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_q_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_k_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_k_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_v_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_v_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)

    to_k_context_weight = torch.randn(hidden_size, joint_attention_dim, device=device, dtype=torch.float32) * 0.02
    to_k_context_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)
    to_v_context_weight = torch.randn(hidden_size, joint_attention_dim, device=device, dtype=torch.float32) * 0.02
    to_v_context_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)

    to_out_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    to_out_bias = torch.zeros(hidden_size, device=device, dtype=torch.float32)

    pos_idx_axis0 = torch.randint(0, rope_axis0_dim, (seq_len,), device=device, dtype=torch.int64)
    pos_idx_axis1 = torch.randint(0, rope_axis1_dim, (seq_len,), device=device, dtype=torch.int64)
    pos_idx_axis2 = torch.randint(0, rope_axis2_dim, (seq_len,), device=device, dtype=torch.int64)

    rope_theta = 10000.0
    inv_freq0 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions0 = torch.arange(rope_axis0_dim, dtype=torch.float32, device=device)
    freqs0 = torch.outer(positions0, inv_freq0)
    rope_cos_axis0 = freqs0.cos()
    rope_sin_axis0 = freqs0.sin()

    inv_freq1 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions1 = torch.arange(rope_axis1_dim, dtype=torch.float32, device=device)
    freqs1 = torch.outer(positions1, inv_freq1)
    rope_cos_axis1 = freqs1.cos()
    rope_sin_axis1 = freqs1.sin()

    inv_freq2 = 1.0 / (rope_theta ** (torch.arange(0, rope_dim_per_axis, dtype=torch.float32, device=device) / rope_dim_per_axis))
    positions2 = torch.arange(rope_axis2_dim, dtype=torch.float32, device=device)
    freqs2 = torch.outer(positions2, inv_freq2)
    rope_cos_axis2 = freqs2.cos()
    rope_sin_axis2 = freqs2.sin()

    return {
        "hidden_states": hidden_states,
        "timestep_embedding": timestep_embedding,
        "encoder_hidden_states": encoder_hidden_states,
        "adaln_linear_weight": adaln_linear_weight,
        "adaln_linear_bias": adaln_linear_bias,
        "to_q_weight": to_q_weight,
        "to_q_bias": to_q_bias,
        "to_k_weight": to_k_weight,
        "to_k_bias": to_k_bias,
        "to_v_weight": to_v_weight,
        "to_v_bias": to_v_bias,
        "to_k_context_weight": to_k_context_weight,
        "to_k_context_bias": to_k_context_bias,
        "to_v_context_weight": to_v_context_weight,
        "to_v_context_bias": to_v_context_bias,
        "to_out_weight": to_out_weight,
        "to_out_bias": to_out_bias,
        "pos_idx_axis0": pos_idx_axis0,
        "pos_idx_axis1": pos_idx_axis1,
        "pos_idx_axis2": pos_idx_axis2,
        "rope_cos_axis0": rope_cos_axis0,
        "rope_sin_axis0": rope_sin_axis0,
        "rope_cos_axis1": rope_cos_axis1,
        "rope_sin_axis1": rope_sin_axis1,
        "rope_cos_axis2": rope_cos_axis2,
        "rope_sin_axis2": rope_sin_axis2,
        "is_joint_block": is_joint_block,
    }


@triton.jit
def _rope_3axis_kernel(
    x_ptr, out_ptr,
    pi0_ptr, pi1_ptr, pi2_ptr,
    cos0_ptr, sin0_ptr,
    cos1_ptr, sin1_ptr,
    cos2_ptr, sin2_ptr,
    S, H,
    P0, P1, P2,
    RDA: tl.constexpr,
    HALF: tl.constexpr,
    REST: tl.constexpr,
    BH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    bs = pid // S
    s = pid % S

    idx0 = tl.load(pi0_ptr + s)
    idx1 = tl.load(pi1_ptr + s)
    idx2 = tl.load(pi2_ptr + s)

    off = tl.arange(0, BH)
    hmask = off < HALF

    c0 = tl.load(cos0_ptr + idx0 * RDA + off, mask=hmask, other=0.0)
    s0 = tl.load(sin0_ptr + idx0 * RDA + off, mask=hmask, other=0.0)
    c1 = tl.load(cos1_ptr + idx1 * RDA + off, mask=hmask, other=0.0)
    s1 = tl.load(sin1_ptr + idx1 * RDA + off, mask=hmask, other=0.0)
    c2 = tl.load(cos2_ptr + idx2 * RDA + off, mask=hmask, other=0.0)
    s2 = tl.load(sin2_ptr + idx2 * RDA + off, mask=hmask, other=0.0)

    row_base = (bs * S + s) * H * D
    roff = tl.arange(0, REST)

    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        hb = row_base + h_off[:, None] * D
        m2 = h_mask[:, None] & hmask[None, :]

        x0f = tl.load(x_ptr + hb + off[None, :], mask=m2, other=0.0)
        x0s = tl.load(x_ptr + hb + (HALF + off)[None, :], mask=m2, other=0.0)
        tl.store(out_ptr + hb + off[None, :], x0f * c0[None, :] - x0s * s0[None, :], mask=m2)
        tl.store(out_ptr + hb + (HALF + off)[None, :], x0f * s0[None, :] + x0s * c0[None, :], mask=m2)

        b1 = RDA
        x1f = tl.load(x_ptr + hb + (b1 + off)[None, :], mask=m2, other=0.0)
        x1s = tl.load(x_ptr + hb + (b1 + HALF + off)[None, :], mask=m2, other=0.0)
        tl.store(out_ptr + hb + (b1 + off)[None, :], x1f * c1[None, :] - x1s * s1[None, :], mask=m2)
        tl.store(out_ptr + hb + (b1 + HALF + off)[None, :], x1f * s1[None, :] + x1s * c1[None, :], mask=m2)

        b2 = 2 * RDA
        x2f = tl.load(x_ptr + hb + (b2 + off)[None, :], mask=m2, other=0.0)
        x2s = tl.load(x_ptr + hb + (b2 + HALF + off)[None, :], mask=m2, other=0.0)
        tl.store(out_ptr + hb + (b2 + off)[None, :], x2f * c2[None, :] - x2s * s2[None, :], mask=m2)
        tl.store(out_ptr + hb + (b2 + HALF + off)[None, :], x2f * s2[None, :] + x2s * c2[None, :], mask=m2)

        if REST > 0:
            m3 = h_mask[:, None]
            xr = tl.load(x_ptr + hb + (3 * RDA + roff)[None, :], mask=m3, other=0.0)
            tl.store(out_ptr + hb + (3 * RDA + roff)[None, :], xr, mask=m3)


def _rope_3axis(x, pi0, pi1, pi2, cos0, sin0, cos1, sin1, cos2, sin2, rda, head_dim):
    B, S, H, D = x.shape
    half = rda // 2
    rest = D - 3 * rda
    out = torch.empty_like(x)
    grid = (B * S,)
    bh = max(triton.next_power_of_2(half), 1)
    _rope_3axis_kernel[grid](
        x, out, pi0, pi1, pi2, cos0, sin0, cos1, sin1, cos2, sin2,
        S, H,
        cos0.shape[0], cos1.shape[0], cos2.shape[0],
        RDA=rda, HALF=half, REST=rest, BH=bh, BLOCK_H=32, D=D, enable_fp_fusion=False,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    timestep_embedding: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    adaln_linear_weight: torch.Tensor,
    adaln_linear_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    to_k_context_weight: torch.Tensor,
    to_k_context_bias: torch.Tensor,
    to_v_context_weight: torch.Tensor,
    to_v_context_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    pos_idx_axis0: torch.Tensor,
    pos_idx_axis1: torch.Tensor,
    pos_idx_axis2: torch.Tensor,
    rope_cos_axis0: torch.Tensor,
    rope_sin_axis0: torch.Tensor,
    rope_cos_axis1: torch.Tensor,
    rope_sin_axis1: torch.Tensor,
    rope_cos_axis2: torch.Tensor,
    rope_sin_axis2: torch.Tensor,
    is_joint_block: int,
):
    batch, seq_len, hidden_size = hidden_states.shape
    num_heads = 24
    head_dim = 128
    rope_dim_per_axis = head_dim // 3  # 42

    residual = hidden_states

    timestep_activated = timestep_embedding * torch.sigmoid(timestep_embedding)
    modulation = F.linear(timestep_activated, adaln_linear_weight, adaln_linear_bias)
    scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp = modulation.chunk(6, dim=-1)

    hidden_states_normalized = F.layer_norm(hidden_states, (hidden_size,))
    hidden_states_modulated = hidden_states_normalized * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

    q = F.linear(hidden_states_modulated, to_q_weight, to_q_bias)
    k = F.linear(hidden_states_modulated, to_k_weight, to_k_bias)
    v = F.linear(hidden_states_modulated, to_v_weight, to_v_bias)

    q = q.view(batch, seq_len, num_heads, head_dim)
    k = k.view(batch, seq_len, num_heads, head_dim)
    v = v.view(batch, seq_len, num_heads, head_dim)

    q = _rope_3axis(q, pos_idx_axis0, pos_idx_axis1, pos_idx_axis2,
                    rope_cos_axis0, rope_sin_axis0,
                    rope_cos_axis1, rope_sin_axis1,
                    rope_cos_axis2, rope_sin_axis2,
                    rope_dim_per_axis, head_dim)
    k = _rope_3axis(k, pos_idx_axis0, pos_idx_axis1, pos_idx_axis2,
                    rope_cos_axis0, rope_sin_axis0,
                    rope_cos_axis1, rope_sin_axis1,
                    rope_cos_axis2, rope_sin_axis2,
                    rope_dim_per_axis, head_dim)

    if is_joint_block == 1:
        text_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states_normalized = F.layer_norm(encoder_hidden_states, (4096,))
        k_context = F.linear(encoder_hidden_states_normalized, to_k_context_weight, to_k_context_bias)
        v_context = F.linear(encoder_hidden_states_normalized, to_v_context_weight, to_v_context_bias)
        k_context = k_context.view(batch, text_seq_len, num_heads, head_dim)
        v_context = v_context.view(batch, text_seq_len, num_heads, head_dim)
        k = torch.cat([k, k_context], dim=1)
        v = torch.cat([v, v_context], dim=1)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scale = 1.0 / math.sqrt(head_dim)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)

    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, hidden_size)
    attn_output = F.linear(attn_output, to_out_weight, to_out_bias)

    output = residual + gate_msa.unsqueeze(1) * attn_output
    return output
