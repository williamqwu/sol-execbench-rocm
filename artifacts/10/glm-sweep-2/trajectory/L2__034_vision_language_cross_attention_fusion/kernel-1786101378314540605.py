import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    lang_seq_len = axes_and_scalars["lang_seq_len"]
    vision_seq_len = axes_and_scalars["vision_seq_len"]
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    
    language_hidden_states = torch.randn(batch_size, lang_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    vision_hidden_states = torch.randn(batch_size, vision_seq_len, hidden_size, dtype=torch.bfloat16, device=device)
    
    language_position_ids = torch.arange(lang_seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    
    vision_grid_thw = torch.zeros(batch_size, vision_seq_len, 3, dtype=torch.int64, device=device)
    for b in range(batch_size):
        for i in range(vision_seq_len):
            t = i // 196
            spatial_idx = i % 196
            h = spatial_idx // 14
            w = spatial_idx % 14
            vision_grid_thw[b, i, 0] = t
            vision_grid_thw[b, i, 1] = h
            vision_grid_thw[b, i, 2] = w
    
    q_proj_weight = torch.randn(num_attention_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    q_proj_bias = torch.randn(num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    k_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_weight = torch.randn(num_key_value_heads * head_dim, hidden_size, dtype=torch.bfloat16, device=device) * 0.02
    v_proj_bias = torch.randn(num_key_value_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    o_proj_weight = torch.randn(hidden_size, num_attention_heads * head_dim, dtype=torch.bfloat16, device=device) * 0.02
    
    return {
        "language_hidden_states": language_hidden_states,
        "vision_hidden_states": vision_hidden_states,
        "language_position_ids": language_position_ids,
        "vision_grid_thw": vision_grid_thw,
        "q_proj_weight": q_proj_weight,
        "q_proj_bias": q_proj_bias,
        "k_proj_weight": k_proj_weight,
        "k_proj_bias": k_proj_bias,
        "v_proj_weight": v_proj_weight,
        "v_proj_bias": v_proj_bias,
        "o_proj_weight": o_proj_weight,
    }


# ---- Triton RoPE kernels ----
# 1D RoPE: applies rope to q [B, H, S, 128] using position_ids [B, S] and inv_freq [64]
@triton.jit
def _rope1d_kernel(q_ptr, pos_ptr, invfreq_ptr, out_ptr,
                   S, H, D,
                   stride_qb, stride_qh, stride_qs, stride_qd,
                   stride_pb, stride_ps,
                   stride_ob, stride_oh, stride_os, stride_od,
                   BLOCK_S: tl.constexpr, HALF_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, HALF_D)
    mask_s = offs_s < S
    mask2d = mask_s[:, None]
    
    # position for this row
    pos = tl.load(pos_ptr + pid_b * stride_pb + offs_s * stride_ps, mask=mask_s, other=0).to(tl.float32)  # (BLOCK_S,)
    invf = tl.load(invfreq_ptr + offs_d).to(tl.float32)  # (HALF_D,)
    freqs = pos[:, None] * invf[None, :]  # (BLOCK_S, HALF_D) fp32
    cos = tl.cos(freqs).to(tl.bfloat16)
    sin = tl.sin(freqs).to(tl.bfloat16)
    
    q_base = pid_b * stride_qb + pid_h * stride_qh + offs_s[:, None] * stride_qs
    q1 = tl.load(q_ptr + q_base + offs_d[None, :] * stride_qd, mask=mask2d, other=0.0)  # bf16
    q2 = tl.load(q_ptr + q_base + (HALF_D + offs_d[None, :]) * stride_qd, mask=mask2d, other=0.0)
    
    o1 = q1 * cos - q2 * sin
    o2 = q2 * cos + q1 * sin
    
    o_base = pid_b * stride_ob + pid_h * stride_oh + offs_s[:, None] * stride_os
    tl.store(out_ptr + o_base + offs_d[None, :] * stride_od, o1, mask=mask2d)
    tl.store(out_ptr + o_base + (HALF_D + offs_d[None, :]) * stride_od, o2, mask=mask2d)


def rope_1d_triton(q, position_ids, inv_freq):
    B, H, S, D = q.shape
    HALF_D = D // 2
    out = torch.empty_like(q)
    BLOCK_S = triton.next_power_of_2(min(S, 64))
    grid = (B, H, triton.cdiv(S, BLOCK_S))
    _rope1d_kernel[grid](q, position_ids, inv_freq, out, S, H, D,
                         q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                         position_ids.stride(0), position_ids.stride(1),
                         out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                         BLOCK_S=BLOCK_S, HALF_D=HALF_D)
    return out


# 3D RoPE: applies rope to k [B, H, S, 128] split into 42/42/44
@triton.jit
def _rope3d_kernel(k_ptr, thw_ptr, inft_ptr, infh_ptr, infw_ptr, out_ptr,
                   S, H, D,
                   stride_kb, stride_kh, stride_ks, stride_kd,
                   stride_tb, stride_ts, stride_td,
                   stride_ob, stride_oh, stride_os, stride_od,
                   DIM_T: tl.constexpr, DIM_H: tl.constexpr, DIM_W: tl.constexpr,
                   HALF_T: tl.constexpr, HALF_H: tl.constexpr, HALF_W: tl.constexpr,
                   HALF_MAX: tl.constexpr, BLOCK_S: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    
    thw_base = pid_b * stride_tb + offs_s * stride_ts
    t_pos = tl.load(thw_ptr + thw_base + 0 * stride_td, mask=mask_s, other=0).to(tl.float32)
    h_pos = tl.load(thw_ptr + thw_base + 1 * stride_td, mask=mask_s, other=0).to(tl.float32)
    w_pos = tl.load(thw_ptr + thw_base + 2 * stride_td, mask=mask_s, other=0).to(tl.float32)
    
    k_base = pid_b * stride_kb + pid_h * stride_kh + offs_s[:, None] * stride_ks
    o_base = pid_b * stride_ob + pid_h * stride_oh + offs_s[:, None] * stride_os
    offs = tl.arange(0, HALF_MAX)
    mask2d = mask_s[:, None]
    
    # --- T component ---
    mt = offs < HALF_T
    mt2d = mask2d & (offs[None, :] < HALF_T)
    inft = tl.load(inft_ptr + offs, mask=mt, other=0.0).to(tl.float32)
    ft = t_pos[:, None] * inft[None, :]
    ct = tl.cos(ft).to(tl.bfloat16)
    st = tl.sin(ft).to(tl.bfloat16)
    kt1 = tl.load(k_ptr + k_base + offs[None, :] * stride_kd, mask=mt2d, other=0.0)
    kt2 = tl.load(k_ptr + k_base + (HALF_T + offs[None, :]) * stride_kd, mask=mt2d, other=0.0)
    ot1 = kt1 * ct - kt2 * st
    ot2 = kt2 * ct + kt1 * st
    tl.store(out_ptr + o_base + offs[None, :] * stride_od, ot1, mask=mt2d)
    tl.store(out_ptr + o_base + (HALF_T + offs[None, :]) * stride_od, ot2, mask=mt2d)
    
    # --- H component ---
    mh2d = mask2d & (offs[None, :] < HALF_H)
    infh = tl.load(infh_ptr + offs, mask=offs < HALF_H, other=0.0).to(tl.float32)
    fh = h_pos[:, None] * infh[None, :]
    ch = tl.cos(fh).to(tl.bfloat16)
    sh = tl.sin(fh).to(tl.bfloat16)
    kh1 = tl.load(k_ptr + k_base + (DIM_T + offs[None, :]) * stride_kd, mask=mh2d, other=0.0)
    kh2 = tl.load(k_ptr + k_base + (DIM_T + HALF_H + offs[None, :]) * stride_kd, mask=mh2d, other=0.0)
    oh1 = kh1 * ch - kh2 * sh
    oh2 = kh2 * ch + kh1 * sh
    tl.store(out_ptr + o_base + (DIM_T + offs[None, :]) * stride_od, oh1, mask=mh2d)
    tl.store(out_ptr + o_base + (DIM_T + HALF_H + offs[None, :]) * stride_od, oh2, mask=mh2d)
    
    # --- W component ---
    mw2d = mask2d & (offs[None, :] < HALF_W)
    infw = tl.load(infw_ptr + offs, mask=offs < HALF_W, other=0.0).to(tl.float32)
    fw = w_pos[:, None] * infw[None, :]
    cw = tl.cos(fw).to(tl.bfloat16)
    sw = tl.sin(fw).to(tl.bfloat16)
    kw1 = tl.load(k_ptr + k_base + (DIM_T + DIM_H + offs[None, :]) * stride_kd, mask=mw2d, other=0.0)
    kw2 = tl.load(k_ptr + k_base + (DIM_T + DIM_H + HALF_W + offs[None, :]) * stride_kd, mask=mw2d, other=0.0)
    ow1 = kw1 * cw - kw2 * sw
    ow2 = kw2 * cw + kw1 * sw
    tl.store(out_ptr + o_base + (DIM_T + DIM_H + offs[None, :]) * stride_od, ow1, mask=mw2d)
    tl.store(out_ptr + o_base + (DIM_T + DIM_H + HALF_W + offs[None, :]) * stride_od, ow2, mask=mw2d)


def rope_3d_triton(k, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, dim_t, dim_h, dim_w):
    B, H, S, D = k.shape
    out = torch.empty_like(k)
    BLOCK_S = triton.next_power_of_2(min(S, 64))
    grid = (B, H, triton.cdiv(S, BLOCK_S))
    _rope3d_kernel[grid](k, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, out,
                         S, H, D,
                         k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                         vision_grid_thw.stride(0), vision_grid_thw.stride(1), vision_grid_thw.stride(2),
                         out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                         DIM_T=dim_t, DIM_H=dim_h, DIM_W=dim_w,
                         HALF_T=dim_t//2, HALF_H=dim_h//2, HALF_W=dim_w//2,
                         HALF_MAX=32, BLOCK_S=BLOCK_S)
    return out


@torch.no_grad()
def run(
    language_hidden_states, vision_hidden_states, language_position_ids, vision_grid_thw,
    q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight,
):
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    num_kv_groups = num_attention_heads // num_key_value_heads
    rope_theta = 10000.0
    
    batch_size, lang_seq_len, _ = language_hidden_states.shape
    vision_seq_len = vision_hidden_states.shape[1]
    device = language_hidden_states.device
    
    inv_freq_1d = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    dim_t = 42; dim_h = 42; dim_w = 44
    inv_freq_t = 1.0 / (rope_theta ** (torch.arange(0, dim_t, 2, dtype=torch.float32, device=device) / dim_t))
    inv_freq_h = 1.0 / (rope_theta ** (torch.arange(0, dim_h, 2, dtype=torch.float32, device=device) / dim_h))
    inv_freq_w = 1.0 / (rope_theta ** (torch.arange(0, dim_w, 2, dtype=torch.float32, device=device) / dim_w))
    
    query_states = F.linear(language_hidden_states, q_proj_weight, q_proj_bias)
    query_states = query_states.view(batch_size, lang_seq_len, num_attention_heads, head_dim).transpose(1, 2)
    
    key_states = F.linear(vision_hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(vision_hidden_states, v_proj_weight, v_proj_bias)
    
    key_states = key_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, vision_seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE via Triton (fused: compute cos/sin + apply in single kernel each)
    query_states = rope_1d_triton(query_states, language_position_ids, inv_freq_1d)
    key_states = rope_3d_triton(key_states, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, dim_t, dim_h, dim_w)
    
    key_states = key_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_kv_groups, vision_seq_len, head_dim).reshape(batch_size, num_attention_heads, vision_seq_len, head_dim)
    value_states = value_states[:, :, None, :, :].expand(batch_size, num_key_value_heads, num_kv_groups, vision_seq_len, head_dim).reshape(batch_size, num_attention_heads, vision_seq_len, head_dim)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, lang_seq_len, num_attention_heads * head_dim)
    output = F.linear(attn_output, o_proj_weight)
    
    return output
