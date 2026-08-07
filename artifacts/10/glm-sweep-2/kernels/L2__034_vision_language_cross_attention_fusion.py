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


# Cache inv_freq tensors (constants, don't depend on inputs)
_INV_FREQ_CACHE = {}

def _get_inv_freq(name, dim, theta, device):
    key = (name, device.index if device else 0)
    if key not in _INV_FREQ_CACHE:
        _INV_FREQ_CACHE[key] = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return _INV_FREQ_CACHE[key]


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
    
    pos = tl.load(pos_ptr + pid_b * stride_pb + offs_s * stride_ps, mask=mask_s, other=0).to(tl.float32)
    invf = tl.load(invfreq_ptr + offs_d).to(tl.float32)
    freqs = pos[:, None] * invf[None, :]
    cos = tl.cos(freqs).to(tl.bfloat16)
    sin = tl.sin(freqs).to(tl.bfloat16)
    
    q_base = pid_b * stride_qb + pid_h * stride_qh + offs_s[:, None] * stride_qs
    q1 = tl.load(q_ptr + q_base + offs_d[None, :] * stride_qd, mask=mask2d, other=0.0)
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
    
    # T component
    mt2d = mask2d & (offs[None, :] < HALF_T)
    inft = tl.load(inft_ptr + offs, mask=offs < HALF_T, other=0.0).to(tl.float32)
    ft = t_pos[:, None] * inft[None, :]
    ct = tl.cos(ft).to(tl.bfloat16)
    st = tl.sin(ft).to(tl.bfloat16)
    kt1 = tl.load(k_ptr + k_base + offs[None, :] * stride_kd, mask=mt2d, other=0.0)
    kt2 = tl.load(k_ptr + k_base + (HALF_T + offs[None, :]) * stride_kd, mask=mt2d, other=0.0)
    ot1 = kt1 * ct - kt2 * st
    ot2 = kt2 * ct + kt1 * st
    tl.store(out_ptr + o_base + offs[None, :] * stride_od, ot1, mask=mt2d)
    tl.store(out_ptr + o_base + (HALF_T + offs[None, :]) * stride_od, ot2, mask=mt2d)
    
    # H component
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
    
    # W component
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


_HEAD_DIM = 128
_NUM_Q_HEADS = 32
_NUM_KV_HEADS = 8
_NUM_KV_GROUPS = 4
_DIM_T = 42
_DIM_H = 42
_DIM_W = 44
_ROPE_THETA = 10000.0
_SCALE = 1.0 / math.sqrt(_HEAD_DIM)


@torch.no_grad()
def run(
    language_hidden_states, vision_hidden_states, language_position_ids, vision_grid_thw,
    q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight,
):
    batch_size, lang_seq_len, _ = language_hidden_states.shape
    vision_seq_len = vision_hidden_states.shape[1]
    device = language_hidden_states.device
    
    inv_freq_1d = _get_inv_freq("1d", _HEAD_DIM, _ROPE_THETA, device)
    inv_freq_t = _get_inv_freq("t", _DIM_T, _ROPE_THETA, device)
    inv_freq_h = _get_inv_freq("h", _DIM_H, _ROPE_THETA, device)
    inv_freq_w = _get_inv_freq("w", _DIM_W, _ROPE_THETA, device)
    
    # Fuse K+V projection (single GEMM, then split)
    kv_w = torch.cat([k_proj_weight, v_proj_weight], dim=0)
    kv_b = torch.cat([k_proj_bias, v_proj_bias], dim=0)
    kv_dim = k_proj_weight.shape[0]
    
    query_states = F.linear(language_hidden_states, q_proj_weight, q_proj_bias)
    query_states = query_states.view(batch_size, lang_seq_len, _NUM_Q_HEADS, _HEAD_DIM).transpose(1, 2)
    
    kv = F.linear(vision_hidden_states, kv_w, kv_b)
    key_states, value_states = kv[..., :kv_dim], kv[..., kv_dim:]
    
    key_states = key_states.view(batch_size, vision_seq_len, _NUM_KV_HEADS, _HEAD_DIM).transpose(1, 2)
    value_states = value_states.view(batch_size, vision_seq_len, _NUM_KV_HEADS, _HEAD_DIM).transpose(1, 2)
    
    query_states = rope_1d_triton(query_states, language_position_ids, inv_freq_1d)
    key_states = rope_3d_triton(key_states, vision_grid_thw, inv_freq_t, inv_freq_h, inv_freq_w, _DIM_T, _DIM_H, _DIM_W)
    
    key_states = key_states[:, :, None, :, :].expand(batch_size, _NUM_KV_HEADS, _NUM_KV_GROUPS, vision_seq_len, _HEAD_DIM).reshape(batch_size, _NUM_Q_HEADS, vision_seq_len, _HEAD_DIM)
    value_states = value_states[:, :, None, :, :].expand(batch_size, _NUM_KV_HEADS, _NUM_KV_GROUPS, vision_seq_len, _HEAD_DIM).reshape(batch_size, _NUM_Q_HEADS, vision_seq_len, _HEAD_DIM)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * _SCALE
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, lang_seq_len, _NUM_Q_HEADS * _HEAD_DIM)
    output = F.linear(attn_output, o_proj_weight)
    
    return output
