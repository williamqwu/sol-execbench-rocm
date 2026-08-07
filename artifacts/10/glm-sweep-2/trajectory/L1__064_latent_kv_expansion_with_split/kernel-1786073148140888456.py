import torch
import triton
import triton.language as tl

NUM_HEADS = 128
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
KV_LORA_RANK = 512

@triton.jit
def _transpose_split_kernel(
    expanded_ptr, k_ptr, v_ptr,
    B, S,
    stride_exp_b, stride_exp_s, stride_exp_h,
    stride_k_b, stride_k_h, stride_k_s,
    DH: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    base = pid_b * stride_exp_b + pid_h * stride_exp_h
    d_k = tl.arange(0, DH)
    d_v = DH + tl.arange(0, DH)
    offs_k = base + s_off[:, None] * stride_exp_s + d_k[None, :]
    offs_v = base + s_off[:, None] * stride_exp_s + d_v[None, :]
    k_part = tl.load(expanded_ptr + offs_k, mask=s_mask[:, None], other=0.0)
    v_part = tl.load(expanded_ptr + offs_v, mask=s_mask[:, None], other=0.0)
    out_offs = pid_b * stride_k_b + pid_h * stride_k_h + s_off[:, None] * stride_k_s + tl.arange(0, DH)[None, :]
    tl.store(k_ptr + out_offs, k_part, mask=s_mask[:, None])
    tl.store(v_ptr + out_offs, v_part, mask=s_mask[:, None])


@torch.no_grad()
def run(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    bsz, seq_len, _ = compressed_kv.shape

    input_dtype = compressed_kv.dtype
    hidden_states = compressed_kv.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized_kv = (kv_a_layernorm_weight.to(torch.float32) * hidden_states).to(input_dtype)

    expanded_kv = torch.matmul(normalized_kv, kv_b_proj_weight.t())
    # [B, S, 32768] -> [B, S, 128, 256] (contiguous view, no copy)
    expanded_4d = expanded_kv.view(bsz, seq_len, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)

    k_nope = torch.empty(bsz, NUM_HEADS, seq_len, QK_NOPE_HEAD_DIM, dtype=input_dtype, device=compressed_kv.device)
    value_states = torch.empty(bsz, NUM_HEADS, seq_len, V_HEAD_DIM, dtype=input_dtype, device=compressed_kv.device)

    BLOCK_S = 64
    grid = (bsz, NUM_HEADS, triton.cdiv(seq_len, BLOCK_S))
    _transpose_split_kernel[grid](
        expanded_4d, k_nope, value_states, bsz, seq_len,
        expanded_4d.stride(0), expanded_4d.stride(1), expanded_4d.stride(2),
        k_nope.stride(0), k_nope.stride(1), k_nope.stride(2),
        DH=QK_NOPE_HEAD_DIM, BLOCK_S=BLOCK_S,
    )

    return k_nope, value_states
