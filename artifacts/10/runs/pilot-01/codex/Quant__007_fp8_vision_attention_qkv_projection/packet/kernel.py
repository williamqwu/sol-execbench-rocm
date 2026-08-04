import torch
import aiter
import triton
import triton.language as tl


E4M3_MAX = 448.0


@triton.jit
def _bias_split_kernel(qkv, bias, q_out, k_out, v_out, total:tl.constexpr, BLOCK:tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    col = offs % 4608
    row = offs // 4608
    val = tl.load(qkv + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias + col, mask=mask, other=0.0).to(tl.float32)
    val = val + b

    q_mask = mask & (col < 1536)
    k_mask = mask & ((col >= 1536) & (col < 3072))
    v_mask = mask & (col >= 3072)
    q_col = col
    k_col = col - 1536
    v_col = col - 3072
    tl.store(q_out + row * 1536 + q_col, val, mask=q_mask)
    tl.store(k_out + row * 1536 + k_col, val, mask=k_mask)
    tl.store(v_out + row * 1536 + v_col, val, mask=v_mask)


@triton.jit
def _weight_scale_kernel(weight, scales, CHUNK_N:tl.constexpr, BLOCK_K:tl.constexpr):
    nb = tl.program_id(0)
    kb = tl.program_id(1)
    offs_k = kb * 128 + tl.arange(0, BLOCK_K)
    max_k = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for start in range(0, 128, CHUNK_N):
        offs_n = nb * 128 + start + tl.arange(0, CHUNK_N)
        vals = tl.load(weight + offs_n[:, None] * 1536 + offs_k[None, :]).to(tl.float32)
        max_k = tl.maximum(max_k, tl.max(tl.abs(vals), axis=0))
    scale = tl.maximum(tl.max(max_k, axis=0) / 448.0, 1.0e-12)
    tl.store(scales + nb * 12 + kb, scale)


@triton.jit
def _weight_quant_kernel(weight, scales, qweight, total:tl.constexpr, BLOCK:tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = offs // 1536
    k = offs % 1536
    scale = tl.load(scales + (n // 128) * 12 + (k // 128))
    vals = tl.load(weight + offs).to(tl.float32) / scale
    vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
    tl.store(qweight + offs, vals)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    seq_len = hidden_states.shape[0]

    qx = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)
    scale_x = torch.empty((seq_len, 12), device=hidden_states.device, dtype=torch.float32)
    aiter.dynamic_per_group_scaled_quant(qx, hidden_states, scale_x, 128, False)

    w_f32 = qkv_weight.to(torch.float32)
    weight_scales = w_f32.reshape(36, 128, 12, 128).abs().amax(dim=3).amax(dim=1)
    weight_scales = torch.clamp(weight_scales / E4M3_MAX, min=1.0e-12)
    qw = torch.empty_like(qkv_weight, dtype=torch.float8_e4m3fn)
    _weight_quant_kernel[(6912,)](
        qkv_weight, weight_scales, qw, 4608 * 1536, BLOCK=1024, num_warps=4
    )

    qkv = aiter.gemm_a8w8_blockscale(
        qx, qw, scale_x, weight_scales, torch.bfloat16
    )
    query_states = torch.empty((seq_len, 16, 96), device=hidden_states.device, dtype=torch.bfloat16)
    key_states = torch.empty_like(query_states)
    value_states = torch.empty_like(query_states)
    total = seq_len * 4608
    _bias_split_kernel[(triton.cdiv(total, 1024),)](
        qkv, qkv_bias, query_states, key_states, value_states, total, BLOCK=1024, num_warps=4
    )
    return query_states, key_states, value_states
