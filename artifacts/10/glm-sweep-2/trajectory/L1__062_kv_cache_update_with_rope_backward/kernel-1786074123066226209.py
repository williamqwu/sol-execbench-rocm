import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bwd_kernel(
    grad_key_cache_ptr,   # [B, H, maxS, D]
    grad_value_cache_ptr,  # [B, H, maxS, D]
    key_states_ptr,        # [B, H, S, D]
    cos_ptr,               # [B, S, D]
    sin_ptr,               # [B, S, D]
    grad_key_states_ptr,   # [B, H, S, D]
    grad_value_states_ptr, # [B, H, S, D]
    grad_cos_ptr,          # [B, S, D]
    grad_sin_ptr,          # [B, S, D]
    H, S, maxS,
    D: tl.constexpr,
    halfD: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    cs_off = pid_b * S * D + pid_s * D
    cos = tl.load(cos_ptr + cs_off + offs_d, mask=d_mask, other=0.0)
    sin = tl.load(sin_ptr + cs_off + offs_d, mask=d_mask, other=0.0)

    gkc_bh_base = pid_b * H * maxS * D + pid_s * D
    ks_bh_base = pid_b * H * S * D + pid_s * D

    acc_cos = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_sin = tl.zeros([BLOCK_D], dtype=tl.float32)

    sin_swap = tl.where(offs_d < halfD,
                        tl.load(sin_ptr + cs_off + offs_d + halfD, mask=(offs_d < halfD) & d_mask, other=0.0),
                        tl.load(sin_ptr + cs_off + offs_d - halfD, mask=(offs_d >= halfD) & d_mask, other=0.0))

    for h in range(H):
        gkc_off = gkc_bh_base + h * maxS * D + offs_d
        gvc_off = gkc_bh_base + h * maxS * D + offs_d
        ks_off = ks_bh_base + h * S * D + offs_d

        gkr = tl.load(grad_key_cache_ptr + gkc_off, mask=d_mask, other=0.0)
        gvc = tl.load(grad_value_cache_ptr + gvc_off, mask=d_mask, other=0.0)
        ks = tl.load(key_states_ptr + ks_off, mask=d_mask, other=0.0)

        tl.store(grad_value_states_ptr + ks_off, gvc, mask=d_mask)

        krh = tl.where(offs_d < halfD,
                       -tl.load(key_states_ptr + ks_off + halfD, mask=(offs_d < halfD) & d_mask, other=0.0),
                       tl.load(key_states_ptr + ks_off - halfD, mask=(offs_d >= halfD) & d_mask, other=0.0))

        acc_cos += (gkr * ks).to(tl.float32)
        acc_sin += (gkr * krh).to(tl.float32)

        gkr_swap = tl.where(offs_d < halfD,
                            tl.load(grad_key_cache_ptr + gkc_off + halfD, mask=(offs_d < halfD) & d_mask, other=0.0),
                            tl.load(grad_key_cache_ptr + gkc_off - halfD, mask=(offs_d >= halfD) & d_mask, other=0.0))
        cos_term = (gkr * cos).to(tl.bfloat16)
        sin_term = (gkr_swap * sin_swap).to(tl.bfloat16)
        gks = tl.where(offs_d < halfD, cos_term + sin_term, cos_term - sin_term)
        tl.store(grad_key_states_ptr + ks_off, gks.to(tl.bfloat16), mask=d_mask)

    out_cs = pid_b * S * D + pid_s * D
    tl.store(grad_cos_ptr + out_cs + offs_d, acc_cos.to(tl.bfloat16), mask=d_mask)
    tl.store(grad_sin_ptr + out_cs + offs_d, acc_sin.to(tl.bfloat16), mask=d_mask)


@triton.jit
def _zero_head_copy_kernel(in_ptr, out_ptr, total, S, maxS, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    seq_idx = (offs % (maxS * D)) // D
    is_head = seq_idx < S
    val = tl.load(in_ptr + offs, mask=mask, other=0.0)
    out = tl.where(is_head, tl.zeros_like(val), val)
    tl.store(out_ptr + offs, out, mask=mask)


@torch.no_grad()
def _zero_head_copy(t, S):
    out = torch.empty_like(t)
    total = t.numel()
    BLOCK = 4096
    grid = (triton.cdiv(total, BLOCK),)
    _zero_head_copy_kernel[grid](t, out, total, S, t.shape[2], t.shape[3], BLOCK=BLOCK)
    return out


@torch.no_grad()
def run(
    grad_key_cache: torch.Tensor,
    grad_value_cache: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
):
    B, H, S, D = key_states.shape
    maxS = grad_key_cache.shape[2]
    halfD = D // 2
    BLOCK_D = triton.next_power_of_2(D)

    grad_key_states = torch.empty_like(key_states)
    grad_value_states = torch.empty_like(key_states)
    grad_cos = torch.empty((B, S, D), dtype=torch.bfloat16, device=key_states.device)
    grad_sin = torch.empty((B, S, D), dtype=torch.bfloat16, device=key_states.device)

    grid = (B, S)
    _rope_bwd_kernel[grid](
        grad_key_cache, grad_value_cache, key_states, cos, sin,
        grad_key_states, grad_value_states, grad_cos, grad_sin,
        H, S, maxS, D, halfD, BLOCK_D,
    )

    grad_key_cache_input = _zero_head_copy(grad_key_cache, S)
    grad_value_cache_input = _zero_head_copy(grad_value_cache, S)

    return (
        grad_key_states,
        grad_value_states,
        grad_cos,
        grad_sin,
        grad_key_cache_input,
        grad_value_cache_input,
    )
