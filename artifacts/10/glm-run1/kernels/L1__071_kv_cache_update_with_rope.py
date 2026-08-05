import torch
import triton
import triton.language as tl


@triton.jit
def _kv_copy_rope_kernel(
    out_ptr,          # [B, H, S_out, D]
    cache_ptr,        # [B, H, S_cur, D]
    new_ptr,          # [B, H, S_new, D]
    cos_ptr,          # [B, 1, S_new, D]
    sin_ptr,          # [B, 1, S_new, D]
    S_cur, S_new, S_out,
    H: tl.constexpr, D: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_S: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    d_offs = tl.arange(0, D)                  # [D]

    s2 = s_offs[:, None]                       # [BLOCK_S, 1]
    d2 = d_offs[None, :]                       # [1, D]
    sd = s2 * D + d2                           # [BLOCK_S, D]

    out_base = pid_b * (H * S_out * D) + pid_h * (S_out * D)
    cache_base = pid_b * (H * S_cur * D) + pid_h * (S_cur * D)
    new_base = pid_b * (H * S_new * D) + pid_h * (S_new * D)
    cos_base = pid_b * (S_new * D)  # cos: [B,1,S_new,D]

    s_mask = s_offs < S_out
    out_mask = s_mask[:, None]

    is_cache = (s_offs < S_cur)[:, None]
    is_new = ((s_offs >= S_cur) & (s_offs < S_out))[:, None]

    if APPLY_ROPE:
        # cache region
        cache_ptrs = cache_ptr + cache_base + sd
        cache_vals = tl.load(cache_ptrs, mask=is_cache, other=0.0)

        # new region (RoPE)
        s_new = (s_offs - S_cur)[:, None]  # [BLOCK_S, 1]
        new_sd = s_new * D + d2            # [BLOCK_S, D]
        new_ptrs = new_ptr + new_base + new_sd
        cos_ptrs = cos_ptr + cos_base + new_sd
        sin_ptrs = sin_ptr + cos_base + new_sd
        new_vals = tl.load(new_ptrs, mask=is_new, other=0.0)
        cos_vals = tl.load(cos_ptrs, mask=is_new, other=0.0)
        sin_vals = tl.load(sin_ptrs, mask=is_new, other=0.0)

        # partner index for rotate_half
        d_partner = tl.where(d_offs < HALF, d_offs + HALF, d_offs - HALF)  # [D]
        new_sd_p = s_new * D + d_partner[None, :]
        partner_vals = tl.load(new_ptr + new_base + new_sd_p, mask=is_new, other=0.0)

        sign = tl.where(d_offs < HALF, -1.0, 1.0)  # [D]
        rot = partner_vals * (sin_vals * sign[None, :])
        result = new_vals * cos_vals + rot

        final = tl.where(is_cache, cache_vals, result)
        tl.store(out_ptr + out_base + sd, final, mask=out_mask)
    else:
        cache_ptrs = cache_ptr + cache_base + sd
        cache_vals = tl.load(cache_ptrs, mask=is_cache, other=0.0)

        s_new = (s_offs - S_cur)[:, None]
        new_sd = s_new * D + d2
        new_ptrs = new_ptr + new_base + new_sd
        new_vals = tl.load(new_ptrs, mask=is_new, other=0.0)

        final = tl.where(is_cache, cache_vals, new_vals)
        tl.store(out_ptr + out_base + sd, final, mask=out_mask)


def _launch(out, cache, new, cos, sin, apply_rope):
    B, H, S_out, D = out.shape
    S_cur = cache.shape[2]
    S_new = new.shape[2]
    BLOCK_S = 32
    num_s = triton.cdiv(S_out, BLOCK_S)
    grid = (B, H, num_s)
    _kv_copy_rope_kernel[grid](
        out, cache, new, cos, sin,
        S_cur, S_new, S_out,
        H=H, D=D, HALF=D // 2, BLOCK_S=BLOCK_S,
        APPLY_ROPE=apply_rope,
        num_warps=4,
    )


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    B, H, S_new, D = key_states.shape
    S_cur = key_cache.shape[2]
    S_out = S_cur + S_new

    out_dtype = key_cache.dtype
    updated_key_cache = torch.empty(B, H, S_out, D, dtype=out_dtype, device=key_states.device)
    updated_value_cache = torch.empty_like(updated_key_cache)

    _launch(updated_key_cache, key_cache, key_states, cos, sin, apply_rope=True)
    _launch(updated_value_cache, value_cache, value_states, cos, sin, apply_rope=False)

    return updated_key_cache, updated_value_cache
