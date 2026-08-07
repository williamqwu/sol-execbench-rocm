import torch
import triton
import triton.language as tl


@triton.jit
def _key_update_kernel(
    out_key_ptr,
    key_cache_ptr,
    key_states_ptr,
    cos_ptr, sin_ptr,
    H, S_old, S_new,
    ok_b, ok_h, ok_s,
    kc_b, kc_h, kc_s,
    ks_b, ks_h, ks_s,
    co_b, co_s,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    S_out = S_old + S_new
    ngroups = tl.cdiv(S_out, ROWS)
    r = pid % ngroups
    rest = pid // ngroups
    h = rest % H
    b = rest // H

    d = tl.arange(0, BLOCK_D)
    half = BLOCK_D // 2
    partner = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, -1.0, 1.0)

    s_start = r * ROWS
    for i in range(ROWS):
        s_out = s_start + i
        if s_out < S_out:
            ok_off = b * ok_b + h * ok_h + s_out * ok_s + d
            if s_out < S_old:
                kc = tl.load(key_cache_ptr + b * kc_b + h * kc_h + s_out * kc_s + d)
                tl.store(out_key_ptr + ok_off, kc)
            else:
                s_new = s_out - S_old
                ks_base = b * ks_b + h * ks_h + s_new * ks_s
                co_base = b * co_b + s_new * co_s
                k = tl.load(key_states_ptr + ks_base + d)
                c = tl.load(cos_ptr + co_base + d)
                s = tl.load(sin_ptr + co_base + d)
                k_partner = tl.load(key_states_ptr + ks_base + partner)
                out_k = k * c + sign * k_partner * s
                tl.store(out_key_ptr + ok_off, out_k)


def _run(key_states, value_states, cos, sin, key_cache, value_cache):
    B, H, S_old, D = key_cache.shape
    S_new = key_states.shape[2]
    S_out = S_old + S_new

    out_key = torch.empty(B, H, S_out, D, dtype=key_cache.dtype, device=key_cache.device)

    def st3(t):
        stb, sth, sts, _ = t.stride()
        return stb, sth, sts

    ok_b, ok_h, ok_s = st3(out_key)
    kc_b, kc_h, kc_s = st3(key_cache)
    ks_b, ks_h, ks_s = st3(key_states)
    co_b = cos.stride(0)
    co_s = cos.stride(2)

    ROWS = 2
    ngroups = (S_out + ROWS - 1) // ROWS
    grid = (B * H * ngroups,)
    _key_update_kernel[grid](
        out_key,
        key_cache,
        key_states,
        cos, sin,
        H, S_old, S_new,
        ok_b, ok_h, ok_s,
        kc_b, kc_h, kc_s,
        ks_b, ks_h, ks_s,
        co_b, co_s,
        BLOCK_D=D,
        ROWS=ROWS,
        num_warps=1,
    )

    # Value path is a pure copy: use torch.cat (highly tuned memcpy).
    updated_value_cache = torch.cat([value_cache, value_states], dim=2)
    return out_key, updated_value_cache


@torch.no_grad()
def run(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    return _run(key_states, value_states, cos, sin, key_cache, value_cache)
