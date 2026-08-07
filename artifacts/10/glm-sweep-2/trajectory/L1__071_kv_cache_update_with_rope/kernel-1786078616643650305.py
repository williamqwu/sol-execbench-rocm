import torch
import triton
import triton.language as tl


@triton.jit
def _kv_update_kernel(
    out_key_ptr, out_val_ptr,
    key_cache_ptr, value_cache_ptr,
    key_states_ptr, value_states_ptr,
    cos_ptr, sin_ptr,
    H, S_old, S_new,
    ok_b, ok_h, ok_s,
    ov_b, ov_h, ov_s,
    kc_b, kc_h, kc_s,
    vc_b, vc_h, vc_s,
    ks_b, ks_h, ks_s,
    vs_b, vs_h, vs_s,
    co_b, co_s,
    HALF: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    S_out = S_old + S_new
    ngroups = tl.cdiv(S_out, ROWS)
    r = pid % ngroups
    rest = pid // ngroups
    h = rest % H
    b = rest // H

    d_lo = tl.arange(0, HALF)
    d_hi = tl.arange(HALF, 2 * HALF)

    s_start = r * ROWS
    for i in range(ROWS):
        s_out = s_start + i
        if s_out < S_out:
            ok_base = b * ok_b + h * ok_h + s_out * ok_s
            ov_base = b * ov_b + h * ov_h + s_out * ov_s
            if s_out < S_old:
                kc_base = b * kc_b + h * kc_h + s_out * kc_s
                vc_base = b * vc_b + h * vc_h + s_out * vc_s
                kc_lo = tl.load(key_cache_ptr + kc_base + d_lo)
                kc_hi = tl.load(key_cache_ptr + kc_base + d_hi)
                tl.store(out_key_ptr + ok_base + d_lo, kc_lo)
                tl.store(out_key_ptr + ok_base + d_hi, kc_hi)
                vc_lo = tl.load(value_cache_ptr + vc_base + d_lo)
                vc_hi = tl.load(value_cache_ptr + vc_base + d_hi)
                tl.store(out_val_ptr + ov_base + d_lo, vc_lo)
                tl.store(out_val_ptr + ov_base + d_hi, vc_hi)
            else:
                s_new = s_out - S_old
                ks_base = b * ks_b + h * ks_h + s_new * ks_s
                vs_base = b * vs_b + h * vs_h + s_new * vs_s
                co_base = b * co_b + s_new * co_s
                k1 = tl.load(key_states_ptr + ks_base + d_lo)
                k2 = tl.load(key_states_ptr + ks_base + d_hi)
                c1 = tl.load(cos_ptr + co_base + d_lo)
                c2 = tl.load(cos_ptr + co_base + d_hi)
                s1 = tl.load(sin_ptr + co_base + d_lo)
                s2 = tl.load(sin_ptr + co_base + d_hi)
                r1 = k1 * c1 - k2 * s1
                r2 = k2 * c2 + k1 * s2
                tl.store(out_key_ptr + ok_base + d_lo, r1)
                tl.store(out_key_ptr + ok_base + d_hi, r2)
                v_lo = tl.load(value_states_ptr + vs_base + d_lo)
                v_hi = tl.load(value_states_ptr + vs_base + d_hi)
                tl.store(out_val_ptr + ov_base + d_lo, v_lo)
                tl.store(out_val_ptr + ov_base + d_hi, v_hi)


def _run(key_states, value_states, cos, sin, key_cache, value_cache):
    B, H, S_old, D = key_cache.shape
    S_new = key_states.shape[2]
    S_out = S_old + S_new
    HALF = D // 2

    out_key = torch.empty(B, H, S_out, D, dtype=key_cache.dtype, device=key_cache.device)
    out_val = torch.empty(B, H, S_out, D, dtype=value_cache.dtype, device=value_cache.device)

    def st3(t):
        stb, sth, sts, _ = t.stride()
        return stb, sth, sts

    ok_b, ok_h, ok_s = st3(out_key)
    ov_b, ov_h, ov_s = st3(out_val)
    kc_b, kc_h, kc_s = st3(key_cache)
    vc_b, vc_h, vc_s = st3(value_cache)
    ks_b, ks_h, ks_s = st3(key_states)
    vs_b, vs_h, vs_s = st3(value_states)
    co_b = cos.stride(0)
    co_s = cos.stride(2)

    ROWS = 2
    ngroups = (S_out + ROWS - 1) // ROWS
    grid = (B * H * ngroups,)
    _kv_update_kernel[grid](
        out_key, out_val,
        key_cache, value_cache,
        key_states, value_states,
        cos, sin,
        H, S_old, S_new,
        ok_b, ok_h, ok_s,
        ov_b, ov_h, ov_s,
        kc_b, kc_h, kc_s,
        vc_b, vc_h, vc_s,
        ks_b, ks_h, ks_s,
        vs_b, vs_h, vs_s,
        co_b, co_s,
        HALF=HALF,
        ROWS=ROWS,
        num_warps=1,
    )
    return out_key, out_val


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
