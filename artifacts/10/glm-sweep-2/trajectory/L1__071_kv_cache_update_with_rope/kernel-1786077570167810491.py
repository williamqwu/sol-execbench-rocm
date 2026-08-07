import torch
import triton
import triton.language as tl


@triton.jit
def _kv_update_kernel(
    out_key_ptr, out_val_ptr,
    key_cache_ptr, value_cache_ptr,
    key_states_ptr, value_states_ptr,
    cos_ptr, sin_ptr,
    B, H, S_old, S_new,
    # strides for [B, H, S, D] tensors
    ok_b, ok_h, ok_s,
    ov_b, ov_h, ov_s,
    kc_b, kc_h, kc_s,
    vc_b, vc_h, vc_s,
    ks_b, ks_h, ks_s,
    vs_b, vs_h, vs_s,
    co_b, co_s,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    S_out = S_old + S_new
    s_out = pid % S_out
    rest = pid // S_out
    h = rest % H
    b = rest // H

    d = tl.arange(0, BLOCK_D)
    half = BLOCK_D // 2

    ok_off = b * ok_b + h * ok_h + s_out * ok_s + d
    ov_off = b * ov_b + h * ov_h + s_out * ov_s + d

    if s_out < S_old:
        kc = tl.load(key_cache_ptr + b * kc_b + h * kc_h + s_out * kc_s + d)
        vc = tl.load(value_cache_ptr + b * vc_b + h * vc_h + s_out * vc_s + d)
        tl.store(out_key_ptr + ok_off, kc)
        tl.store(out_val_ptr + ov_off, vc)
    else:
        s_new = s_out - S_old
        ks_base = b * ks_b + h * ks_h + s_new * ks_s
        vs_off2 = b * vs_b + h * vs_h + s_new * vs_s + d
        co_base = b * co_b + s_new * co_s

        k = tl.load(key_states_ptr + ks_base + d)
        v = tl.load(value_states_ptr + vs_off2)
        c = tl.load(cos_ptr + co_base + d)
        s = tl.load(sin_ptr + co_base + d)

        # rotate_half(k) = cat(-k2, k1); partner gather
        partner = tl.where(d < half, d + half, d - half)
        k_partner = tl.load(key_states_ptr + ks_base + partner)
        sign = tl.where(d < half, -1.0, 1.0)
        out_k = k * c + sign * k_partner * s

        tl.store(out_key_ptr + ok_off, out_k)
        tl.store(out_val_ptr + ov_off, v)


def _run_triton(key_states, value_states, cos, sin, key_cache, value_cache):
    B, H, S_old, D = key_cache.shape
    S_new = key_states.shape[2]
    S_out = S_old + S_new

    out_key = torch.empty(B, H, S_out, D, dtype=key_cache.dtype, device=key_cache.device)
    out_val = torch.empty(B, H, S_out, D, dtype=value_cache.dtype, device=value_cache.device)

    def st(t):
        # strides for contiguous [B,H,S,D]
        stb, sth, sts, _ = t.stride()
        return stb, sth, sts

    ok_b, ok_h, ok_s = st(out_key)
    ov_b, ov_h, ov_s = st(out_val)
    kc_b, kc_h, kc_s = st(key_cache)
    vc_b, vc_h, vc_s = st(value_cache)
    ks_b, ks_h, ks_s = st(key_states)
    vs_b, vs_h, vs_s = st(value_states)
    # cos/sin: [B,1,S_new,D] contiguous -> strides (S_new*D, D, D) but use (co_b, co_s)
    co_b = cos.stride(0)
    co_s = cos.stride(2)

    grid = (B * H * S_out,)
    _kv_update_kernel[grid](
        out_key, out_val,
        key_cache, value_cache,
        key_states, value_states,
        cos, sin,
        B, H, S_old, S_new,
        ok_b, ok_h, ok_s,
        ov_b, ov_h, ov_s,
        kc_b, kc_h, kc_s,
        vc_b, vc_h, vc_s,
        ks_b, ks_h, ks_s,
        vs_b, vs_h, vs_s,
        co_b, co_s,
        BLOCK_D=D,
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
    return _run_triton(key_states, value_states, cos, sin, key_cache, value_cache)
