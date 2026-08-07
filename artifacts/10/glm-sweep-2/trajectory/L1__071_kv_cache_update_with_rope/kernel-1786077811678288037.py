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
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    S_out = S_old + S_new
    rows_total = tl.cdiv(S_out, ROWS)
    r = pid % rows_total
    rest = pid // rows_total
    h = rest % H
    b = rest // H

    d = tl.arange(0, BLOCK_D)
    half = BLOCK_D // 2
    partner = tl.where(d < half, d + half, d - half)
    sign = tl.where(d < half, -1.0, 1.0)

    s_base = r * ROWS
    s_offs = s_base + tl.arange(0, ROWS)
    mask = s_offs < S_out

    ok_row = b * ok_b + h * ok_h + s_offs * ok_s  # [ROWS]
    ov_row = b * ov_b + h * ov_h + s_offs * ov_s

    is_cache = s_offs < S_old

    # Cache path loads
    kc_row = b * kc_b + h * kc_h + s_offs * kc_s
    vc_row = b * vc_b + h * vc_h + s_offs * vc_s

    kc = tl.load(key_cache_ptr + kc_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    vc = tl.load(value_cache_ptr + vc_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)

    # New path loads
    s_new_offs = s_offs - S_old
    ks_row = b * ks_b + h * ks_h + s_new_offs * ks_s
    vs_row = b * vs_b + h * vs_h + s_new_offs * vs_s
    co_row = b * co_b + s_new_offs * co_s

    k_new = tl.load(key_states_ptr + ks_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    v_new = tl.load(value_states_ptr + vs_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    c = tl.load(cos_ptr + co_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    s = tl.load(sin_ptr + co_row[:, None] + d[None, :], mask=mask[:, None], other=0.0)
    k_partner = tl.load(key_states_ptr + ks_row[:, None] + partner[None, :], mask=mask[:, None], other=0.0)

    out_k_new = k_new * c + sign[None, :] * k_partner * s

    out_k = tl.where(is_cache[:, None], kc, out_k_new)
    out_v = tl.where(is_cache[:, None], vc, v_new)

    tl.store(out_key_ptr + ok_row[:, None] + d[None, :], out_k, mask=mask[:, None])
    tl.store(out_val_ptr + ov_row[:, None] + d[None, :], out_v, mask=mask[:, None])


def _run_triton(key_states, value_states, cos, sin, key_cache, value_cache):
    B, H, S_old, D = key_cache.shape
    S_new = key_states.shape[2]
    S_out = S_old + S_new

    out_key = torch.empty(B, H, S_out, D, dtype=key_cache.dtype, device=key_cache.device)
    out_val = torch.empty(B, H, S_out, D, dtype=value_cache.dtype, device=value_cache.device)

    def st(t):
        stb, sth, sts, _ = t.stride()
        return stb, sth, sts

    ok_b, ok_h, ok_s = st(out_key)
    ov_b, ov_h, ov_s = st(out_val)
    kc_b, kc_h, kc_s = st(key_cache)
    vc_b, vc_h, vc_s = st(value_cache)
    ks_b, ks_h, ks_s = st(key_states)
    vs_b, vs_h, vs_s = st(value_states)
    co_b = cos.stride(0)
    co_s = cos.stride(2)

    ROWS = 4
    n_row_groups = (S_out + ROWS - 1) // ROWS
    grid = (B * H * n_row_groups,)
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
        BLOCK_D=D,
        ROWS=ROWS,
        num_warps=2,
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
