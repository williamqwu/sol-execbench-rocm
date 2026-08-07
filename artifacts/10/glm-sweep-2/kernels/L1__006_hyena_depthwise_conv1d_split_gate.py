import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(
    u_ptr, w_ptr, b_ptr,
    v_gated_ptr, x0_ptr, x1_ptr,
    SQ,
    BLOCK: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = tl.program_id(2)
    c0 = pid_c
    c1 = pid_c + 256
    c2 = pid_c + 512
    j = tl.arange(0, BLOCK)
    s = pid_s * BLOCK + j
    mask = s < SQ
    u_base_b = pid_b * 768 * SQ

    base0 = u_base_b + c0 * SQ
    a = tl.load(u_ptr + base0 + s - 2, mask=(mask & (s - 2 >= 0)), other=0.0)
    b_ = tl.load(u_ptr + base0 + s - 1, mask=(mask & (s - 1 >= 0)), other=0.0)
    c = tl.load(u_ptr + base0 + s, mask=mask, other=0.0)
    w0 = tl.load(w_ptr + c0 * 3 + 0)
    w1 = tl.load(w_ptr + c0 * 3 + 1)
    w2 = tl.load(w_ptr + c0 * 3 + 2)
    bb = tl.load(b_ptr + c0)
    x0 = w0 * a + w1 * b_ + w2 * c + bb

    base1 = u_base_b + c1 * SQ
    a = tl.load(u_ptr + base1 + s - 2, mask=(mask & (s - 2 >= 0)), other=0.0)
    b_ = tl.load(u_ptr + base1 + s - 1, mask=(mask & (s - 1 >= 0)), other=0.0)
    c = tl.load(u_ptr + base1 + s, mask=mask, other=0.0)
    w0 = tl.load(w_ptr + c1 * 3 + 0)
    w1 = tl.load(w_ptr + c1 * 3 + 1)
    w2 = tl.load(w_ptr + c1 * 3 + 2)
    bb = tl.load(b_ptr + c1)
    x1 = w0 * a + w1 * b_ + w2 * c + bb

    base2 = u_base_b + c2 * SQ
    a = tl.load(u_ptr + base2 + s - 2, mask=(mask & (s - 2 >= 0)), other=0.0)
    b_ = tl.load(u_ptr + base2 + s - 1, mask=(mask & (s - 1 >= 0)), other=0.0)
    c = tl.load(u_ptr + base2 + s, mask=mask, other=0.0)
    w0 = tl.load(w_ptr + c2 * 3 + 0)
    w1 = tl.load(w_ptr + c2 * 3 + 1)
    w2 = tl.load(w_ptr + c2 * 3 + 2)
    bb = tl.load(b_ptr + c2)
    v = w0 * a + w1 * b_ + w2 * c + bb

    v_gated = v * x0
    out_base = pid_b * 256 * SQ + c0 * SQ
    tl.store(v_gated_ptr + out_base + s, v_gated, mask=mask)
    tl.store(x0_ptr + out_base + s, x0, mask=mask)
    tl.store(x1_ptr + out_base + s, x1, mask=mask)


@torch.no_grad()
def run(u: torch.Tensor, short_filter_weight: torch.Tensor, short_filter_bias: torch.Tensor):
    batch_size, inner_width, seq_len = u.shape
    BLOCK = 128
    v_gated = torch.empty(batch_size, 256, seq_len, device=u.device, dtype=u.dtype)
    x0 = torch.empty(batch_size, 256, seq_len, device=u.device, dtype=u.dtype)
    x1 = torch.empty(batch_size, 256, seq_len, device=u.device, dtype=u.dtype)
    grid = (256, triton.cdiv(seq_len, BLOCK), batch_size)
    _fused_kernel[grid](u, short_filter_weight, short_filter_bias,
                        v_gated, x0, x1, seq_len, BLOCK=BLOCK)
    return v_gated, x0, x1
