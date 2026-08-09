import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _wkv_kernel(
    decay_p, key_p, first_p, value_p,
    max_in_p, num_in_p, den_in_p,
    output_p, max_out_p, num_out_p, den_out_p,
    T: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    cblk = tl.program_id(1)
    c = cblk * BLOCK + tl.arange(0, BLOCK)
    mask = c < H
    state_off = b * H + c

    m = tl.load(max_in_p + state_off, mask=mask).to(tl.float32)
    n = tl.load(num_in_p + state_off, mask=mask).to(tl.float32)
    d = tl.load(den_in_p + state_off, mask=mask).to(tl.float32)
    w = -libdevice.exp(tl.load(decay_p + c, mask=mask).to(tl.float32))
    u = tl.load(first_p + c, mask=mask).to(tl.float32)

    for t in range(T):
        off = (b * T + t) * H + c
        k = tl.load(key_p + off, mask=mask).to(tl.float32)
        v = tl.load(value_p + off, mask=mask).to(tl.float32)

        ku = k + u
        mo = tl.maximum(m, ku)
        a = libdevice.exp(m - mo)
        bb = libdevice.exp(ku - mo)
        y = (a * n + bb * v) / (a * d + bb)
        tl.store(output_p + off, y, mask=mask)

        mw = m + w
        ms = tl.maximum(mw, k)
        a = libdevice.exp(mw - ms)
        bb = libdevice.exp(k - ms)
        n = a * n + bb * v
        d = a * d + bb
        m = ms

    tl.store(max_out_p + state_off, m, mask=mask)
    tl.store(num_out_p + state_off, n, mask=mask)
    tl.store(den_out_p + state_off, d, mask=mask)


@torch.no_grad()
def run(
    time_decay: torch.Tensor,
    key: torch.Tensor,
    time_first: torch.Tensor,
    value: torch.Tensor,
    max_state: torch.Tensor,
    num_state: torch.Tensor,
    den_state: torch.Tensor,
):
    B, T, H = key.shape
    output = torch.empty_like(key, dtype=torch.float32)
    max_out = torch.empty_like(max_state, dtype=torch.float32)
    num_out = torch.empty_like(num_state, dtype=torch.float32)
    den_out = torch.empty_like(den_state, dtype=torch.float32)
    block = 64
    _wkv_kernel[(B, triton.cdiv(H, block))](
        time_decay, key, time_first, value,
        max_state, num_state, den_state,
        output, max_out, num_out, den_out,
        T=T, H=H, BLOCK=block,
        num_warps=1,
    )
    return output, max_out, num_out, den_out
