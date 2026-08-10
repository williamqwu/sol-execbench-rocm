import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mul_rn(x, y):
    # An ISA boundary prevents LLVM from contracting this multiply with the
    # following add.  The eager reference rounds at that same boundary.
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2", "=v,v,v", [x, y],
        dtype=tl.float32, is_pure=True, pack=1,
    )


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
        y = (_mul_rn(a, n) + _mul_rn(bb, v)) / (_mul_rn(a, d) + bb)
        tl.store(output_p + off, y, mask=mask)

        mw = m + w
        ms = tl.maximum(mw, k)
        a = libdevice.exp(mw - ms)
        bb = libdevice.exp(k - ms)
        n = _mul_rn(a, n) + _mul_rn(bb, v)
        d = _mul_rn(a, d) + bb
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
    if B <= 2:
        m = max_state.clone().float()
        n = num_state.clone().float()
        d = den_state.clone().float()
        w = -torch.exp(time_decay.float())
        output = torch.zeros_like(key, dtype=torch.float32)
        for t in range(T):
            k = key[:, t].float()
            v = value[:, t].float()
            mo = torch.maximum(m, k + time_first)
            a = torch.exp(m - mo)
            bb = torch.exp(k + time_first - mo)
            output[:, t] = (a * n + bb * v) / (a * d + bb)
            ms = torch.maximum(m + w, k)
            a = torch.exp(m + w - ms)
            bb = torch.exp(k - ms)
            n = a * n + bb * v
            d = a * d + bb
            m = ms
        return output, m, n, d

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
