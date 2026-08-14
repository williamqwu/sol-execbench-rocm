import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mod_kernel(H, MEAN, VAR, MOD, OUT, M, S, stride_mod, eps,
                K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < M
    rows = tl.where(mask_m, rm, 0)
    b = rows // S
    mean = tl.load(MEAN + rows, mask=mask_m, other=0.0)
    var = tl.load(VAR + rows, mask=mask_m, other=0.0)
    den = tl.math.sqrt_rn(var + eps)
    base = H + rows[:, None] * K
    mb = MOD + b[:, None] * stride_mod
    ob = OUT + rows[:, None] * K
    for k0 in tl.static_range(0, K, BLOCK_K):
        ok = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(base + ok[None, :], mask=mask_m[:, None], other=0.0)
        s1 = tl.load(mb + ok[None, :], mask=mask_m[:, None], other=0.0)
        s2 = tl.load(mb + K + ok[None, :], mask=mask_m[:, None], other=0.0)
        xn = tl.math.div_rn(x - mean[:, None], den[:, None])
        t = 1.0 + s2
        # force a rounded multiply (no FMA contraction) to match torch elementwise
        u = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", "=v,v,v",
                                      [xn, t], dtype=tl.float32,
                                      is_pure=True, pack=1)
        y = u + s1
        tl.store(ob + ok[None, :], y, mask=mask_m[:, None])


def run(hidden_states, temb, linear_weight, linear_bias,
        proj_out_weight, proj_out_bias, eps):
    B, S, K = hidden_states.shape
    M = B * S
    h = hidden_states.reshape(M, K)

    mean = h.mean(dim=-1)
    var = h.var(dim=-1, unbiased=False)

    temb_silu = temb * torch.sigmoid(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)

    mod_out = torch.empty_like(h)
    BLOCK_M = 4
    BLOCK_K = 1024
    grid = (triton.cdiv(M, BLOCK_M),)
    _mod_kernel[grid](h, mean, var, modulation, mod_out, M, S,
                      modulation.stride(0), eps,
                      K=K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                      num_warps=8, num_stages=1)
    return F.linear(mod_out.view(B, S, K), proj_out_weight, proj_out_bias)
