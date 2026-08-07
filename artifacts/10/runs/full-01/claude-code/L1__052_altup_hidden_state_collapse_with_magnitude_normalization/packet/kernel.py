import torch
import triton
import triton.language as tl


@triton.jit
def _collapse_kernel(
    X0, P1, P2, OUT,
    M, H, eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < H)
    offs = rm[:, None] * H + rn[None, :]

    x0 = tl.load(X0 + offs, mask=mask, other=0.0).to(tl.float32)
    p1 = tl.load(P1 + offs, mask=mask, other=0.0).to(tl.float32)
    p2 = tl.load(P2 + offs, mask=mask, other=0.0).to(tl.float32)

    inv_h = 1.0 / H
    t = tl.sqrt(tl.sum(x0 * x0, axis=1) * inv_h)
    m1 = tl.sqrt(tl.maximum(tl.sum(p1 * p1, axis=1) * inv_h, eps))
    m2 = tl.sqrt(tl.maximum(tl.sum(p2 * p2, axis=1) * inv_h, eps))

    s1 = (t / m1)[:, None]
    s2 = (t / m2)[:, None]

    out = (x0 + p1 * s1 + p2 * s2) * (1.0 / 3.0)
    tl.store(OUT + offs, out.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, unembed_proj_1, unembed_proj_2, epsilon):
    H = hidden_states.shape[-1]
    out_shape = hidden_states.shape[1:]
    x = hidden_states.reshape(3, -1, H)
    M = x.shape[1]

    p1 = torch.matmul(x[1], unembed_proj_1.t())
    p2 = torch.matmul(x[2], unembed_proj_2.t())

    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)

    BLOCK_N = triton.next_power_of_2(H)
    BLOCK_M = 1
    grid = (triton.cdiv(M, BLOCK_M),)
    _collapse_kernel[grid](
        x[0], p1, p2, out, M, H, float(epsilon),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=1,
    )
    return out.view(out_shape)
