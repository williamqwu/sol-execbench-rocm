import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x, w, y, n_rows: tl.constexpr, BLOCK_M: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]
    mask = rows[:, None] < n_rows
    xf = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(xf * xf, axis=1) * (1.0 / 128.0)
    scale = tl.rsqrt(mean_sq + 1.0e-6)
    wf = tl.load(w + cols).to(tl.float32)
    out = xf * scale[:, None] * wf[None, :]
    tl.store(y + offsets, out, mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    n_rows, hidden_size = hidden_states.shape
    assert hidden_size == 128
    output = torch.empty_like(hidden_states)
    block_m = 4
    _rmsnorm[(triton.cdiv(n_rows, block_m),)](
        hidden_states, weight, output, n_rows=n_rows, BLOCK_M=block_m,
        num_warps=4,
    )
    return output
