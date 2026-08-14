import torch
import triton
import triton.language as tl


@triton.jit
def _backward_partials(
    grad_output, gate, hidden_states, mask, grad_hidden, partials,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    go = tl.load(grad_output + offsets)
    hs = tl.load(hidden_states + offsets)
    m = tl.load(mask + (block * BLOCK) // 4096)
    g = tl.load(gate).to(tl.float32)
    t = 2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0
    tl.store(grad_hidden + offsets, go.to(tl.float32) * t * m.to(tl.float32))
    masked_hs = (hs.to(tl.float32) * m.to(tl.float32)).to(tl.bfloat16)
    products = go.to(tl.float32) * masked_hs.to(tl.float32)
    tl.store(partials + block, tl.sum(products, axis=0))


@triton.jit
def _finish_reduction(partials, gate, grad_gate, n_partials: tl.constexpr, REDUCE: tl.constexpr):
    offsets = tl.arange(0, REDUCE)
    values = tl.load(partials + offsets, mask=offsets < n_partials, other=0.0)
    g = tl.load(gate).to(tl.float32)
    t = 2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0
    scale = 1.0 - t * t
    tl.store(grad_gate, tl.sum(values, axis=0) * scale)


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    n = grad_output.numel()
    block = 4096 if n < 8_000_000 else 2048
    n_partials = triton.cdiv(n, block)
    grad_hidden = torch.empty_like(grad_output)
    partials = torch.empty((n_partials,), device=grad_output.device, dtype=torch.float32)
    grad_gate = torch.empty((), device=grad_output.device, dtype=torch.bfloat16)
    _backward_partials[(n_partials,)](
        grad_output, gate, hidden_states, mask, grad_hidden, partials,
        BLOCK=block, num_warps=8,
    )
    reduce = triton.next_power_of_2(n_partials)
    _finish_reduction[(1,)](
        partials, gate, grad_gate,
        n_partials=n_partials, REDUCE=reduce, num_warps=8,
    )
    return grad_output, grad_hidden, grad_gate
