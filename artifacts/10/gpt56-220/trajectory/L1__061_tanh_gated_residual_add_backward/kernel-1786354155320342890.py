import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_rows(go, gate, hs, mask, residual, grad_hs, row_sums,
                n_rows: tl.constexpr, H: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, H)
    offsets = row * H + cols
    g = tl.load(go + offsets).to(tl.float32)
    h = tl.load(hs + offsets).to(tl.float32)
    m = tl.load(mask + row).to(tl.float32)
    gv = libdevice.fast_tanhf(tl.load(gate).to(tl.float32))

    # Preserve the reference's materialized BF16 hidden_states * mask.
    masked = (h * m).to(tl.bfloat16)
    tl.store(residual + offsets, g)
    tl.store(grad_hs + offsets, g * gv * m)
    tl.store(row_sums + row, tl.sum(g * masked.to(tl.float32), axis=0))


@torch.compile(fullgraph=True)
def _finish_reduce(row_sums, gate):
    gate_value = torch.tanh(gate.float())
    return (torch.sum(row_sums) * (1.0 - gate_value * gate_value)).bfloat16()


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    residual = torch.empty_like(grad_output)
    grad_hs = torch.empty_like(grad_output)
    n_rows = grad_output.numel() // 4096
    row_sums = torch.empty((n_rows,), device=grad_output.device, dtype=torch.float32)
    _fused_rows[(n_rows,)](
        grad_output, gate, hidden_states, mask, residual, grad_hs,
        row_sums, n_rows=n_rows, H=4096, num_warps=16, waves_per_eu=2
    )
    return residual, grad_hs, _finish_reduce(row_sums, gate)
