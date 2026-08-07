import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh as _libdevice_tanh


@triton.jit
def _tanh_gated_bwd_kernel(
    grad_output_ptr,
    hidden_states_ptr,
    mask_ptr,
    gate_ptr,
    grad_residual_ptr,
    grad_hidden_states_ptr,
    partial_sum_ptr,
    n_rows,
    hidden_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, hidden_size)

    g = tl.load(gate_ptr).to(tl.float32)
    gv = _libdevice_tanh(g)
    ss = 1.0 - gv * gv

    go_ptr = grad_output_ptr + row * hidden_size
    hs_ptr = hidden_states_ptr + row * hidden_size

    go = tl.load(go_ptr + cols)
    hs = tl.load(hs_ptr + cols)
    m = tl.load(mask_ptr + row).to(tl.float32)

    out_ptr = grad_residual_ptr + row * hidden_size
    outh_ptr = grad_hidden_states_ptr + row * hidden_size

    tl.store(out_ptr + cols, go)

    gh = go.to(tl.float32) * gv * m
    tl.store(outh_ptr + cols, gh.to(tl.bfloat16))

    masked_hs = (hs * m.to(tl.bfloat16)).to(tl.float32)
    partial = tl.sum(go.to(tl.float32) * masked_hs) * ss
    tl.store(partial_sum_ptr + row, partial)


@triton.jit
def _reduce_cast_kernel(
    partial_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(vals)
    tl.atomic_add(out_ptr, acc)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    batch_size, seq_len, hidden_size = grad_output.shape[0], grad_output.shape[1], grad_output.shape[2]
    n_rows = batch_size * seq_len

    grad_residual = torch.empty_like(grad_output)
    grad_hidden_states = torch.empty_like(grad_output)
    partial = torch.empty(n_rows, dtype=torch.float32, device=grad_output.device)
    grad_gate = torch.zeros((), dtype=torch.float32, device=grad_output.device)

    grid = (n_rows,)
    _tanh_gated_bwd_kernel[grid](
        grad_output,
        hidden_states,
        mask,
        gate,
        grad_residual,
        grad_hidden_states,
        partial,
        n_rows,
        hidden_size=hidden_size,
        num_warps=8,
    )

    BLOCK = 1024
    rgrid = (triton.cdiv(n_rows, BLOCK),)
    _reduce_cast_kernel[rgrid](
        partial,
        grad_gate,
        n_rows,
        BLOCK=BLOCK,
    )

    return grad_residual, grad_hidden_states, grad_gate.to(torch.bfloat16)
