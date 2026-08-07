import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_gated_bwd_kernel(
    grad_output_ptr,
    hidden_states_ptr,
    mask_ptr,
    grad_residual_ptr,
    grad_hidden_states_ptr,
    partial_sum_ptr,
    gate_value: tl.constexpr,
    hidden_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, hidden_size)

    go_ptr = grad_output_ptr + row * hidden_size
    hs_ptr = hidden_states_ptr + row * hidden_size

    go = tl.load(go_ptr + cols).to(tl.float32)
    hs = tl.load(hs_ptr + cols).to(tl.float32)
    m = tl.load(mask_ptr + row).to(tl.float32)

    out_ptr = grad_residual_ptr + row * hidden_size
    outh_ptr = grad_hidden_states_ptr + row * hidden_size

    # grad_residual = grad_output
    tl.store(out_ptr + cols, go.to(tl.bfloat16))

    # grad_hidden_states = grad_output * tanh(gate) * mask
    gh = go * gate_value * m
    tl.store(outh_ptr + cols, gh.to(tl.bfloat16))

    # partial sum for grad_gate: sum(go * hs * m)  (multiply by sech^2 later)
    partial = tl.sum(go * hs * m)
    tl.store(partial_sum_ptr + row, partial)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    n_rows, hidden_size = grad_output.shape[0], grad_output.shape[2]

    gate_float = gate.to(torch.float32)
    gate_value_t = torch.tanh(gate_float)
    sech_squared = 1.0 - gate_value_t * gate_value_t

    grad_residual = torch.empty_like(grad_output)
    grad_hidden_states = torch.empty_like(grad_output)
    partial = torch.empty(n_rows, dtype=torch.float32, device=grad_output.device)

    gate_value = gate_value_t.item()

    grid = (n_rows,)
    _tanh_gated_bwd_kernel[grid](
        grad_output,
        hidden_states,
        mask,
        grad_residual,
        grad_hidden_states,
        partial,
        gate_value,
        hidden_size=hidden_size,
    )

    grad_gate = partial.sum() * sech_squared
    return grad_residual, grad_hidden_states, grad_gate.to(torch.bfloat16)
