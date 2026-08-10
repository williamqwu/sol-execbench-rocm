import torch
import triton
import triton.language as tl


@triton.jit
def _pointwise_kernel(go, hs, mask, gate_value, residual, grad_hs, masked_hs,
                      n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    g = tl.load(go + offsets, mask=valid).to(tl.float32)
    h = tl.load(hs + offsets, mask=valid).to(tl.float32)
    m = tl.load(mask + offsets // 4096, mask=valid).to(tl.float32)
    gv = tl.load(gate_value).to(tl.float32)
    tl.store(residual + offsets, g, mask=valid)
    tl.store(grad_hs + offsets, g * gv * m, mask=valid)
    tl.store(masked_hs + offsets, h * m, mask=valid)


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    gate_value = torch.tanh(gate.float())
    residual = torch.empty_like(grad_output)
    grad_hs = torch.empty_like(grad_output)
    masked_hs = torch.empty_like(hidden_states)
    n = grad_output.numel()
    _pointwise_kernel[(triton.cdiv(n, 65536),)](
        grad_output, hidden_states, mask, gate_value,
        residual, grad_hs, masked_hs, n, BLOCK=65536, num_warps=8
    )
    sech_squared = 1.0 - gate_value * gate_value
    grad_gate = torch.sum(grad_output.float() * masked_hs.float()) * sech_squared
    return residual, grad_hs, grad_gate.bfloat16()
