import torch

@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
):
    # gate is scalar bf16. Compute tanh(gate) and sech^2 in fp32.
    gate_float = gate.to(torch.float32)
    gate_value = torch.tanh(gate_float)
    sech_squared = 1.0 - gate_value * gate_value

    # grad_residual = grad_output (identity) -> copy out
    grad_residual = grad_output.clone()

    # grad_hidden_states = grad_output * tanh(gate) * mask  (bf16 elementwise)
    # Use bf16 scalar to keep dtype; compute tanh(gate) as bf16 scalar multiply.
    gate_bf16 = gate_value.to(torch.bfloat16)
    grad_hidden_states = grad_output * gate_bf16 * mask

    # grad_gate = sum(grad_output * hidden_states * mask * sech^2)
    # Fuse: one elementwise mul in fp32 then sum.
    go_f32 = grad_output.to(torch.float32)
    mh_f32 = (hidden_states * mask).to(torch.float32)
    grad_gate = (go_f32 * mh_f32).sum() * sech_squared

    return grad_residual.to(torch.bfloat16), grad_hidden_states.to(torch.bfloat16), grad_gate.to(torch.bfloat16)
