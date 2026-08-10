import torch


@torch.compile(fullgraph=True)
def _impl(grad_output, gate, hidden_states, mask):
    gate_value = torch.tanh(gate.float())
    grad_residual = grad_output.clone()
    grad_hidden_states = grad_output * gate_value * mask
    sech_squared = 1.0 - gate_value * gate_value
    grad_gate = torch.sum(
        grad_output.float() * (hidden_states * mask).float()
    ) * sech_squared
    return grad_residual, grad_hidden_states.bfloat16(), grad_gate.bfloat16()


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    return _impl(grad_output, gate, hidden_states, mask)
