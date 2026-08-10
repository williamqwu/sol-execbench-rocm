import torch


@torch.compile(fullgraph=True)
def _pointwise(grad_output, gate, hidden_states, mask):
    gate_value = torch.tanh(gate.float())
    grad_residual = grad_output.clone()
    grad_hidden_states = grad_output * gate_value * mask
    sech_squared = 1.0 - gate_value * gate_value
    products = grad_output.float() * (hidden_states * mask).float()
    return grad_residual, grad_hidden_states.bfloat16(), products, sech_squared


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    grad_residual, grad_hidden_states, products, sech_squared = _pointwise(
        grad_output, gate, hidden_states, mask
    )
    grad_gate = torch.sum(products) * sech_squared
    return grad_residual, grad_hidden_states, grad_gate.bfloat16()
