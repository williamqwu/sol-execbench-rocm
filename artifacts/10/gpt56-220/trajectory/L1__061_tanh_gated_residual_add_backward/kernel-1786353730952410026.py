import torch


@torch.compile(fullgraph=True)
def _pointwise(grad_output, gate, hidden_states, mask):
    gate_value = torch.tanh(gate.float())
    grad_residual = grad_output.clone()
    grad_hidden_states = grad_output * gate_value * mask
    masked_hidden_states = hidden_states * mask
    return grad_residual, grad_hidden_states.bfloat16(), masked_hidden_states, gate_value


@torch.compile(fullgraph=True)
def _row_reduce(grad_output, masked_hidden_states):
    return torch.sum(
        grad_output.float() * masked_hidden_states.float(), dim=-1
    )


@torch.no_grad()
def run(grad_output, gate, hidden_states, mask):
    grad_residual, grad_hidden_states, masked_hidden_states, gate_value = _pointwise(
        grad_output, gate, hidden_states, mask
    )
    sech_squared = 1.0 - gate_value * gate_value
    grad_gate = torch.sum(_row_reduce(grad_output, masked_hidden_states)) * sech_squared
    return grad_residual, grad_hidden_states, grad_gate.bfloat16()
