import torch
import torch.nn.functional as F
import math


def gen_inputs(axes_and_scalars, device):
    num_tokens = axes_and_scalars['num_tokens']
    hidden_size = 2048
    intermediate_size = 768

    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) / math.sqrt(intermediate_size)
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)

    with torch.no_grad():
        gate = F.linear(x, gate_weight)
        gate_sigmoid = torch.sigmoid(gate.to(torch.float32)).to(torch.bfloat16)
        gate_silu = gate * gate_sigmoid
        up = F.linear(x, up_weight)
        intermediate = gate_silu * up

    grad_output = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)

    return {
        'grad_output': grad_output, 'x': x, 'gate_weight': gate_weight, 'up_weight': up_weight,
        'down_weight': down_weight, 'gate': gate, 'gate_sigmoid': gate_sigmoid,
        'gate_silu': gate_silu, 'up': up, 'intermediate': intermediate,
    }


@torch.no_grad()
def _elementwise(grad_intermediate, up, gate_silu, gate_sigmoid, gate):
    gi = grad_intermediate
    up_f = up.to(torch.float32)
    gsilu_f = gate_silu.to(torch.float32)
    gsig_f = gate_sigmoid.to(torch.float32)
    gate_f = gate.to(torch.float32)
    grad_gate_silu = gi * up_f
    grad_up = gi * gsilu_f
    grad_gate = grad_gate_silu * gsig_f * (1.0 + gate_f * (1.0 - gsig_f))
    return grad_gate, grad_up


_elem_compiled = torch.compile(_elementwise, mode='max-autotune')


@torch.no_grad()
def run(
    grad_output, x, gate_weight, up_weight, down_weight,
    gate, gate_sigmoid, gate_silu, up, intermediate,
):
    grad_down_weight = grad_output.t().mm(intermediate)
    grad_intermediate = grad_output.to(torch.float32).mm(down_weight.to(torch.float32))
    grad_gate, grad_up = _elem_compiled(grad_intermediate, up, gate_silu, gate_sigmoid, gate)
    grad_up_weight = grad_up.to(torch.bfloat16).t().mm(x)
    grad_gate_weight = grad_gate.to(torch.bfloat16).t().mm(x)
    grad_x = grad_gate.mm(gate_weight.to(torch.float32)) + grad_up.mm(up_weight.to(torch.float32))
    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight.to(torch.bfloat16),
        grad_up_weight.to(torch.bfloat16),
        grad_down_weight.to(torch.bfloat16),
    )
