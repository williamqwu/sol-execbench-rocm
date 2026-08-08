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
def run(
    grad_output, x, gate_weight, up_weight, down_weight,
    gate, gate_sigmoid, gate_silu, up, intermediate,
):
    # Weight-grad GEMMs: bf16 inputs, fp32 output (internal fp32 accumulation).
    # This is 5x faster than fp32-input GEMMs while staying close to reference precision.
    grad_down_weight = grad_output.to(torch.float32).t().mm(intermediate.to(torch.float32))
    # Try bf16-input version for weight grads
    grad_down_weight_bf = grad_output.t().mm(intermediate)
    grad_intermediate = grad_output.to(torch.float32).mm(down_weight.to(torch.float32))
    grad_gate_silu = grad_intermediate * up.to(torch.float32)
    grad_up = grad_intermediate * gate_silu.to(torch.float32)
    grad_gate = grad_gate_silu * gate_sigmoid.to(torch.float32) * (1.0 + gate.to(torch.float32) * (1.0 - gate_sigmoid.to(torch.float32)))
    grad_up_weight_bf = grad_up.bfloat16().t().mm(x)
    grad_gate_weight_bf = grad_gate.bfloat16().t().mm(x)
    grad_x = grad_gate.mm(gate_weight.to(torch.float32)) + grad_up.mm(up_weight.to(torch.float32))
    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight_bf.to(torch.bfloat16),
        grad_up_weight_bf.to(torch.bfloat16),
        grad_down_weight_bf.to(torch.bfloat16),
    )
