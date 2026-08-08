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
    grad_output_f32 = grad_output.to(torch.float32)
    x_f32 = x.to(torch.float32)
    down_weight_f32 = down_weight.to(torch.float32)
    intermediate_f32 = intermediate.to(torch.float32)
    gate_weight_f32 = gate_weight.to(torch.float32)
    up_weight_f32 = up_weight.to(torch.float32)
    gate_f32 = gate.to(torch.float32)
    gate_sigmoid_f32 = gate_sigmoid.to(torch.float32)
    gate_silu_f32 = gate_silu.to(torch.float32)
    up_f32 = up.to(torch.float32)

    grad_down_weight = grad_output_f32.t().mm(intermediate_f32)
    grad_intermediate = grad_output_f32.mm(down_weight_f32)
    grad_gate_silu = grad_intermediate * up_f32
    grad_up = grad_intermediate * gate_silu_f32
    grad_gate = grad_gate_silu * gate_sigmoid_f32 * (1.0 + gate_f32 * (1.0 - gate_sigmoid_f32))
    grad_up_weight = grad_up.t().mm(x_f32)
    grad_gate_weight = grad_gate.t().mm(x_f32)
    # grad_x fp32 (matching reference) to confirm all pass
    grad_x = grad_gate.mm(gate_weight_f32) + grad_up.mm(up_weight_f32)

    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight.to(torch.bfloat16),
        grad_up_weight.to(torch.bfloat16),
        grad_down_weight.to(torch.bfloat16),
    )
