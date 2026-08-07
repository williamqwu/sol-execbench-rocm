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
        'grad_output': grad_output,
        'x': x,
        'gate_weight': gate_weight,
        'up_weight': up_weight,
        'down_weight': down_weight,
        'gate': gate,
        'gate_sigmoid': gate_sigmoid,
        'gate_silu': gate_silu,
        'up': up,
        'intermediate': intermediate,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate: torch.Tensor,
    gate_sigmoid: torch.Tensor,
    gate_silu: torch.Tensor,
    up: torch.Tensor,
    intermediate: torch.Tensor,
):
    # bf16 GEMMs (rocBLAS accumulates in fp32); fp32 only for elementwise sigmoid-derivative math.

    # grad_down_weight = grad_output.T @ intermediate   [hidden, intermediate]
    grad_down_weight = grad_output.t().mm(intermediate)

    # grad_intermediate = grad_output @ down_weight      [N, intermediate]
    grad_intermediate = grad_output.mm(down_weight)

    # elementwise in fp32
    gi = grad_intermediate.to(torch.float32)
    up_f32 = up.to(torch.float32)
    gate_silu_f32 = gate_silu.to(torch.float32)
    gate_sigmoid_f32 = gate_sigmoid.to(torch.float32)
    gate_f32 = gate.to(torch.float32)

    grad_gate_silu = gi * up_f32
    grad_up = gi * gate_silu_f32
    grad_gate = grad_gate_silu * gate_sigmoid_f32 * (1.0 + gate_f32 * (1.0 - gate_sigmoid_f32))

    grad_gate_bf = grad_gate.to(torch.bfloat16)
    grad_up_bf = grad_up.to(torch.bfloat16)

    # grad_up_weight = grad_up.T @ x          [intermediate, hidden]
    grad_up_weight = grad_up_bf.t().mm(x)

    # grad_gate_weight = grad_gate.T @ x      [intermediate, hidden]
    grad_gate_weight = grad_gate_bf.t().mm(x)

    # grad_x = grad_gate @ gate_weight + grad_up @ up_weight   [N, hidden]
    grad_x = grad_gate_bf.mm(gate_weight) + grad_up_bf.mm(up_weight)

    return (grad_x, grad_gate_weight, grad_up_weight, grad_down_weight)
