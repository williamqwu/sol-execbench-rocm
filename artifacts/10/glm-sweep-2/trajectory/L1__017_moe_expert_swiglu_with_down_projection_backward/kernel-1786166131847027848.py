import os
os.environ.setdefault('TORCHINDUCTOR_DISABLE_TRITON_MM', '1')
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
def _run_impl(
    grad_output, x, gate_weight, up_weight, down_weight,
    gate, gate_sigmoid, gate_silu, up, intermediate,
):
    grad_down_weight = grad_output.t().mm(intermediate)
    # fp16 throughout for GEMMs and elementwise
    grad_intermediate = grad_output.to(torch.float16).mm(down_weight.to(torch.float16))
    gi = grad_intermediate
    up_h = up.to(torch.float16)
    gsilu_h = gate_silu.to(torch.float16)
    gsig_h = gate_sigmoid.to(torch.float16)
    gate_h = gate.to(torch.float16)
    grad_gate_silu = gi * up_h
    grad_up = gi * gsilu_h
    grad_gate = grad_gate_silu * gsig_h * (1.0 + gate_h * (1.0 - gsig_h))
    grad_up_weight = grad_up.to(torch.bfloat16).t().mm(x)
    grad_gate_weight = grad_gate.to(torch.bfloat16).t().mm(x)
    # Fused grad_x: [grad_gate|grad_up] @ [gate_weight; up_weight]
    grad_x = torch.cat([grad_gate.to(torch.float16), grad_up.to(torch.float16)], dim=1).mm(
        torch.cat([gate_weight.to(torch.float16), up_weight.to(torch.float16)], dim=0))
    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight.to(torch.bfloat16),
        grad_up_weight.to(torch.bfloat16),
        grad_down_weight.to(torch.bfloat16),
    )


_run_compiled = torch.compile(_run_impl, mode='max-autotune-no-cudagraphs')


@torch.no_grad()
def run(
    grad_output, x, gate_weight, up_weight, down_weight,
    gate, gate_sigmoid, gate_silu, up, intermediate,
):
    return _run_compiled(
        grad_output, x, gate_weight, up_weight, down_weight,
        gate, gate_sigmoid, gate_silu, up, intermediate,
    )
