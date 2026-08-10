import torch
import torch.nn.functional as F
import math


@torch.compile(dynamic=True)
def _swiglu_backward(grad_intermediate, up, gate_silu, gate, gate_sigmoid):
    gi = grad_intermediate.float()
    gf = gate.float()
    gs = gate_sigmoid.float()
    grad_up = gi * gate_silu.float()
    grad_gate = (gi * up.float()) * gs * (1.0 + gf * (1.0 - gs))
    return grad_gate.bfloat16(), grad_up.bfloat16()


def gen_inputs(axes_and_scalars, device):
    num_tokens = axes_and_scalars['num_tokens']
    hidden_size = 2048
    intermediate_size = 768

    # Xavier-scaled weights
    gate_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    up_weight = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)
    down_weight = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device) / math.sqrt(intermediate_size)

    # Activation-scale input
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size)

    # Run forward pass to produce consistent saved tensors
    with torch.no_grad():
        gate = F.linear(x, gate_weight)
        gate_sigmoid = torch.sigmoid(gate.to(torch.float32)).to(torch.bfloat16)
        gate_silu = gate * gate_sigmoid
        up = F.linear(x, up_weight)
        intermediate = gate_silu * up

    # Small-magnitude upstream gradient
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
    # Tiny reductions are accuracy-bound: BF16 input rounding is visible when
    # only a handful of token contributions are accumulated.
    if x.shape[0] <= 128:
        go = grad_output.float()
        uw = up_weight.float()
        gf = gate.float()
        gs = gate_sigmoid.float()
        gsil = gate_silu.float()
        upf = up.float()
        grad_dw = grad_output.t().mm(intermediate)
        if x.shape[0] <= 16:
            grad_inter = go.mm(down_weight.float())
        else:
            grad_inter = grad_output.mm(down_weight).float()
        grad_up = grad_inter * gsil
        grad_gate = (grad_inter * upf) * gs * (1.0 + gf * (1.0 - gs))
        grad_x = (grad_gate.bfloat16().mm(gate_weight)
                  + grad_up.mm(uw)).bfloat16()
        grad_gw = grad_gate.bfloat16().t().mm(x)
        grad_uw = grad_up.bfloat16().t().mm(x)
        return grad_x, grad_gw, grad_uw, grad_dw.bfloat16()

    # Run the large contractions with native BF16 matrix instructions.  The
    # saved sigmoid path stays in FP32, where its extra precision matters.
    
    # Gradient w.r.t. down_weight
    # down_proj: output = intermediate @ down_weight.T
    # grad_down_weight = grad_output.T @ intermediate
    grad_down_weight = grad_output.t().mm(intermediate)
    
    # Gradient w.r.t. intermediate
    grad_intermediate = grad_output.mm(down_weight)
    
    # Gradient w.r.t. gate_silu and up (element-wise multiplication)
    # intermediate = gate_silu * up
    grad_gate_bf16, grad_up_bf16 = _swiglu_backward(
        grad_intermediate, up, gate_silu, gate, gate_sigmoid
    )
    
    # Gradient w.r.t. up_weight
    # up_proj: up = x @ up_weight.T
    # grad_up_weight = grad_up.T @ x
    grad_up_weight = grad_up_bf16.t().mm(x)
    
    # Gradient w.r.t. gate_weight
    # gate_proj: gate = x @ gate_weight.T
    # grad_gate_weight = grad_gate.T @ x
    grad_gate_weight = grad_gate_bf16.t().mm(x)
    
    # Gradient w.r.t. x (input)
    # Accumulate gradients from both gate and up projections
    grad_x = grad_gate_bf16.mm(gate_weight) + grad_up_bf16.mm(up_weight)
    
    return (
        grad_x.to(torch.bfloat16),
        grad_gate_weight.to(torch.bfloat16),
        grad_up_weight.to(torch.bfloat16),
        grad_down_weight.to(torch.bfloat16),
    )
