import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps: float,
):
    # Step 1: Residual connection
    hidden_states = residual + hidden_states

    # Step 2: RMSNorm
    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    variance = hidden_states_f32.pow(2).mean(dim=-1, keepdim=True)
    hidden_states_f32 = hidden_states_f32 * torch.rsqrt(variance + eps)
    hidden_states = (norm_weight * hidden_states_f32).to(input_dtype)

    # Step 3: SwiGLU MLP
    gate_output = F.linear(hidden_states, gate_proj_weight)
    up_output = F.linear(hidden_states, up_proj_weight)
    intermediate = F.silu(gate_output) * up_output
    output = F.linear(intermediate, down_proj_weight)

    return output

_run = torch.compile(run, mode="max-autotune-no-cudagraphs", dynamic=False)

@torch.no_grad()
def run(*args, **kwargs):
    return _run(*args, **kwargs)
