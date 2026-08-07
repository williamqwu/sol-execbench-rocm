import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    unembed_proj_1: torch.Tensor,
    unembed_proj_2: torch.Tensor,
    epsilon: float,
):
    dev = hidden_states.device
    eps_t = torch.tensor(epsilon, dtype=torch.float32, device=dev)

    first = hidden_states[0]                      # bf16 [B,S,H]
    first_f = first.to(torch.float32)
    target_mag = torch.sqrt(torch.mean(first_f * first_f, dim=-1, keepdim=True))  # [B,S,1]

    # bf16 MFMA GEMM (fp32 accumulation), upcast output for normalization
    p1 = torch.matmul(hidden_states[1], unembed_proj_1.t()).to(torch.float32)
    p2 = torch.matmul(hidden_states[2], unembed_proj_2.t()).to(torch.float32)

    mag1 = torch.sqrt(torch.maximum(torch.mean(p1 * p1, dim=-1, keepdim=True), eps_t))
    mag2 = torch.sqrt(torch.maximum(torch.mean(p2 * p2, dim=-1, keepdim=True), eps_t))

    n1 = p1 * (target_mag / mag1)
    n2 = p2 * (target_mag / mag2)

    out = (first_f + n1 + n2) * (1.0 / 3.0)
    return out.to(torch.bfloat16)
