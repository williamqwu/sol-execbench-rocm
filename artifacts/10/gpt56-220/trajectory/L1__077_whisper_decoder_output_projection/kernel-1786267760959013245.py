import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    b, s, k = hidden_states.shape
    if b * s == 1:
        return torch.matmul(hidden_states, weight.t())
    # Compute the transposed GEMM directly.  The final transpose/reshape are views.
    logits_t = torch.mm(weight, hidden_states.view(b * s, k).t())
    return logits_t.t().reshape(b, s, weight.shape[0])
