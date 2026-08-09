import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    b, s, k = hidden_states.shape
    if b * s <= 16:
        return torch.matmul(hidden_states, weight.t())
    # Write the transposed GEMM through a transpose view of contiguous logits.
    logits = torch.empty((b, s, weight.shape[0]), device=hidden_states.device,
                         dtype=hidden_states.dtype)
    torch.mm(weight, hidden_states.view(b * s, k).t(),
             out=logits.view(b * s, weight.shape[0]).t())
    return logits
