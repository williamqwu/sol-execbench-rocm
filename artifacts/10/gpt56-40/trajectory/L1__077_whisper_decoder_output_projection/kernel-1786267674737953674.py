import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = hidden_states.numel() // hidden_states.shape[-1]
    n = weight.shape[0]
    out = torch.empty((*hidden_states.shape[:-1], n), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    torch.mm(hidden_states.view(m, -1), weight.t(), out=out.view(m, n))
    return out
