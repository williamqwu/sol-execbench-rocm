import torch


@torch.compile(fullgraph=True, dynamic=True)
def _transposed_projection(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m = hidden_states.numel() // hidden_states.shape[-1]
    logits_t = torch.mm(weight, hidden_states.view(m, -1).t())
    return logits_t.t().reshape(*hidden_states.shape[:-1], weight.shape[0])


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    b, s, k = hidden_states.shape
    if b * s <= 16:
        return torch.matmul(hidden_states, weight.t())
    return _transposed_projection(hidden_states, weight)
