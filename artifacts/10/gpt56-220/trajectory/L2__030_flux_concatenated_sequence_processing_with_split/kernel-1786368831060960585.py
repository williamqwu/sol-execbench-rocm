import torch


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_t = process_weight.t()
    return (
        torch.matmul(encoder_hidden_states, weight_t),
        torch.matmul(hidden_states, weight_t),
    )
