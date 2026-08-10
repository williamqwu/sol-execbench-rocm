import torch


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, text_len, hidden_dim = encoder_hidden_states.shape
    image_len = hidden_states.shape[1]
    combined = torch.cat((encoder_hidden_states, hidden_states), dim=1)
    projected = torch.mm(combined.view(-1, hidden_dim), process_weight.t())
    projected = projected.view(batch, text_len + image_len, hidden_dim)
    return projected[:, :text_len], projected[:, text_len:]
