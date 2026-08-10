import torch


@torch.compile(fullgraph=True, dynamic=True)
def _project(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
):
    text_len = encoder_hidden_states.shape[1]
    combined = torch.cat((encoder_hidden_states, hidden_states), dim=1)
    projected = torch.mm(combined.flatten(0, 1), process_weight.t())
    projected = projected.view(
        encoder_hidden_states.shape[0],
        text_len + hidden_states.shape[1],
        process_weight.shape[0],
    )
    return projected[:, :text_len], projected[:, text_len:]


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _project(hidden_states, encoder_hidden_states, process_weight)
