import torch


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    return torch.index_add(
        final_hidden_states, 0, token_indices, expert_outputs
    )
