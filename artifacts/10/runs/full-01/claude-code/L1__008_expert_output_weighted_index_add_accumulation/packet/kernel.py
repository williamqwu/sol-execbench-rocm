import torch


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output
