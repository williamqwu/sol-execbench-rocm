import torch


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    index = token_indices[:, None].expand_as(expert_outputs)
    output.scatter_add_(0, index, expert_outputs)
    return output
