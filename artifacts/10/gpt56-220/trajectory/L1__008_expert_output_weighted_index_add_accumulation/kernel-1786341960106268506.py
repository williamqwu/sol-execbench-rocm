import torch


def _impl(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    output.index_add_(0, token_indices, expert_outputs)
    return output


_compiled = torch.compile(_impl, fullgraph=True, dynamic=True)


@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    return _compiled(final_hidden_states, expert_outputs, token_indices)
