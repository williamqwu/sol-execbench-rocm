import torch


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    # Strategy: sort indices, segment-reduce expert outputs (non-atomic), add to buffer.
    # For small workloads the sort overhead dominates; fall back to atomic index_add_.
    nsel = expert_outputs.shape[0]
    bsq = final_hidden_states.shape[0]

    # Threshold: only use segment-reduce when the atomic scatter cost outweighs sort overhead.
    # Sort costs ~50-70us fixed; index_add_ scales with nsel. Crossover ~ nsel >= 32768.
    if nsel >= 32768:
        sorted_idx, perm = torch.sort(token_indices)
        eo_sorted = expert_outputs[perm]
        counts = torch.bincount(token_indices, minlength=bsq)
        reduced = torch.segment_reduce(eo_sorted, "sum", lengths=counts)
        return final_hidden_states + reduced
    else:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output
