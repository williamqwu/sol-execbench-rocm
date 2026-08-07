import torch


@torch.no_grad()
def run(
    final_hidden_states: torch.Tensor,
    expert_outputs: torch.Tensor,
    token_indices: torch.Tensor,
):
    nsel = expert_outputs.shape[0]
    bsq = final_hidden_states.shape[0]

    # Segment-reduce (non-atomic) wins only for the largest workloads where the
    # atomic scatter cost dominates the fixed sort/bincount overhead.
    if nsel >= 65536:
        sorted_idx, perm = torch.sort(token_indices)
        eo_sorted = expert_outputs[perm]
        counts = torch.bincount(token_indices, minlength=bsq)
        reduced = torch.segment_reduce(eo_sorted, "sum", lengths=counts)
        return final_hidden_states + reduced
    else:
        output = final_hidden_states.clone()
        output.index_add_(dim=0, index=token_indices, source=expert_outputs)
        return output
