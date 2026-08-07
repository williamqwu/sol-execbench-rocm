import torch


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    """
    MoE token sorting using counting sort with prefix sum.

    The stable sort on expert IDs is the counting-sort permutation. Expert
    offsets are derived from the sorted keys via searchsorted instead of a
    separate bincount+cumsum pass, since the sort already produces sorted keys.
    """
    flat = topk_idx.reshape(-1)
    sorted_keys, sorted_token_indices = flat.sort(stable=True)
    arange = torch.arange(257, device=flat.device, dtype=flat.dtype)
    expert_offsets = torch.searchsorted(sorted_keys, arange, right=False).to(torch.int32)
    return sorted_token_indices.to(torch.int32), expert_offsets
