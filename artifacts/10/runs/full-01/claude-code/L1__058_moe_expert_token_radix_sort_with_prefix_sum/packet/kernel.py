import torch


@torch.no_grad()
def run(topk_idx: torch.Tensor):
    num_experts = 256
    flat = topk_idx.reshape(-1)
    _, sorted_token_indices = flat.sort(stable=True)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    expert_offsets[1:] = torch.bincount(flat.long(), minlength=num_experts).cumsum(0).to(torch.int32)
    return sorted_token_indices.to(torch.int32), expert_offsets
