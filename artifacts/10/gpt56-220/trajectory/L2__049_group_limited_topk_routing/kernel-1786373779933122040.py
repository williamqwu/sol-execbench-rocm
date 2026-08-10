import torch
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    Group-limited top-k expert routing.
    
    Two-stage selection:
    1. Organize 256 experts into 8 groups of 32
    2. Compute group scores by summing top-2 expert scores within each group
    3. Select top-4 groups based on group scores
    4. Mask out experts from non-selected groups
    5. Select top-8 experts from remaining candidates
    6. Normalize weights and apply scaling factor
    """
    # Constants
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32
    
    num_tokens = hidden_states.shape[0]

    logits = F.linear(
        hidden_states.to(torch.float32),
        weight.to(torch.float32)
    )
    
    # Apply sigmoid activation to get routing scores
    scores = logits.sigmoid_()  # [num_tokens, 256]
    
    # Add learned expert bias for routing adjustment
    scores_for_routing = scores + expert_bias.to(torch.float32)
    
    # Step 1: Reshape scores into groups [num_tokens, 8, 32]
    group_scores_reshaped = scores_for_routing.view(num_tokens, n_group, experts_per_group)
    
    # Step 2: Get top-2 scores within each group and sum them
    # [num_tokens, 8, 2] -> [num_tokens, 8]
    top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=-1)  # [num_tokens, 8]
    
    # Step 3: Select top-4 groups based on aggregated scores
    # [num_tokens, 4]
    _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)
    
    # Compact the selected groups, so the final top-k examines 128 rather than
    # 256 values per token.
    candidates = torch.gather(
        group_scores_reshaped,
        1,
        group_idx.unsqueeze(-1).expand(-1, -1, experts_per_group),
    ).reshape(num_tokens, topk_group * experts_per_group)
    _, candidate_idx = torch.topk(candidates, k=top_k, dim=-1, sorted=False)
    candidate_group = candidate_idx // experts_per_group
    expert_in_group = candidate_idx % experts_per_group
    topk_idx = torch.gather(group_idx, 1, candidate_group) * experts_per_group + expert_in_group
    
    # Gather selected expert scores (use original scores without bias)
    selected_scores = torch.gather(scores, dim=1, index=topk_idx)
    
    # Normalize routing weights
    topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    
    # Apply routing scaling factor
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight
