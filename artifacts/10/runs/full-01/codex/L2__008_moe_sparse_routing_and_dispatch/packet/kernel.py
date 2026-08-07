import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states,
    gate_weight,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    norm_topk_prob,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    x = hidden_states.view(-1, hidden_dim)

    router_logits = torch.matmul(x, gate_weight.t())
    routing_weights = F.softmax(router_logits.float(), dim=1).to(x.dtype)
    routing_weights, selected_experts = torch.topk(routing_weights, 8, dim=-1)
    if norm_topk_prob:
        routing_weights = routing_weights / (
            routing_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

    # Sort the token/expert assignments once, then issue three grouped GEMMs
    # instead of launching three separate GEMMs for every expert.
    flat_experts = selected_experts.reshape(-1)
    sorted_experts, permutation = torch.sort(flat_experts)
    offsets = torch.bincount(sorted_experts, minlength=128).cumsum(0).to(torch.int32)
    sorted_tokens = permutation // 8
    current = x[sorted_tokens]

    gate = torch._grouped_mm(
        current, expert_gate_proj.transpose(1, 2), offsets
    )
    up = torch._grouped_mm(
        current, expert_up_proj.transpose(1, 2), offsets
    )
    intermediate = (gate * torch.sigmoid(gate)) * up
    expert_output = torch._grouped_mm(
        intermediate, expert_down_proj.transpose(1, 2), offsets
    )

    sorted_weights = routing_weights.reshape(-1)[permutation]
    weighted = expert_output * sorted_weights[:, None]
    weighted_by_token = torch.empty_like(weighted)
    weighted_by_token[permutation] = weighted
    weighted_by_token = weighted_by_token.view(x.shape[0], 8, hidden_dim)

    # Match the reference's BF16 accumulation order (increasing expert id).
    expert_order = torch.argsort(selected_experts, dim=1)
    rows = torch.arange(x.shape[0], device=x.device)
    output = torch.zeros_like(x)
    for rank in range(8):
        output.add_(weighted_by_token[rows, expert_order[:, rank]])

    return output.view(batch_size, sequence_length, hidden_dim)
