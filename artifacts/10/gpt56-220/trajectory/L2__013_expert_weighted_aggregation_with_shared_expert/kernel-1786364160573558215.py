import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
    shared_expert_gate_proj_weight: torch.Tensor,
    shared_expert_up_proj_weight: torch.Tensor,
    shared_expert_down_proj_weight: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
):
    # Put the token/expert assignments into contiguous expert groups.  The
    # grouped GEMM primitive consumes cumulative group ends (not lengths).
    tokens, top_k = selected_experts.shape
    flat_experts = selected_experts.reshape(-1)
    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[order]
    token_ids = torch.arange(tokens, device=hidden_states.device).repeat_interleave(top_k)
    sorted_tokens = token_ids[order]
    offsets = torch.cumsum(
        torch.bincount(sorted_experts, minlength=expert_gate_proj_weights.shape[0]), 0
    ).to(torch.int32)

    routed = hidden_states[sorted_tokens]
    gate = torch._grouped_mm(routed, expert_gate_proj_weights.transpose(1, 2), offsets)
    up = torch._grouped_mm(routed, expert_up_proj_weights.transpose(1, 2), offsets)
    intermediate = F.silu(gate.float()).to(hidden_states.dtype) * up
    expert_out = torch._grouped_mm(
        intermediate, expert_down_proj_weights.transpose(1, 2), offsets
    )

    flat_routing = routing_weights.reshape(-1)
    weighted = expert_out * flat_routing[order, None]
    final_hidden = torch.zeros_like(hidden_states)
    final_hidden.index_add_(0, sorted_tokens, weighted)

    shared_gate = hidden_states @ shared_expert_gate_proj_weight.t()
    shared_up = hidden_states @ shared_expert_up_proj_weight.t()
    shared_intermediate = F.silu(shared_gate.float()).to(hidden_states.dtype) * shared_up
    shared_out = shared_intermediate @ shared_expert_down_proj_weight.t()
    shared_weight = torch.sigmoid(
        (hidden_states @ shared_expert_gate_weight.t()).float()
    ).to(hidden_states.dtype)
    return final_hidden + shared_weight * shared_out
