import torch
import torch.nn.functional as F

_shared_stream = torch.cuda.Stream()


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
    current_stream = torch.cuda.current_stream()
    _shared_stream.wait_stream(current_stream)
    with torch.cuda.stream(_shared_stream):
        shared_gate_up_weight = torch.cat(
            (shared_expert_gate_proj_weight, shared_expert_up_proj_weight), dim=0
        )
        shared_gate_up = hidden_states @ shared_gate_up_weight.t()
        shared_gate, shared_up = shared_gate_up.chunk(2, dim=1)
        shared_gate_silu = shared_gate / (
            1.0 + torch.exp(-shared_gate.float())
        ).to(hidden_states.dtype)
        shared_intermediate = shared_gate_silu * shared_up
        shared_out = shared_intermediate @ shared_expert_down_proj_weight.t()

    shared_weight = torch.sigmoid(
        (hidden_states @ shared_expert_gate_weight.t()).float()
    ).to(hidden_states.dtype)

    flat_experts = selected_experts.reshape(-1)
    order = torch.argsort(flat_experts)
    token_ids = torch.arange(tokens, device=hidden_states.device).repeat_interleave(top_k)
    sorted_tokens = token_ids[order]
    offsets = torch.cumsum(
        torch.bincount(flat_experts, minlength=expert_gate_proj_weights.shape[0]), 0
    ).to(torch.int32)

    routed = hidden_states[sorted_tokens]
    gate_up_weights = torch.cat(
        (expert_gate_proj_weights, expert_up_proj_weights), dim=1
    )
    gate_up = torch._grouped_mm(routed, gate_up_weights.transpose(1, 2), offsets)
    gate, up = gate_up.chunk(2, dim=1)
    silu_gate = gate / (1.0 + torch.exp(-gate.float())).to(hidden_states.dtype)
    intermediate = silu_gate * up
    expert_out = torch._grouped_mm(
        intermediate, expert_down_proj_weights.transpose(1, 2), offsets
    )

    flat_routing = routing_weights.reshape(-1)
    weighted = expert_out * flat_routing[order, None]
    final_hidden = torch.zeros_like(hidden_states)
    final_hidden.index_add_(0, sorted_tokens, weighted)

    current_stream.wait_stream(_shared_stream)
    return final_hidden + shared_weight * shared_out
