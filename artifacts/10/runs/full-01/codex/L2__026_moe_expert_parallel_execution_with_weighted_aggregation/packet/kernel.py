import torch
import torch.nn.functional as F


_gate_stream = None
_up_stream = None


@torch.no_grad()
def run(
    hidden_states,
    routing_weights,
    selected_experts,
    gate_proj_weights,
    up_proj_weights,
    down_proj_weights,
):
    global _gate_stream, _up_stream

    x = hidden_states.view(-1, hidden_states.shape[-1])
    num_experts = gate_proj_weights.shape[0]
    top_k = selected_experts.shape[1]

    # Group all routed rows by expert.  `_grouped_mm` consumes cumulative row
    # offsets, and keeps all sixteen expert GEMMs in one GPU launch.
    expert_for_route = selected_experts.reshape(-1)
    sorted_experts, order = torch.sort(expert_for_route)
    grouped_x = x[order // top_k]
    offsets = torch.bincount(
        sorted_experts, minlength=num_experts
    ).cumsum(0).to(torch.int32)

    if _gate_stream is None:
        _gate_stream = torch.cuda.Stream(device=hidden_states.device)
        _up_stream = torch.cuda.Stream(device=hidden_states.device)

    current = torch.cuda.current_stream(hidden_states.device)
    _gate_stream.wait_stream(current)
    _up_stream.wait_stream(current)

    # These projections are independent and the skinny expert GEMMs leave
    # enough machine capacity for useful overlap.
    with torch.cuda.stream(_gate_stream):
        gate = torch._grouped_mm(
            grouped_x, gate_proj_weights.transpose(1, 2), offsets
        )
    with torch.cuda.stream(_up_stream):
        up = torch._grouped_mm(
            grouped_x, up_proj_weights.transpose(1, 2), offsets
        )

    current.wait_stream(_gate_stream)
    current.wait_stream(_up_stream)
    intermediate = F.silu(gate) * up
    expert_output = torch._grouped_mm(
        intermediate, down_proj_weights.transpose(1, 2), offsets
    )

    sorted_route_weights = routing_weights.reshape(-1)[order]
    weighted = (expert_output.float() * sorted_route_weights[:, None]).to(
        torch.bfloat16
    )

    # Restore original token/top-k order.  Adding the two BF16 contributions
    # reproduces the reference's BF16 index_add accumulation boundary.
    routed = torch.empty_like(weighted)
    routed.index_copy_(0, order, weighted)
    output = routed[0::2] + routed[1::2]
    return output.reshape_as(hidden_states)
