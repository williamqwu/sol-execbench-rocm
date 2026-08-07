import torch


_expert_streams = None


def _parallel_expert_mm(x, weights, bounds, out):
    """Run reference-identical expert GEMMs, two at a time."""
    global _expert_streams
    if _expert_streams is None:
        _expert_streams = (torch.cuda.Stream(), torch.cuda.Stream())

    default = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record(default)
    for stream in _expert_streams:
        stream.wait_event(ready)

    begin = 0
    for expert, end in enumerate(bounds):
        if end != begin:
            with torch.cuda.stream(_expert_streams[expert & 1]):
                torch.mm(
                    x[begin:end],
                    weights[expert],
                    out=out[begin:end],
                )
        begin = end

    for stream in _expert_streams:
        done = torch.cuda.Event()
        done.record(stream)
        default.wait_event(done)


@torch.no_grad()
def run(
    hidden_states,
    router_indices,
    routing_weights,
    gate_up_proj,
    gate_up_proj_bias,
    down_proj,
    down_proj_bias,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts = gate_up_proj.shape[0]
    top_k = router_indices.shape[1]

    # Match torch.where(expert_mask[e])'s order: top-k slot first, token second.
    route_experts = router_indices.transpose(0, 1).contiguous().view(-1)
    route_tokens = torch.arange(num_tokens, device=hidden_states.device).repeat(top_k)
    sorted_experts, order = torch.sort(route_experts, stable=True)
    sorted_tokens = route_tokens[order]
    counts = torch.bincount(route_experts, minlength=num_experts)
    # One synchronization supplies all 128 slice bounds.  Avoiding the
    # reference's per-expert .item() synchronization is the important part.
    bounds = counts.cumsum(0).cpu().tolist()

    current_state = hidden_states[sorted_tokens]
    gate_up = torch.empty(
        (top_k * num_tokens, gate_up_proj.shape[2]),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _parallel_expert_mm(current_state, gate_up_proj, bounds, gate_up)
    gate_up.add_(gate_up_proj_bias[sorted_experts])
    gate = gate_up[..., ::2].clamp(max=7.0)
    up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
    glu = gate * torch.sigmoid(gate * 1.702)
    gated_output = (up + 1) * glu
    expert_output = torch.empty_like(current_state)
    _parallel_expert_mm(gated_output, down_proj, bounds, expert_output)
    expert_output.add_(down_proj_bias[sorted_experts])
    expert_output.mul_(routing_weights[sorted_tokens, sorted_experts, None])

    # Restore [token, top-k, hidden] and add routes in ascending expert order,
    # exactly matching the reference's expert-by-expert index_add_ sequence.
    by_route = torch.empty_like(expert_output)
    by_route[order] = expert_output
    by_route = by_route.view(top_k, num_tokens, hidden_size).permute(1, 0, 2)
    slot_order = torch.argsort(router_indices, dim=1, stable=True)
    ordered = torch.gather(
        by_route,
        1,
        slot_order[..., None].expand(-1, -1, hidden_size),
    )
    output = ordered[:, 0] + ordered[:, 1]
    for slot in range(2, top_k):
        output.add_(ordered[:, slot])
    return output
