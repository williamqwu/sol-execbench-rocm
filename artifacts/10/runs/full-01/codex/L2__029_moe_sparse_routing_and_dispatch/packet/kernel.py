import torch
import torch.nn.functional as F


_gate_stream = None
_up_stream = None


def _get_streams():
    global _gate_stream, _up_stream
    if _gate_stream is None:
        _gate_stream = torch.cuda.Stream()
        _up_stream = torch.cuda.Stream()
    return _gate_stream, _up_stream


@torch.no_grad()
def run(
    hidden_states,
    gate_weight,
    e_score_correction_bias,
    expert_gate_proj,
    expert_up_proj,
    expert_down_proj,
    shared_gate_proj,
    shared_up_proj,
    shared_down_proj,
    norm_min,
):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    x = hidden_states.view(-1, hidden_dim)

    # Shared expert.  Keeping these as ordinary BF16 matmuls preserves the
    # reference's materialization and rounding after every projection.
    shared_gate = F.silu(torch.matmul(x, shared_gate_proj.t()))
    shared_up = torch.matmul(x, shared_up_proj.t())
    shared_output = torch.matmul(shared_gate * shared_up, shared_down_proj.t())

    router_logits = torch.matmul(x.float(), gate_weight.t())
    routing = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing = routing + e_score_correction_bias
    selected_weights, selected_experts = torch.topk(routing, 8, dim=-1)
    selected_weights = selected_weights / torch.clamp(
        selected_weights.sum(dim=-1, keepdim=True), min=norm_min
    )
    selected_weights = selected_weights.to(hidden_states.dtype)

    # Materialize each expert's token list without reading GPU scalars back to
    # Python.  The reference's .item() in its loop serializes every expert with
    # the host; iterating the fixed expert set also handles empty lists because
    # PyTorch's zero-row matmuls and index_add are well-defined no-ops.
    expert_mask = F.one_hot(selected_experts, num_classes=64).permute(2, 1, 0)
    dispatch = [torch.where(expert_mask[e]) for e in range(64)]
    states = [x[token] for _, token in dispatch]

    # Gate and up projections are independent.  Running the two chains on
    # separate streams fills otherwise idle CUs for the small expert groups.
    current = torch.cuda.current_stream()
    gate_stream, up_stream = _get_streams()
    gate_stream.wait_stream(current)
    up_stream.wait_stream(current)
    with torch.cuda.stream(gate_stream):
        gates = [
            F.silu(torch.matmul(states[e], expert_gate_proj[e].t()))
            for e in range(64)
        ]
    with torch.cuda.stream(up_stream):
        ups = [
            torch.matmul(states[e], expert_up_proj[e].t())
            for e in range(64)
        ]
    current.wait_stream(gate_stream)
    current.wait_stream(up_stream)

    # Down projections and accumulation remain in expert-index order, exactly
    # matching the reference's BF16 index_add rounding order.
    final = torch.zeros_like(x)
    for e in range(64):
        slot, token = dispatch[e]
        expert_output = torch.matmul(
            gates[e] * ups[e], expert_down_proj[e].t()
        )
        final.index_add_(
            0,
            token,
            expert_output * selected_weights[token, slot, None],
        )

    output = (final + shared_output).view(batch_size, seq_len, hidden_dim)
    return output, router_logits
