import torch
import torch.nn.functional as F


_NUM_EXPERTS = 64
_TOP_K = 8


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj_weights: torch.Tensor,
    expert_up_proj_weights: torch.Tensor,
    expert_down_proj_weights: torch.Tensor,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_tokens = batch_size * sequence_length
    hidden_states_flat = hidden_states.reshape(num_tokens, hidden_dim)

    # Routing is kept byte-for-byte equivalent to the reference path.
    router_logits = F.linear(hidden_states_flat, gate_weight)
    routing_weights = F.softmax(
        router_logits.float(), dim=1, dtype=torch.float32
    )
    routing_weights, selected_experts = torch.topk(
        routing_weights, _TOP_K, dim=-1
    )
    routing_weights = routing_weights / routing_weights.sum(
        dim=-1, keepdim=True
    )
    routing_weights = routing_weights.to(torch.bfloat16)

    # Sort the 8 routes per token by expert.  Each expert is then one item in
    # a strided batch, eliminating the reference's 192 small GEMM launches.
    flat_experts = selected_experts.reshape(-1)
    num_routes = num_tokens * _TOP_K
    route_order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[route_order]
    expert_counts = torch.bincount(
        flat_experts, minlength=_NUM_EXPERTS
    )
    expert_offsets = torch.cat(
        (expert_counts.new_zeros(1), expert_counts.cumsum(0))
    )

    # One tiny device-to-host transfer determines the padded batch extent and
    # gives loop bounds for the reference-order bf16 reduction at the end.
    offsets_cpu = expert_offsets.cpu().tolist()
    max_tokens_per_expert = max(
        offsets_cpu[i + 1] - offsets_cpu[i]
        for i in range(_NUM_EXPERTS)
    )

    local_rows = torch.arange(
        num_routes, device=hidden_states.device, dtype=torch.int64
    ) - expert_offsets[sorted_experts]
    token_rows = route_order.div(_TOP_K, rounding_mode="floor")

    grouped_states = torch.zeros(
        (_NUM_EXPERTS, max_tokens_per_expert, hidden_dim),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    grouped_states[sorted_experts, local_rows] = hidden_states_flat[
        token_rows
    ]

    gate_out = torch.bmm(
        grouped_states, expert_gate_proj_weights.transpose(1, 2)
    )
    up_out = torch.bmm(
        grouped_states, expert_up_proj_weights.transpose(1, 2)
    )

    # Preserve both bf16 rounding points in the reference SwiGLU expression.
    silu_gate = gate_out / (
        1.0 + torch.exp(-gate_out.float())
    ).to(torch.bfloat16)
    intermediate = silu_gate * up_out
    expert_outputs = torch.bmm(
        intermediate, expert_down_proj_weights.transpose(1, 2)
    )

    # The outer expert loop and bf16 index_add order are observable parts of
    # the reference numerics, so retain them while reusing the batched GEMMs.
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    for expert_idx in range(_NUM_EXPERTS):
        begin = offsets_cpu[expert_idx]
        end = offsets_cpu[expert_idx + 1]
        expert_routes = route_order[begin:end]
        token_idx = expert_routes.div(_TOP_K, rounding_mode="floor")
        topk_slot = expert_routes.remainder(_TOP_K)
        weighted = expert_outputs[expert_idx, : end - begin] * routing_weights[
            token_idx, topk_slot, None
        ]
        final_hidden_states.index_add_(0, token_idx, weighted)

    return (
        final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
        router_logits,
    )
