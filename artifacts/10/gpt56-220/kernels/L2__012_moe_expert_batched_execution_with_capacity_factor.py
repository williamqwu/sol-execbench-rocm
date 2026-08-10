import torch
import torch.nn.functional as F
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Generate valid expert indices - each token selects num_experts_per_tok unique experts
    selected_experts = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm

    # Generate routing weights that sum to 1 for each token
    routing_logits = torch.randn(num_tokens, num_experts_per_tok, dtype=dtype, device=device)
    routing_weights = F.softmax(routing_logits.float(), dim=-1).to(dtype)

    # Expert weights: Xavier init with 1/sqrt(fan_in)
    # gate/up: shape [num_experts, hidden_size, moe_intermediate_size], fan_in = hidden_size (dim 1)
    expert_gate_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    expert_up_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    # down: shape [num_experts, moe_intermediate_size, hidden_size], fan_in = moe_intermediate_size (dim 1)
    expert_down_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(moe_intermediate_size)

    return {
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "expert_gate_weights": expert_gate_weights,
        "expert_up_weights": expert_up_weights,
        "expert_down_weights": expert_down_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    capacity = max(int((num_tokens * num_experts_per_tok / num_experts) * 1.25), 1)

    # Flatten all token-expert assignments: (num_tokens * K,)
    flat_experts = selected_experts.reshape(-1).to(torch.uint8)
    flat_weights = routing_weights.reshape(-1)

    # Sort by expert ID (stable to match original sequential assignment order)
    sorted_experts, sorted_indices = flat_experts.sort(stable=True)
    sorted_indices = sorted_indices.to(torch.int32)
    # OPT 1: Vectorized within-expert position computation.
    # After sorting, tokens for the same expert are contiguous.
    # Position within group = global_sorted_index - start_of_expert_group.
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    starts = counts.cumsum(0) - counts

    # Materialize the compact expert-capacity grid, rather than computing and
    # filtering positions for every assignment (including rejected overflow).
    positions = torch.arange(capacity, device=device).expand(num_experts, -1)
    valid_slots = positions < counts.unsqueeze(1)
    v_slot = valid_slots.reshape(-1).nonzero().squeeze(1)
    v_exp = torch.div(v_slot, capacity, rounding_mode="floor")
    v_pos = v_slot.remainder(capacity)
    admitted_sorted_pos = starts.index_select(0, v_exp) + v_pos
    admitted_assignment = sorted_indices.index_select(0, admitted_sorted_pos)
    v_tok = torch.div(admitted_assignment, num_experts_per_tok,
                      rounding_mode="floor")
    v_wt = flat_weights.index_select(0, admitted_assignment)

    # Scatter tokens into padded expert batches using advanced indexing
    expert_inputs = torch.zeros(num_experts, capacity, hidden_size, dtype=dtype, device=device)
    expert_inputs.view(-1, hidden_size).index_copy_(
        0, v_slot, hidden_states.index_select(0, v_tok))

    # Batched expert forward pass (same 3 BMMs as original)
    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)

    # SwiGLU with in-place mul to save one allocation
    activated = F.silu(gate_out).mul_(up_out)

    # The down projection is linear, so apply routing weights while rows are
    # only intermediate-width rather than multiplying the wider output rows.
    slot_weights = torch.zeros(num_experts * capacity, dtype=dtype, device=device)
    slot_weights.index_copy_(0, v_slot, v_wt)
    activated.mul_(slot_weights.view(num_experts, capacity, 1))

    expert_outputs = torch.bmm(activated, expert_down_weights)

    # OPT 2: Vectorized gather + weighted aggregation.
    # Gather only valid positions (skip mask — invalid positions never read).
    valid_out = expert_outputs.view(-1, hidden_size).index_select(0, v_slot)
    result = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    result.index_add_(0, v_tok, valid_out)

    return result
