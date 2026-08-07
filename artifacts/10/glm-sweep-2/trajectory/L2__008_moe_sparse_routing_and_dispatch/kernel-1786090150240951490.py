import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    norm_topk_prob: bool,
):
    """Sparse MoE routing + expert dispatch via grouped (padded) batched GEMM.

    Replaces the reference's 128-iteration Python loop (which issues a
    host-syncing ``torch.where`` per expert) with:
      * one sort to group token-expert assignments by expert,
      * one host sync to learn the per-expert token count (for padding),
      * a single batched matmul (bmm) over all 128 experts at once.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = gate_weight.shape[0]
    top_k = 8

    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = hidden_states_flat.shape[0]

    # --- Routing ---
    router_logits = torch.matmul(hidden_states_flat, gate_weight.t())
    routing_weights = F.softmax(router_logits.float(), dim=1).to(hidden_states.dtype)
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    if norm_topk_prob:
        routing_weights_topk = routing_weights_topk / (
            routing_weights_topk.sum(dim=-1, keepdim=True) + 1e-9
        )

    # --- Flatten token-expert assignments: (num_tokens * top_k,) ---
    # token_idx: which token each assignment belongs to
    # expert_idx: which expert it is routed to
    token_idx = (
        torch.arange(num_tokens, device=hidden_states.device)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    expert_flat = selected_experts.reshape(-1)  # (num_tokens*top_k,)
    weight_flat = routing_weights_topk.reshape(-1)  # (num_tokens*top_k,)

    # --- Sort by expert so assignments for the same expert are contiguous ---
    sorted_expert, sort_order = torch.sort(expert_flat)
    sorted_token = token_idx[sort_order]
    sorted_weight = weight_flat[sort_order]

    # --- Count tokens per expert (one host sync) ---
    counts = torch.bincount(sorted_expert, minlength=num_experts).to(torch.int32)
    max_count = int(counts.max().item())
    if max_count == 0:
        return torch.zeros_like(hidden_states)

    # --- Build padded (num_experts, max_count) index tensor, -1 padding ---
    # offsets[e] = number of assignments for experts < e
    offsets = torch.cumsum(counts, dim=0).to(torch.int32) - counts
    # arange over max_count, broadcast: (num_experts, max_count)
    pos = torch.arange(max_count, device=hidden_states.device, dtype=torch.int32)
    pos = pos.unsqueeze(0).expand(num_experts, max_count)  # (E, C)
    # valid mask: pos < counts[e]
    valid = pos < counts.unsqueeze(1)  # (E, C)
    # global index into sorted_token for each (expert, slot)
    global_idx = offsets.unsqueeze(1) + pos  # (E, C)
    global_idx = torch.where(valid, global_idx, torch.zeros_like(global_idx))
    # gather token idx and weight per (expert, slot); pad token idx with 0 (we'll mask)
    pad_token = sorted_token[global_idx]  # (E, C)
    pad_weight = sorted_weight[global_idx]  # (E, C)

    # --- Gather hidden states: (E, C, H) ---
    x = hidden_states_flat[pad_token]  # (E, C, H)
    # zero out padded rows so they contribute nothing
    x = x * valid.unsqueeze(-1).to(x.dtype)

    # --- Expert MLP via batched matmul ---
    # gate_proj: (E, INTER, H) -> need (E, H, INTER) for bmm x@W^T
    gate_w = expert_gate_proj.transpose(-1, -2)  # (E, H, INTER)
    up_w = expert_up_proj.transpose(-1, -2)  # (E, H, INTER)
    down_w = expert_down_proj.transpose(-1, -2)  # (E, INTER, H)

    gate_out = torch.bmm(x, gate_w)  # (E, C, INTER)
    up_out = torch.bmm(x, up_w)  # (E, C, INTER)
    silu_gate = gate_out * torch.sigmoid(gate_out)
    intermediate = silu_gate * up_out  # (E, C, INTER)
    expert_out = torch.bmm(intermediate, down_w)  # (E, C, H)

    # --- Weight and scatter-add back to tokens ---
    expert_out = expert_out * pad_weight.unsqueeze(-1).to(expert_out.dtype)  # (E, C, H)
    # zero padded slots
    expert_out = expert_out * valid.unsqueeze(-1).to(expert_out.dtype)

    # Scatter-add: for each (expert, slot), add expert_out[e, slot] to output[pad_token[e,slot]]
    out = torch.zeros(
        (num_tokens, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    out.index_add_(0, pad_token.reshape(-1), expert_out.reshape(-1, hidden_dim))

    return out.reshape(batch_size, sequence_length, hidden_dim)
