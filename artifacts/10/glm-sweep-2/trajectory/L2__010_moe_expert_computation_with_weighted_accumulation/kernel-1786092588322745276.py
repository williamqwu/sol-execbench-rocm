import torch
import torch.nn.functional as F

@torch.no_grad()
def run(hidden_states, routing_weights, selected_experts, gate_proj_weights, up_proj_weights, down_proj_weights):
    batch_seq_len, hidden_dim = hidden_states.shape
    num_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = selected_experts.shape[1]

    # Preprocess weights: [E, hidden, intermediate] and [E, intermediate, hidden], fp32
    gate_w = gate_proj_weights.transpose(1, 2).to(torch.float32)  # [E, hidden, intermediate]
    up_w = up_proj_weights.transpose(1, 2).to(torch.float32)
    down_w = down_proj_weights.transpose(1, 2).to(torch.float32)  # [E, intermediate, hidden]

    # Flatten (token, slot) assignments: each token appears num_experts_per_tok times
    # selected_experts: [B, K]
    flat_experts = selected_experts.reshape(-1)  # [B*K]
    B = batch_seq_len
    K = num_experts_per_tok
    # token index for each flat entry
    flat_tokens = torch.arange(B, device=hidden_states.device).unsqueeze(1).expand(B, K).reshape(-1)  # [B*K]
    flat_slots = torch.arange(K, device=hidden_states.device).unsqueeze(0).expand(B, K).reshape(-1)  # [B*K]

    # Sort by expert to get contiguous segments per expert
    sorted_experts, sort_idx = torch.sort(flat_experts)
    sorted_tokens = flat_tokens[sort_idx]   # [B*K]
    sorted_slots = flat_slots[sort_idx]     # [B*K]

    # Count tokens per expert
    counts = torch.bincount(sorted_experts, minlength=num_experts)  # [E]
    max_count = int(counts.max().item())

    # Build offsets: cumulative start per expert
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32) - counts.to(torch.int32)  # [E]

    # Build padded token index: [E, max_count], with -1 padding
    # For each expert e, rows [offsets[e]:offsets[e]+counts[e]] are its tokens
    token_idx = torch.full((num_experts, max_count), -1, dtype=torch.int64, device=hidden_states.device)
    slot_idx = torch.zeros((num_experts, max_count), dtype=torch.int64, device=hidden_states.device)
    # scatter the sorted tokens into padded layout
    # position within expert = global_pos - offset[expert]
    arange_max = torch.arange(max_count, device=hidden_states.device)
    # build a mask of valid positions
    pos_in_expert = arange_max.unsqueeze(0) < counts.unsqueeze(1)  # [E, max_count]
    # global positions for valid entries
    global_pos = offsets.unsqueeze(1).long() + arange_max.unsqueeze(0)  # [E, max_count]
    global_pos = global_pos * pos_in_expert  # zero out invalid (will mask later)
    # gather: token_idx[e, j] = sorted_tokens[global_pos[e,j]] if valid else -1
    valid_global = pos_in_expert
    token_idx[valid_global] = sorted_tokens[global_pos[valid_global]]
    slot_idx[valid_global] = sorted_slots[global_pos[valid_global]]

    # Gather hidden states: [E, max_count, hidden]; pad rows are index 0 (will mask)
    safe_token_idx = token_idx.clamp(min=0)
    gathered = hidden_states[safe_token_idx]  # [E, max_count, hidden] bf16
    # mask: [E, max_count, 1]
    mask = (token_idx >= 0).to(torch.float32).unsqueeze(-1)  # [E, max_count, 1]

    current_state = gathered.to(torch.float32) * mask  # zero out padded rows

    # Batched GEMMs
    # gate: [E, max_count, hidden] @ [E, hidden, intermediate] -> [E, max_count, intermediate]
    gate_output = torch.bmm(current_state, gate_w) * mask
    gate_activated = F.silu(gate_output)
    up_output = torch.bmm(current_state, up_w) * mask
    intermediate = gate_activated * up_output
    expert_output = torch.bmm(intermediate, down_w)  # [E, max_count, hidden]

    # Weight by routing weights
    # routing_weights: [B, K]; need [E, max_count] aligned with token_idx/slot_idx
    rw_gathered = routing_weights[safe_token_idx, slot_idx]  # [E, max_count]
    weighted = expert_output * (rw_gathered.unsqueeze(-1)) * mask  # [E, max_count, hidden]

    # Scatter-add back: for each (e, j) valid, add weighted[e,j] to row token_idx[e,j]
    final_hidden_states = torch.zeros((B, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    valid = token_idx >= 0  # [E, max_count]
    out_idx = token_idx[valid]  # [num_valid]
    out_vals = weighted[valid].to(hidden_states.dtype)  # [num_valid, hidden]
    final_hidden_states.index_add_(0, out_idx, out_vals)

    return final_hidden_states
