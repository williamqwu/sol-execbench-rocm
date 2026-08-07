import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    norm_min: float,
):
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_experts = 64
    top_k = 8

    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # === Shared Experts (process all tokens) ===
    # Fuse gate+up: concat shared weights -> [2*shared_inter, hidden], single GEMM
    shared_gate_up = torch.cat([shared_gate_proj, shared_up_proj], dim=0)  # [14336, hidden]
    shared_gu_out = torch.matmul(hidden_states_flat, shared_gate_up.t())  # [ntok, 14336]
    inter_size = shared_gate_proj.shape[0]
    shared_silu = F.silu(shared_gu_out[:, :inter_size])
    shared_up_out = shared_gu_out[:, inter_size:]
    shared_intermediate = shared_silu * shared_up_out
    shared_output = torch.matmul(shared_intermediate, shared_down_proj.t())

    # === Gating and Routing (float32 for stability) ===
    router_logits = torch.matmul(hidden_states_flat.float(), gate_weight.t())
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights = routing_weights + e_score_correction_bias
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights_selected = torch.gather(routing_weights, dim=-1, index=selected_experts)
    routing_weights_normalized = routing_weights_selected / torch.clamp(
        routing_weights_selected.sum(dim=-1, keepdim=True), min=norm_min
    )
    routing_weights_normalized = routing_weights_normalized.to(hidden_states.dtype)

    # === Sorted dispatch ===
    ntok = batch_size * seq_len
    total_assign = ntok * top_k
    token_ids = torch.arange(ntok, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
    slot_ids = torch.arange(top_k, device=hidden_states.device).unsqueeze(0).expand(ntok, -1).reshape(-1)
    flat_experts = selected_experts.reshape(-1)

    sort_experts, sort_idx = torch.sort(flat_experts)
    sorted_tokens = token_ids[sort_idx]
    sorted_slots = slot_ids[sort_idx]

    counts = torch.bincount(sort_experts, minlength=num_experts)
    ends = counts.cumsum(0).to(torch.int32)
    offs_cpu = torch.cat([torch.zeros(1, dtype=torch.int64, device=hidden_states.device), ends.to(torch.int64)]).cpu().tolist()

    sorted_weights = routing_weights_normalized[sorted_tokens, sorted_slots]
    sorted_states = hidden_states_flat[sorted_tokens]

    # Fuse expert gate+up weights: [E, 2*intermediate, hidden]
    expert_gate_up = torch.cat([expert_gate_proj, expert_up_proj], dim=1)  # [E, 7168, hidden]
    intermediate_size = expert_gate_proj.shape[1]

    final_hidden_states = torch.zeros(
        (ntok, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    for e in range(num_experts):
        start = offs_cpu[e]
        end = offs_cpu[e + 1]
        if end == start:
            continue
        states = sorted_states[start:end]
        # Single GEMM for gate+up
        gu_out = torch.matmul(states, expert_gate_up[e].t())  # [m, 7168]
        silu_out = F.silu(gu_out[:, :intermediate_size])
        up_out = gu_out[:, intermediate_size:]
        intermediate = silu_out * up_out
        expert_output = torch.matmul(intermediate, expert_down_proj[e].t())
        weights = sorted_weights[start:end].unsqueeze(1)
        weighted_output = expert_output * weights
        toks = sorted_tokens[start:end]
        final_hidden_states.index_add_(0, toks, weighted_output)

    final_hidden_states = final_hidden_states + shared_output
    output = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return output, router_logits
