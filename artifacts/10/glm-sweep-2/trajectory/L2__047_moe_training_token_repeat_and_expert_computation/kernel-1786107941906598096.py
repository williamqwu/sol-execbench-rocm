import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states, topk_idx, topk_weight,
    expert_gate_projs, expert_up_projs, expert_down_projs,
):
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    moe_intermediate_size = expert_gate_projs.shape[1]
    K = topk_idx.shape[1]
    N = batch_seq_len
    M = moe_intermediate_size
    H = hidden_size

    hidden_states_repeated = hidden_states.repeat_interleave(K, dim=0)  # [N*K, H]
    flat_topk_idx = topk_idx.reshape(-1)  # [N*K]

    # Sort by expert.
    sorted_expert_idx = torch.argsort(flat_topk_idx, stable=True)
    sorted_experts = flat_topk_idx[sorted_expert_idx]
    x_sorted = hidden_states_repeated.index_select(0, sorted_expert_idx)  # [N*K, H]

    counts = torch.bincount(sorted_experts, minlength=num_experts)
    counts_cpu = counts.cpu()
    max_count = int(counts_cpu.max().item())

    # Pad x_sorted into [num_experts, max_count, H] with zeros.
    # Build per-expert padded layout.
    x_padded = torch.zeros(num_experts, max_count, H, device=hidden_states.device, dtype=hidden_states.dtype)
    # scatter x_sorted rows into the right (expert, local_pos) slots.
    # local position within expert = arange(N*K) - offset[expert]
    offsets = torch.cumsum(counts, dim=0) - counts  # start offset per expert [num_experts]
    arange_nk = torch.arange(N * K, device=hidden_states.device)
    expert_of_row = sorted_experts  # [N*K]
    local_pos = arange_nk - offsets[expert_of_row]  # [N*K]
    x_padded[expert_of_row, local_pos] = x_sorted

    # gate: bmm [E, maxC, H] x [E, H, M] -> [E, maxC, M]
    # weights are [E, M, H]; need [E, H, M] => transpose(1,2)
    gate_w = expert_gate_projs.transpose(1, 2)  # [E, H, M]
    up_w = expert_up_projs.transpose(1, 2)       # [E, H, M]
    gate_out = torch.bmm(x_padded, gate_w)  # [E, maxC, M]
    up_out = torch.bmm(x_padded, up_w)      # [E, maxC, M]
    inter = F.silu(gate_out) * up_out       # [E, maxC, M]

    # down: bmm [E, maxC, M] x [E, M, H] -> [E, maxC, H]
    down_w = expert_down_projs.transpose(1, 2)  # [E, M, H]
    y_padded = torch.bmm(inter, down_w)     # [E, maxC, H]

    # Gather back: y_sorted[e, local_pos] -> y_sorted_flat
    y_sorted = y_padded[expert_of_row, local_pos]  # [N*K, H]

    # Scatter back to original positions.
    y = torch.empty(N * K, H, device=hidden_states.device, dtype=hidden_states.dtype)
    y.index_copy_(0, sorted_expert_idx, y_sorted)

    y = y.view(N, K, H)
    output = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
    return output
