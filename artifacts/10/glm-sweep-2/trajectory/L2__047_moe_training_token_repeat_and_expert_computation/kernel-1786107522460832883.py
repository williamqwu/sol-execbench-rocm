import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
) -> torch.Tensor:
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    K = num_experts_per_tok
    N = batch_seq_len

    # Repeat tokens: [N, H] -> [N*K, H]
    hidden_states_repeated = hidden_states.repeat_interleave(K, dim=0)
    flat_topk_idx = topk_idx.reshape(-1)  # [N*K]

    # Sort by expert so each expert's tokens are contiguous.
    sorted_expert_idx = torch.argsort(flat_topk_idx, stable=True)  # [N*K]
    sorted_experts = flat_topk_idx[sorted_expert_idx]  # [N*K], non-decreasing

    # Gather hidden states in sorted order -> contiguous per expert.
    x_sorted = hidden_states_repeated.index_select(0, sorted_expert_idx)  # [N*K, H]

    # Compute per-expert token counts and offsets.
    counts = torch.bincount(sorted_experts, minlength=num_experts)  # [num_experts]
    # offsets: cumulative start index of each expert.
    offsets = torch.cumsum(counts, dim=0)  # [num_experts], exclusive end
    # Move to CPU to drive the per-expert loop (counts are small, 256 values).
    counts_cpu = counts.cpu()
    offsets_cpu = offsets.cpu()

    y_sorted = torch.empty(N * K, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

    start = 0
    for expert_idx in range(num_experts):
        cnt = int(counts_cpu[expert_idx].item())
        if cnt == 0:
            continue
        end = start + cnt
        expert_input = x_sorted[start:end]  # [cnt, H], contiguous

        gate_output = F.linear(expert_input, expert_gate_projs[expert_idx])
        up_output = F.linear(expert_input, expert_up_projs[expert_idx])
        intermediate = F.silu(gate_output) * up_output
        expert_output = F.linear(intermediate, expert_down_projs[expert_idx])

        y_sorted[start:end] = expert_output
        start = end

    # Scatter sorted outputs back to original (token*K + slot) positions.
    y = torch.empty(N * K, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
    y.index_copy_(0, sorted_expert_idx, y_sorted)

    # Reshape and apply routing weights, sum across experts.
    y = y.view(N, K, hidden_size)
    output = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
    return output
