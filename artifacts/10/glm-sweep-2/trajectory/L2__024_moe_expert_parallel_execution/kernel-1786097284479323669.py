import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE expert parallel execution."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    n_routed_experts = axes_and_scalars["n_routed_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    topk_indices = torch.randint(
        0, n_routed_experts, (num_tokens, num_experts_per_tok), dtype=torch.int64, device=device
    )
    topk_weights = torch.randn(num_tokens, num_experts_per_tok, dtype=torch.bfloat16, device=device)
    topk_weights = F.softmax(topk_weights.float(), dim=-1).to(torch.bfloat16)
    gate_proj_weights = torch.randn(
        n_routed_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    up_proj_weights = torch.randn(
        n_routed_experts, moe_intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    down_proj_weights = torch.randn(
        n_routed_experts, hidden_size, moe_intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    return {
        "hidden_states": hidden_states,
        "topk_indices": topk_indices,
        "topk_weights": topk_weights,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    """MoE expert parallel execution via grouped GEMM (fp32 accumulation)."""
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = topk_indices.shape[1]
    dev = hidden_states.device

    # Weights: [E, N_out, K_in] -> transpose to [E, K_in, N_out] for _grouped_mm.
    gate_w = gate_proj_weights.transpose(-1, -2).contiguous().float()  # [E, hidden, intermediate]
    up_w = up_proj_weights.transpose(-1, -2).contiguous().float()
    down_w = down_proj_weights.transpose(-1, -2).contiguous().float()  # [E, intermediate, hidden]

    # Flatten routing across (token, slot).
    flat_expert = topk_indices.reshape(-1)  # [N*k]
    tok_id = (
        torch.arange(num_tokens, device=dev)
        .unsqueeze(-1)
        .expand(num_tokens, num_experts_per_tok)
        .reshape(-1)
    )
    flat_weight = topk_weights.reshape(-1).float()  # [N*k]

    # Sort by expert so each expert's tokens are contiguous.
    sorted_expert, sort_idx = torch.sort(flat_expert, stable=True)
    sorted_tok = tok_id[sort_idx]
    sorted_weight = flat_weight[sort_idx]

    counts = torch.bincount(sorted_expert, minlength=n_routed_experts)
    nonzero_idx = torch.where(counts > 0)[0]
    active_counts = counts[counts > 0]
    offs = torch.cumsum(active_counts, dim=0).to(torch.int32)  # cumulative end offsets

    expert_input = hidden_states.index_select(0, sorted_tok).float()  # [M_total, hidden]

    gate_w_active = gate_w.index_select(0, nonzero_idx)
    up_w_active = up_w.index_select(0, nonzero_idx)
    down_w_active = down_w.index_select(0, nonzero_idx)

    gate_out = torch._grouped_mm(expert_input, gate_w_active, offs)
    gate_out = F.silu(gate_out)
    up_out = torch._grouped_mm(expert_input, up_w_active, offs)
    intermediate = gate_out * up_out
    expert_output = torch._grouped_mm(intermediate, down_w_active, offs)  # [M_total, hidden]

    weighted_output = expert_output * sorted_weight.unsqueeze(-1)

    final_hidden_states = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=dev
    )
    final_hidden_states.index_add_(0, sorted_tok, weighted_output)

    return final_hidden_states.to(torch.bfloat16)
