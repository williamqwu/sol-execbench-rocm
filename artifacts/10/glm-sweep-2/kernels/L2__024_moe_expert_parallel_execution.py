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
    """MoE expert parallel execution via grouped GEMM.

    gate/up projections run in bf16 (memory-bound); down_proj runs in fp32 to
    preserve output precision for the scatter aggregation. All 256 experts are
    dispatched in a single grouped GEMM using cumulative offsets (empty experts
    produce zero-length groups), avoiding per-expert weight copies.
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = topk_indices.shape[1]
    dev = hidden_states.device

    # Weight views as [E, K_in, N_out] for _grouped_mm (no copy needed; strides valid).
    gate_w = gate_proj_weights.transpose(-1, -2)  # [E, hidden, intermediate]
    up_w = up_proj_weights.transpose(-1, -2)      # [E, hidden, intermediate]
    down_w = down_proj_weights.transpose(-1, -2)  # [E, intermediate, hidden]

    # Flatten routing across (token, slot) and sort by expert id.
    flat_expert = topk_indices.reshape(-1)
    tok_id = (
        torch.arange(num_tokens, device=dev)
        .unsqueeze(-1)
        .expand(num_tokens, num_experts_per_tok)
        .reshape(-1)
    )
    flat_weight = topk_weights.reshape(-1).float()

    sorted_expert, sort_idx = torch.sort(flat_expert, stable=True)
    sorted_tok = tok_id[sort_idx]
    sorted_weight = flat_weight[sort_idx]

    # Cumulative end offsets over all experts (empty groups → zero-length slices).
    counts = torch.bincount(sorted_expert, minlength=n_routed_experts)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)

    # Gather input tokens in sorted order (bf16).
    expert_input = hidden_states.index_select(0, sorted_tok)

    # gate_proj + up_proj in bf16, promote to fp32 for the SwiGLU multiply.
    gate_out = torch._grouped_mm(expert_input, gate_w, offs).float()
    gate_out = F.silu(gate_out)
    up_out = torch._grouped_mm(expert_input, up_w, offs).float()
    intermediate = gate_out * up_out  # [M_total, intermediate], fp32

    # down_proj in fp32 for output precision.
    expert_output = torch._grouped_mm(intermediate, down_w.float(), offs).float()

    # Apply routing weights and scatter-add back to original token positions.
    weighted_output = expert_output * sorted_weight.unsqueeze(-1)
    final_hidden_states = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=dev
    )
    final_hidden_states.index_add_(0, sorted_tok, weighted_output)

    return final_hidden_states.to(torch.bfloat16)
