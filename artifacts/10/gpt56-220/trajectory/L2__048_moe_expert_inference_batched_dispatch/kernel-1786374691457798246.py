import torch
import torch.nn.functional as F


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs including non-random topk indices."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    intermediate_size = axes_and_scalars["intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Random hidden states
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    
    # Generate valid topk indices - each token selects num_experts_per_tok unique experts
    topk_idx = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(num_tokens)
    ]).to(torch.int64)
    
    # Random weights that sum to 1 per token (softmax-like)
    topk_weight_raw = torch.rand(num_tokens, num_experts_per_tok, device=device)
    topk_weight = (topk_weight_raw / topk_weight_raw.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    
    # Expert weights
    gate_weights = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    up_weights = torch.randn(
        num_experts, intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    down_weights = torch.randn(
        num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "topk_idx": topk_idx,
        "topk_weight": topk_weight,
        "gate_weights": gate_weights,
        "up_weights": up_weights,
        "down_weights": down_weights,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    """
    MoE expert inference with batched token dispatch.
    
    Args:
        hidden_states: Input tokens [num_tokens, hidden_size]
        topk_idx: Selected expert indices [num_tokens, num_experts_per_tok]
        topk_weight: Expert weights [num_tokens, num_experts_per_tok]
        gate_weights: Gate projection weights [num_experts, intermediate_size, hidden_size]
        up_weights: Up projection weights [num_experts, intermediate_size, hidden_size]
        down_weights: Down projection weights [num_experts, hidden_size, intermediate_size]
        
    Returns:
        output: Weighted expert outputs [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = gate_weights.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    
    # Keep GEMMs in bf16 so they use the matrix engine.  Accumulate the final
    # mixture in fp32, matching the numerically sensitive part of the reference.
    hidden_states_f32 = hidden_states
    topk_weight_f32 = topk_weight.to(torch.float32)
    gate_weights_f32 = gate_weights
    up_weights_f32 = up_weights
    down_weights_f32 = down_weights
    gate_up_weights = torch.cat((gate_weights, up_weights), dim=1)
    
    # Step 1/2: Count and sort the flattened expert assignments.  Bincount
    # avoids materializing and reducing a dense [tokens, experts] matrix.
    flat_topk_idx = topk_idx.view(-1)  # [num_tokens * num_experts_per_tok]
    tokens_per_expert = torch.bincount(flat_topk_idx, minlength=num_experts)
    idxs = flat_topk_idx.to(torch.int32).argsort()  # Sort by expert ID
    
    # Gather tokens in sorted order
    token_indices = idxs // num_experts_per_tok
    sorted_tokens = hidden_states_f32[token_indices]  # [num_tokens * num_experts_per_tok, hidden_size]
    
    # Step 3: Process tokens through experts in batches
    outputs = []
    start_idx = 0
    tokens_per_expert_cpu = tokens_per_expert.cpu().numpy()
    
    for expert_id in range(num_experts):
        num_tokens_for_expert = int(tokens_per_expert_cpu[expert_id])
        
        if num_tokens_for_expert == 0:
            continue
            
        end_idx = start_idx + num_tokens_for_expert
        
        # Get tokens for this expert
        expert_input = sorted_tokens[start_idx:end_idx]  # [num_tokens_for_expert, hidden_size]
        
        # Expert computation: SwiGLU MLP
        # gate_proj: [num_tokens_for_expert, hidden_size] @ [hidden_size, intermediate_size]
        gate_up_out = torch.matmul(expert_input, gate_up_weights[expert_id].t())
        gate_out, up_out = gate_up_out.chunk(2, dim=1)
        
        # SwiGLU activation: silu(gate) * up
        # silu(x) = x * sigmoid(x)
        activated = F.silu(gate_out, inplace=True)
        activated.mul_(up_out)
        
        # down_proj
        expert_output = torch.matmul(activated, down_weights_f32[expert_id].t())
        
        outputs.append(expert_output)
        start_idx = end_idx
    
    # Step 4: Concatenate all expert outputs
    if len(outputs) == 0:
        return torch.zeros_like(hidden_states)
        
    all_outputs = torch.cat(outputs, dim=0)  # [num_tokens * num_experts_per_tok, hidden_size]
    
    # Step 5/6: Accumulate expert results directly into their source tokens.
    # `all_outputs` is in sorted-pair order, so gather weights by the same
    # permutation and avoid a full pair-sized scatter/reshape/reduction.
    sorted_weight = topk_weight_f32.view(-1)[idxs]
    final_output = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.float32, device=hidden_states.device
    )
    weighted_outputs = all_outputs.float()
    weighted_outputs.mul_(sorted_weight.unsqueeze(1))
    final_output.index_add_(0, token_indices, weighted_outputs)
    
    return final_output.to(torch.bfloat16)
