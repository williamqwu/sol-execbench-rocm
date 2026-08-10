import torch
import torch.nn.functional as F


@torch.compile
def _swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


@torch.compile
def _route(y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (y * weight.unsqueeze(-1)).sum(dim=1)


@torch.compile
def _gather_tokens(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return hidden[torch.bitwise_right_shift(positions, 3)]


def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for MoE training token repeat and expert computation."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]
    
    # Hidden states - random input tokens
    hidden_states = torch.randn(batch_seq_len, hidden_size, device=device, dtype=torch.float16)
    
    # TopK indices - each token selects num_experts_per_tok unique experts
    topk_idx = torch.stack([
        torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        for _ in range(batch_seq_len)
    ]).to(torch.int64)
    
    # TopK weights - normalized routing weights (softmax-like)
    topk_weight_raw = torch.rand(batch_seq_len, num_experts_per_tok, device=device, dtype=torch.float16)
    topk_weight = topk_weight_raw / topk_weight_raw.sum(dim=-1, keepdim=True)
    
    # Expert weights - small initialization
    expert_gate_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_up_projs = torch.randn(
        num_experts, moe_intermediate_size, hidden_size, device=device, dtype=torch.float16
    ) * 0.02
    expert_down_projs = torch.randn(
        num_experts, hidden_size, moe_intermediate_size, device=device, dtype=torch.float16
    ) * 0.02
    
    return {
        "hidden_states": hidden_states,
        "topk_idx": topk_idx,
        "topk_weight": topk_weight,
        "expert_gate_projs": expert_gate_projs,
        "expert_up_projs": expert_up_projs,
        "expert_down_projs": expert_down_projs,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    expert_gate_projs: torch.Tensor,
    expert_up_projs: torch.Tensor,
    expert_down_projs: torch.Tensor,
) -> torch.Tensor:
    """
    MoE training forward pass with token repetition and masked expert computation.
    
    Steps:
    1. Repeat each token num_experts_per_tok times
    2. Process each expert using masked indexing (SwiGLU FFN)
    3. Reshape and apply routing weights
    4. Sum across experts
    
    Args:
        hidden_states: [batch_seq_len, hidden_size]
        topk_idx: [batch_seq_len, num_experts_per_tok] - selected expert indices
        topk_weight: [batch_seq_len, num_experts_per_tok] - routing weights
        expert_gate_projs: [num_experts, moe_intermediate_size, hidden_size]
        expert_up_projs: [num_experts, moe_intermediate_size, hidden_size]
        expert_down_projs: [num_experts, hidden_size, moe_intermediate_size]
    
    Returns:
        output: [batch_seq_len, hidden_size]
    """
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    
    # Step 1: Repeat tokens for each selected expert
    # [batch_seq_len, hidden_size] -> [batch_seq_len * num_experts_per_tok, hidden_size]
    flat_topk_idx = topk_idx.view(-1)
    positions = torch.argsort(flat_topk_idx, stable=False)
    counts = torch.bincount(flat_topk_idx, minlength=num_experts)
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    y = torch.empty(
        (batch_seq_len * num_experts_per_tok, hidden_size),
        device=hidden_states.device, dtype=hidden_states.dtype
    )

    # Sort tokens by expert and execute each projection as one grouped GEMM.
    expert_input = _gather_tokens(hidden_states, positions)
    gate_output = torch._grouped_mm(
        expert_input, expert_gate_projs.transpose(1, 2), offs=offsets
    )
    up_output = torch._grouped_mm(
        expert_input, expert_up_projs.transpose(1, 2), offs=offsets
    )
    intermediate = _swiglu(gate_output, up_output)
    expert_output = torch._grouped_mm(
        intermediate, expert_down_projs.transpose(1, 2), offs=offsets
    )
    y.index_copy_(0, positions, expert_output)
    
    # Step 4: Reshape and apply routing weights
    # [batch_seq_len * num_experts_per_tok, hidden_size] ->
    # [batch_seq_len, num_experts_per_tok, hidden_size]
    y = y.view(batch_seq_len, num_experts_per_tok, hidden_size)
    output = _route(y, topk_weight)
    return output
