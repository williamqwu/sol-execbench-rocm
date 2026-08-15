import torch


def _fp8_router_impl(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    scale_hidden: torch.Tensor,
    scale_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Core implementation of FP8 router projection.
    """
    M, K = hidden_states.shape
    N_padded, _ = gate_weight.shape
    num_experts = 64

    # Dequantize activations with BlockWise1x128 scaling
    # Reshape to apply per-block scaling
    hidden_f32 = hidden_states.to(torch.float32).view(M, K // 128, 128)
    hidden_dequant = (hidden_f32 * scale_hidden.unsqueeze(-1)).view(M, K)

    # Dequantize weights with BlockWise128x128 scaling
    # Weight is (padded_experts, hidden_size), transposed view is (hidden_size, padded_experts)
    weight_t = gate_weight.T  # (K, N_padded)
    weight_f32 = weight_t.to(torch.float32).view(K // 128, 128, N_padded // 128, 128)
    # scale_weight is (1, K//128), need to broadcast correctly
    # For BlockWise128x128 on transposed weight: (K//128, N_padded//128)
    scale_weight_expanded = scale_weight.T.unsqueeze(-1).unsqueeze(-1)  # (K//128, 1, 1, 1)
    weight_dequant = (weight_f32 * scale_weight_expanded).view(K, N_padded)

    # Matrix multiply in FP32 and convert to bfloat16
    result = torch.matmul(hidden_dequant, weight_dequant).to(torch.bfloat16)

    # Return only the first num_experts columns
    return result[:, :num_experts]


# Compile the implementation for better performance
_compiled_impl = torch.compile(_fp8_router_impl, mode="max-autotune")


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    scale_hidden: torch.Tensor,
    scale_weight: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 router projection using compiled implementation.

    Args:
        hidden_states: [M, hidden_size] in FP8
        gate_weight: [padded_experts, hidden_size] in FP8
        scale_hidden: [M, hidden_size//128] in FP32
        scale_weight: [1, hidden_size//128] in FP32

    Returns:
        router_logits: [M, num_experts] in BF16
    """
    return _compiled_impl(hidden_states, gate_weight, scale_hidden, scale_weight)
