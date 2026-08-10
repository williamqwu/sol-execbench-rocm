import torch


@torch.compile(fullgraph=True, dynamic=False)
def _gating_backward(grad_gated_output, up_output, gate_output, w1_output):
    grad_up_output = grad_gated_output * gate_output
    x = w1_output.float()
    sigmoid_x = torch.sigmoid(x)
    grad_w1_output = (
        (grad_gated_output * up_output).float()
        * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
    ).to(torch.bfloat16)
    return torch.cat((grad_w1_output, grad_up_output), dim=1)


@torch.compile(fullgraph=True, dynamic=True)
def _routing_backward(grad_output, token_indices, expert_output, selected_weights):
    gathered = grad_output[token_indices].float()
    grad_selected_weights = (gathered * expert_output.float()).sum(dim=-1)
    grad_routing_weights = torch.zeros(
        grad_output.shape[0], dtype=torch.bfloat16, device=grad_output.device
    )
    grad_routing_weights[token_indices] = grad_selected_weights.to(torch.bfloat16)
    grad_expert_output = (
        gathered * selected_weights.float().unsqueeze(-1)
    ).to(torch.bfloat16)
    return grad_expert_output, grad_routing_weights


def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_dim = axes_and_scalars["hidden_dim"]
    ffn_dim = axes_and_scalars["ffn_dim"]
    
    # Ensure num_tokens <= batch_seq_len
    num_tokens = min(num_tokens, batch_seq_len)
    
    # Generate random token indices (unique indices into batch_seq_len)
    perm = torch.randperm(batch_seq_len, device=device)
    token_indices = perm[:num_tokens].sort().values.to(torch.int64)
    
    # Gradient from next layer
    grad_output = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Selected tokens (gathered from hidden_states)
    selected_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Forward pass intermediate values
    w1_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gate_output = torch.nn.functional.silu(w1_output.float()).to(torch.bfloat16)
    up_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gated_output = gate_output * up_output
    expert_output = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    
    # Routing weights for selected tokens
    selected_weights = torch.rand(num_tokens, device=device, dtype=torch.bfloat16)
    
    # Expert MLP weights
    w1_weight = torch.randn(ffn_dim, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02
    w2_weight = torch.randn(hidden_dim, ffn_dim, device=device, dtype=torch.bfloat16) * 0.02
    w3_weight = torch.randn(ffn_dim, hidden_dim, device=device, dtype=torch.bfloat16) * 0.02
    
    return {
        "grad_output": grad_output,
        "token_indices": token_indices,
        "selected_tokens": selected_tokens,
        "w1_output": w1_output,
        "gate_output": gate_output,
        "up_output": up_output,
        "gated_output": gated_output,
        "expert_output": expert_output,
        "selected_weights": selected_weights,
        "w1_weight": w1_weight,
        "w2_weight": w2_weight,
        "w3_weight": w3_weight,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    token_indices: torch.Tensor,
    selected_tokens: torch.Tensor,
    w1_output: torch.Tensor,
    gate_output: torch.Tensor,
    up_output: torch.Tensor,
    gated_output: torch.Tensor,
    expert_output: torch.Tensor,
    selected_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
):
    """
    Backward pass for expert token scatter with weighted forward.
    
    Gradient flow:
    1. Gather gradients from scattered positions
    2. Backprop through routing weight multiplication
    3. Backprop through w2 linear layer
    4. Backprop through element-wise gating
    5. Backprop through w3 linear layer (up branch)
    6. Backprop through SiLU activation and w1 linear layer (gate branch)
    7. Accumulate gradients and scatter back
    """
    batch_seq_len = grad_output.shape[0]
    hidden_dim = grad_output.shape[1]
    num_tokens = token_indices.shape[0]
    ffn_dim = w1_weight.shape[0]
    
    device = grad_output.device
    
    # Keep matrix multiplies in bf16 (MFMA accumulates in fp32 on CDNA), and
    # use fp32 only for the nonlinear/reduction portions that need it.
    grad_expert_output, grad_routing_weights = _routing_backward(
        grad_output, token_indices, expert_output, selected_weights
    )
    
    # Step 3: Backprop through w2 linear layer
    # expert_output = gated_output @ w2_weight.T
    # grad_w2_weight = grad_expert_output.T @ gated_output
    grad_w2_weight = grad_expert_output.t() @ gated_output
    
    # Gradient w.r.t. gated_output
    grad_gated_output = grad_expert_output @ w2_weight
    
    # Step 4: Backprop through element-wise gating
    # gated_output = gate_output * up_output
    grad_w13_output = _gating_backward(
        grad_gated_output, up_output, gate_output, w1_output
    )
    grad_w1_output, grad_up_output = grad_w13_output.split(ffn_dim, dim=1)
    
    # Step 5: Backprop through w3 linear layer (up branch)
    # up_output = selected_tokens @ w3_weight.T
    grad_selected_tokens_w3 = grad_up_output @ w3_weight

    # Step 6: Backprop through SiLU activation and w1 linear layer
    # gate_output = silu(w1_output) = w1_output * sigmoid(w1_output)
    # d(silu)/d(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    # w1_output = selected_tokens @ w1_weight.T
    # Both weight gradients share the same right operand.  A single taller
    # GEMM reduces launch overhead and gives small-token cases more work/CU.
    grad_w13_weight = grad_w13_output.t() @ selected_tokens
    grad_w1_weight, grad_w3_weight = grad_w13_weight.split(ffn_dim, dim=0)
    # Step 7: Accumulate the w1 branch in the GEMM epilogue.
    grad_selected_tokens = torch.addmm(
        grad_selected_tokens_w3, grad_w1_output, w1_weight
    )

    # Step 8: Scatter gradients back to full hidden_states tensor
    grad_hidden_states = torch.zeros(batch_seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
    grad_hidden_states[token_indices] = grad_selected_tokens
    
    # Convert back to bfloat16
    return (
        grad_hidden_states,
        grad_routing_weights,
        grad_w1_weight,
        grad_w2_weight,
        grad_w3_weight,
    )
