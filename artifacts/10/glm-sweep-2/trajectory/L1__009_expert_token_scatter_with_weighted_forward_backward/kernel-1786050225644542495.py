import torch


def get_inputs(
    axes_and_scalars: dict[str, int], device: torch.device
) -> dict[str, torch.Tensor]:
    batch_seq_len = axes_and_scalars["batch_seq_len"]
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_dim = axes_and_scalars["hidden_dim"]
    ffn_dim = axes_and_scalars["ffn_dim"]
    num_tokens = min(num_tokens, batch_seq_len)
    perm = torch.randperm(batch_seq_len, device=device)
    token_indices = perm[:num_tokens].sort().values.to(torch.int64)
    grad_output = torch.randn(batch_seq_len, hidden_dim, device=device, dtype=torch.bfloat16)
    selected_tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    w1_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gate_output = torch.nn.functional.silu(w1_output.float()).to(torch.bfloat16)
    up_output = torch.randn(num_tokens, ffn_dim, device=device, dtype=torch.bfloat16)
    gated_output = gate_output * up_output
    expert_output = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    selected_weights = torch.rand(num_tokens, device=device, dtype=torch.bfloat16)
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
    batch_seq_len = grad_output.shape[0]
    hidden_dim = grad_output.shape[1]
    num_tokens = token_indices.shape[0]
    device = grad_output.device

    # Keep inputs in bf16; upcast only what elementwise precision needs.
    # Step 1: gather gradients (bf16 gather is free)
    grad_weighted_output = grad_output[token_indices]  # (num_tokens, hidden_dim) bf16

    # Step 2: backprop routing weight mult — do reduction in fp32 for stability
    grad_selected_weights = (grad_weighted_output.float() * expert_output.float()).sum(dim=-1)
    grad_routing_weights = torch.zeros(batch_seq_len, dtype=torch.float32, device=device)
    grad_routing_weights[token_indices] = grad_selected_weights

    # grad_expert_output = grad_weighted_output * selected_weights (fp32, then bf16 for matmul)
    grad_expert_output_f = grad_weighted_output.float() * selected_weights.float().unsqueeze(-1)
    grad_expert_output = grad_expert_output_f.to(torch.bfloat16)

    # Step 3: w2 backward — bf16 matmuls (fp32 accumulate)
    grad_w2_weight = grad_expert_output.t() @ gated_output  # (hidden_dim, ffn_dim) bf16
    grad_gated_output_f = (grad_expert_output @ w2_weight).float()  # (num_tokens, ffn_dim)

    # Step 4: element-wise gating (fp32)
    grad_gate_output_f = grad_gated_output_f * up_output.float()
    grad_up_output_f = grad_gated_output_f * gate_output.float()
    grad_up_output = grad_up_output_f.to(torch.bfloat16)

    # Step 5: w3 backward (bf16 matmuls)
    grad_w3_weight = grad_up_output.t() @ selected_tokens  # (ffn_dim, hidden_dim)
    grad_selected_tokens_w3 = (grad_up_output @ w3_weight).float()  # (num_tokens, hidden_dim)

    # Step 6: SiLU backward + w1 backward (fp32 elementwise, bf16 matmul)
    w1_output_f = w1_output.float()
    sigmoid_w1 = torch.sigmoid(w1_output_f)
    grad_w1_output_f = grad_gate_output_f * (sigmoid_w1 * (1 + w1_output_f * (1 - sigmoid_w1)))
    grad_w1_output = grad_w1_output_f.to(torch.bfloat16)

    grad_w1_weight = grad_w1_output.t() @ selected_tokens  # (ffn_dim, hidden_dim)
    grad_selected_tokens_w1 = (grad_w1_output @ w1_weight).float()  # (num_tokens, hidden_dim)

    # Step 7: accumulate selected_tokens gradients
    grad_selected_tokens = grad_selected_tokens_w1 + grad_selected_tokens_w3

    # Step 8: scatter back
    grad_hidden_states = torch.zeros(batch_seq_len, hidden_dim, dtype=torch.float32, device=device)
    grad_hidden_states[token_indices] = grad_selected_tokens

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_routing_weights.to(torch.bfloat16),
        grad_w1_weight.to(torch.bfloat16),
        grad_w2_weight.to(torch.bfloat16),
        grad_w3_weight.to(torch.bfloat16),
    )
