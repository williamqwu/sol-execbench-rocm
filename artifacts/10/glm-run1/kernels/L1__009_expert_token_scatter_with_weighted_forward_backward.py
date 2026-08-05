import torch

_COMPILED = None


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
        "grad_output": grad_output, "token_indices": token_indices,
        "selected_tokens": selected_tokens, "w1_output": w1_output,
        "gate_output": gate_output, "up_output": up_output,
        "gated_output": gated_output, "expert_output": expert_output,
        "selected_weights": selected_weights, "w1_weight": w1_weight,
        "w2_weight": w2_weight, "w3_weight": w3_weight,
    }


@torch.no_grad()
def _run_impl(
    grad_output, token_indices, selected_tokens,
    w1_output, gate_output, up_output, gated_output, expert_output,
    selected_weights, w1_weight, w2_weight, w3_weight,
):
    batch_seq_len = grad_output.shape[0]
    hidden_dim = grad_output.shape[1]
    device = grad_output.device

    grad_weighted_output = grad_output[token_indices]

    grad_selected_weights = (grad_weighted_output.float() * expert_output.float()).sum(dim=-1)
    grad_routing_weights = torch.zeros(batch_seq_len, dtype=torch.bfloat16, device=device)
    grad_routing_weights[token_indices] = grad_selected_weights.to(torch.bfloat16)

    grad_expert_output = grad_weighted_output * selected_weights.unsqueeze(-1)

    grad_w2_weight = grad_expert_output.t() @ gated_output
    grad_gated_output = grad_expert_output @ w2_weight

    grad_gated_output_f = grad_gated_output.float()
    up_output_f = up_output.float()
    gate_output_f = gate_output.float()
    grad_gate_output = grad_gated_output_f * up_output_f
    grad_up_output = grad_gated_output_f * gate_output_f

    grad_up_output_b = grad_up_output.to(torch.bfloat16)
    grad_w3_weight = grad_up_output_b.t() @ selected_tokens
    grad_selected_tokens_w3 = grad_up_output_b @ w3_weight

    w1_output_f = w1_output.float()
    sigmoid_w1 = torch.sigmoid(w1_output_f)
    grad_w1_output = grad_gate_output * (sigmoid_w1 * (1.0 + w1_output_f * (1.0 - sigmoid_w1)))
    grad_w1_output_b = grad_w1_output.to(torch.bfloat16)
    grad_w1_weight = grad_w1_output_b.t() @ selected_tokens
    grad_selected_tokens_w1 = grad_w1_output_b @ w1_weight

    grad_selected_tokens = grad_selected_tokens_w1 + grad_selected_tokens_w3

    grad_hidden_states = torch.zeros(batch_seq_len, hidden_dim, dtype=torch.bfloat16, device=device)
    grad_hidden_states[token_indices] = grad_selected_tokens

    return (
        grad_hidden_states, grad_routing_weights,
        grad_w1_weight, grad_w2_weight, grad_w3_weight,
    )


def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(_run_impl)
    return _COMPILED


@torch.no_grad()
def run(
    grad_output, token_indices, selected_tokens,
    w1_output, gate_output, up_output, gated_output, expert_output,
    selected_weights, w1_weight, w2_weight, w3_weight,
):
    return _get_compiled()(
        grad_output, token_indices, selected_tokens,
        w1_output, gate_output, up_output, gated_output, expert_output,
        selected_weights, w1_weight, w2_weight, w3_weight,
    )
