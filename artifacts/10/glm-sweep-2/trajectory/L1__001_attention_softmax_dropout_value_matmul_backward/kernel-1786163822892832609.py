import torch

_compiled_run = None


def _run_impl(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    # Expand value states for GQA (broadcast, no copy)
    value_states_expanded = value_states
    if num_key_value_groups > 1:
        value_states_expanded = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        ).reshape(batch_size, num_attention_heads, seq_len_kv, head_dim)

    # 1. Transpose gradient
    grad_attn_output_transposed = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. Gradient w.r.t. attn_weights_dropped — fp32 matmul (reduction over head_dim=128)
    grad_attn_weights_dropped = torch.matmul(
        grad_attn_output_transposed,
        value_states_expanded.to(torch.float32).transpose(-2, -1),
    )

    # 3. Gradient through dropout
    if attention_dropout > 0.0:
        grad_attn_weights = grad_attn_weights_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        grad_attn_weights = grad_attn_weights_dropped

    # 4. Gradient through softmax
    attn_weights_f32 = attn_weights.to(torch.float32)
    sum_term = (grad_attn_weights * attn_weights_f32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights_f32 * (grad_attn_weights - sum_term)
    grad_attn_scores = grad_attn_scores.to(torch.bfloat16)

    # 5. Gradient w.r.t. value_states_expanded — bf16 matmul (reduction over seq_len_q)
    grad_value_states_expanded = torch.matmul(
        attn_weights_dropped.transpose(-2, -1),
        grad_attn_output_transposed.to(torch.bfloat16),
    ).to(torch.float32)

    # 6. GQA gradient aggregation
    if num_key_value_groups > 1:
        grad_value_states_expanded = grad_value_states_expanded.reshape(
            batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
        )
        grad_value_states = grad_value_states_expanded.sum(dim=2)
    else:
        grad_value_states = grad_value_states_expanded

    grad_value_states = grad_value_states.to(torch.bfloat16)

    return grad_attn_scores, grad_value_states


@torch.no_grad()
def run(
    grad_attn_output: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_weights_dropped: torch.Tensor,
    value_states: torch.Tensor,
    dropout_mask: torch.Tensor,
    attention_dropout: float,
):
    global _compiled_run
    if _compiled_run is None:
        _compiled_run = torch.compile(_run_impl, dynamic=True)
    return _compiled_run(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
