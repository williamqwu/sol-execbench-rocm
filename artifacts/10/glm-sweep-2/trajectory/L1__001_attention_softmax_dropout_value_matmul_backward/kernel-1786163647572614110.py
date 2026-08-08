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
    num_key_value_groups = num_attention_heads // num_key_value_heads  # 10

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    # 1. Transpose gradient: (batch, seq_q, heads, head_dim) -> (batch, heads, seq_q, head_dim)
    grad_attn_output_transposed = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. Gradient w.r.t. attn_weights_dropped — avoid 10x expansion via grouped broadcast
    # grad_attn_output_transposed: [B, 80, S_q, D] -> [B, 8, 10, S_q, D]
    gao_grouped = grad_attn_output_transposed.reshape(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_q, head_dim
    )
    # value_states: [B, 8, S_kv, D] -> [B, 8, 1, D, S_kv]
    vs_t = value_states.to(torch.float32).transpose(-1, -2).unsqueeze(2)
    # matmul: [B, 8, 10, S_q, D] @ [B, 8, 1, D, S_kv] -> [B, 8, 10, S_q, S_kv]
    grad_attn_weights_dropped = torch.matmul(gao_grouped, vs_t).reshape(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv
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

    # 5. Gradient w.r.t. value_states_expanded (grouped to avoid expansion)
    # attn_weights_dropped: [B, 80, S_q, S_kv] -> [B, 8, 10, S_q, S_kv] -> transpose -> [B, 8, 10, S_kv, S_q]
    awd_t = attn_weights_dropped.to(torch.float32).reshape(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_q, seq_len_kv
    ).transpose(-1, -2)
    # gao_grouped: [B, 8, 10, S_q, D]
    # matmul: [B, 8, 10, S_kv, S_q] @ [B, 8, 10, S_q, D] -> [B, 8, 10, S_kv, D]
    grad_value_states_expanded = torch.matmul(awd_t, gao_grouped)
    # sum over group dim -> [B, 8, S_kv, D]
    grad_value_states = grad_value_states_expanded.sum(dim=2)
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
