import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    batch_size, seq_len, _ = hidden_states.shape
    num_attention_heads = 4
    num_key_value_heads = 1
    head_dim = 256

    # Flatten (batch, seq) into M rows so addmm can fuse bias in-epilogue.
    x = hidden_states.view(-1, 640)

    # addmm: out = beta*bias + alpha*(x @ W.T)  -> fused GEMM+bias, one kernel.
    query_states = torch.addmm(q_bias, x, q_weight.t())
    key_states = torch.addmm(k_bias, x, k_weight.t())
    value_states = torch.addmm(v_bias, x, v_weight.t())

    query_states = query_states.view(batch_size, seq_len, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_key_value_heads, head_dim)

    return query_states, key_states, value_states
