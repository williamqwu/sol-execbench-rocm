import torch

@torch.no_grad()
def _run_ref(attention_weights, value):
    batch_size = attention_weights.shape[0]
    seq_len_q = attention_weights.shape[2]
    num_heads = 20
    head_dim = 64
    hidden_size = num_heads * head_dim
    attn_output = torch.matmul(attention_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len_q, hidden_size)
    return attn_output

_COMPILED = torch.compile(_run_ref, fullgraph=True, mode="max-autotune")

@torch.no_grad()
def run(attention_weights, value):
    return _COMPILED(attention_weights, value)
