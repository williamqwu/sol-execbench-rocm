import torch

@torch.compile(dynamic=True, fullgraph=True)
def _compiled(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128

    bsz, seq_len, _ = compressed_kv.shape

    input_dtype = compressed_kv.dtype
    hidden_states = compressed_kv.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized_kv = (kv_a_layernorm_weight.to(torch.float32) * hidden_states).to(input_dtype)

    expanded_kv = torch.matmul(normalized_kv, kv_b_proj_weight.t())

    kv = expanded_kv.view(bsz, seq_len, num_heads, qk_nope_head_dim + v_head_dim)
    kv = kv.transpose(1, 2)
    k_nope = kv[:, :, :, :qk_nope_head_dim].contiguous()
    value_states = kv[:, :, :, qk_nope_head_dim:].contiguous()
    return k_nope, value_states

@torch.no_grad()
def run(
    compressed_kv: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    eps: float,
):
    return _compiled(compressed_kv, kv_a_layernorm_weight, kv_b_proj_weight, eps)
