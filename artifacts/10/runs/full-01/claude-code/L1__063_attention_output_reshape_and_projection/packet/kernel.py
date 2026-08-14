import torch


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    intermediate_size = num_heads * v_head_dim
    a = attn_output.transpose(1, 2).reshape(bsz, seq_len, intermediate_size)
    return torch.matmul(a, o_proj_weight.t())
