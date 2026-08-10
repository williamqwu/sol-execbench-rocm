import torch


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    x = attn_output.transpose(1, 2).reshape(
        bsz * seq_len, num_heads * v_head_dim
    )
    return torch._C._nn.linear(x, o_proj_weight).view(
        bsz, seq_len, o_proj_weight.shape[0])
