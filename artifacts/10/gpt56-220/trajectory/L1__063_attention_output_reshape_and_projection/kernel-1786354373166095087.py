import torch


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    x = attn_output.permute(0, 2, 1, 3).contiguous()
    x = x.view(bsz * seq_len, num_heads * v_head_dim)
    return torch.mm(x, o_proj_weight.t()).view(bsz, seq_len, o_proj_weight.shape[0])
