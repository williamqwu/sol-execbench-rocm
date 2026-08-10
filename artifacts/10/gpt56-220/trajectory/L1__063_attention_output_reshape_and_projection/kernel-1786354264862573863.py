import torch


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    weight = o_proj_weight.view(o_proj_weight.shape[0], num_heads, v_head_dim)
    return torch.einsum("bhsv,ohv->bso", attn_output, weight)
