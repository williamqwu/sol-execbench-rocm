import torch
import torch.nn.functional as F

_compiled = None

def _get_compiled():
    global _compiled
    if _compiled is None:
        def _fn(attn_output, o_proj_weight):
            bsz, num_heads, seq_len, v_head_dim = attn_output.shape
            intermediate_size = num_heads * v_head_dim
            x = attn_output.transpose(1, 2).reshape(bsz, seq_len, intermediate_size)
            return F.linear(x, o_proj_weight)
        _compiled = torch.compile(_fn, mode="max-autotune-no-cudagraphs")
    return _compiled

@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    return _get_compiled()(attn_output, o_proj_weight)
