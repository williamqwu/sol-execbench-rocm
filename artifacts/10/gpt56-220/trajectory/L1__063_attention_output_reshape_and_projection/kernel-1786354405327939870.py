import torch
import triton
import triton.language as tl


@triton.jit
def _reorder(src, dst, M: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
             D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    b = row // S
    s = row - b * S
    h = cols // D
    d = cols - h * D
    src_offsets = ((b * (K // D) + h) * S + s) * D + d
    tl.store(dst + row * K + cols, tl.load(src + src_offsets, mask=mask), mask=mask)


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    m = bsz * seq_len
    k = num_heads * v_head_dim
    x = torch.empty((m, k), device=attn_output.device, dtype=attn_output.dtype)
    _reorder[(m, triton.cdiv(k, 256))](
        attn_output, x, M=m, S=seq_len, K=k, D=v_head_dim, BLOCK=256
    )
    return torch.mm(x, o_proj_weight.t()).view(bsz, seq_len, o_proj_weight.shape[0])
