import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _transpose_kernel(
    src_ptr, dst_ptr,
    B, NH, S,
    VHD: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_NH: tl.constexpr,
):
    pid_nh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_b = tl.program_id(2)
    offs_nh = pid_nh * BLOCK_NH + tl.arange(0, BLOCK_NH)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_v = tl.arange(0, VHD)
    src_ptrs = (
        src_ptr
        + pid_b * NH * S * VHD
        + offs_nh[:, None, None] * S * VHD
        + offs_s[None, :, None] * VHD
        + offs_v[None, None, :]
    )
    mask_src = (offs_nh[:, None, None] < NH) & (offs_s[None, :, None] < S)
    val = tl.load(src_ptrs, mask=mask_src, other=0.0)
    val_t = tl.permute(val, (1, 0, 2))
    dst_ptrs = (
        dst_ptr
        + pid_b * S * NH * VHD
        + offs_s[:, None, None] * NH * VHD
        + offs_nh[None, :, None] * VHD
        + offs_v[None, None, :]
    )
    mask_dst = (offs_s[:, None, None] < S) & (offs_nh[None, :, None] < NH)
    tl.store(dst_ptrs, val_t, mask=mask_dst)


@torch.no_grad()
def run(attn_output: torch.Tensor, o_proj_weight: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, seq_len, v_head_dim = attn_output.shape
    intermediate_size = num_heads * v_head_dim
    dst = torch.empty(
        bsz, seq_len, intermediate_size,
        dtype=attn_output.dtype, device=attn_output.device,
    )
    BLOCK_S = 16
    BLOCK_NH = 8
    grid = (
        triton.cdiv(num_heads, BLOCK_NH),
        triton.cdiv(seq_len, BLOCK_S),
        bsz,
    )
    _transpose_kernel[grid](
        attn_output, dst, bsz, num_heads, seq_len,
        VHD=v_head_dim, BLOCK_S=BLOCK_S, BLOCK_NH=BLOCK_NH,
    )
    return F.linear(dst, o_proj_weight)
