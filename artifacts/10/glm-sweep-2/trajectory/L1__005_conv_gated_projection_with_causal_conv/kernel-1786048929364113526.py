import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_gate_conv_gate_kernel(
    B_ptr, C_ptr, XP_ptr, W_ptr, BIAS_ptr, Y_ptr,
    sb, ss, sh,
    S,
    K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    bias = tl.load(BIAS_ptr + pid_h).to(tl.float32)
    base = pid_b * sb + pid_h * sh

    s_start = pid_s * BLOCK_S
    offs = tl.arange(0, BLOCK_S)
    s_idx = s_start + offs
    mask = s_idx < S

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for k in tl.static_range(K):
        src_idx = s_idx - k
        smask = mask & (src_idx >= 0)
        sc = tl.where(smask, src_idx, 0)
        b_val = tl.load(B_ptr + base + sc * ss, mask=smask, other=0.0).to(tl.float32)
        xp_val = tl.load(XP_ptr + base + sc * ss, mask=smask, other=0.0).to(tl.float32)
        bx = b_val * xp_val
        wk = tl.load(W_ptr + pid_h * K + (K - 1 - k)).to(tl.float32)
        acc += wk * bx
    acc = acc + bias
    c_val = tl.load(C_ptr + base + s_idx * ss, mask=mask, other=0.0).to(tl.float32)
    y = acc * c_val
    tl.store(Y_ptr + base + s_idx * ss, y.to(Y_ptr.dtype.element_ty), mask=mask)


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]

    # Step 1: Triple linear projection -> [B, S, 3H] (contiguous)
    BCx = F.linear(x, in_proj_weight, in_proj_bias)
    B, C, x_proj = BCx.chunk(3, dim=-1)
    B = B.contiguous()
    C = C.contiguous()
    x_proj = x_proj.contiguous()

    # Steps 2-4 fused: gate(B,x_proj) -> causal depthwise conv -> gate(C) -> [B, S, H]
    Y = torch.empty_like(B)
    w = conv_weight.squeeze(1).contiguous()
    cb = conv_bias.contiguous()
    BLOCK_S = 1024
    grid = (batch_size, hidden_size, triton.cdiv(seq_len, BLOCK_S))
    _fused_gate_conv_gate_kernel[grid](
        B, C, x_proj, w, cb, Y,
        B.stride(0), B.stride(1), B.stride(2),
        seq_len, K=conv_kernel_size, BLOCK_S=BLOCK_S,
    )

    # Step 5: Final output projection
    output = F.linear(Y, out_proj_weight, out_proj_bias)
    return output
