import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _grn_apply_kernel(x_ptr, nf_ptr, gw_ptr, gb_ptr, out_ptr,
                      HWC, hidden, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    idx = offs
    b = idx // HWC
    chan = idx % hidden
    nf = tl.load(nf_ptr + b * hidden + chan)
    gw_v = tl.load(gw_ptr + chan)
    gb_v = tl.load(gb_ptr + chan)
    x = tl.load(x_ptr + offs)
    # exact order matching reference: gw * (x * nf) + gb + x
    t = x * nf
    t = gw_v * t
    t = gb_v + t
    t = t + x
    tl.store(out_ptr + offs, t)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
):
    # Expansion linear: (B, H, W, dim) -> (B, H, W, hidden_dim)
    x = F.linear(hidden_states, pwconv1_weight, pwconv1_bias)

    # GELU activation
    x = F.gelu(x)

    # Global Response Normalization (GRN)
    global_features = torch.linalg.vector_norm(x, ord=2, dim=(1, 2), keepdim=True)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)

    # Fused apply: grn_weight * (x * norm_features) + grn_bias + x
    B, H, W, C = x.shape
    nf_flat = norm_features.view(B, C).contiguous()
    gw_flat = grn_weight.view(-1).contiguous()
    gb_flat = grn_bias.view(-1).contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    HWC = H * W * C
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _grn_apply_kernel[grid](x, nf_flat, gw_flat, gb_flat, out, HWC, C, BLOCK=BLOCK)

    # Projection linear: (B, H, W, hidden_dim) -> (B, H, W, dim)
    output = F.linear(out, pwconv2_weight, pwconv2_bias)
    return output
