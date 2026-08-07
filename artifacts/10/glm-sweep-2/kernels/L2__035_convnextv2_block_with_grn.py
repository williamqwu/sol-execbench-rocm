import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _grn_apply_kernel(gel_ptr, nf_ptr, gw_ptr, gb_ptr, out_ptr,
                      n_total, HWC4, C4, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_total
    b = offs // HWC4
    rem = offs % HWC4
    c = rem % C4
    nf_idx = b * C4 + c
    g = tl.load(gel_ptr + offs, mask=mask, other=0.0)
    nf = tl.load(nf_ptr + nf_idx, mask=mask, other=0.0)
    gw = tl.load(gw_ptr + c, mask=mask, other=0.0)
    gb = tl.load(gb_ptr + c, mask=mask, other=0.0)
    # exact match to eager: grn_weight * (out * norm_features) + grn_bias + out
    res = gw * (g * nf) + gb + g
    tl.store(out_ptr + offs, res, mask=mask)


def _grn_apply(gel, norm_features, grn_weight, grn_bias):
    B = gel.shape[0]
    C4 = gel.shape[-1]
    HW = gel.shape[1] * gel.shape[2]
    gel = gel.contiguous()
    nf_flat = norm_features.reshape(B, C4).contiguous()
    gw_flat = grn_weight.reshape(-1).contiguous()
    gb_flat = grn_bias.reshape(-1).contiguous()
    out = torch.empty_like(gel)
    n_total = gel.numel()
    HWC4 = HW * C4
    BLOCK = 1024
    grid = (triton.cdiv(n_total, BLOCK),)
    _grn_apply_kernel[grid](gel, nf_flat, gw_flat, gb_flat, out,
                            n_total, HWC4, C4, BLOCK=BLOCK, enable_fp_fusion=False)
    return out


@torch.no_grad()
def run(
    x: torch.Tensor,
    dwconv_weight: torch.Tensor,
    dwconv_bias: torch.Tensor,
    layernorm_weight: torch.Tensor,
    layernorm_bias: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    pwconv1_bias: torch.Tensor,
    grn_weight: torch.Tensor,
    grn_bias: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    pwconv2_bias: torch.Tensor,
    eps: float,
    layer_norm_eps: float,
):
    residual = x
    B, C, H, W = x.shape

    out = F.conv2d(x, dwconv_weight, dwconv_bias, padding=3, groups=C)
    out = out.permute(0, 2, 3, 1)
    out = F.layer_norm(out, (C,), layernorm_weight, layernorm_bias, eps=layer_norm_eps)
    out = torch.matmul(out, pwconv1_weight.T) + pwconv1_bias
    out = F.gelu(out)

    global_features = torch.linalg.vector_norm(out, ord=2, dim=(1, 2), keepdim=True)
    norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + eps)

    out = _grn_apply(out, norm_features, grn_weight, grn_bias)

    out = torch.matmul(out, pwconv2_weight.T) + pwconv2_bias
    out = out.permute(0, 3, 1, 2)
    out = residual + out
    return out
