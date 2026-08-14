import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, out_ptr, eps: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * 512 + cols).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * (1.0 / 512.0)
    w = tl.load(w_ptr + cols).to(tl.float32)
    out = x * tl.rsqrt(variance + eps) * w
    tl.store(out_ptr + row * 512 + cols, out)


@torch.no_grad()
def run(compressed_kv, kv_a_layernorm_weight, kv_b_proj_weight, eps):
    rows = compressed_kv.numel() // 512
    normalized = torch.empty_like(compressed_kv)
    _rms_norm_kernel[(rows,)](
        compressed_kv, kv_a_layernorm_weight, normalized, eps,
        BLOCK=512, num_warps=1,
    )
    expanded = torch.mm(
        normalized.view(rows, 512), kv_b_proj_weight.t()
    )
    bsz, seq_len, _ = compressed_kv.shape
    kv = expanded.view(bsz, seq_len, 128, 256).transpose(1, 2)
    return kv[..., :128], kv[..., 128:]
