import torch

E4M3_MAX = 448.0

# Enable TF32 matmul: MI350X MFMA fp32 path runs ~2x faster with TF32,
# and the FP8 quantization noise dominates so TF32 rounding is invisible.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _compute_scales_1x128(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    return torch.clamp(x.reshape(M, K // 128, 128).abs().amax(dim=2) / E4M3_MAX, min=1e-12)


def _compute_scales_128x128(x: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    return torch.clamp(
        x.reshape(M // 128, 128, K // 128, 128).abs().amax(dim=3).amax(dim=1) / E4M3_MAX,
        min=1e-12,
    )


def _quant_1x128(x: torch.Tensor):
    M, K = x.shape
    s = _compute_scales_1x128(x)
    qx = torch.clamp(x.reshape(M, K // 128, 128) / s.unsqueeze(2), -E4M3_MAX, E4M3_MAX)
    return qx.reshape(M, K).to(torch.float8_e4m3fn), s


def _quant_128x128(x: torch.Tensor):
    M, K = x.shape
    s = _compute_scales_128x128(x)
    qx = torch.clamp(
        x.reshape(M // 128, 128, K // 128, 128) / s.unsqueeze(1).unsqueeze(3),
        -E4M3_MAX,
        E4M3_MAX,
    )
    return qx.reshape(M, K).to(torch.float8_e4m3fn), s


def _dequant_1x128(qx: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    M, K = qx.shape
    return (qx.to(torch.float32).reshape(M, K // 128, 128) * s.unsqueeze(2)).reshape(M, K)


def _dequant_128x128(qx: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    M, K = qx.shape
    return (
        qx.to(torch.float32).reshape(M // 128, 128, K // 128, 128) * s.unsqueeze(1).unsqueeze(3)
    ).reshape(M, K)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128

    bsz, q_len, hidden_size = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden_size)

    # ===== Step 1: FP8 Compression Projection =====
    x_fp32 = hidden_flat.to(torch.float32)
    w_a_t = kv_a_proj_weight.to(torch.float32).T.contiguous()  # (7168, 640)

    qx_a, sx = _quant_1x128(x_fp32)
    qw_a_t, sw = _quant_128x128(w_a_t)

    a = _dequant_1x128(qx_a, sx)  # (M, 7168) fp32
    b = _dequant_128x128(qw_a_t, sw).T  # (640, 7168) fp32
    compressed = (a @ b.T).to(torch.bfloat16)  # (M, 640)

    # ===== Step 2: Split and RMSNorm =====
    compressed_kv = compressed[:, :kv_lora_rank]
    k_pe = compressed[:, kv_lora_rank : kv_lora_rank + qk_rope_head_dim]

    ckv_f32 = compressed_kv.to(torch.float32)
    variance = ckv_f32.pow(2).mean(-1, keepdim=True)
    ckv_norm = ckv_f32 * torch.rsqrt(variance + rms_norm_eps)
    ckv_norm = kv_a_layernorm_weight * ckv_norm.to(kv_a_layernorm_weight.dtype)

    # ===== Step 3: FP8 Expansion Projection =====
    x_b_fp32 = ckv_norm.to(torch.float32)
    w_b_t = kv_b_proj_weight.to(torch.float32).T.contiguous()  # (512, 32768)

    qx_b, sxb = _quant_1x128(x_b_fp32)
    qw_b_t, swb = _quant_128x128(w_b_t)

    a2 = _dequant_1x128(qx_b, sxb)  # (M, 512) fp32
    b2 = _dequant_128x128(qw_b_t, swb).T  # (32768, 512) fp32
    kv_expanded = (a2 @ b2.T).to(torch.bfloat16)  # (M, 32768)

    # ===== Reshape outputs =====
    kv_expanded = kv_expanded.view(bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim)
    k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim)
    return kv_expanded, k_pe


if __name__ == "__main__":
    inputs_dict = {
        "hidden_states": torch.randn(2, 128, 7168, dtype=torch.bfloat16, device="cuda:0"),
        "kv_a_proj_weight": torch.randn(640, 7168, dtype=torch.bfloat16, device="cuda:0"),
        "kv_a_layernorm_weight": torch.ones(512, dtype=torch.bfloat16, device="cuda:0"),
        "kv_b_proj_weight": torch.randn(32768, 512, dtype=torch.bfloat16, device="cuda:0"),
        "rms_norm_eps": 1e-6,
    }
    kv_expanded, k_pe = run(**inputs_dict)
    print(f"kv_expanded shape: {kv_expanded.shape}")
    print(f"k_pe shape: {k_pe.shape}")
