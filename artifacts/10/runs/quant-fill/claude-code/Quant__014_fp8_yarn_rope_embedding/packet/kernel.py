import torch
import math


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """Find dimension for correction based on number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dimension range bounds based on rotations."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1):
    """Compute mscale factor for YaRN."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim, device):
    """Create linear ramp mask for frequency interpolation."""
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


E4M3_MAX = 448.0


@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    mscale: float,
    mscale_all_dim: float,
):
    """
    Compute FP8-quantized YaRN rotary position embeddings.

    This implements YaRN (Yet another RoPE extensioN method) with FP8 quantization
    for memory bandwidth reduction. The computation involves:
    1. Computing base inverse frequencies
    2. Applying YaRN interpolation with correction ranges
    3. FP8 quantization of frequency computations
    4. Computing cos/sin embeddings with mscale

    Args:
        position_ids: Position indices (seq_len,)
        mscale: Mscale factor for YaRN
        mscale_all_dim: Mscale all dimension factor

    Returns:
        Tuple of (cos_emb, sin_emb) tensors for rotary embedding application
    """
    device = position_ids.device
    seq_len = position_ids.shape[0]

    # Constants from definition
    dim = 64  # qk_rope_head_dim
    base = 10000  # rope_theta
    scaling_factor = 40
    original_max_position_embeddings = 4096
    beta_fast = 32
    beta_slow = 1
    block_size_k = 128

    # Compute base frequencies
    arange_vals = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    freq_extra = 1.0 / (base ** (arange_vals / dim))
    freq_inter = 1.0 / (scaling_factor * base ** (arange_vals / dim))

    # Find correction range for YaRN interpolation
    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        base,
        original_max_position_embeddings,
    )

    # Create interpolation mask
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2, device)

    # Interpolate frequencies using YaRN method
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    # Create position tensor from position_ids
    t = position_ids.to(torch.float32)

    # Compute frequencies via outer product
    freqs = torch.outer(t, inv_freq)  # (seq_len, dim//2)

    # Pad to multiple of 128 for blockwise FP8 scaling
    half_dim = dim // 2
    pad_size = (block_size_k - (half_dim % block_size_k)) % block_size_k
    if pad_size > 0:
        freqs_padded = torch.nn.functional.pad(freqs, (0, pad_size), value=0.0)
    else:
        freqs_padded = freqs

    # FP8 quantization and dequantization with BlockWise1x128
    M, K = freqs_padded.shape
    block_size_m = 1

    # Reshape to blocks
    new_shape = (
        M // block_size_m,
        block_size_m,
        K // block_size_k,
        block_size_k,
    )
    tensor_blocked = freqs_padded.reshape(new_shape)

    # Compute max over block dimensions (dims 1 and 3)
    block_max = tensor_blocked.abs().amax(dim=3, keepdim=True).amax(dim=1, keepdim=True)

    # Compute inverse scales
    scales = block_max / E4M3_MAX
    scales = torch.clamp(scales, min=1e-12)

    # Apply scaling (quantization)
    freqs_scaled = tensor_blocked / scales
    freqs_scaled = torch.clamp(freqs_scaled, min=-E4M3_MAX, max=E4M3_MAX)

    # Convert to FP8 and back (dequantization)
    freqs_fp8 = freqs_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    freqs_dequant = freqs_fp8.to(torch.float32).reshape(new_shape)

    # Apply inverse scaling
    freqs_result = (freqs_dequant * scales).reshape(M, K)

    # Remove padding
    freqs_final = freqs_result[:, :half_dim]

    # Compute mscale factor
    _mscale = float(
        yarn_get_mscale(scaling_factor, mscale)
        / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )

    # Concatenate for full embedding dimension
    emb = torch.cat((freqs_final, freqs_final), dim=-1)  # (seq_len, dim)

    # Compute cos/sin with mscale
    cos_emb = (emb.cos() * _mscale).to(torch.bfloat16)
    sin_emb = (emb.sin() * _mscale).to(torch.bfloat16)

    return cos_emb, sin_emb
