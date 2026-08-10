import torch
import triton
import triton.language as tl


@triton.jit
def _norm_mod_kernel(x, mod, out, n_rows: tl.constexpr, seq_len: tl.constexpr,
                     width: tl.constexpr, eps: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    vals = tl.load(x + row * width + cols, mask=mask, other=0.0)
    mean = tl.sum(vals, axis=0) / width
    centered = tl.where(mask, vals - mean, 0.0)
    var = tl.sum(vals * vals, axis=0) / width - mean * mean
    norm = centered * tl.rsqrt(var + eps)
    batch = row // seq_len
    scale = tl.load(mod + batch * (2 * width) + cols, mask=mask)
    shift = tl.load(mod + batch * (2 * width) + width + cols, mask=mask)
    tl.store(out + row * width + cols, norm * (1.0 + scale) + shift,
             mask=mask)


def _triton_impl(hidden_states, temb, linear_weight, linear_bias, eps):
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    out = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0] * hidden_states.shape[1]
    _norm_mod_kernel[(rows,)](
        hidden_states, modulation, out, rows, hidden_states.shape[1],
        hidden_states.shape[2], eps, BLOCK=4096, num_warps=2, waves_per_eu=4,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    eps: float,
):
    """
    Adaptive Layer Normalization with continuous modulation.
    
    1. Compute mean and variance across the feature dimension
    2. Normalize: (x - mean) / sqrt(variance + eps)
    3. Apply learned linear projection to temb to get scale and shift
    4. Modulate: normalized * (1 + scale) + shift
    
    Args:
        hidden_states: (batch, seq_len, inner_dim) - Input features to normalize
        temb: (batch, inner_dim) - Timestep conditioning embeddings
        linear_weight: (inner_dim*2, inner_dim) - Weight for temb projection
        linear_bias: (inner_dim*2,) - Bias for temb projection
        eps: Epsilon for numerical stability
    
    Returns:
        (batch, seq_len, inner_dim) - Normalized and modulated features
    """
    return _triton_impl(hidden_states, temb, linear_weight, linear_bias, eps)
