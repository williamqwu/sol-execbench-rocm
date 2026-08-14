import torch
import triton
import triton.language as tl


_INNER_DIM = 2432


@triton.jit
def _layer_norm_modulate(
    x_ptr,
    modulation_ptr,
    out_ptr,
    eps,
    seq_len: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    batch = tl.program_id(1)
    row = batch * seq_len + seq
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    normalized = centered / tl.sqrt(variance + eps)

    mod_base = batch * (2 * n_cols)
    scale = tl.load(modulation_ptr + mod_base + cols, mask=mask)
    shift = tl.load(modulation_ptr + mod_base + n_cols + cols, mask=mask)
    result = normalized * (1.0 + scale) + shift
    tl.store(out_ptr + row * n_cols + cols, result, mask=mask)


def run(hidden_states, temb, linear_weight, linear_bias, eps):
    modulation = torch.nn.functional.linear(temb, linear_weight, linear_bias)
    output = torch.empty_like(hidden_states)
    batch, seq_len, inner_dim = hidden_states.shape
    assert inner_dim == _INNER_DIM
    _layer_norm_modulate[(seq_len, batch)](
        hidden_states,
        modulation,
        output,
        eps,
        seq_len=seq_len,
        n_cols=inner_dim,
        BLOCK=4096,
        num_warps=4,
        num_stages=5,
        waves_per_eu=8,
    )
    return output
