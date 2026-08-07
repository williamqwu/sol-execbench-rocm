import torch
import triton
import triton.language as tl


@triton.jit
def _sinusoidal_emb_kernel(
    timestep_ptr, freqs_ptr, out_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    HALF: tl.constexpr = N // 2
    ts = tl.load(timestep_ptr + offs_m, mask=mask_m, other=0.0) * 1000.0
    is_first_half = offs_n < HALF
    freq_idx = tl.where(is_first_half, offs_n, offs_n - HALF)
    mask_n = freq_idx >= 0
    freqs = tl.load(freqs_ptr + freq_idx, mask=mask_n, other=0.0)
    args = ts[:, None] * freqs[None, :]
    cos_v = tl.cos(args)
    sin_v = tl.sin(args)
    val = tl.where(is_first_half[None, :], cos_v, sin_v)
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], val, mask=mask)


def _fused_sinusoidal(timestep, freqs, time_embed_dim):
    bs = timestep.shape[0]
    out = torch.empty(bs, time_embed_dim, dtype=torch.float32, device=timestep.device)
    BLOCK_M = 16
    BLOCK_N = 64
    grid = (triton.cdiv(bs, BLOCK_M), triton.cdiv(time_embed_dim, BLOCK_N))
    _sinusoidal_emb_kernel[grid](timestep, freqs, out, bs, time_embed_dim, BLOCK_M, BLOCK_N)
    return out


@torch.no_grad()
def run(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
):
    time_embed_dim = timestep_linear1_weight.shape[1]
    emb = _fused_sinusoidal(timestep, freqs, time_embed_dim)

    x = torch.nn.functional.linear(emb, timestep_linear1_weight, timestep_linear1_bias)
    x = x * torch.sigmoid(x)

    text_embed = torch.nn.functional.linear(
        pooled_projections, text_embedder_weight, text_embedder_bias
    )
    conditioning = torch.addmm(text_embed, x, timestep_linear2_weight.t())
    conditioning = conditioning + timestep_linear2_bias
    return conditioning
