import torch
import torch.nn.functional as F

@torch.no_grad()
@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def _run_compiled(
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    num_heads = 128
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    q_head_dim = 192
    kv_lora_rank = 512

    bsz, seq_len, _ = hidden_states.shape

    q_latent = F.linear(hidden_states, q_a_proj_weight)

    input_dtype = q_latent.dtype
    q_latent_fp32 = q_latent.to(torch.float32)
    variance = q_latent_fp32.pow(2).mean(-1, keepdim=True)
    q_latent_normed = q_latent_fp32 * torch.rsqrt(variance + rms_norm_eps)
    q_latent = (q_a_layernorm_weight * q_latent_normed.to(input_dtype))

    q = F.linear(q_latent, q_b_proj_weight)
    q = q.view(bsz, seq_len, num_heads, q_head_dim)

    q_nope = q[..., :qk_nope_head_dim].contiguous()
    q_pe = q[..., qk_nope_head_dim:].contiguous()

    kv_combined = F.linear(hidden_states, kv_a_proj_weight)

    compressed_kv = kv_combined[..., :kv_lora_rank].contiguous()
    k_pe = kv_combined[..., kv_lora_rank:].contiguous()
    k_pe = k_pe.view(bsz, seq_len, 1, qk_rope_head_dim)

    return q_nope, q_pe, compressed_kv, k_pe


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    return _run_compiled(hidden_states, q_a_proj_weight, q_a_layernorm_weight,
                         q_b_proj_weight, kv_a_proj_weight, rms_norm_eps)
