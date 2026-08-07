import torch
import torch.nn.functional as F


def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    scaling: float,
):
    """
    Backward pass for GQA with RoPE.

    Instead of manually materializing the [B,H,S,S] attention matrix and
    writing the backward by hand, we rebuild the forward graph and let
    torch.autograd compute all gradients.  The attention uses
    F.scaled_dot_product_attention (flash / memory-efficient) so the score
    matrix is never materialised, and the Q/K/V / O projection gradients fall
    out of the autodiff'd GEMMs.
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 128
    device = hidden_states.device
    original_dtype = hidden_states.dtype

    # Leaf tensors that require grad.
    hs = hidden_states.detach().clone().requires_grad_(True)
    qw = q_weight.detach().clone().requires_grad_(True)
    kw = k_weight.detach().clone().requires_grad_(True)
    vw = v_weight.detach().clone().requires_grad_(True)
    ow = o_weight.detach().clone().requires_grad_(True)

    # ---- forward recompute ----
    query_states = F.linear(hs, qw).view(batch_size, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    key_states = F.linear(hs, kw).view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = F.linear(hs, vw).view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    # RoPE
    position_ids = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0)
    inv_freq_expanded = inv_freq[None, :, None].float()
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    cos_unsqueezed = cos.unsqueeze(1)
    sin_unsqueezed = sin.unsqueeze(1)
    cos_interleaved = cos_unsqueezed[..., : cos_unsqueezed.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin_interleaved = sin_unsqueezed[..., : sin_unsqueezed.shape[-1] // 2].repeat_interleave(2, dim=-1)

    q_float = query_states.float()
    k_float = key_states.float()

    q1 = q_float[..., 0::2]
    q2 = q_float[..., 1::2]
    q_rotated = torch.stack((-q2, q1), dim=-1).flatten(-2)
    k1 = k_float[..., 0::2]
    k2 = k_float[..., 1::2]
    k_rotated = torch.stack((-k2, k1), dim=-1).flatten(-2)

    query_states_rope = (q_float * cos_interleaved) + (q_rotated * sin_interleaved)
    key_states_rope = (k_float * cos_interleaved) + (k_rotated * sin_interleaved)

    q_bf16 = query_states_rope.to(original_dtype)
    k_bf16 = key_states_rope.to(original_dtype)
    v_bf16 = value_states.to(original_dtype)

    attn_output = F.scaled_dot_product_attention(
        q_bf16, k_bf16, v_bf16, is_causal=True, scale=scaling, enable_gqa=True
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_size)

    output = F.linear(attn_output, ow)

    # ---- backward via autograd ----
    output.backward(grad_output)

    return (
        hs.grad.to(original_dtype),
        qw.grad.to(original_dtype),
        kw.grad.to(original_dtype),
        vw.grad.to(original_dtype),
        ow.grad.to(original_dtype),
    )
