import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dual_rmsnorm_kernel(
    embed_ptr, hidden_ptr, out_ptr,
    enorm_w_ptr, hnorm_w_ptr,
    eps,
    n_rows, H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < H

    # Embedding RMSNorm -> out[row, 0:H]
    e_ptr = embed_ptr + row * H
    e = tl.load(e_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    e_var = tl.sum(e * e, axis=0) / H
    e_rinv = tl.rsqrt(e_var + eps)
    e_w = tl.load(enorm_w_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    e_normed = (e * e_rinv * e_w).to(tl.bfloat16)
    tl.store(out_ptr + row * (2 * H) + col_offs, e_normed, mask=mask)

    # Hidden RMSNorm -> out[row, H:2H]
    h_ptr = hidden_ptr + row * H
    hv = tl.load(h_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    h_var = tl.sum(hv * hv, axis=0) / H
    h_rinv = tl.rsqrt(h_var + eps)
    h_w = tl.load(hnorm_w_ptr + col_offs, mask=mask, other=0.0).to(tl.float32)
    h_normed = (hv * h_rinv * h_w).to(tl.bfloat16)
    tl.store(out_ptr + row * (2 * H) + H + col_offs, h_normed, mask=mask)


def _fused_dual_rmsnorm(embed, hidden, enorm_w, hnorm_w, eps):
    bs, seq, H = embed.shape
    n_rows = bs * seq
    out = torch.empty(bs, seq, 2 * H, dtype=torch.bfloat16, device=embed.device)
    BLOCK = triton.next_power_of_2(H)
    grid = (n_rows,)
    _fused_dual_rmsnorm_kernel[grid](
        embed, hidden, out, enorm_w, hnorm_w, eps, n_rows, H=H, BLOCK=BLOCK
    )
    return out


@torch.no_grad()
def run(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    word_embeddings: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eh_proj_weight: torch.Tensor,
    rms_norm_eps: float,
):
    # Shift input_ids left by 1 (next-token prediction); last position -> 0.
    shifted_input_ids = torch.empty_like(input_ids)
    shifted_input_ids[:, :-1] = input_ids[:, 1:]
    shifted_input_ids[:, -1] = 0

    # Embed shifted ids
    input_embeds = word_embeddings[shifted_input_ids]

    # Fused dual RMSNorm writing directly into a (..., 2*H) concat buffer.
    concat = _fused_dual_rmsnorm(
        input_embeds, hidden_states, enorm_weight, hnorm_weight, rms_norm_eps
    )

    # Project back to hidden_size
    fused_hidden_states = torch.matmul(concat, eh_proj_weight.t())
    return fused_hidden_states
