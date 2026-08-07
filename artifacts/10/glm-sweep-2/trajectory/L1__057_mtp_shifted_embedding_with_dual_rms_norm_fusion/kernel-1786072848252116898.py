import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_dualnorm_kernel(
    input_ids_ptr, hidden_ptr, word_emb_ptr, out_ptr,
    enorm_w_ptr, hnorm_w_ptr,
    eps, seq_len,
    n_rows, H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    batch = row // seq_len
    pos = row % seq_len
    # Shift left by 1: shifted[b,p] = input_ids[b, p+1] for p < seq-1, else 0
    if pos < seq_len - 1:
        tok = tl.load(input_ids_ptr + batch * seq_len + (pos + 1))
    else:
        tok = tl.full([], 0, tl.int64)

    col_offs = tl.arange(0, BLOCK)
    mask = col_offs < H

    # Embedding RMSNorm -> out[row, 0:H]
    e = tl.load(word_emb_ptr + tok * H + col_offs, mask=mask, other=0.0).to(tl.float32)
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
    bs, seq, H = hidden_states.shape
    n_rows = bs * seq
    concat = torch.empty(bs, seq, 2 * H, dtype=torch.bfloat16, device=hidden_states.device)
    BLOCK = triton.next_power_of_2(H)
    grid = (n_rows,)
    _fused_gather_dualnorm_kernel[grid](
        input_ids, hidden_states, word_embeddings, concat,
        enorm_weight, hnorm_weight, rms_norm_eps, seq, n_rows, H=H, BLOCK=BLOCK,
    )
    fused_hidden_states = torch.matmul(concat, eh_proj_weight.t())
    return fused_hidden_states
