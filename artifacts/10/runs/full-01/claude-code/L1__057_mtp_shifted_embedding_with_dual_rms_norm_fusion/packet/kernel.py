import torch
import triton
import triton.language as tl


@triton.jit
def _norm_concat_kernel(
    ids_ptr,          # int64 (B, S)
    hs_ptr,           # bf16  (B, S, H)
    emb_ptr,          # bf16  (V, H)
    ew_ptr,           # bf16  (H,)
    hw_ptr,           # bf16  (H,)
    out_ptr,          # bf16  (M, 2H)
    S,                # seq len
    eps,
    H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = row // S
    s = row - b * S
    not_last = s < (S - 1)
    # shifted (roll by -1) token id; last position of each sequence -> 0
    off = tl.where(not_last, row + 1, 0)
    idx = tl.load(ids_ptr + off)
    idx = tl.where(not_last, idx, 0).to(tl.int64)

    cols = tl.arange(0, H)

    e = tl.load(emb_ptr + idx * H + cols).to(tl.float32)
    h = tl.load(hs_ptr + row * H + cols).to(tl.float32)

    ve = tl.sum(e * e, axis=0) / H
    vh = tl.sum(h * h, axis=0) / H

    ne = (e * tl.rsqrt(ve + eps)).to(tl.bfloat16).to(tl.float32)
    nh = (h * tl.rsqrt(vh + eps)).to(tl.bfloat16).to(tl.float32)

    ew = tl.load(ew_ptr + cols).to(tl.float32)
    hw = tl.load(hw_ptr + cols).to(tl.float32)

    base = out_ptr + row * (2 * H)
    tl.store(base + cols, (ne * ew).to(tl.bfloat16))
    tl.store(base + H + cols, (nh * hw).to(tl.bfloat16))


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
    B, S = input_ids.shape
    H = hidden_states.shape[-1]
    M = B * S

    concat = torch.empty((M, 2 * H), dtype=torch.bfloat16, device=hidden_states.device)

    _norm_concat_kernel[(M,)](
        input_ids,
        hidden_states,
        word_embeddings,
        enorm_weight,
        hnorm_weight,
        concat,
        S,
        rms_norm_eps,
        H=H,
        num_warps=8,
        num_stages=1,
    )

    out = torch.mm(concat, eh_proj_weight.t())
    return out.view(B, S, H)
