import torch
import triton
import triton.language as tl
import aiter


aiter.hipb_create_extension()
_HIPB_SOLUTIONS = {512: 439118, 1024: 438976, 2048: 439189}


@triton.jit
def _prepare_kernel(
    input_ids,
    hidden_states,
    word_embeddings,
    enorm_weight,
    hnorm_weight,
    prepared,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    eps: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    which = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    if which == 0:
        seq_pos = row % seq_len
        next_token = tl.load(input_ids + row + 1, mask=seq_pos + 1 < seq_len, other=0)
        token = next_token
        x = tl.load(word_embeddings + token * H + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(enorm_weight + cols, mask=mask, other=0.0)
        out_off = cols
    else:
        x = tl.load(hidden_states + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(hnorm_weight + cols, mask=mask, other=0.0)
        out_off = H + cols

    variance = tl.sum(x * x, axis=0) * (1.0 / H)
    normalized_bf16 = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    result = normalized_bf16 * weight
    tl.store(prepared + row * (2 * H) + out_off, result, mask=mask)


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
    batch_size, seq_len, hidden_size = hidden_states.shape
    n_rows = batch_size * seq_len
    prepared = torch.empty(
        (n_rows, 2 * hidden_size), device=hidden_states.device, dtype=torch.bfloat16
    )
    _prepare_kernel[(n_rows, 2)](
        input_ids,
        hidden_states,
        word_embeddings,
        enorm_weight,
        hnorm_weight,
        prepared,
        n_rows=n_rows,
        seq_len=seq_len,
        eps=rms_norm_eps,
        H=hidden_size,
        BLOCK=4096,
        num_warps=8,
    )
    if n_rows in _HIPB_SOLUTIONS:
        output = aiter.hipb_mm(
            prepared, eh_proj_weight.t(), _HIPB_SOLUTIONS[n_rows]
        )
    elif 4096 < n_rows < 8192:
        # hipBLASLt selects a much slower kernel for a slightly ragged M.
        # Keep the large aligned prefix on its fast path and project the tail.
        output = torch.empty(
            (n_rows, hidden_size),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )
        torch.mm(prepared[:4096], eh_proj_weight.t(), out=output[:4096])
        torch.mm(prepared[4096:], eh_proj_weight.t(), out=output[4096:])
    else:
        output = torch.mm(prepared, eh_proj_weight.t())
    return output.reshape(batch_size, seq_len, hidden_size)
