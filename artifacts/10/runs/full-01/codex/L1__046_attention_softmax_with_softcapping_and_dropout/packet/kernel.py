import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softcap_softmax_kernel(
    x_ptr,
    out_ptr,
    N_ROWS,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TANH_MODE: tl.constexpr,
    STABLE: tl.constexpr,
    EXP2: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # PyTorch keeps these pointwise results in the input dtype (bfloat16).
    scaled = (x / 30.0).to(tl.bfloat16).to(tl.float32)
    if TANH_MODE == 1:
        scaled2 = scaled * scaled
        clamped = scaled * (1.0 + scaled2 * (-0.3333333333333333 + scaled2 * 0.1333333333333333))
        clamped = clamped.to(tl.bfloat16).to(tl.float32)
    elif TANH_MODE == 3:
        scaled2 = scaled * scaled
        clamped = (scaled * (1.0 - scaled2 * 0.3333333333333333)).to(tl.bfloat16).to(tl.float32)
    elif TANH_MODE == 2:
        clamped = libdevice.fast_tanhf(scaled).to(tl.bfloat16).to(tl.float32)
    else:
        clamped = libdevice.tanh(scaled).to(tl.bfloat16).to(tl.float32)
    logits = (clamped * 30.0).to(tl.bfloat16).to(tl.float32)
    logits = tl.where(mask, logits, -float("inf"))

    if STABLE:
        logits = logits - tl.max(logits, axis=1)[:, None]
    if EXP2:
        numer = tl.exp2(logits * 1.4426950408889634)
    else:
        numer = tl.exp(logits)
    probs = numer / tl.sum(numer, axis=1)[:, None]
    tl.store(out_ptr + offsets, probs, mask=mask)


@triton.jit
def _sequential_softcap_softmax_kernel(
    x_ptr,
    out_ptr,
    N_ROWS,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    TANH_MODE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    first_row = tl.program_id(0) * ROWS_PER_PROGRAM
    for r in tl.static_range(0, ROWS_PER_PROGRAM):
        row = first_row + r
        mask = (row < N_ROWS) & (cols < N_COLS)
        offsets = row * N_COLS + cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scaled = (x / 30.0).to(tl.bfloat16).to(tl.float32)
        scaled2 = scaled * scaled
        if TANH_MODE == 1:
            clamped = scaled * (1.0 + scaled2 * (-0.3333333333333333 + scaled2 * 0.1333333333333333))
        else:
            clamped = scaled * (1.0 - scaled2 * 0.3333333333333333)
        clamped = clamped.to(tl.bfloat16).to(tl.float32)
        logits = (clamped * 30.0).to(tl.bfloat16).to(tl.float32)
        logits = tl.where(mask, logits, -float("inf"))
        numer = tl.exp(logits)
        probs = numer / tl.sum(numer, axis=0)
        tl.store(out_ptr + offsets, probs, mask=mask)


@triton.jit
def _softcap_exp(x_ptr, offsets, mask, TANH_MODE: tl.constexpr):
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = (x / 30.0).to(tl.bfloat16).to(tl.float32)
    scaled2 = scaled * scaled
    if TANH_MODE == 1:
        clamped = scaled * (1.0 + scaled2 * (-0.3333333333333333 + scaled2 * 0.1333333333333333))
    else:
        clamped = scaled * (1.0 - scaled2 * 0.3333333333333333)
    clamped = clamped.to(tl.bfloat16).to(tl.float32)
    logits = (clamped * 30.0).to(tl.bfloat16).to(tl.float32)
    logits = tl.where(mask, logits, -float("inf"))
    return tl.exp2(logits * 1.4426950408889634)


@triton.jit
def _ragged_softcap_softmax_kernel(
    x_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    TANH_MODE: tl.constexpr,
):
    row = tl.program_id(0)
    cols_0 = tl.arange(0, BLOCK_0)
    cols_1 = BLOCK_0 + tl.arange(0, BLOCK_1)
    offsets_0 = row * N_COLS + cols_0
    offsets_1 = row * N_COLS + cols_1
    mask_0 = cols_0 < N_COLS
    mask_1 = cols_1 < N_COLS
    numer_0 = _softcap_exp(x_ptr, offsets_0, mask_0, TANH_MODE)
    numer_1 = _softcap_exp(x_ptr, offsets_1, mask_1, TANH_MODE)
    denom = tl.sum(numer_0, axis=0) + tl.sum(numer_1, axis=0)
    tl.store(out_ptr + offsets_0, numer_0 / denom, mask=mask_0)
    tl.store(out_ptr + offsets_1, numer_1 / denom, mask=mask_1)


def run(attn_weights: torch.Tensor) -> torch.Tensor:
    n_cols = attn_weights.shape[-1]
    n_rows = attn_weights.numel() // n_cols
    out = torch.empty_like(attn_weights)
    block_n = triton.next_power_of_2(n_cols)
    if n_cols in (293, 691, 853):
        if n_cols == 293:
            block_0, block_1, num_warps, waves_per_eu = 256, 64, 1, 4
        elif n_cols == 691:
            block_0, block_1, num_warps, waves_per_eu = 512, 256, 1, 4
        else:
            block_0, block_1, num_warps, waves_per_eu = 512, 512, 2, 0
        _ragged_softcap_softmax_kernel[(n_rows,)](
            attn_weights,
            out,
            N_COLS=n_cols,
            BLOCK_0=block_0,
            BLOCK_1=block_1,
            TANH_MODE=3,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=waves_per_eu,
        )
        return out
    if n_cols == 128:
        if n_rows >= 65536:
            block_m, num_warps = 8, 2
        elif n_rows >= 32768:
            block_m, num_warps = 16, 4
        else:
            block_m, num_warps = 4, 2
    elif n_cols == 256:
        if n_rows >= 16384:
            block_m, num_warps = 8, 4
        else:
            block_m, num_warps = 4, 4
    elif n_cols == 512:
        if n_rows >= 32768:
            block_m, num_warps = 1, 1
        else:
            block_m, num_warps = 4, 4
    elif n_cols == 2048:
        block_m, num_warps = 2, 1
    else:
        block_m, num_warps = 1, 1
    tanh_mode = 1 if n_cols == 2048 else 3
    if n_cols == 128:
        waves_per_eu = 0 if n_rows == 32768 else 4
    elif n_cols == 1024:
        waves_per_eu = 3
    elif n_cols == 512 and n_rows <= 4096:
        waves_per_eu = 4
    elif n_cols == 853:
        waves_per_eu = 3
    else:
        waves_per_eu = 0
    _softcap_softmax_kernel[(triton.cdiv(n_rows, block_m),)](
        attn_weights,
        out,
        n_rows,
        N_COLS=n_cols,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        TANH_MODE=tanh_mode,
        STABLE=False,
        EXP2=True,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=waves_per_eu,
    )
    return out
