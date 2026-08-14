import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_single_row(x_ptr, w_ptr, y_ptr):
    row = tl.program_id(0)
    cols = tl.arange(0, 128)
    offsets = row * 128 + cols

    x = tl.load(x_ptr + offsets).to(tl.float32)
    w = tl.load(w_ptr + cols).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) * (1.0 / 128.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    tl.store(y_ptr + offsets, x * inv_rms * w)


@triton.jit
def _rmsnorm_eight_rows(
    x_ptr,
    w_ptr,
    y_ptr,
    n_rows,
    EVEN_N: tl.constexpr,
    STREAMING_STORE: tl.constexpr,
):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]

    if EVEN_N:
        x = tl.load(x_ptr + offsets, cache_modifier=".cg").to(tl.float32)
    else:
        mask = rows[:, None] < n_rows
        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)

    w = tl.load(w_ptr + cols).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1) * (1.0 / 128.0)
    inv_rms = tl.rsqrt(mean_square + 1.0e-6)
    y = x * inv_rms[:, None] * w[None, :]

    if EVEN_N:
        if STREAMING_STORE:
            tl.store(y_ptr + offsets, y, cache_modifier=".wt")
        else:
            tl.store(y_ptr + offsets, y)
    else:
        if STREAMING_STORE:
            tl.store(y_ptr + offsets, y, mask=mask, cache_modifier=".wt")
        else:
            tl.store(y_ptr + offsets, y, mask=mask)


def run(hidden_states, weight):
    n_rows = hidden_states.shape[0]
    output = torch.empty_like(hidden_states)

    # Very short tensors are launch-latency limited.  One independently
    # scheduled row also retains the reference reduction order there.
    if n_rows < 4096:
        _rmsnorm_single_row[(n_rows,)](
            hidden_states,
            weight,
            output,
            num_warps=2,
        )
    else:
        # Eight rows per wave amortizes workgroup scheduling and performs eight
        # independent reductions concurrently.  Large outputs bypass L1 and
        # use write-through stores to avoid displacing the streaming input.
        _rmsnorm_eight_rows[(triton.cdiv(n_rows, 8),)](
            hidden_states,
            weight,
            output,
            n_rows,
            EVEN_N=(n_rows % 8 == 0),
            STREAMING_STORE=(n_rows >= 60000),
            num_warps=1,
            waves_per_eu=5 if n_rows >= 100000 else 0,
        )

    return output
