import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_persistent(
    x_ptr,
    w_ptr,
    y_ptr,
    B,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_PIDS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    for row in range(pid, B, NUM_PIDS):
        x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / H
        inv_rms = tl.rsqrt(mean_sq + EPS)
        y = (x * inv_rms) * w
        tl.store(y_ptr + row * H + cols, y.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(hidden_states, weight):
    B, H = hidden_states.shape
    assert H == 512
    y = torch.empty_like(hidden_states)
    # Use 256 programs (one per CU) for the persistent kernel
    num_pids = min(B, 256)
    _rmsnorm_persistent[(num_pids,)](
        hidden_states, weight, y,
        B, H=H, EPS=1e-6, BLOCK=512, NUM_PIDS=num_pids,
        num_warps=1,
    )
    return y
