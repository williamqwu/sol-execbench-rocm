import os
import torch
import triton
import triton.language as tl

_hip_fn = None

def _get_hip_fn():
    global _hip_fn
    if _hip_fn is None:
        import importlib.util
        so_path = "/job/fused_rmsnorm_hip7.so"
        if os.path.exists(so_path):
            spec = importlib.util.spec_from_file_location("fused_rmsnorm_hip7", so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _hip_fn = mod.fused_add_rmsnorm
        else:
            _hip_fn = False
    return _hip_fn


@triton.jit
def _fused_add_rmsnorm_triton(
    hidden_ptr, residual_ptr, weight_ptr, output_ptr,
    stride_h, stride_r, stride_o,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    h = tl.load(hidden_ptr + row * stride_h + cols, mask=cols < H, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * stride_r + cols, mask=cols < H, other=0.0).to(tl.float32)
    x = h + r

    sum_sq = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_sq / H + 1e-5)

    w = tl.load(weight_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(output_ptr + row * stride_o + cols, y.to(tl.bfloat16), mask=cols < H)


_TRITON_BLOCK = 4096


@torch.no_grad()
def run(hidden_states, residual, weight):
    _, hidden_size = hidden_states.shape
    assert hidden_size == 4096

    rows = hidden_states.shape[0]

    if rows < 2000:
        hip_fn = _get_hip_fn()
        if hip_fn:
            return hip_fn(hidden_states, residual, weight)

    out = torch.empty_like(hidden_states)
    num_warps = 4 if rows >= 1024 else 16
    grid = (rows,)
    _fused_add_rmsnorm_triton[grid](
        hidden_states, residual, weight, out,
        hidden_states.stride(0), residual.stride(0), out.stride(0),
        H=hidden_size, BLOCK=_TRITON_BLOCK,
        num_warps=num_warps, num_stages=1,
    )
    return out
