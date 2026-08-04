import torch
import triton
import triton.language as tl


@triton.jit
def _cos_sin_emb_kernel(
    freqs_ptr, cos_ptr, sin_ptr, n_rows, scale,
    D: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, D)
    mask = r[:, None] < n_rows
    off = r[:, None] * D + c[None, :]
    x = tl.load(freqs_ptr + off, mask=mask, other=0.0)
    co = (tl.cos(x) * scale).to(tl.bfloat16)
    si = (tl.sin(x) * scale).to(tl.bfloat16)
    o = r[:, None] * (2 * D) + c[None, :]
    tl.store(cos_ptr + o, co, mask=mask)
    tl.store(cos_ptr + o + D, co, mask=mask)
    tl.store(sin_ptr + o, si, mask=mask)
    tl.store(sin_ptr + o + D, si, mask=mask)


_BLOCK_R = 8
_WARPS = 4

# cache: D -> (raw_run, function, packed_metadata)
_LAUNCH = {}
_current_stream = torch.cuda.current_stream


def _build(D, device):
    """Compile once and extract the raw launcher, bypassing Triton's
    per-call JIT dispatch (which costs ~10us of pure Python)."""
    dummy_in = torch.empty(_BLOCK_R, D, dtype=torch.float32, device=device)
    dummy_out = torch.empty(_BLOCK_R, 2 * D, dtype=torch.bfloat16, device=device)
    ck = _cos_sin_emb_kernel[(1,)](
        dummy_in, dummy_out, dummy_out, 0, 1.0,
        D=D, BLOCK_R=_BLOCK_R, num_warps=_WARPS, num_stages=1,
    )
    ck._init_handles()
    ent = (ck.run, ck.function, ck.packed_metadata)
    _LAUNCH[D] = ent
    return ent


@torch.no_grad()
def run(freqs: torch.Tensor, attention_scaling: float):
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    shape = freqs.shape
    D = shape[-1]
    n_rows = freqs.numel() // D

    out = torch.empty((2,) + shape[:-1] + (2 * D,),
                      dtype=torch.bfloat16, device=freqs.device)
    cos = out[0]
    sin = out[1]

    if n_rows == 0:
        return cos, sin

    ent = _LAUNCH.get(D)
    if ent is None:
        ent = _build(D, freqs.device)
    run_, func, pm = ent

    grid = (n_rows + _BLOCK_R - 1) // _BLOCK_R
    run_(grid, 1, 1, _current_stream().cuda_stream, func, pm,
         None, None, None,
         freqs, cos, sin, n_rows, float(attention_scaling), D, _BLOCK_R)
    return cos, sin
