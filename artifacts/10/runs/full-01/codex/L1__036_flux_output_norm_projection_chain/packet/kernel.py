import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _norm_mod_kernel(
    hidden_ptr,
    modulation_ptr,
    mean_ptr,
    var_ptr,
    output_ptr,
    eps,
    seq_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    mask = col < 3072
    batch = row // seq_len
    hidden = tl.load(hidden_ptr + row * 3072 + col, mask=mask)
    mean = tl.load(mean_ptr + row)
    var = tl.load(var_ptr + row)
    denom = libdevice.sqrt(var + eps.to(tl.float32))
    shift = tl.load(modulation_ptr + batch * 6144 + col, mask=mask)
    scale = tl.load(modulation_ptr + batch * 6144 + 3072 + col, mask=mask)
    norm = (hidden - mean) / denom
    one_plus_scale = 1.0 + scale
    product = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [norm, one_plus_scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [product, shift],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + row * 3072 + col, value, mask=mask)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
    eps: float,
):
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
    temb_silu = torch.sigmoid(temb)
    temb_silu.mul_(temb)
    modulation = F.linear(temb_silu, linear_weight, linear_bias)
    hidden_states_mod = torch.empty_like(hidden_states)
    rows = hidden_states.shape[0] * hidden_states.shape[1]
    if rows <= 1024 or rows >= 4096:
        block = 4096
        num_warps = 8
    else:
        block = 1024
        num_warps = 4
    _norm_mod_kernel[(rows, triton.cdiv(3072, block))](
        hidden_states,
        modulation,
        mean,
        var,
        hidden_states_mod,
        eps,
        hidden_states.shape[1],
        BLOCK=block,
        num_warps=num_warps,
    )
    return F.linear(hidden_states_mod, proj_out_weight, proj_out_bias)
