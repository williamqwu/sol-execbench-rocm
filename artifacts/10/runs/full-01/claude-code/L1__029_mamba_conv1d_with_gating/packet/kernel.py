import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _post_kernel(
    proj_ptr, mask_ptr, cw_ptr, cb_ptr, outh_ptr, outg_ptr,
    L,
    C: tl.constexpr,
    PROJ_STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_l = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    l_ok = offs_l < L

    base = proj_ptr + (pid_b.to(tl.int64) * L) * PROJ_STRIDE
    mbase = mask_ptr + pid_b.to(tl.int64) * L

    acc = tl.zeros([BLOCK_L, BLOCK_C], tl.float32)

    for k in tl.static_range(4):
        ls = offs_l + k - 3
        ok = (ls >= 0) & (ls < L)
        p = base + ls[:, None].to(tl.int64) * PROJ_STRIDE + offs_c[None, :]
        x = tl.load(p, mask=ok[:, None], other=0.0)
        m = tl.load(mbase + ls, mask=ok, other=0.0)
        xm = (x.to(tl.float32) * m.to(tl.float32)[:, None]).to(tl.bfloat16)
        w = tl.load(cw_ptr + offs_c * 4 + k)
        acc += xm.to(tl.float32) * w.to(tl.float32)[None, :]

    # MIOpen depthwise conv rounds the accumulator to bf16, then adds the bias
    # in a separate elementwise kernel (aten::add_) -- reproduce both roundings.
    cv = acc.to(tl.bfloat16).to(tl.float32)
    cv = (cv + tl.load(cb_ptr + offs_c).to(tl.float32)[None, :]).to(tl.bfloat16).to(
        tl.float32
    )
    s = tl.sigmoid(cv).to(tl.bfloat16).to(tl.float32)
    y = (cv * s).to(tl.bfloat16).to(tl.float32)
    mcur = tl.load(mbase + offs_l, mask=l_ok, other=0.0).to(tl.float32)
    y = (y * mcur[:, None]).to(tl.bfloat16)

    g = tl.load(
        base + offs_l[:, None].to(tl.int64) * PROJ_STRIDE + (offs_c + C)[None, :],
        mask=l_ok[:, None],
        other=0.0,
    )

    yT = tl.trans(y)
    gT = tl.trans(g)

    out_off = (
        pid_b.to(tl.int64) * (C * L)
        + offs_c[:, None].to(tl.int64) * L
        + offs_l[None, :]
    )
    st = l_ok[None, :]
    tl.store(outh_ptr + out_off, yT, mask=st)
    tl.store(outg_ptr + out_off, gT, mask=st)


def _pick(L):
    if L >= 256:
        return 64, 64, 4
    if L >= 64:
        return 64, 64, 4
    return 128, 32, 4


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
):
    batch_size, seq_len, _ = hidden_states.shape
    C = in_proj_weight.shape[0] // 2

    x2d = hidden_states.reshape(batch_size * seq_len, hidden_states.shape[-1])
    proj = F.linear(x2d, in_proj_weight, in_proj_bias)

    outh = torch.empty(
        (batch_size, C, seq_len), device=proj.device, dtype=torch.bfloat16
    )
    outg = torch.empty_like(outh)

    cw = conv1d_weight.reshape(-1).contiguous()
    cb = conv1d_bias.contiguous()
    am = attention_mask.contiguous()

    BC, BL, nw = _pick(seq_len)
    grid = (triton.cdiv(seq_len, BL), C // BC, batch_size)
    _post_kernel[grid](
        proj, am, cw, cb, outh, outg,
        seq_len,
        C=C,
        PROJ_STRIDE=2 * C,
        BLOCK_C=BC,
        BLOCK_L=BL,
        num_warps=nw,
    )
    return outh, outg
