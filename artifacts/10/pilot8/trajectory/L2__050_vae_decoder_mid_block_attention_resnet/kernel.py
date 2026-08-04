import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SC = 4096.0
_RSC = 1.0 / 4096.0


@triton.jit
def _conv3x3_k(X, Wh, Wm, Bias, Out,
               C: tl.constexpr, H, W, HW, N,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    nm = C // BM
    nn = tl.cdiv(N, BN)
    ng = GM * nn
    gid = pid // ng
    fm = gid * GM
    gs = min(nm - fm, GM)
    pid_m = fm + ((pid % ng) % gs)
    pid_n = (pid % ng) // gs

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mn = rn < N
    rnc = tl.where(mn, rn, 0)
    bb = rnc // HW
    hw = rnc % HW
    hh = hw // W
    ww = hw % W
    base = bb * (C * HW) + hh * W + ww

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    acl = tl.zeros((BM, BN), dtype=tl.float32)
    rk = tl.arange(0, BK)
    for khw in range(9):
        kh = khw // 3
        kw = khw % 3
        ih = hh + kh - 1
        iw = ww + kw - 1
        vm = mn & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        xb = X + base + (kh - 1) * W + (kw - 1)
        wb = Wh + khw * (C * C) + rm[:, None] * C
        wbm = Wm + khw * (C * C) + rm[:, None] * C
        for c0 in range(0, C, BK):
            ci = c0 + rk
            x = tl.load(xb[None, :] + ci[:, None] * HW, mask=vm[None, :], other=0.)
            ah = tl.load(wb + ci[None, :])
            am = tl.load(wbm + ci[None, :])
            xh = x.to(tl.float16)
            xm = ((x - xh.to(tl.float32)) * _SC).to(tl.float16)
            acc = tl.dot(ah, xh, acc)
            acl = tl.dot(ah, xm, acl)
            acl = tl.dot(am, xh, acl)

    o = acc + acl * _RSC + tl.load(Bias + rm)[:, None]
    ob = bb * (C * HW) + rm[:, None] * HW + hh * W + ww
    tl.store(Out + ob, o, mask=mn[None, :])


_wcache = {}


def _prep_w(w):
    key = w.data_ptr()
    ent = _wcache.get(key)
    if ent is not None and ent[0] is w:
        return ent[1], ent[2]
    C = w.shape[0]
    w2 = w.permute(2, 3, 0, 1).contiguous().view(9, C, C)
    wh = w2.to(torch.float16)
    wm = ((w2 - wh.float()) * _SC).to(torch.float16)
    _wcache[key] = (w, wh, wm)
    return wh, wm


def _cfg(N):
    if N >= 16384:
        return 256, 128, 32, 4, 8, 2
    if N >= 4096:
        return 256, 64, 32, 1, 8, 2
    if N >= 2048:
        return 256, 32, 64, 8, 8, 2
    return 64, 32, 64, 8, 4, 2


def _conv(x, w, bias):
    B, C, H, W = x.shape
    N = B * H * W
    wh, wm = _prep_w(w)
    out = torch.empty_like(x)
    BM, BN, BK, GM, nw, ns = _cfg(N)
    grid = ((C // BM) * triton.cdiv(N, BN),)
    _conv3x3_k[grid](x, wh, wm, bias, out, C, H, W, H * W, N,
                     BM, BN, BK, GM, num_warps=nw, num_stages=ns)
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    resnet1_norm1_weight: torch.Tensor,
    resnet1_norm1_bias: torch.Tensor,
    resnet1_conv1_weight: torch.Tensor,
    resnet1_conv1_bias: torch.Tensor,
    resnet1_time_emb_proj_weight: torch.Tensor,
    resnet1_time_emb_proj_bias: torch.Tensor,
    resnet1_norm2_weight: torch.Tensor,
    resnet1_norm2_bias: torch.Tensor,
    resnet1_conv2_weight: torch.Tensor,
    resnet1_conv2_bias: torch.Tensor,
    attn_group_norm_weight: torch.Tensor,
    attn_group_norm_bias: torch.Tensor,
    attn_to_q_weight: torch.Tensor,
    attn_to_q_bias: torch.Tensor,
    attn_to_k_weight: torch.Tensor,
    attn_to_k_bias: torch.Tensor,
    attn_to_v_weight: torch.Tensor,
    attn_to_v_bias: torch.Tensor,
    attn_to_out_weight: torch.Tensor,
    attn_to_out_bias: torch.Tensor,
    resnet2_norm1_weight: torch.Tensor,
    resnet2_norm1_bias: torch.Tensor,
    resnet2_conv1_weight: torch.Tensor,
    resnet2_conv1_bias: torch.Tensor,
    resnet2_time_emb_proj_weight: torch.Tensor,
    resnet2_time_emb_proj_bias: torch.Tensor,
    resnet2_norm2_weight: torch.Tensor,
    resnet2_norm2_bias: torch.Tensor,
    resnet2_conv2_weight: torch.Tensor,
    resnet2_conv2_bias: torch.Tensor,
    eps: float,
):
    batch, channels, height, width = hidden_states.shape
    num_groups = 32
    num_heads = 1
    head_dim = channels
    scale = head_dim ** -0.5

    residual1 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet1_norm1_weight, resnet1_norm1_bias, eps)
    h = F.silu(h)
    h = _conv(h, resnet1_conv1_weight, resnet1_conv1_bias)

    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet1_time_emb_proj_weight, resnet1_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet1_norm2_weight, resnet1_norm2_bias, eps)
    h = F.silu(h)
    h = _conv(h, resnet1_conv2_weight, resnet1_conv2_bias)

    hidden_states = h + residual1

    attn_residual = hidden_states

    h = F.group_norm(hidden_states, num_groups, attn_group_norm_weight, attn_group_norm_bias, eps)

    h = h.view(batch, channels, height * width).transpose(1, 2)

    query = F.linear(h, attn_to_q_weight, attn_to_q_bias)
    key = F.linear(h, attn_to_k_weight, attn_to_k_bias)
    value = F.linear(h, attn_to_v_weight, attn_to_v_bias)

    seq_len = height * width
    query = query.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attention_probs = F.softmax(attention_scores, dim=-1)

    h = torch.matmul(attention_probs, value)

    h = h.transpose(1, 2).reshape(batch, seq_len, channels)

    h = F.linear(h, attn_to_out_weight, attn_to_out_bias)

    h = h.transpose(1, 2).view(batch, channels, height, width)

    hidden_states = h + attn_residual

    residual2 = hidden_states

    h = F.group_norm(hidden_states, num_groups, resnet2_norm1_weight, resnet2_norm1_bias, eps)
    h = F.silu(h)
    h = _conv(h, resnet2_conv1_weight, resnet2_conv1_bias)

    temb_proj = F.silu(temb)
    temb_proj = F.linear(temb_proj, resnet2_time_emb_proj_weight, resnet2_time_emb_proj_bias)
    h = h + temb_proj[:, :, None, None]

    h = F.group_norm(h, num_groups, resnet2_norm2_weight, resnet2_norm2_bias, eps)
    h = F.silu(h)
    h = _conv(h, resnet2_conv2_weight, resnet2_conv2_bias)

    output = h + residual2

    return output
