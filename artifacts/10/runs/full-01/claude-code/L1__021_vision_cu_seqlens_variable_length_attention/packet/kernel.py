import torch
import torch.nn.functional as F

HIDDEN = 1280
NUM_HEADS = 16
HEAD_DIM = 80
HALF = 40


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
):
    T = hidden_states.shape[0]
    scaling = HEAD_DIM ** -0.5

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(T, 3, NUM_HEADS, HEAD_DIM).permute(1, 0, 2, 3)
    q, k, v = qkv.unbind(0)

    qf = q.float()
    kf = k.float()
    ce = cos.unsqueeze(1).float()
    se = sin.unsqueeze(1).float()
    qr = torch.cat((-qf[..., HALF:], qf[..., :HALF]), dim=-1)
    kr = torch.cat((-kf[..., HALF:], kf[..., :HALF]), dim=-1)
    qe = (qf * ce) + (qr * se)
    ke = (kf * ce) + (kr * se)

    Q = qe.transpose(0, 1).unsqueeze(0)
    K = ke.transpose(0, 1).unsqueeze(0)
    V = v.transpose(0, 1).unsqueeze(0)

    cul = cu_seqlens.tolist()
    outs = []
    for i in range(len(cul) - 1):
        s, e = cul[i], cul[i + 1]
        if e - s == 0:
            continue
        aw = torch.matmul(Q[:, :, s:e], K[:, :, s:e].transpose(2, 3)) * scaling
        aw = F.softmax(aw, dim=-1, dtype=torch.float32)
        outs.append(torch.matmul(aw, V[:, :, s:e]).transpose(1, 2))

    if outs:
        attn = torch.cat(outs, dim=1)
    else:
        attn = torch.zeros(1, 0, NUM_HEADS, HEAD_DIM, device=qkv.device, dtype=qkv.dtype)
    attn = attn.reshape(T, HIDDEN).contiguous()

    return F.linear(attn, proj_weight, proj_bias)
