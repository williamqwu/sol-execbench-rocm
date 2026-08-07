import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


def _dequant_impl(hidden_states, weight):
    M, K = hidden_states.shape
    N = weight.shape[0]
    x_fp32 = hidden_states.to(torch.float32)
    w_fp32 = weight.to(torch.float32)
    x_blk = x_fp32.reshape(M, K // 128, 128)
    scale_x = torch.clamp(x_blk.abs().amax(dim=2) / E4M3_MAX, min=1e-12)
    w_blk = w_fp32.reshape(N // 128, 128, K // 128, 128)
    scale_w = torch.clamp(w_blk.abs().amax(dim=3).amax(dim=1) / E4M3_MAX, min=1e-12)
    sx = scale_x.unsqueeze(2)
    qx = torch.clamp(x_blk / sx, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    a_bf16 = (qx.to(torch.float32) * sx).reshape(M, K).to(torch.bfloat16)
    sw = scale_w.unsqueeze(1).unsqueeze(3)
    qw = torch.clamp(w_blk / sw, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    b_bf16 = (qw.to(torch.float32) * sw).reshape(N, K).to(torch.bfloat16)
    return a_bf16 @ b_bf16.T


_compiled = torch.compile(_dequant_impl, mode="max-autotune-no-cuda-graphs", dynamic=True)


@torch.no_grad()
def run(hidden_states, weight):
    return _compiled(hidden_states, weight)
