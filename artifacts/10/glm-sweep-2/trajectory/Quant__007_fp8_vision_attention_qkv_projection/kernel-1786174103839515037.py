import torch

E4M3_MAX = 448.0
NUM_HEADS = 16
HEAD_DIM = 96
HIDDEN_SIZE = 1536
QKV_OUT = 4608  # 3 * 16 * 96


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    seq_length = hidden_states.shape[0]
    M, K, N = seq_length, HIDDEN_SIZE, QKV_OUT

    # --- Activation: BlockWise1x128 scaling -> FP8 -> dequant to bf16 ---
    x_f32 = hidden_states.to(torch.float32)
    x_blocked = x_f32.view(M, 12, 128)          # K=1536=12*128
    sx = x_blocked.abs().amax(dim=2).clamp(min=1e-12) / E4M3_MAX   # (M, 12)
    x_scaled = (x_blocked / sx.unsqueeze(2)).clamp(-E4M3_MAX, E4M3_MAX)
    qx = x_scaled.reshape(M, K).to(torch.float8_e4m3fn)
    a = (qx.view(M, 12, 128).to(torch.float32) * sx.unsqueeze(2)).reshape(M, K).to(torch.bfloat16)

    # --- Weight: BlockWise128x128 scaling -> FP8 -> dequant to bf16 ---
    # weight is (N, K); work with (K, N) for blocking
    w_f32 = qkv_weight.T.to(torch.float32)       # (K, N) = (1536, 4608)
    w_blocked = w_f32.view(12, 128, 36, 128)     # K//128=12, N//128=36
    sw = w_blocked.abs().amax(dim=3).amax(dim=1).clamp(min=1e-12) / E4M3_MAX  # (12, 36)
    w_scaled = (w_blocked / sw.unsqueeze(1).unsqueeze(3)).clamp(-E4M3_MAX, E4M3_MAX)
    qw = w_scaled.reshape(K, N).T.to(torch.float8_e4m3fn)  # (N, K)
    # dequant qw (N, K): N=36*128, K=12*128
    qw_v = qw.to(torch.float32).view(36, 128, 12, 128)
    b_f32 = qw_v * sw.T.unsqueeze(1).unsqueeze(3)  # sw.T (36,12) -> (36,1,12,1)
    b = b_f32.reshape(N, K).to(torch.bfloat16)

    # --- bf16 GEMM + bias ---
    qkv = a @ b.T                                 # (M, N) bf16
    qkv = qkv + qkv_bias
    qkv = qkv.view(seq_length, 3, NUM_HEADS, HEAD_DIM)
    q, k, v = qkv.unbind(dim=1)
    return q.contiguous(), k.contiguous(), v.contiguous()
