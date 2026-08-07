import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _act_qd_fused(x_ptr, out_ptr, M, K, xsm, xsk, osm, osk, BLOCK: tl.constexpr):
    """Fully fused 1x128 quant+dequant. Computes scale in-kernel."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + pid_m * xsm + (pid_k * BLOCK + offs) * xsk).to(tl.float32)
    block_max = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(block_max / 448.0, 1e-12)
    x_q = tl.clamp(x / scale, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(out_ptr + pid_m * osm + (pid_k * BLOCK + offs) * osk,
             (x_q.to(tl.float32) * scale).to(tl.bfloat16))


@triton.jit
def _weight_qd_fused(w_ptr, out_ptr, K, N, wsk, wsn, osn, osk, BLOCK: tl.constexpr):
    """Fully fused 128x128 quant+dequant. w(K,N)->out(N,K) bf16."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    w_ptrs = (w_ptr + (pid_k*BLOCK+offs_k[:,None])*wsk + (pid_n*BLOCK+offs_n[None,:])*wsn)
    w = tl.load(w_ptrs).to(tl.float32)
    block_max = tl.max(tl.abs(w), axis=0)
    block_max = tl.max(block_max, axis=0)
    scale = tl.maximum(block_max / 448.0, 1e-12)
    w_q = tl.clamp(w / scale, -448.0, 448.0).to(tl.float8e4nv)
    w_dq = w_q.to(tl.float32) * scale
    out_ptrs = (out_ptr + (pid_n*BLOCK+offs_n[:,None])*osn + (pid_k*BLOCK+offs_k[None,:])*osk)
    tl.store(out_ptrs, w_dq.T.to(tl.bfloat16))


@torch.no_grad()
def run(hidden_states, qkv_weight, qkv_bias):
    num_heads = 16
    head_dim = 96
    seq_length = hidden_states.shape[0]

    x_f32 = hidden_states.to(torch.float32).contiguous()
    M, K = x_f32.shape
    a_bf16 = torch.empty((M, K), device=x_f32.device, dtype=torch.bfloat16)
    _act_qd_fused[(M, K // 128)](x_f32, a_bf16, M, K, x_f32.stride(0), x_f32.stride(1), a_bf16.stride(0), a_bf16.stride(1), BLOCK=128)

    w_fp32 = qkv_weight.T.to(torch.float32).contiguous()
    K2, N = w_fp32.shape
    b_bf16 = torch.empty((N, K2), device=w_fp32.device, dtype=torch.bfloat16)
    _weight_qd_fused[(K2 // 128, N // 128)](w_fp32, b_bf16, K2, N, w_fp32.stride(0), w_fp32.stride(1), b_bf16.stride(0), b_bf16.stride(1), BLOCK=128)

    qkv = torch.addmm(qkv_bias, a_bf16, b_bf16.t())
    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=1)
    return q.contiguous(), k.contiguous(), v.contiguous()
