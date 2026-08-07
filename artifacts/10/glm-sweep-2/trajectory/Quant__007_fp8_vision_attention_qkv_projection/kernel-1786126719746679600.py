import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _act_qd_kernel(
    x_ptr, scale_ptr, out_ptr,
    M, K, n_blocks_k,
    xsm, xsk, ssm, ssk, osm, osk,
    BLOCK: tl.constexpr,
):
    """Quantize+dequant one 1x128 block using precomputed scale. x(fp32)->bf16."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    x_ptrs = x_ptr + pid_m * xsm + (pid_k * BLOCK + offs) * xsk
    x = tl.load(x_ptrs).to(tl.float32)
    scale = tl.load(scale_ptr + pid_m * ssm + pid_k * ssk)
    x_q = x / scale
    x_q = tl.clamp(x_q, -448.0, 448.0)
    x_fp8 = x_q.to(tl.float8e4nv)
    x_dq = x_fp8.to(tl.float32) * scale
    out_ptrs = out_ptr + pid_m * osm + (pid_k * BLOCK + offs) * osk
    tl.store(out_ptrs, x_dq.to(tl.bfloat16))


@triton.jit
def _weight_qd_kernel(
    w_ptr, scale_ptr, out_ptr,
    K, N, n_blocks_n,
    wsk, wsn, ssk, ssn, osn, osk,
    BLOCK: tl.constexpr,
):
    """Quantize+dequant one 128x128 block of w(K,N) using precomputed scale.
    Output is transposed (N,K) bf16."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK)
    offs_n = tl.arange(0, BLOCK)
    w_ptrs = (w_ptr
              + (pid_k * BLOCK + offs_k[:, None]) * wsk
              + (pid_n * BLOCK + offs_n[None, :]) * wsn)
    w = tl.load(w_ptrs).to(tl.float32)
    scale = tl.load(scale_ptr + pid_k * ssk + pid_n * ssn)
    w_q = w / scale
    w_q = tl.clamp(w_q, -448.0, 448.0)
    w_fp8 = w_q.to(tl.float8e4nv)
    w_dq = w_fp8.to(tl.float32) * scale
    out_ptrs = (out_ptr
                + (pid_n * BLOCK + offs_n[:, None]) * osn
                + (pid_k * BLOCK + offs_k[None, :]) * osk)
    tl.store(out_ptrs, w_dq.T.to(tl.bfloat16))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
):
    num_heads = 16
    head_dim = 96
    seq_length = hidden_states.shape[0]

    x_f32 = hidden_states.to(torch.float32).contiguous()
    M, K = x_f32.shape

    # Activation scales (1x128) - computed in PyTorch to match reference exactly
    x_blocked = x_f32.view(M, K // 128, 128)
    block_max = x_blocked.abs().amax(dim=2)
    scale_x = torch.clamp(block_max / E4M3_MAX, min=1e-12).contiguous()

    a_bf16 = torch.empty((M, K), device=x_f32.device, dtype=torch.bfloat16)
    _act_qd_kernel[(M, K // 128)](
        x_f32, scale_x, a_bf16, M, K, K // 128,
        x_f32.stride(0), x_f32.stride(1),
        scale_x.stride(0), scale_x.stride(1),
        a_bf16.stride(0), a_bf16.stride(1),
        BLOCK=128,
    )

    # Weight: (N, K_in) -> transpose to (K, N) for 128x128 blocking
    w_fp32 = qkv_weight.T.to(torch.float32).contiguous()  # (K, N)
    K2, N = w_fp32.shape
    w_blocked = w_fp32.view(K2 // 128, 128, N // 128, 128)
    w_block_max = w_blocked.abs().amax(dim=3).amax(dim=1)
    weight_scales = torch.clamp(w_block_max / E4M3_MAX, min=1e-12).contiguous()  # (K//128, N//128)

    b_bf16 = torch.empty((N, K2), device=w_fp32.device, dtype=torch.bfloat16)
    _weight_qd_kernel[(K2 // 128, N // 128)](
        w_fp32, weight_scales, b_bf16, K2, N, N // 128,
        w_fp32.stride(0), w_fp32.stride(1),
        weight_scales.stride(0), weight_scales.stride(1),
        b_bf16.stride(0), b_bf16.stride(1),
        BLOCK=128,
    )

    qkv = a_bf16 @ b_bf16.t()
    qkv = qkv + qkv_bias

    qkv = qkv.view(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=1)
    return query_states.contiguous(), key_states.contiguous(), value_states.contiguous()
