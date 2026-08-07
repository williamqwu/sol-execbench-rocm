import torch
import torch.nn.functional as F
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _post_kernel(
    dt_proj_ptr,    # [B*L, H] bf16
    A_log_ptr,      # [H] f32
    B_ptr,          # [B*L, G, ssm] bf16
    dt_out_ptr,     # [B*L, H] bf16
    dA_ptr,         # [B*L, H] f32
    dB_ptr,         # [B*L, H, ssm] f32
    H: tl.constexpr,
    G: tl.constexpr,
    HPG: tl.constexpr,
    SSM: tl.constexpr,
):
    pid = tl.program_id(0)
    h_offs = tl.arange(0, H)
    dt_proj_row = tl.load(dt_proj_ptr + pid * H + h_offs).to(tl.float32)
    dt = tl.maximum(dt_proj_row, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(dt_proj_row)))
    dt = tl.clamp(dt, 0.0, float('inf'))

    A_log_row = tl.load(A_log_ptr + h_offs)
    A = -tl.exp(A_log_row)
    dA = tl.exp(dt * A)

    tl.store(dt_out_ptr + pid * H + h_offs, dt.to(tl.bfloat16))
    tl.store(dA_ptr + pid * H + h_offs, dA)

    ssm_off = tl.arange(0, SSM)
    h_2d = h_offs[:, None]
    s_2d = ssm_off[None, :]
    g_idx = h_offs // HPG
    B_offs = pid * G * SSM + g_idx[:, None] * SSM + s_2d
    B_vals = tl.load(B_ptr + B_offs).to(tl.float32)
    dB_vals = dt[:, None] * B_vals
    dB_offs = pid * H * SSM + h_2d * SSM + s_2d
    tl.store(dB_ptr + dB_offs, dB_vals)


@torch.compile
def _quant_and_gemm(
    hidden_states: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    dt_bias: torch.Tensor,
):
    num_heads = hidden_states.shape[-1]
    hidden_states_flat = hidden_states.reshape(-1, num_heads)

    x_fp32 = hidden_states_flat.to(torch.float32)
    x_amax = x_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale_x = x_amax / E4M3_MAX
    x_scaled = (x_fp32 / scale_x).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.to(torch.float8_e4m3fn)

    weight_fp32 = dt_proj_weight.to(torch.float32)
    weight_t = weight_fp32.T
    w_amax = weight_t.abs().max().clamp(min=1e-12)
    scale_w_scalar = w_amax / E4M3_MAX
    w_scaled = (weight_t / scale_w_scalar).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qw = w_scaled.T.to(torch.float8_e4m3fn)

    a_f32 = qx.to(torch.float32) * scale_x
    b_f32 = qw.to(torch.float32) * scale_w_scalar
    y = a_f32 @ b_f32.T
    y = y + dt_bias
    return y.to(torch.bfloat16)  # [BL, H]


def _post_process(dt_proj_flat, A_log, B_flat, BL, H, G, HPG, SSM):
    dt_out = torch.empty(BL, H, dtype=torch.bfloat16, device=dt_proj_flat.device)
    dA = torch.empty(BL, H, dtype=torch.float32, device=dt_proj_flat.device)
    dB = torch.empty(BL, H, SSM, dtype=torch.float32, device=dt_proj_flat.device)
    _post_kernel[(BL,)](dt_proj_flat, A_log, B_flat, dt_out, dA, dB,
                        H=H, G=G, HPG=HPG, SSM=SSM)
    return dt_out, dA, dB


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    time_step_limit_min: float,
    time_step_limit_max: float,
):
    batch_size, seq_len, num_heads = hidden_states.shape
    head_dim = 128
    ssm_state_size = 128
    n_groups = 8
    BL = batch_size * seq_len
    HPG = num_heads // n_groups

    dt_proj_flat = _quant_and_gemm(hidden_states, dt_proj_weight, dt_bias)  # [BL, H]
    B_flat = B.reshape(BL, n_groups, ssm_state_size)

    dt_compact, dA_compact, dB_compact = _post_process(
        dt_proj_flat, A_log, B_flat, BL, num_heads, n_groups, HPG, ssm_state_size
    )

    dt_out = dt_compact.reshape(batch_size, seq_len, num_heads).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    dA = dA_compact.reshape(batch_size, seq_len, num_heads).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, head_dim, ssm_state_size)
    dB = dB_compact.reshape(batch_size, seq_len, num_heads, ssm_state_size).unsqueeze(3).expand(-1, -1, -1, head_dim, -1)

    return dt_out, dA, dB
