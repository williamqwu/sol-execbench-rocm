import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


@torch.compile(mode="max-autotune")
def _fused_run(
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

    hidden_states_flat = hidden_states.reshape(-1, num_heads)  # [BL, H]

    # ---- FP8 quantization of activations: BlockWise1x128 ----
    # Each row of 128 elements is one block.
    x_fp32 = hidden_states_flat.to(torch.float32)
    x_amax = x_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [BL,1]
    scale_x = x_amax / E4M3_MAX  # [BL,1]
    x_scaled = (x_fp32 / scale_x).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.to(torch.float8_e4m3fn)

    # ---- FP8 quantization of weight: BlockWise128x128 ----
    # weight is [H,H]=[128,128]. Transpose -> [H,H], one 128x128 block.
    weight_fp32 = dt_proj_weight.to(torch.float32)
    weight_t = weight_fp32.T  # [H, H]
    w_amax = weight_t.abs().max().clamp(min=1e-12)
    scale_w_scalar = w_amax / E4M3_MAX  # scalar
    w_scaled = (weight_t / scale_w_scalar).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [H, H]

    # ---- Dequantize-then-matmul (matches reference: dequant to fp32, matmul) ----
    a_f32 = qx.to(torch.float32) * scale_x  # [BL, H]
    b_f32 = qw.to(torch.float32) * scale_w_scalar  # [H, H]
    y = a_f32 @ b_f32.T  # [BL, H]
    y = y + dt_bias
    dt_proj = y.to(torch.bfloat16).reshape(batch_size, seq_len, num_heads)

    # ---- Softplus + clamp ----
    dt = F.softplus(dt_proj.float())
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)

    # ---- dA: [B,L,H] ----
    A = -torch.exp(A_log.float())
    dA_compact = torch.exp(dt * A)

    # ---- dB: [B,L,H,ssm] via group-wise multiply ----
    heads_per_group = num_heads // n_groups
    dt_groups = dt.reshape(batch_size, seq_len, n_groups, heads_per_group)
    dB_compact = (
        dt_groups.unsqueeze(-1) * B.unsqueeze(-2).float()
    ).reshape(batch_size, seq_len, num_heads, ssm_state_size)

    # ---- Expand via views ----
    dt_out = dt.to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    dA = dA_compact.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, head_dim, ssm_state_size)
    dB = dB_compact.unsqueeze(3).expand(-1, -1, -1, head_dim, -1)

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
    return _fused_run(
        hidden_states, B, dt_proj_weight, dt_bias, A_log,
        time_step_limit_min, time_step_limit_max,
    )
