import torch
import torch.nn.functional as F

E4M3_MAX = 448.0


@torch.compile
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

    # ---- FP8 quantization of activations: BlockWise1x128 (per-row) ----
    x_fp32 = hidden_states_flat.to(torch.float32)
    x_amax = x_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [BL,1]
    scale_x = x_amax / E4M3_MAX  # [BL,1]
    x_scaled = (x_fp32 / scale_x).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qx = x_scaled.to(torch.float8_e4m3fn)  # [BL, H]

    # ---- FP8 quantization of weight: BlockWise128x128 (single block) ----
    weight_fp32 = dt_proj_weight.to(torch.float32)
    weight_t = weight_fp32.T  # [H, H]
    w_amax = weight_t.abs().max().clamp(min=1e-12)
    scale_w_scalar = w_amax / E4M3_MAX  # scalar
    w_scaled = (weight_t / scale_w_scalar).clamp(min=-E4M3_MAX, max=E4M3_MAX)
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [H, H] row-major

    # ---- FP8 scaled GEMM via _scaled_mm (row-wise scaling) ----
    scale_a = scale_x.contiguous().float()  # [BL, 1]
    scale_b = scale_w_scalar.float().reshape(1, 1).expand(1, num_heads).contiguous()  # [1, H]
    dt_proj = torch._scaled_mm(
        qx, qw.T, scale_a=scale_a, scale_b=scale_b,
        out_dtype=torch.bfloat16, bias=dt_bias.to(torch.bfloat16),
    ).reshape(batch_size, seq_len, num_heads)

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
