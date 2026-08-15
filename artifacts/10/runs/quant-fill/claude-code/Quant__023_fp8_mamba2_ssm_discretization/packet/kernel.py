import torch
import torch.nn.functional as F
from enum import StrEnum


class ScalingType(StrEnum):
    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"


class BlockwiseScaler:
    E4M3_MAX = 448.0

    def __init__(self, scaling_type: ScalingType):
        self.scaling_type = scaling_type
        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }
        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scaling_type == ScalingType.TensorWise:
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)

        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(
                    tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX
                )

        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)


@torch.compile(mode="max-autotune")
@torch.no_grad()
def _run_compiled(
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

    # FP8 GEMM - exactly matching reference
    hidden_states_flat = hidden_states.reshape(-1, num_heads)

    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    hidden_states_fp32 = hidden_states_flat.to(torch.float32)
    weight_fp32 = dt_proj_weight.to(torch.float32)

    scale_x = activation_scaler.compute_scales(hidden_states_fp32)
    weight_fp32_t = weight_fp32.T
    scales_w = weight_scaler.compute_scales(weight_fp32_t)

    x_scaled = activation_scaler.apply_scaling(
        hidden_states_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    w_scaled = weight_scaler.apply_scaling(
        weight_fp32_t, scales_w, inverse=False, clamp_to_fp8_range=True
    )

    qx = x_scaled.to(torch.float8_e4m3fn)
    qw = w_scaled.T.to(torch.float8_e4m3fn)
    scale_w_cublas = scales_w.T.contiguous()

    dt_proj = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=dt_bias,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    dt_proj = dt_proj.reshape(batch_size, seq_len, num_heads)

    # Softplus + clamp
    dt = F.softplus(dt_proj.float())
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)

    # Compact dA
    A = -torch.exp(A_log.float())
    dA_compact = torch.exp(dt * A)

    # Compact dB
    heads_per_group = num_heads // n_groups
    dt_groups = dt.reshape(batch_size, seq_len, n_groups, heads_per_group)
    dB_compact = (
        dt_groups.unsqueeze(-1) * B.unsqueeze(-2)
    ).reshape(batch_size, seq_len, num_heads, ssm_state_size)

    # Expand outputs
    dt_out = dt.to(torch.bfloat16).unsqueeze(-1).expand(-1, -1, -1, head_dim)
    dA = dA_compact.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, head_dim, ssm_state_size)
    dB = dB_compact.unsqueeze(3).expand(-1, -1, -1, head_dim, -1)

    return dt_out, dA, dB


def run(
    hidden_states: torch.Tensor,
    B: torch.Tensor,
    dt_proj_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    time_step_limit_min: float,
    time_step_limit_max: float,
):
    return _run_compiled(
        hidden_states, B, dt_proj_weight, dt_bias, A_log,
        time_step_limit_min, time_step_limit_max
    )
