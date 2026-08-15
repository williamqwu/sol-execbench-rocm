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

    @property
    def shape(self) -> tuple[int, int]:
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]


class BlockwiseScaler:
    E4M3_MAX = 448.0

    def __init__(self, scaling_type: ScalingType):
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape
        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }
        self.block_size_m, self.block_size_k = scaling_map[scaling_type]
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.scaling_type == ScalingType.TensorWise:
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        assert M % self.block_size_m == 0
        assert K % self.block_size_k == 0

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

        # Dequantize: FP8 values * inverse_scales -> float32
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        # Single matmul in float32
        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
):
    # Initialize scalers
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    # Step 1: Gate-up projection with FP8
    hidden_fp32 = hidden_states.to(torch.float32)
    scale_hidden = activation_scaler.compute_scales(hidden_fp32)

    gate_up_weight_fp32 = gate_up_weight.to(torch.float32)
    gate_up_weight_t = gate_up_weight_fp32.T
    scale_gate_up = weight_scaler.compute_scales(gate_up_weight_t)

    hidden_scaled = activation_scaler.apply_scaling(
        hidden_fp32, scale_hidden, inverse=False, clamp_to_fp8_range=True
    )
    gate_up_scaled = weight_scaler.apply_scaling(
        gate_up_weight_t, scale_gate_up, inverse=False, clamp_to_fp8_range=True
    )

    hidden_fp8 = hidden_scaled.to(torch.float8_e4m3fn)
    gate_up_fp8 = gate_up_scaled.T.to(torch.float8_e4m3fn)

    scale_gate_up_cublas = scale_gate_up.T.contiguous()

    gate_up_output = gemm_ref.scaled_mm(
        mat_a=hidden_fp8,
        mat_b=gate_up_fp8,
        scale_a=scale_hidden,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_gate_up_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    # Step 2: Split and apply activation
    gate, up = gate_up_output.chunk(2, dim=-1)
    gated_output = F.silu(gate) * up

    # Step 3: Down projection with FP8
    gated_fp32 = gated_output.to(torch.float32)
    scale_gated = activation_scaler.compute_scales(gated_fp32)

    down_weight_fp32 = down_weight.to(torch.float32)
    down_weight_t = down_weight_fp32.T
    scale_down = weight_scaler.compute_scales(down_weight_t)

    gated_scaled = activation_scaler.apply_scaling(
        gated_fp32, scale_gated, inverse=False, clamp_to_fp8_range=True
    )
    down_scaled = weight_scaler.apply_scaling(
        down_weight_t, scale_down, inverse=False, clamp_to_fp8_range=True
    )

    gated_fp8 = gated_scaled.to(torch.float8_e4m3fn)
    down_fp8 = down_scaled.T.to(torch.float8_e4m3fn)

    scale_down_cublas = scale_down.T.contiguous()

    output = gemm_ref.scaled_mm(
        mat_a=gated_fp8,
        mat_b=down_fp8,
        scale_a=scale_gated,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_down_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )

    # Step 4: Apply routing weight
    weighted_output = output * routing_weight

    return weighted_output
