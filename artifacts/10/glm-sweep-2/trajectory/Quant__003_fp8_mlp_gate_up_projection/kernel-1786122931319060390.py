import torch
import torch.nn.functional as F
# --- inlined fp8_reference ---
import torch

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
        new_shape = (M // self.block_size_m, self.block_size_m, K // self.block_size_k, self.block_size_k)
        tensor_blocked = tensor.reshape(new_shape)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(self, tensor, scales, inverse=False, clamp_to_fp8_range=False):
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            M, K = tensor.shape
            new_shape = (M // self.block_size_m, self.block_size_m, K // self.block_size_k, self.block_size_k)
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)
        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX)
        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    def scaled_mm(self, mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b,
                 bias=None, output_dtype=torch.bfloat16, use_fast_accum=True):
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)
        y = a_f32 @ b_f32.T
        if bias is not None and bias.numel():
            y = y + bias
        return y.to(output_dtype)

# --- end inlined fp8_reference ---


def _dequant_act(a: torch.Tensor, sa: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    Kb = K // 128
    return (a.view(M, Kb, 128).to(torch.bfloat16) * sa.to(torch.bfloat16).unsqueeze(2)).view(M, K)


def _dequant_wt(w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    N, K = w.shape
    Kb = K // 128
    Nb = N // 128
    return (w.view(Nb, 128, Kb, 128).to(torch.bfloat16) * sw.to(torch.bfloat16).view(Nb, 1, Kb, 1)).view(N, K)


@torch.no_grad()
def run(x, scale_x, gate_proj_weight, scale_gate, up_proj_weight, scale_up):
    xb = _dequant_act(x, scale_x)
    wgb = _dequant_wt(gate_proj_weight, scale_gate)
    wub = _dequant_wt(up_proj_weight, scale_up)
    gate_output = xb @ wgb.T
    up_output = xb @ wub.T
    gate_activated = F.silu(gate_output)
    output = gate_activated * up_output
    return output
