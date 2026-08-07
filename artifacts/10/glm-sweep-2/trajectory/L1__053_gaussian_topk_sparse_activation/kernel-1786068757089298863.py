import torch
import torch.nn.functional as F
import math


def _ndtri_scalar(p: float) -> float:
    """Inverse of the standard normal CDF (quantile function), pure Python."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / (
            ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


@torch.compile(dynamic=True, fullgraph=True)
def _fused_compute(inputs: torch.Tensor, std_multiplier: float):
    inputs_f32 = inputs.to(torch.float32)
    inputs_mean = inputs_f32.mean(dim=-1, keepdim=True)
    inputs_sq_mean = (inputs_f32 * inputs_f32).mean(dim=-1, keepdim=True)
    inputs_var = (inputs_sq_mean - inputs_mean * inputs_mean).clamp(min=0.0)
    inputs_std = torch.sqrt(inputs_var)
    cutoff_threshold = inputs_mean + inputs_std * std_multiplier
    sparse_output = F.relu(inputs_f32 - cutoff_threshold)
    return sparse_output.to(torch.bfloat16)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs

    std_multiplier = _ndtri_scalar(target_sparsity)
    return _fused_compute(inputs, std_multiplier)
