import torch
import torch.nn.functional as F
import aiter


E4M3_MAX = 448.0
_quant_1x128 = aiter.get_hip_quant(aiter.QuantType.per_1x128)


def _activation_scales(x):
    m, k = x.shape
    return torch.clamp(
        x.reshape(m, k // 128, 128).abs().amax(dim=2) / E4M3_MAX,
        min=1.0e-12,
    )


def _weight_scales_transposed(w_t):
    m, k = w_t.shape
    return torch.clamp(
        w_t.reshape(m // 128, 128, k // 128, 128).abs().amax(dim=3).amax(dim=1)
        / E4M3_MAX,
        min=1.0e-12,
    )


def _weight_scales(w):
    n, k = w.shape
    return torch.clamp(
        w.reshape(n // 128, 128, k // 128, 128).abs().amax(dim=3).amax(dim=1)
        / E4M3_MAX,
        min=1.0e-12,
    )


def _quantize_activation(x, scales):
    m, k = x.shape
    y = x.reshape(m, k // 128, 128) / scales.unsqueeze(2)
    y = torch.clamp(y, min=-E4M3_MAX, max=E4M3_MAX)
    return y.reshape(m, k).to(torch.float8_e4m3fn)


def _quantize_activation_hip(x):
    return _quant_1x128(x, None, torch.float8_e4m3fn)


def _quantize_weight_t(w_t, scales):
    m, k = w_t.shape
    y = w_t.reshape(m // 128, 128, k // 128, 128) / scales.unsqueeze(1).unsqueeze(3)
    y = torch.clamp(y, min=-E4M3_MAX, max=E4M3_MAX)
    return y.reshape(m, k).to(torch.float8_e4m3fn)


def _quantize_weight(w, scales):
    n, k = w.shape
    y = w.reshape(n // 128, 128, k // 128, 128) / scales.unsqueeze(1).unsqueeze(3)
    y = torch.clamp(y, min=-E4M3_MAX, max=E4M3_MAX)
    return y.reshape(n, k).to(torch.float8_e4m3fn)


def _quantize_weight_full(w):
    w_fp32 = w.to(torch.float32)
    scales = _weight_scales(w_fp32)
    return _quantize_weight(w_fp32, scales), scales


_quantize_weight_full_compiled = torch.compile(
    _quantize_weight_full, dynamic=False, fullgraph=True
)


def _scaled_mm(a_fp8, b_fp8, scale_a, scale_b_cublas):
    m = a_fp8.shape[0]
    n = b_fp8.shape[0]
    out = torch.empty((m, n), device=a_fp8.device, dtype=torch.bfloat16)
    kernel_id = 0
    if (n == 4096 and (m == 384 or m == 1152)) or (
        n == 3584 and (m == 384 or m == 1536)
    ):
        kernel_id = 2
    return aiter.gemm_a8w8_blockscale_tune(
        a_fp8, b_fp8, scale_a, scale_b_cublas, out, kernel_id, 0
    )


@torch.no_grad()
def run(hidden_states, routing_weight, gate_up_weight, down_weight):
    hidden_fp8, scale_hidden = _quantize_activation_hip(hidden_states)
    gate_up_fp8, scale_gate_up = _quantize_weight_full_compiled(gate_up_weight)
    gate_up_out = _scaled_mm(hidden_fp8, gate_up_fp8, scale_hidden, scale_gate_up)

    gate, up = gate_up_out.chunk(2, dim=-1)
    gated = F.silu(gate) * up

    gated_fp8, scale_gated = _quantize_activation_hip(gated)
    down_fp8, scale_down = _quantize_weight_full_compiled(down_weight)
    out = _scaled_mm(gated_fp8, down_fp8, scale_gated, scale_down)

    return out * routing_weight
