import os
import torch

_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>

// Fused gated-GLU activation, bit-exact to eager reference:
//   gate = gate_up[2i]; up = gate_up[2i+1]
//   gate = min(gate, limit); up = min(max(up,-limit), limit)
//   out  = (up + 1) * (gate * sigmoid(gate*alpha))
//   sigmoid(z) = 1/(1+expf(-z))  -- matches torch.sigmoid on ROCm (accurate expf)
// NO fast-math / FMA contraction: ops emitted in reference order.
__global__ void glu_kernel(const float* __restrict__ gate_up,
                           float* __restrict__ out,
                           int64_t n, float alpha, float limit) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate_up[2 * idx];
    float u = gate_up[2 * idx + 1];
    g = fminf(g, limit);
    u = fmaxf(u, -limit);
    u = fminf(u, limit);
    float ga = g * alpha;
    float sg = 1.0f / (1.0f + expf(-ga));
    float glu = g * sg;
    float up1 = u + 1.0f;
    out[idx] = up1 * glu;
}

void fused_glu(torch::Tensor gate_up, torch::Tensor out, double alpha, double limit) {
    int64_t n = out.numel();
    int64_t threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    glu_kernel<<<(unsigned int)blocks, (unsigned int)threads>>>(
        gate_up.data_ptr<float>(), out.data_ptr<float>(), n,
        (float)alpha, (float)limit);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_glu", &fused_glu, "Fused gated GLU (bit-exact)");
}
"""

_glu_mod = None
def _glu():
    global _glu_mod
    if _glu_mod is None:
        from torch.utils.cpp_extension import load
        _src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_glu_ext.cu")
        with open(_src_path, "w") as f:
            f.write(_SRC)
        _glu_mod = load(name="_glu_ext", sources=[_src_path],
                        extra_cuda_cflags=["-O3"], verbose=False)
    return _glu_mod

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    alpha: float,
    limit: float,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    num_experts = gate_up_proj.shape[0]
    expert_dim = down_proj.shape[1]

    hidden_flat = hidden_states.reshape(-1, hidden_size)
    hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, hidden_size)

    gate_up = torch.bmm(hidden_batched, gate_up_proj)
    gate_up = gate_up + gate_up_proj_bias.unsqueeze(1)

    # Fused gated-GLU (bit-exact, single kernel).
    gated_output = torch.empty(
        num_experts, hidden_flat.shape[0], expert_dim,
        device=gate_up.device, dtype=gate_up.dtype)
    _glu().fused_glu(gate_up, gated_output, float(alpha), float(limit))

    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)

    expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
    routing_weights_reshaped = routing_weights.transpose(0, 1).view(
        num_experts, batch_size, seq_len, 1
    )
    output = (expert_outputs * routing_weights_reshaped).sum(dim=0)
    return output
