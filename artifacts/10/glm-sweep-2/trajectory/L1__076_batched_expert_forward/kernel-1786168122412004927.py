import torch
import triton
import triton.language as tl


@triton.jit
def _fused_act_kernel(
    gate_up_ptr,         # (E, T, 2D)
    bias_ptr,            # (E, 2D)
    gated_out_ptr,       # (E, T, D)
    E, T, D2,
    alpha,
    limit,
    stride_ge, stride_gt, stride_gd,
    stride_be, stride_bd,
    stride_oe, stride_ot, stride_od,
    BLOCK_D: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < (D2 // 2)

    # gate indices: even -> 0,2,4,...  up indices: odd -> 1,3,5,...
    gate_idx = 2 * d_off
    up_idx = 2 * d_off + 1

    # Load gate value: gate_up[e, t, 2*d]
    g_ptr = gate_up_ptr + pid_e * stride_ge + pid_t * stride_gt
    gate = tl.load(g_ptr + gate_idx * stride_gd, mask=d_mask, other=0.0).to(tl.float32)
    up = tl.load(g_ptr + up_idx * stride_gd, mask=d_mask, other=0.0).to(tl.float32)

    # Load bias
    b_ptr = bias_ptr + pid_e * stride_be
    gate_b = tl.load(b_ptr + gate_idx * stride_bd, mask=d_mask, other=0.0).to(tl.float32)
    up_b = tl.load(b_ptr + up_idx * stride_bd, mask=d_mask, other=0.0).to(tl.float32)

    gate = gate + gate_b
    up = up + up_b

    # clamp
    gate = tl.minimum(gate, limit)
    up = tl.maximum(up, -limit)
    up = tl.minimum(up, limit)

    # GLU: gate * sigmoid(gate*alpha) * (up+1)
    glu = gate * tl.sigmoid(gate * alpha)
    out = (up + 1.0) * glu

    # Store gated_out[e, t, d]
    o_ptr = gated_out_ptr + pid_e * stride_oe + pid_t * stride_ot
    tl.store(o_ptr + d_off * stride_od, out, mask=d_mask)


def _fused_activation(gate_up, gate_up_bias, alpha, limit):
    E, T, D2 = gate_up.shape
    D = D2 // 2
    gated_out = torch.empty((E, T, D), device=gate_up.device, dtype=gate_up.dtype)
    BLOCK_D = 256
    grid = (E, T, triton.cdiv(D, BLOCK_D))
    _fused_act_kernel[grid](
        gate_up, gate_up_bias, gated_out,
        E, T, D2,
        float(alpha), float(limit),
        gate_up.stride(0), gate_up.stride(1), gate_up.stride(2),
        gate_up_bias.stride(0), gate_up_bias.stride(1),
        gated_out.stride(0), gated_out.stride(1), gated_out.stride(2),
        BLOCK_D=BLOCK_D,
    )
    return gated_out


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
    hidden_batched = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)

    gate_up = torch.matmul(hidden_batched, gate_up_proj)

    # Fused: bias + split + clamp + GLU -> gated_output
    gated_output = _fused_activation(gate_up, gate_up_proj_bias, alpha, limit)

    expert_outputs = torch.bmm(gated_output, down_proj)
    expert_outputs = expert_outputs + down_proj_bias.unsqueeze(1)

    expert_outputs = expert_outputs.view(num_experts, batch_size, seq_len, hidden_size)
    routing_weights_reshaped = routing_weights.transpose(0, 1).view(
        num_experts, batch_size, seq_len, 1
    )
    output = (expert_outputs * routing_weights_reshaped).sum(dim=0)
    return output
