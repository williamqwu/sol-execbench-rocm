import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = axes_and_scalars["hidden_size"]
    moe_intermediate_size = axes_and_scalars["moe_intermediate_size"]
    num_experts = axes_and_scalars["num_experts"]
    num_experts_per_tok = axes_and_scalars["num_experts_per_tok"]

    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    selected_experts = torch.zeros(num_tokens, num_experts_per_tok, dtype=torch.int64, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:num_experts_per_tok]
        selected_experts[i] = perm

    routing_logits = torch.randn(num_tokens, num_experts_per_tok, dtype=dtype, device=device)
    routing_weights = F.softmax(routing_logits.float(), dim=-1).to(dtype)

    expert_gate_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    expert_up_weights = torch.randn(num_experts, hidden_size, moe_intermediate_size, dtype=dtype, device=device) / math.sqrt(hidden_size)
    expert_down_weights = torch.randn(num_experts, moe_intermediate_size, hidden_size, dtype=dtype, device=device) / math.sqrt(moe_intermediate_size)

    return {
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "expert_gate_weights": expert_gate_weights,
        "expert_up_weights": expert_up_weights,
        "expert_down_weights": expert_down_weights,
    }


@triton.jit
def _scatter_kernel(ei_ptr, src_ptr, flat_idx_ptr, N,
                    H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n_offs < N
    fi = tl.load(flat_idx_ptr + n_offs, mask=nmask, other=0)
    for h in range(0, H, BLOCK_H):
        h_offs = h + tl.arange(0, BLOCK_H)
        hmask = h_offs < H
        vals = tl.load(src_ptr + n_offs[:, None].to(tl.int64) * H + h_offs[None, :],
                       mask=nmask[:, None] & hmask[None, :], other=0.0)
        tl.store(ei_ptr + fi[:, None].to(tl.int64) * H + h_offs[None, :],
                 vals.to(tl.bfloat16), mask=nmask[:, None] & hmask[None, :])


@triton.jit
def _gather_weight_kernel(out_ptr, eo_ptr, flat_idx_ptr, wt_ptr, N,
                          H: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n_offs < N
    fi = tl.load(flat_idx_ptr + n_offs, mask=nmask, other=0)
    w = tl.load(wt_ptr + n_offs, mask=nmask, other=0.0)
    for h in range(0, H, BLOCK_H):
        h_offs = h + tl.arange(0, BLOCK_H)
        hmask = h_offs < H
        vals = tl.load(eo_ptr + fi[:, None].to(tl.int64) * H + h_offs[None, :],
                       mask=nmask[:, None] & hmask[None, :], other=0.0)
        vals = vals * w[:, None]
        tl.store(out_ptr + n_offs[:, None].to(tl.int64) * H + h_offs[None, :],
                 vals.to(tl.bfloat16), mask=nmask[:, None] & hmask[None, :])


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_gate_weights: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, moe_intermediate_size = expert_gate_weights.shape
    num_experts_per_tok = selected_experts.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    capacity = max(int((num_tokens * num_experts_per_tok / num_experts) * 1.25), 1)

    # Flatten all token-expert assignments: (num_tokens * K,)
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    flat_token_ids = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_experts_per_tok).reshape(-1)

    # Sort by expert ID (stable to match original sequential assignment order)
    sorted_experts, sorted_indices = flat_experts.sort(stable=True)
    sorted_weights = flat_weights[sorted_indices]
    sorted_token_ids = flat_token_ids[sorted_indices]

    # Vectorized within-expert position computation.
    counts = torch.bincount(sorted_experts, minlength=num_experts)
    starts = torch.zeros(num_experts, dtype=torch.long, device=device)
    starts[1:] = counts[:-1].cumsum(0)

    within_pos = torch.arange(len(sorted_experts), device=device) - starts[sorted_experts]

    # Apply capacity constraint
    valid = within_pos < capacity
    v_exp = sorted_experts[valid]
    v_pos = within_pos[valid]
    v_tok = sorted_token_ids[valid]
    v_wt = sorted_weights[valid]
    num_valid = v_exp.numel()

    # Flat index into [num_experts, capacity, hidden_size] viewed as [num_experts*capacity, hidden_size]
    flat_scatter_idx = (v_exp.to(torch.int64) * capacity + v_pos).to(torch.int32)

    # Gather source tokens contiguously
    src = hidden_states[v_tok]

    # Scatter tokens into padded expert batches via Triton (fused gather-source + scatter-write)
    expert_inputs = torch.zeros(num_experts, capacity, hidden_size, dtype=dtype, device=device)
    BLOCK_N = 64
    BLOCK_H = 256
    grid_s = ((num_valid + BLOCK_N - 1) // BLOCK_N,)
    _scatter_kernel[grid_s](expert_inputs, src, flat_scatter_idx, num_valid,
                            H=hidden_size, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, num_warps=4)

    # Batched expert forward pass
    gate_out = torch.bmm(expert_inputs, expert_gate_weights)
    up_out = torch.bmm(expert_inputs, expert_up_weights)

    # SwiGLU with in-place mul
    activated = F.silu(gate_out).mul_(up_out)

    expert_outputs = torch.bmm(activated, expert_down_weights)

    # Gather + weight via Triton (fused), then index_add aggregation
    weighted_out = torch.empty(num_valid, hidden_size, dtype=dtype, device=device)
    _gather_weight_kernel[grid_s](weighted_out, expert_outputs, flat_scatter_idx, v_wt, num_valid,
                                  H=hidden_size, BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, num_warps=4)

    result = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    result.index_add_(0, v_tok, weighted_out)

    return result
