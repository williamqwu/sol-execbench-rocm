import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _select_kernel(scores_ptr, bias_ptr, idx_ptr, weight_ptr, scaling_factor):
    row = tl.program_id(0)
    expert = tl.arange(0, 256)

    scores = tl.load(scores_ptr + row * 256 + expert)
    bias = tl.load(bias_ptr + expert).to(tl.float32)
    routing = scores + bias

    grouped = tl.reshape(routing, (8, 32))
    top_two = tl.topk(grouped, 2, dim=1)
    group_scores = tl.sum(top_two, axis=1)

    group = tl.arange(0, 8)
    group_signed_bits = group_scores.to(tl.int32, bitcast=True)
    group_unsigned_bits = group_signed_bits.to(tl.int64) & 0xFFFFFFFF
    group_ordered_bits = tl.where(
        group_signed_bits < 0,
        0xFFFFFFFF - group_unsigned_bits,
        group_unsigned_bits ^ 0x80000000,
    )
    group_keys = group_ordered_bits * 8 + (7 - group).to(tl.int64)
    top_group_keys = tl.topk(group_keys, 4, dim=0)
    selected_group_idx = 7 - (top_group_keys & 7)
    group_matches = group[:, None] == selected_group_idx[None, :]
    selected_groups = tl.sum(group_matches.to(tl.int32), axis=1) != 0

    selected_matrix = tl.broadcast_to(selected_groups[:, None], (8, 32))
    selected_experts = tl.reshape(selected_matrix, (256,))

    # A sortable 40-bit key preserves every float bit and uses the low byte to
    # reproduce topk's lower-index tie break.
    signed_bits = routing.to(tl.int32, bitcast=True)
    unsigned_bits = signed_bits.to(tl.int64) & 0xFFFFFFFF
    ordered_bits = tl.where(
        signed_bits < 0,
        0xFFFFFFFF - unsigned_bits,
        unsigned_bits ^ 0x80000000,
    )
    keys = ordered_bits * 256 + (255 - expert).to(tl.int64)
    keys = tl.where(selected_experts, keys, -1)
    top_keys = tl.topk(keys, 8, dim=0)
    selected_idx = 255 - (top_keys & 255)
    selected_scores = tl.load(scores_ptr + row * 256 + selected_idx)
    pair_sums = tl.sum(tl.reshape(selected_scores, (4, 2)), axis=1)
    quad_sums = tl.sum(tl.reshape(pair_sums, (2, 2)), axis=1)
    denominator = tl.sum(quad_sums, axis=0) + 1.0e-20
    ranks = tl.arange(0, 8)
    base = row * 8 + ranks
    tl.store(idx_ptr + base, selected_idx.to(tl.int64))
    tl.store(weight_ptr + base, selected_scores / denominator * scaling_factor)


@torch.no_grad()
def run(hidden_states, weight, expert_bias, routed_scaling_factor):
    n_rows = hidden_states.shape[0]
    logits = F.linear(hidden_states.float(), weight.float())
    scores = logits.sigmoid_()
    topk_idx = torch.empty(
        (n_rows, 8), device=hidden_states.device, dtype=torch.int64
    )
    topk_weight = torch.empty(
        (n_rows, 8), device=hidden_states.device, dtype=torch.float32
    )
    if n_rows == 2048:
        waves_per_eu = 2
    elif n_rows < 3000:
        waves_per_eu = 3 if n_rows >= 2500 else 0
    elif n_rows < 5000:
        waves_per_eu = 4
    else:
        waves_per_eu = 5
    _select_kernel[(n_rows,)](
        scores,
        expert_bias,
        topk_idx,
        topk_weight,
        routed_scaling_factor,
        num_warps=1,
        waves_per_eu=waves_per_eu,
    )
    return topk_idx, topk_weight
