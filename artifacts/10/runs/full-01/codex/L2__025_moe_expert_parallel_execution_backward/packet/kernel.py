import torch
import torch.nn.functional as F


def _grouped_mm(a, b, offsets):
    return torch._grouped_mm(a, b, offsets)


@torch.no_grad()
def run(grad_output, hidden_states, topk_indices, topk_weights,
        gate_weights, up_weights, down_weights):
    # A stable expert sort produces exactly the same row order as
    # torch.where(expert_mask[e]) in the reference, but lets all experts be
    # submitted to the BLAS grouped-GEMM implementation together.
    n_tokens, hidden = hidden_states.shape
    n_experts, intermediate_size, _ = gate_weights.shape
    k = topk_indices.shape[1]
    assignment_experts, assignment_order = torch.sort(
        topk_indices.reshape(-1), stable=True)
    token_order = torch.div(assignment_order, k, rounding_mode="floor")

    counts = torch.bincount(assignment_experts, minlength=n_experts)
    offsets = counts.cumsum(0, dtype=torch.int32)
    starts = offsets.to(torch.int64) - counts
    positions = (torch.arange(assignment_experts.numel(),
                              device=assignment_experts.device)
                 - torch.repeat_interleave(starts, counts))

    x = hidden_states.index_select(0, token_order)
    dy = grad_output.index_select(0, token_order)
    routing = topk_weights.reshape(-1).index_select(0, assignment_order)

    gate_pre = _grouped_mm(x, gate_weights.transpose(1, 2), offsets)
    gate = F.silu(gate_pre)
    up = _grouped_mm(x, up_weights.transpose(1, 2), offsets)
    intermediate = gate * up
    if n_tokens < 3000:
        # For this exact (M,K,N,transpose) regime rocBLAS's strided-batched
        # solution is bitwise identical to its single-GEMM solution for
        # 49..96 rows.  Handle those experts together and retain the grouped
        # reference path for all count outliers.
        cap = 96
        fast_experts = (counts >= 49) & (counts <= cap)
        fast_rows = fast_experts.index_select(0, assignment_experts)
        all_rows = torch.arange(intermediate.shape[0], device=intermediate.device)
        fast_indices = all_rows[fast_rows]
        slow_indices = all_rows[~fast_rows]
        cap_slots = assignment_experts * cap + positions
        fast_slots = cap_slots[fast_rows]
        intermediate_batch = intermediate.new_zeros(
            (n_experts * cap, intermediate_size))
        intermediate_batch.index_copy_(
            0, fast_slots, intermediate.index_select(0, fast_indices))
        intermediate_batch = intermediate_batch.view(
            n_experts, cap, intermediate_size)
        batch_output = torch.bmm(
            intermediate_batch, down_weights.transpose(1, 2))

        slow_counts = torch.where(fast_experts, 0, counts)
        slow_offsets = slow_counts.cumsum(0, dtype=torch.int32)
        slow_output = _grouped_mm(
            intermediate.index_select(0, slow_indices),
            down_weights.transpose(1, 2), slow_offsets)
        expert_output = hidden_states.new_empty(
            (intermediate.shape[0], hidden))
        expert_output.index_copy_(
            0, fast_indices,
            batch_output.view(-1, hidden).index_select(0, fast_slots))
        expert_output.index_copy_(0, slow_indices, slow_output)
    else:
        expert_output = _grouped_mm(
            intermediate, down_weights.transpose(1, 2), offsets)

    grad_expert_output = dy * routing.unsqueeze(1)
    grad_routing_sorted = (dy * expert_output).sum(dim=1)
    grad_topk_weights = torch.empty_like(topk_weights)
    grad_topk_weights.reshape(-1).index_copy_(
        0, assignment_order, grad_routing_sorted)

    grad_intermediate = _grouped_mm(
        grad_expert_output, down_weights, offsets)

    grad_gate_output = grad_intermediate * up
    grad_up_output = grad_intermediate * gate
    sigmoid_gate = torch.sigmoid(gate_pre)
    silu_grad = sigmoid_gate * (
        1.0 + gate_pre * (1.0 - sigmoid_gate))
    grad_gate_pre = grad_gate_output * silu_grad

    grad_x_gate = _grouped_mm(grad_gate_pre, gate_weights, offsets)
    grad_x_up = _grouped_mm(grad_up_output, up_weights, offsets)
    grad_x_sorted = grad_x_gate + grad_x_up

    # Restore the reference's expert-ascending accumulation order.  Each token
    # has unique experts, so its eight additions can be done without atomics.
    grad_x_by_route = torch.empty_like(grad_x_sorted)
    grad_x_by_route.index_copy_(0, assignment_order, grad_x_sorted)
    grad_x_by_route = grad_x_by_route.view(n_tokens, k, hidden)
    expert_rank = torch.argsort(topk_indices, dim=1)
    grad_hidden_states = torch.zeros_like(hidden_states)
    for j in range(k):
        route = expert_rank[:, j]
        contribution = torch.gather(
            grad_x_by_route, 1,
            route[:, None, None].expand(-1, 1, hidden)).squeeze(1)
        grad_hidden_states.add_(contribution)

    # BLAS strided-batched GEMM is considerably faster than 256 individual
    # launches for the three parameter gradients.  Padding is only along the
    # relatively small expert-token axis; trailing zero rows preserve the
    # reference GEMM's accumulation exactly.
    max_count = int(counts.max().item())
    slots = assignment_experts * max_count + positions

    def padded(values):
        out = values.new_zeros((n_experts * max_count, values.shape[1]))
        out.index_copy_(0, slots, values)
        return out.view(n_experts, max_count, values.shape[1])

    x_pad = padded(x)
    expert_grad_pad = padded(grad_expert_output)
    intermediate_pad = padded(intermediate)
    grad_down_weights = torch.bmm(
        expert_grad_pad.transpose(1, 2), intermediate_pad)

    gate_grad_pad = padded(grad_gate_pre)
    grad_gate_weights = torch.bmm(gate_grad_pad.transpose(1, 2), x_pad)
    up_grad_pad = padded(grad_up_output)
    grad_up_weights = torch.bmm(up_grad_pad.transpose(1, 2), x_pad)

    return (grad_hidden_states, grad_topk_weights, grad_gate_weights,
            grad_up_weights, grad_down_weights)
