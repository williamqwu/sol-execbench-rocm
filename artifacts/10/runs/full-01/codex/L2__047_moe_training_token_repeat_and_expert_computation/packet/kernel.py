import torch
import torch.nn.functional as F


_UP_STREAM = None


@torch.no_grad()
def run(
    hidden_states,
    topk_idx,
    topk_weight,
    expert_gate_projs,
    expert_up_projs,
    expert_down_projs,
):
    """Dense strided-batched MoE forward pass."""
    n, hidden = hidden_states.shape
    k = topk_idx.shape[1]
    experts = expert_gate_projs.shape[0]
    routes = n * k

    # Pack routes by expert.  A dense leading M dimension lets rocBLAS use its
    # much faster strided-batched GEMM path.  Only the short tail of each expert
    # is padding; those rows never participate in the output.
    sorted_experts, order = torch.sort(topk_idx.reshape(-1))
    counts = torch.bincount(sorted_experts, minlength=experts)
    expert_m = int(counts.max().item())
    starts = counts.cumsum(0) - counts
    within_expert = torch.arange(routes, device=hidden_states.device) - starts[sorted_experts]

    padded_x = torch.empty(
        (experts, expert_m, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    padded_x[sorted_experts, within_expert] = hidden_states[
        torch.div(order, k, rounding_mode="floor")
    ]

    # The two first-layer projections are independent.  Separate streams let
    # rocBLAS overlap their under-filled expert tiles on the large CDNA device.
    global _UP_STREAM
    current_stream = torch.cuda.current_stream(hidden_states.device)
    if _UP_STREAM is None:
        _UP_STREAM = torch.cuda.Stream(device=hidden_states.device)
    _UP_STREAM.wait_stream(current_stream)
    with torch.cuda.stream(_UP_STREAM):
        up = torch.bmm(padded_x, expert_up_projs.transpose(1, 2))
    gate = torch.bmm(padded_x, expert_gate_projs.transpose(1, 2))
    current_stream.wait_stream(_UP_STREAM)
    up.record_stream(current_stream)
    intermediate = F.silu(gate) * up
    padded_y = torch.bmm(intermediate, expert_down_projs.transpose(1, 2))

    # Restore route order before multiplying and reducing, matching the
    # reference's fp16 elementwise and reduction rounding behavior.
    y = torch.empty((routes, hidden), device=hidden_states.device, dtype=hidden_states.dtype)
    y[order] = padded_y[sorted_experts, within_expert]
    y = y.view(n, k, hidden)
    return (y * topk_weight.unsqueeze(-1)).sum(dim=1)
