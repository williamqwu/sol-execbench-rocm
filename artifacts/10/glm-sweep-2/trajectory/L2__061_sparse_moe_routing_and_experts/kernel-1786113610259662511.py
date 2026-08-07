import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    expert_gate_proj: torch.Tensor,
    expert_up_proj: torch.Tensor,
    expert_down_proj: torch.Tensor,
    shared_gate_proj: torch.Tensor,
    shared_up_proj: torch.Tensor,
    shared_down_proj: torch.Tensor,
    shared_expert_gate_weight: torch.Tensor,
):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    num_experts = router_weight.shape[0]
    top_k = 10
    inter_size = expert_gate_proj.shape[1]

    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = batch_size * sequence_length

    # Router
    router_logits = torch.matmul(hidden_states_flat, router_weight.t())
    routing_weights = F.softmax(router_logits.float(), dim=1)
    routing_weights_topk, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights_topk = routing_weights_topk / routing_weights_topk.sum(dim=-1, keepdim=True)
    routing_weights_topk = routing_weights_topk.to(torch.bfloat16)

    # Flatten (token, slot) pairs and sort by expert id
    sel_flat = selected_experts.reshape(-1)
    tok_idx = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)
    order = torch.argsort(sel_flat, stable=True)
    sel_sorted = sel_flat[order]
    tok_sorted = tok_idx[order]

    # Per-expert token counts and capacity
    counts = torch.bincount(sel_sorted, minlength=num_experts)
    cap = int(counts.max().item())
    if cap == 0:
        cap = 1

    offsets = torch.cumsum(counts, 0)
    starts = offsets - counts
    ar = torch.arange(cap, device=hidden_states.device)
    mask = ar[None, :] < counts[:, None]
    pos = (starts[:, None] + ar[None, :]).clamp(max=tok_sorted.numel() - 1)
    pad_tok = torch.where(mask, tok_sorted[pos], torch.zeros((), device=hidden_states.device, dtype=torch.long))
    pad_slot = torch.where(mask, order[pos] % top_k, torch.zeros((), device=hidden_states.device, dtype=torch.long))

    # Gather tokens for all experts: (num_experts, cap, hidden_dim)
    x = hidden_states_flat[pad_tok]
    x = x * mask[:, :, None].to(x.dtype)

    gate_w = expert_gate_proj.transpose(1, 2)
    up_w = expert_up_proj.transpose(1, 2)
    down_w = expert_down_proj.transpose(1, 2)

    gate_out = torch.bmm(x, gate_w)
    up_out = torch.bmm(x, up_w)
    silu_out = gate_out * torch.sigmoid(gate_out)
    intermediate = silu_out * up_out
    expert_output = torch.bmm(intermediate, down_w)

    # Weight by routing weights (zero invalid slots via weight, not output mask)
    w_for_pad = routing_weights_topk[pad_tok, pad_slot]  # (E, cap)
    w_for_pad = w_for_pad * mask.to(w_for_pad.dtype)
    expert_output = expert_output * w_for_pad[:, :, None]

    # Scatter-add back
    final_hidden_states = torch.zeros(
        (num_tokens, hidden_dim), dtype=torch.bfloat16, device=hidden_states.device
    )
    final_hidden_states.index_add_(0, pad_tok.reshape(-1), expert_output.reshape(-1, hidden_dim))

    # Shared expert
    shared_gate_out = torch.matmul(hidden_states_flat, shared_gate_proj.t())
    shared_up_out = torch.matmul(hidden_states_flat, shared_up_proj.t())
    shared_silu = shared_gate_out * torch.sigmoid(shared_gate_out)
    shared_intermediate = shared_silu * shared_up_out
    shared_expert_output = torch.matmul(shared_intermediate, shared_down_proj.t())
    shared_gate = torch.sigmoid(torch.matmul(hidden_states_flat, shared_expert_gate_weight.t()))
    shared_expert_output = shared_gate * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output
    output = final_hidden_states.view(batch_size, sequence_length, hidden_dim)

    return output, router_logits.to(torch.bfloat16)
