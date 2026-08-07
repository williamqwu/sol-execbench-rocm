import torch


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_experts = gate_weights.shape[0]
    num_experts_per_tok = topk_idx.shape[1]
    device = hidden_states.device

    topk_weight_f32 = topk_weight.to(torch.float32)

    # Step 1: Count tokens per expert
    cnts = torch.zeros(
        (num_tokens, num_experts), dtype=torch.int32, device=device
    )
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim=0)  # [num_experts]

    # Step 2: Sort token-expert pairs by expert id
    flat_topk_idx = topk_idx.view(-1)  # [num_tokens * ept]
    idxs = flat_topk_idx.argsort()  # sorted by expert id
    sorted_expert_ids = flat_topk_idx[idxs]
    token_indices = idxs // num_experts_per_tok
    sorted_tokens = hidden_states[token_indices]  # [N*ept, hidden], bf16

    # Step 3: Build padded [num_experts, max_tok, hidden] buffer
    offsets = torch.cumsum(tokens_per_expert, dim=0)
    start_offsets = offsets - tokens_per_expert
    max_tok = int(tokens_per_expert.max().item())

    n_pairs = num_tokens * num_experts_per_tok
    pos_in_expert = (torch.arange(n_pairs, device=device) - start_offsets[sorted_expert_ids]).to(torch.int32)

    padded = torch.zeros(num_experts, max_tok, hidden_size, dtype=torch.bfloat16, device=device)
    padded[sorted_expert_ids, pos_in_expert] = sorted_tokens

    # mask [num_experts, max_tok]
    ar = torch.arange(max_tok, device=device)
    valid_mask = ar[None, :] < tokens_per_expert[:, None]  # bool

    # Step 4: Batched GEMM (bf16 -> fp32 accumulate)
    # gate: [E, max_tok, H] @ [E, H, I] -> [E, max_tok, I]
    gate_out = torch.bmm(padded, gate_weights.transpose(1, 2))
    up_out = torch.bmm(padded, up_weights.transpose(1, 2))

    # SwiGLU in fp32, zero out invalid rows
    activated = (gate_out * torch.sigmoid(gate_out)) * up_out  # [E, max_tok, I], fp32
    activated = activated * valid_mask[:, :, None].to(activated.dtype)
    activated_bf16 = activated.to(torch.bfloat16)

    # down: [E, max_tok, I] @ [E, I, H] -> [E, max_tok, H]
    expert_output = torch.bmm(activated_bf16, down_weights.transpose(1, 2))  # fp32

    # Step 5: Gather back to sorted order
    # expert_output[sorted_expert_ids, pos_in_expert] -> [N*ept, hidden]
    all_outputs = expert_output[sorted_expert_ids, pos_in_expert]  # [N*ept, hidden]

    # Step 6: Scatter back to original token positions and apply weights
    new_x = torch.zeros(n_pairs, hidden_size, dtype=torch.float32, device=device)
    new_x[idxs] = all_outputs

    new_x = new_x.view(num_tokens, num_experts_per_tok, hidden_size)
    weighted_output = new_x * topk_weight_f32.unsqueeze(-1)
    final_output = weighted_output.sum(dim=1)

    return final_output.to(torch.bfloat16)
