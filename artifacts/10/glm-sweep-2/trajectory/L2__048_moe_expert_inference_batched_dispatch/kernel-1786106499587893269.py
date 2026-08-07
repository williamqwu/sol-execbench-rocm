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

    # Keep weights in bf16; convert only the small input/output tensors.
    hidden_states_f32 = hidden_states.to(torch.float32)
    topk_weight_f32 = topk_weight.to(torch.float32)

    # Step 1: Count tokens per expert using scatter
    cnts = torch.zeros(
        (num_tokens, num_experts),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    cnts.scatter_(1, topk_idx, 1)
    tokens_per_expert = cnts.sum(dim=0)  # [num_experts]

    # Step 2: Sort tokens by expert assignment
    flat_topk_idx = topk_idx.view(-1)
    idxs = flat_topk_idx.argsort()
    token_indices = idxs // num_experts_per_tok
    sorted_tokens = hidden_states_f32[token_indices]

    # Step 3: Process tokens through experts in batches (bf16 weights, fp32 input)
    # Cast sorted_tokens to bf16 for the matmul (MFMA bf16->fp32 accumulate)
    sorted_tokens_bf16 = sorted_tokens.to(torch.bfloat16)

    outputs = []
    start_idx = 0
    tokens_per_expert_cpu = tokens_per_expert.cpu().numpy()

    for expert_id in range(num_experts):
        num_tokens_for_expert = int(tokens_per_expert_cpu[expert_id])
        if num_tokens_for_expert == 0:
            continue
        end_idx = start_idx + num_tokens_for_expert
        expert_input = sorted_tokens_bf16[start_idx:end_idx]

        # bf16 matmul -> fp32 output (default accumulation)
        gate_out = torch.matmul(expert_input, gate_weights[expert_id].t())
        up_out = torch.matmul(expert_input, up_weights[expert_id].t())

        # SwiGLU in fp32
        activated = (gate_out * torch.sigmoid(gate_out)) * up_out
        activated_bf16 = activated.to(torch.bfloat16)

        # down_proj: bf16 -> fp32
        expert_output = torch.matmul(activated_bf16, down_weights[expert_id].t())
        outputs.append(expert_output)
        start_idx = end_idx

    if len(outputs) == 0:
        return torch.zeros_like(hidden_states)

    all_outputs = torch.cat(outputs, dim=0)
    all_outputs_f32 = all_outputs.to(torch.float32)

    new_x = torch.zeros_like(all_outputs_f32)
    new_x[idxs] = all_outputs_f32

    new_x = new_x.view(num_tokens, num_experts_per_tok, hidden_size)
    weighted_output = new_x * topk_weight_f32.unsqueeze(-1)
    final_output = weighted_output.sum(dim=1)

    return final_output.to(torch.bfloat16)
