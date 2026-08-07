import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_backward_kernel(
    grad_activated,
    up_output,
    gate_output,
    grad_outputs,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    ga = tl.load(grad_activated + offsets, mask=mask)
    up = tl.load(up_output + offsets, mask=mask)
    gate = tl.load(gate_output + offsets, mask=mask).to(tl.float32)

    # Match the eager implementation's BF16 intermediates.
    grad_gate_silu = (ga * up).to(tl.bfloat16)
    sigmoid_gate = tl.sigmoid(gate)
    silu_gate = (gate * sigmoid_gate).to(tl.bfloat16)
    grad_up = (ga * silu_gate).to(tl.bfloat16)
    silu_derivative = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_gate = (grad_gate_silu.to(tl.float32) * silu_derivative).to(tl.bfloat16)

    row = offsets // 1408
    column = offsets - row * 1408
    output_offsets = row * 2816 + column
    tl.store(grad_outputs + output_offsets, grad_up, mask=mask)
    tl.store(grad_outputs + output_offsets + 1408, grad_gate, mask=mask)


@triton.jit
def _scatter_router_kernel(
    grad_selected,
    topk_indices,
    score_mask,
    scores,
    grad_router_logits,
    grad_router_logits_bf16,
):
    token = tl.program_id(0)
    experts = tl.arange(0, 128)
    selected = tl.zeros((128,), tl.float32)
    for k in range(8):
        index = tl.load(topk_indices + token * 8 + k)
        value = tl.load(grad_selected + token * 8 + k)
        selected += tl.where(experts == index, value, 0.0)

    offsets = token * 128 + experts
    value = selected * tl.load(score_mask + offsets)
    score = tl.load(scores + offsets)
    value = value * score
    value = value * (1.0 - score)
    tl.store(grad_router_logits + offsets, value)
    tl.store(grad_router_logits_bf16 + offsets, value.to(tl.bfloat16))


@triton.jit
def _add_hidden_kernel(
    grad_up,
    grad_gate,
    grad_router,
    grad_hidden,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    up = tl.load(grad_up + offsets, mask=mask)
    gate = tl.load(grad_gate + offsets, mask=mask)
    router = tl.load(grad_router + offsets, mask=mask)
    shared = (up + gate).to(tl.bfloat16)
    result = (shared + router).to(tl.bfloat16)
    tl.store(grad_hidden + offsets, result, mask=mask)


@torch.no_grad()
def run(
    grad_output,
    hidden_states,
    router_weight,
    e_score_correction_bias,
    router_logits,
    scores,
    topk_indices,
    topk_weights,
    score_mask,
    shared_expert_gate_weight,
    shared_expert_up_weight,
    shared_expert_down_weight,
    shared_gate_output,
    shared_up_output,
    shared_activated,
):
    batch_seq_len = hidden_states.shape[0]

    grad_shared_activated = grad_output @ shared_expert_down_weight
    grad_shared_expert_down_weight = grad_output.t() @ shared_activated

    grad_shared_outputs = torch.empty(
        batch_seq_len, 2816, dtype=torch.bfloat16, device=hidden_states.device
    )
    n_elements = shared_gate_output.numel()
    _swiglu_backward_kernel[(triton.cdiv(n_elements, 1024),)](
        grad_shared_activated,
        shared_up_output,
        shared_gate_output,
        grad_shared_outputs,
        n_elements=n_elements,
        BLOCK=1024,
        num_warps=4,
    )

    grad_shared_up_output = grad_shared_outputs[:, :1408]
    grad_shared_gate_output = grad_shared_outputs[:, 1408:]
    grad_hidden_from_shared_up = grad_shared_up_output @ shared_expert_up_weight
    grad_shared_expert_weights = grad_shared_outputs.t() @ hidden_states
    grad_shared_expert_up_weight = grad_shared_expert_weights[:1408]
    grad_shared_expert_gate_weight = grad_shared_expert_weights[1408:]

    hidden_f32 = hidden_states.to(torch.float32)
    grad_output_f32 = grad_output.to(torch.float32)
    grad_norm_sq = (grad_output_f32 * grad_output_f32).sum(dim=-1, keepdim=True)
    grad_topk_weights = grad_norm_sq.expand_as(topk_weights) / 8
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    sum_grad = (grad_topk_weights * topk_weights).sum(dim=-1, keepdim=True) / denominator
    grad_topk_weights_before_norm = (grad_topk_weights - sum_grad) / denominator

    grad_router_logits = torch.empty(
        batch_seq_len, 128, dtype=torch.float32, device=hidden_states.device
    )
    grad_router_logits_bf16 = torch.empty(
        batch_seq_len, 128, dtype=torch.bfloat16, device=hidden_states.device
    )
    _scatter_router_kernel[(batch_seq_len,)](
        grad_topk_weights_before_norm,
        topk_indices,
        score_mask,
        scores,
        grad_router_logits,
        grad_router_logits_bf16,
        num_warps=4,
    )
    grad_router_weight = grad_router_logits.t() @ hidden_f32

    grad_hidden_from_shared_gate = grad_shared_gate_output @ shared_expert_gate_weight
    grad_hidden_from_router = grad_router_logits_bf16 @ router_weight
    if batch_seq_len < 3500:
        grad_hidden_states = (
            grad_hidden_from_shared_up
            + grad_hidden_from_shared_gate
            + grad_hidden_from_router
        )
    else:
        grad_hidden_states = torch.empty_like(hidden_states)
        hidden_elements = hidden_states.numel()
        _add_hidden_kernel[(triton.cdiv(hidden_elements, 512),)](
            grad_hidden_from_shared_up,
            grad_hidden_from_shared_gate,
            grad_hidden_from_router,
            grad_hidden_states,
            n_elements=hidden_elements,
            BLOCK=512,
            num_warps=4,
        )
    return (
        grad_hidden_states,
        grad_router_weight,
        grad_shared_expert_gate_weight,
        grad_shared_expert_up_weight,
        grad_shared_expert_down_weight,
    )
