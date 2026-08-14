import torch
import triton
import triton.language as tl


@triton.jit
def _routing_backward(
    routing_probs,
    selected_experts,
    routing_weights_sum,
    grad_routing_weights,
    grad_expert_mask,
    grad_router_logits,
    grad_logits,
    num_tokens: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, 128)
    slot = tl.arange(0, 8)

    probs = tl.load(routing_probs + token * 128 + expert)
    direct = tl.load(
        grad_router_logits + token * 128 + expert
    ).to(tl.float32)
    selected = tl.load(selected_experts + token * 8 + slot)
    grad_weights = tl.load(
        grad_routing_weights + token * 8 + slot
    ).to(tl.float32)
    selected_probs = tl.load(routing_probs + token * 128 + selected)
    weight_sum = tl.load(routing_weights_sum + token)

    grad_sum = tl.sum(
        grad_weights * selected_probs / weight_sum, axis=0
    )
    sparse_grad = (
        grad_weights / weight_sum - grad_sum / weight_sum
    )
    sparse_grad += tl.load(
        grad_expert_mask
        + selected * (8 * num_tokens)
        + slot * num_tokens
        + token
    ).to(tl.float32)

    # Splitting the softmax dot into its dense direct term and eight sparse
    # terms avoids materializing a 128x8 comparison tile.
    dot = tl.sum(direct * probs, axis=0) + tl.sum(
        sparse_grad * selected_probs, axis=0
    )
    tl.store(
        grad_logits + token * 128 + expert,
        probs * (direct - dot),
    )
    selected_direct = tl.load(
        grad_router_logits + token * 128 + selected
    ).to(tl.float32)
    tl.store(
        grad_logits + token * 128 + selected,
        selected_probs * (selected_direct + sparse_grad - dot),
    )


@triton.jit
def _routing_backward_two_tokens(
    routing_probs,
    selected_experts,
    routing_weights_sum,
    grad_routing_weights,
    grad_expert_mask,
    grad_router_logits,
    grad_logits,
    num_tokens: tl.constexpr,
):
    token = tl.program_id(0) * 2 + tl.arange(0, 2)
    token_mask = token < num_tokens
    expert = tl.arange(0, 128)
    slot = tl.arange(0, 8)

    probs = tl.load(
        routing_probs + token[:, None] * 128 + expert[None, :],
        mask=token_mask[:, None],
        other=0.0,
    )
    direct = tl.load(
        grad_router_logits + token[:, None] * 128 + expert[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    selected = tl.load(
        selected_experts + token[:, None] * 8 + slot[None, :],
        mask=token_mask[:, None],
        other=0,
    )
    grad_weights = tl.load(
        grad_routing_weights + token[:, None] * 8 + slot[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    selected_probs = tl.load(
        routing_probs + token[:, None] * 128 + selected,
        mask=token_mask[:, None],
        other=0.0,
    )
    weight_sum = tl.load(
        routing_weights_sum + token,
        mask=token_mask,
        other=1.0,
    )
    grad_sum = tl.sum(
        grad_weights * selected_probs / weight_sum[:, None], axis=1
    )
    sparse_grad = (
        grad_weights / weight_sum[:, None]
        - grad_sum[:, None] / weight_sum[:, None]
    )
    sparse_grad += tl.load(
        grad_expert_mask
        + selected * (8 * num_tokens)
        + slot[None, :] * num_tokens
        + token[:, None],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    dot = tl.sum(direct * probs, axis=1) + tl.sum(
        sparse_grad * selected_probs, axis=1
    )
    tl.store(
        grad_logits + token[:, None] * 128 + expert[None, :],
        probs * (direct - dot[:, None]),
        mask=token_mask[:, None],
    )
    selected_direct = tl.load(
        grad_router_logits + token[:, None] * 128 + selected,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_logits + token[:, None] * 128 + selected,
        selected_probs
        * (selected_direct + sparse_grad - dot[:, None]),
        mask=token_mask[:, None],
    )

@torch.no_grad()
def run(
    hidden_states,
    gate_weight,
    router_logits,
    routing_probs,
    selected_experts,
    routing_weights_sum,
    grad_routing_weights,
    grad_expert_mask,
    grad_router_logits,
):
    num_tokens = hidden_states.shape[0]
    grad_logits = torch.empty_like(router_logits)
    args = (
        routing_probs,
        selected_experts,
        routing_weights_sum,
        grad_routing_weights,
        grad_expert_mask,
        grad_router_logits,
        grad_logits,
        num_tokens,
    )
    if num_tokens > 16384:
        _routing_backward_two_tokens[(triton.cdiv(num_tokens, 2),)](
            *args, num_warps=1
        )
    else:
        _routing_backward[(num_tokens,)](*args, num_warps=1)

    grad_hidden = torch.mm(grad_logits, gate_weight)
    grad_gate = torch.mm(grad_logits.t(), hidden_states)
    return grad_hidden, grad_gate
