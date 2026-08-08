import torch

_compiled_run = None


def _run_impl(
    hidden_states, gate_weight, router_logits, routing_probs, selected_experts,
    routing_weights_sum, grad_routing_weights, grad_expert_mask, grad_router_logits,
):
    input_dtype = hidden_states.dtype

    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, dim=1, index=selected_experts)
    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm / routing_weights_sum).sum(dim=-1, keepdim=True)
    grad_routing_weights_unnorm = grad_routing_weights_f32 / routing_weights_sum - grad_sum / routing_weights_sum

    grad_expert_mask_permuted = grad_expert_mask.permute(2, 1, 0)
    grad_from_mask = torch.gather(grad_expert_mask_permuted, dim=2, index=selected_experts.unsqueeze(2)).squeeze(2).float()

    combined = grad_routing_weights_unnorm + grad_from_mask
    grad_routing_probs = grad_router_logits.float()
    grad_routing_probs.scatter_add_(dim=1, index=selected_experts, src=combined)

    dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)
    return grad_hidden_states, grad_gate_weight


@torch.no_grad()
def run(*args, **kwargs):
    global _compiled_run
    if _compiled_run is None:
        _compiled_run = torch.compile(_run_impl, mode="reduce-overhead", fullgraph=True)
    return _compiled_run(*args, **kwargs)
