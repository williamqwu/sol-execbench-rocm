import torch
import torch._inductor.config as inductor_config

# Allow torch.compile to use custom fused kernels
inductor_config.triton.cooperative_reductions = True

_compiled_run = None


def _run_impl(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    routing_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights_sum: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_dtype = hidden_states.dtype

    # --- Normalization backward (quotient rule) ---
    grad_routing_weights_f32 = grad_routing_weights.float()
    routing_weights_unnorm = torch.gather(routing_probs, dim=1, index=selected_experts)

    grad_sum = (grad_routing_weights_f32 * routing_weights_unnorm / routing_weights_sum).sum(
        dim=-1, keepdim=True
    )
    grad_routing_weights_unnorm = (
        grad_routing_weights_f32 / routing_weights_sum
        - grad_sum / routing_weights_sum
    )

    # --- Expert mask backward ---
    grad_expert_mask_permuted = grad_expert_mask.permute(2, 1, 0)  # view
    grad_from_mask = torch.gather(
        grad_expert_mask_permuted,
        dim=2,
        index=selected_experts.unsqueeze(2),
    ).squeeze(2).float()

    # --- Combine and scatter once ---
    combined = grad_routing_weights_unnorm + grad_from_mask
    grad_routing_probs = grad_router_logits.float()
    grad_routing_probs.scatter_add_(dim=1, index=selected_experts, src=combined)

    # --- Softmax backward ---
    dot_product = (grad_routing_probs * routing_probs).sum(dim=1, keepdim=True)
    grad_router_logits_computed = routing_probs * (grad_routing_probs - dot_product)
    grad_router_logits_computed = grad_router_logits_computed.to(input_dtype)

    # --- Linear projection backward ---
    grad_hidden_states = torch.matmul(grad_router_logits_computed, gate_weight)
    grad_gate_weight = torch.matmul(grad_router_logits_computed.t(), hidden_states)

    return grad_hidden_states, grad_gate_weight


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits: torch.Tensor,
    routing_probs: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights_sum: torch.Tensor,
    grad_routing_weights: torch.Tensor,
    grad_expert_mask: torch.Tensor,
    grad_router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    global _compiled_run
    if _compiled_run is None:
        _compiled_run = torch.compile(_run_impl, mode="default", fullgraph=True)
    return _compiled_run(
        hidden_states,
        gate_weight,
        router_logits,
        routing_probs,
        selected_experts,
        routing_weights_sum,
        grad_routing_weights,
        grad_expert_mask,
        grad_router_logits,
    )
