import torch

# Per-shape CUDA graph cache.
_graph_cache = {}


def _make_graph(
    grad_output, hidden_states, fc1_weight, fc1_output, gelu_output,
    fc2_weight, residual_output, normalized, var, ln_weight, eps,
):
    B, S, H = grad_output.shape
    I = fc1_weight.shape[0]
    device = grad_output.device

    # Static input buffers (copies of the current inputs).
    s_grad_output = grad_output.clone()
    s_hidden_states = hidden_states.clone()
    s_fc1_weight = fc1_weight.clone()
    s_fc1_output = fc1_output.clone()
    s_gelu_output = gelu_output.clone()
    s_fc2_weight = fc2_weight.clone()
    s_residual_output = residual_output.clone()
    s_normalized = normalized.clone()
    s_var = var.clone()
    s_ln_weight = ln_weight.clone()

    # Warmup the ops on a side stream to ensure lazy init is done.
    def _compute():
        grad_ln_weight = (s_grad_output * s_normalized).sum(dim=(0, 1))
        grad_ln_bias = s_grad_output.sum(dim=(0, 1))
        grad_normalized = s_grad_output * s_ln_weight
        std = torch.sqrt(s_var + eps)
        grad_normalized_mean = grad_normalized.mean(dim=-1, keepdim=True)
        grad_normalized_normalized_mean = (grad_normalized * s_normalized).mean(dim=-1, keepdim=True)
        grad_residual_output = (1.0 / std) * (
            grad_normalized - grad_normalized_mean - s_normalized * grad_normalized_normalized_mean
        )
        grad_fc2_output = grad_residual_output
        grad_residual = grad_residual_output
        grad_fc2_bias = grad_fc2_output.sum(dim=(0, 1))
        grad_fc2_output_reshaped = grad_fc2_output.view(-1, H)
        gelu_output_reshaped = s_gelu_output.view(-1, I)
        grad_fc2_weight = grad_fc2_output_reshaped.t() @ gelu_output_reshaped
        grad_gelu_output = grad_fc2_output @ s_fc2_weight
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        x = s_fc1_output
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
        tanh_out = torch.tanh(tanh_arg)
        dtanh_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
        sech_sq = 1.0 - tanh_out * tanh_out
        gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * dtanh_arg_dx
        grad_fc1_output = grad_gelu_output * gelu_grad
        grad_fc1_bias = grad_fc1_output.sum(dim=(0, 1))
        grad_fc1_output_reshaped = grad_fc1_output.view(-1, I)
        hidden_states_reshaped = s_hidden_states.view(-1, H)
        grad_fc1_weight = grad_fc1_output_reshaped.t() @ hidden_states_reshaped
        grad_hidden_states_fc1 = grad_fc1_output @ s_fc1_weight
        grad_hidden_states = grad_hidden_states_fc1 + grad_residual
        return (grad_hidden_states, grad_fc1_weight, grad_fc1_bias, grad_fc2_weight,
                grad_fc2_bias, grad_ln_weight, grad_ln_bias)

    # Warmup
    for _ in range(3):
        out = _compute()
    torch.cuda.synchronize()

    # Static output buffers
    s_outputs = [o.clone() for o in out]

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        captured = _compute()
        for buf, t in zip(s_outputs, captured):
            buf.copy_(t)

    static_inputs = [s_grad_output, s_hidden_states, s_fc1_weight, s_fc1_output,
                     s_gelu_output, s_fc2_weight, s_residual_output, s_normalized,
                     s_var, s_ln_weight]
    return g, static_inputs, s_outputs


@torch.no_grad()
def run(
    grad_output, hidden_states, fc1_weight, fc1_output, gelu_output,
    fc2_weight, residual_output, normalized, var, ln_weight, eps,
):
    key = (grad_output.shape, hidden_states.shape, fc1_weight.shape)
    entry = _graph_cache.get(key)
    if entry is None:
        entry = _make_graph(
            grad_output, hidden_states, fc1_weight, fc1_output, gelu_output,
            fc2_weight, residual_output, normalized, var, ln_weight, eps,
        )
        _graph_cache[key] = entry
    g, static_inputs, s_outputs = entry
    (s_grad_output, s_hidden_states, s_fc1_weight, s_fc1_output,
     s_gelu_output, s_fc2_weight, s_residual_output, s_normalized,
     s_var, s_ln_weight) = static_inputs
    s_grad_output.copy_(grad_output)
    s_hidden_states.copy_(hidden_states)
    s_fc1_weight.copy_(fc1_weight)
    s_fc1_output.copy_(fc1_output)
    s_gelu_output.copy_(gelu_output)
    s_fc2_weight.copy_(fc2_weight)
    s_residual_output.copy_(residual_output)
    s_normalized.copy_(normalized)
    s_var.copy_(var)
    s_ln_weight.copy_(ln_weight)
    g.replay()
    return tuple(o.clone() for o in s_outputs)
