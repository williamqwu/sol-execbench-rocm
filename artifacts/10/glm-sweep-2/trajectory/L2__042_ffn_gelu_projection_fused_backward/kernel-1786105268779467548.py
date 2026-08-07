import torch

_compiled = None


def _build():
    @torch.no_grad()
    def run(
        grad_output: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_output: torch.Tensor,
        gelu_output: torch.Tensor,
        fc2_weight: torch.Tensor,
        residual_output: torch.Tensor,
        normalized: torch.Tensor,
        var: torch.Tensor,
        ln_weight: torch.Tensor,
        eps: float,
    ):
        B, S, H = grad_output.shape
        I = fc1_weight.shape[0]

        # Layer norm backward
        grad_ln_weight = (grad_output * normalized).sum(dim=(0, 1))
        grad_ln_bias = grad_output.sum(dim=(0, 1))
        grad_normalized = grad_output * ln_weight

        std = torch.sqrt(var + eps)
        grad_normalized_mean = grad_normalized.mean(dim=-1, keepdim=True)
        grad_normalized_normalized_mean = (grad_normalized * normalized).mean(dim=-1, keepdim=True)
        grad_residual_output = (1.0 / std) * (
            grad_normalized - grad_normalized_mean - normalized * grad_normalized_normalized_mean
        )

        # Residual backward
        grad_fc2_output = grad_residual_output
        grad_residual = grad_residual_output

        # FC2 backward
        grad_fc2_bias = grad_fc2_output.sum(dim=(0, 1))
        grad_fc2_output_reshaped = grad_fc2_output.view(-1, H)
        gelu_output_reshaped = gelu_output.view(-1, I)
        grad_fc2_weight = grad_fc2_output_reshaped.t() @ gelu_output_reshaped
        grad_gelu_output = grad_fc2_output @ fc2_weight

        # GELU backward
        sqrt_2_over_pi = 0.7978845608028654
        coeff = 0.044715
        x = fc1_output
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed)
        tanh_out = torch.tanh(tanh_arg)
        dtanh_arg_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
        sech_sq = 1.0 - tanh_out * tanh_out
        gelu_grad = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech_sq * dtanh_arg_dx
        grad_fc1_output = grad_gelu_output * gelu_grad

        # FC1 backward
        grad_fc1_bias = grad_fc1_output.sum(dim=(0, 1))
        grad_fc1_output_reshaped = grad_fc1_output.view(-1, I)
        hidden_states_reshaped = hidden_states.view(-1, H)
        grad_fc1_weight = grad_fc1_output_reshaped.t() @ hidden_states_reshaped
        grad_hidden_states_fc1 = grad_fc1_output @ fc1_weight
        grad_hidden_states = grad_hidden_states_fc1 + grad_residual

        return (
            grad_hidden_states,
            grad_fc1_weight,
            grad_fc1_bias,
            grad_fc2_weight,
            grad_fc2_bias,
            grad_ln_weight,
            grad_ln_bias,
        )

    return torch.compile(run, dynamic=True)


def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_output: torch.Tensor,
    gelu_output: torch.Tensor,
    fc2_weight: torch.Tensor,
    residual_output: torch.Tensor,
    normalized: torch.Tensor,
    var: torch.Tensor,
    ln_weight: torch.Tensor,
    eps: float,
):
    global _compiled
    if _compiled is None:
        _compiled = _build()
    return _compiled(
        grad_output, hidden_states, fc1_weight, fc1_output, gelu_output,
        fc2_weight, residual_output, normalized, var, ln_weight, eps,
    )
