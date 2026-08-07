import torch
import torch.nn.functional as F


def _run_impl(
    grad_corrected,
    hidden_states,
    activated,
    prediction_coef_weight,
    correction_coef_weight,
    router_weight,
    norm_weight,
    altup_active_idx,
    rms_norm_eps,
):
    altup_num_inputs = 3
    hidden_size = 2304
    router_scale = hidden_size ** -1.0

    batch_size = hidden_states.shape[1]
    seq_len = hidden_states.shape[2]

    # ========== FORWARD RECOMPUTATION FOR PREDICT STEP ==========
    active_input_predict = hidden_states[altup_active_idx]
    x_float_predict = active_input_predict.float()
    variance_predict = x_float_predict.pow(2).mean(-1, keepdim=True)
    rstd_predict = torch.rsqrt(variance_predict + rms_norm_eps)
    normalized_predict = x_float_predict * rstd_predict
    normed_predict = normalized_predict * norm_weight.float()
    scaled_predict = normed_predict * router_scale
    routed_predict = F.linear(scaled_predict, router_weight.float())
    modalities_predict = torch.tanh(routed_predict)

    all_coefs_flat = F.linear(modalities_predict, prediction_coef_weight.float())
    all_coefs = all_coefs_flat.reshape(batch_size, seq_len, altup_num_inputs, altup_num_inputs)
    all_coefs = all_coefs.permute(0, 1, 3, 2)

    h_permuted = hidden_states.float().permute(1, 2, 3, 0)
    predictions_before_residual = torch.matmul(h_permuted, all_coefs)
    predictions_permuted = predictions_before_residual.permute(3, 0, 1, 2)
    predictions = predictions_permuted + hidden_states.float()

    # ========== FORWARD RECOMPUTATION FOR CORRECT STEP ==========
    x_float_correct = activated.float()
    variance_correct = x_float_correct.pow(2).mean(-1, keepdim=True)
    rstd_correct = torch.rsqrt(variance_correct + rms_norm_eps)
    normalized_correct = x_float_correct * rstd_correct
    normed_correct = normalized_correct * norm_weight.float()
    scaled_correct = normed_correct * router_scale
    routed_correct = F.linear(scaled_correct, router_weight.float())
    modalities_correct = torch.tanh(routed_correct)

    innovation = activated.float() - predictions[altup_active_idx]
    innovation_repeated = innovation.unsqueeze(0).expand(altup_num_inputs, -1, -1, -1)

    all_coefs_correct = F.linear(modalities_correct, correction_coef_weight.float()) + 1.0
    all_coefs_expanded = all_coefs_correct.permute(2, 0, 1).unsqueeze(-1)

    # ========== BACKWARD PASS FOR CORRECT STEP ==========
    grad_corrected_float = grad_corrected.float()

    grad_predictions = grad_corrected_float.clone()

    grad_innovation_repeated = grad_corrected_float * all_coefs_expanded
    grad_all_coefs_expanded = (grad_corrected_float * innovation_repeated).sum(dim=-1, keepdim=True)

    grad_all_coefs_correct = grad_all_coefs_expanded.squeeze(-1).permute(1, 2, 0)

    grad_correction_coef_weight = torch.matmul(
        grad_all_coefs_correct.reshape(-1, altup_num_inputs).t(),
        modalities_correct.reshape(-1, altup_num_inputs)
    )

    grad_modalities_correct = torch.matmul(
        grad_all_coefs_correct,
        correction_coef_weight.float()
    )

    grad_innovation = grad_innovation_repeated.sum(dim=0)

    grad_activated = grad_innovation.clone()
    grad_predictions[altup_active_idx] = grad_predictions[altup_active_idx] - grad_innovation

    tanh_modalities_correct = modalities_correct
    grad_routed_correct = grad_modalities_correct * (1 - tanh_modalities_correct.pow(2))

    grad_router_weight_correct = torch.matmul(
        grad_routed_correct.reshape(-1, altup_num_inputs).t(),
        scaled_correct.reshape(-1, hidden_size)
    )

    grad_scaled_correct = torch.matmul(grad_routed_correct, router_weight.float())
    grad_normed_correct = grad_scaled_correct * router_scale

    grad_norm_weight_correct = (grad_normed_correct * normalized_correct).sum(dim=[0, 1])

    grad_normalized_correct = grad_normed_correct * norm_weight.float()
    mean_grad_normalized_x_correct = (grad_normalized_correct * x_float_correct).mean(-1, keepdim=True)
    grad_activated_from_router = grad_normalized_correct * rstd_correct - x_float_correct * rstd_correct.pow(3) * mean_grad_normalized_x_correct
    grad_activated = grad_activated + grad_activated_from_router

    # ========== BACKWARD PASS FOR PREDICT STEP ==========
    grad_hidden_states = grad_predictions.clone()

    grad_predictions_permuted = grad_predictions.permute(1, 2, 3, 0)

    grad_h_permuted = torch.matmul(
        grad_predictions_permuted,
        all_coefs.transpose(-2, -1)
    )

    grad_all_coefs_matmul = torch.matmul(
        h_permuted.transpose(-2, -1),
        grad_predictions_permuted
    )

    grad_hidden_states = grad_hidden_states + grad_h_permuted.permute(3, 0, 1, 2)

    grad_all_coefs_before_transpose = grad_all_coefs_matmul.permute(0, 1, 3, 2)

    grad_all_coefs_flat = grad_all_coefs_before_transpose.reshape(
        batch_size, seq_len, altup_num_inputs * altup_num_inputs
    )

    grad_prediction_coef_weight = torch.matmul(
        grad_all_coefs_flat.reshape(-1, altup_num_inputs * altup_num_inputs).t(),
        modalities_predict.reshape(-1, altup_num_inputs)
    )

    grad_modalities_predict = torch.matmul(
        grad_all_coefs_flat,
        prediction_coef_weight.float()
    )

    tanh_modalities_predict = modalities_predict
    grad_routed_predict = grad_modalities_predict * (1 - tanh_modalities_predict.pow(2))

    grad_router_weight_predict = torch.matmul(
        grad_routed_predict.reshape(-1, altup_num_inputs).t(),
        scaled_predict.reshape(-1, hidden_size)
    )

    grad_scaled_predict = torch.matmul(grad_routed_predict, router_weight.float())
    grad_normed_predict = grad_scaled_predict * router_scale

    grad_norm_weight_predict = (grad_normed_predict * normalized_predict).sum(dim=[0, 1])

    grad_normalized_predict = grad_normed_predict * norm_weight.float()
    mean_grad_normalized_x_predict = (grad_normalized_predict * x_float_predict).mean(-1, keepdim=True)
    grad_active_input = grad_normalized_predict * rstd_predict - x_float_predict * rstd_predict.pow(3) * mean_grad_normalized_x_predict

    grad_hidden_states[altup_active_idx] = grad_hidden_states[altup_active_idx] + grad_active_input

    grad_router_weight = grad_router_weight_predict + grad_router_weight_correct
    grad_norm_weight = grad_norm_weight_predict + grad_norm_weight_correct

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_activated.to(torch.bfloat16),
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )


_compiled = torch.compile(_run_impl, dynamic=True)


@torch.no_grad()
def run(
    grad_corrected: torch.Tensor,
    hidden_states: torch.Tensor,
    activated: torch.Tensor,
    prediction_coef_weight: torch.Tensor,
    correction_coef_weight: torch.Tensor,
    router_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    altup_active_idx: int,
    rms_norm_eps: float,
):
    return _compiled(
        grad_corrected,
        hidden_states,
        activated,
        prediction_coef_weight,
        correction_coef_weight,
        router_weight,
        norm_weight,
        altup_active_idx,
        rms_norm_eps,
    )
