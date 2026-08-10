import torch
import torch.nn.functional as F


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
    """
    Backward pass for AltUp predict-correct cycle.
    
    This computes gradients through:
    1. Correct step backward
    2. Predict step backward
    
    Returns gradients for all learnable parameters and inputs.
    """
    altup_num_inputs = 3
    hidden_size = 2304
    router_scale = hidden_size ** -1.0
    
    batch_size = hidden_states.shape[1]
    seq_len = hidden_states.shape[2]
    router_weight_float = router_weight.float()
    norm_weight_float = norm_weight.float()
    prediction_coef_weight_float = prediction_coef_weight.float()
    correction_coef_weight_float = correction_coef_weight.float()
    
    # ========== FORWARD RECOMPUTATION FOR PREDICT STEP ==========
    # We need intermediate values from forward pass
    
    # Predict step forward recomputation
    hidden_float = hidden_states.float()
    x_float_predict = hidden_float[altup_active_idx]
    variance_predict = (x_float_predict * x_float_predict).mean(-1, keepdim=True)
    rstd_predict = torch.rsqrt(variance_predict + rms_norm_eps)
    normalized_predict = x_float_predict * rstd_predict
    scaled_predict = normalized_predict * norm_weight_float
    scaled_predict.mul_(router_scale)
    routed_predict = F.linear(scaled_predict, router_weight_float)
    modalities_predict = torch.tanh(routed_predict)
    
    all_coefs_flat = F.linear(modalities_predict, prediction_coef_weight_float)
    all_coefs = all_coefs_flat.reshape(batch_size, seq_len, altup_num_inputs, altup_num_inputs)
    all_coefs = all_coefs.permute(0, 1, 3, 2)
    
    h_permuted = hidden_float.permute(1, 2, 3, 0)
    # The correction step consumes only the active prediction. Avoid materializing
    # all three full [B,S,H] predicted states during forward recomputation.
    prediction_active = (
        h_permuted * all_coefs[..., altup_active_idx].unsqueeze(-2)
    ).sum(-1)
    prediction_active.add_(hidden_float[altup_active_idx])
    
    # ========== FORWARD RECOMPUTATION FOR CORRECT STEP ==========
    x_float_correct = activated.float()
    variance_correct = (x_float_correct * x_float_correct).mean(-1, keepdim=True)
    rstd_correct = torch.rsqrt(variance_correct + rms_norm_eps)
    normalized_correct = x_float_correct * rstd_correct
    scaled_correct = normalized_correct * norm_weight_float
    scaled_correct.mul_(router_scale)
    routed_correct = F.linear(scaled_correct, router_weight_float)
    modalities_correct = torch.tanh(routed_correct)
    
    innovation = x_float_correct - prediction_active
    
    all_coefs_correct = F.linear(modalities_correct, correction_coef_weight_float) + 1.0
    all_coefs_expanded = all_coefs_correct.permute(2, 0, 1).unsqueeze(-1)
    
    # ========== BACKWARD PASS FOR CORRECT STEP ==========
    grad_corrected_float = grad_corrected.float()
    
    # Gradient through residual connection
    grad_predictions = grad_corrected_float
    
    # Gradient through element-wise multiplication
    # corrected = innovation_repeated * all_coefs_expanded + predictions
    grad_all_coefs_expanded = (grad_corrected_float * innovation.unsqueeze(0)).sum(dim=-1, keepdim=True)
    
    # Gradient through unsqueeze and permute of all_coefs
    grad_all_coefs_correct = grad_all_coefs_expanded.squeeze(-1).permute(1, 2, 0)
    
    # Gradient through +1.0 (identity)
    # Gradient through linear projection for correction coefficients
    grad_correction_coef_weight = torch.matmul(
        grad_all_coefs_correct.reshape(-1, altup_num_inputs).t(),
        modalities_correct.reshape(-1, altup_num_inputs)
    )
    
    grad_modalities_correct = torch.matmul(
        grad_all_coefs_correct,
        correction_coef_weight_float
    )
    
    # Gradient through repeat of innovation (sum over repeated dimension)
    grad_innovation = grad_corrected_float[0] * all_coefs_expanded[0]
    grad_innovation.addcmul_(grad_corrected_float[1], all_coefs_expanded[1])
    grad_innovation.addcmul_(grad_corrected_float[2], all_coefs_expanded[2])
    
    # Gradient through innovation computation
    # innovation = activated - predictions[altup_active_idx]
    grad_activated = grad_innovation
    grad_predictions[altup_active_idx].sub_(grad_innovation)
    
    # Gradient through router modalities for correct step
    tanh_modalities_correct = modalities_correct
    grad_routed_correct = grad_modalities_correct * (
        1 - tanh_modalities_correct * tanh_modalities_correct
    )
    
    grad_router_weight_correct = torch.matmul(
        grad_routed_correct.reshape(-1, altup_num_inputs).t(),
        scaled_correct.reshape(-1, hidden_size)
    )
    
    grad_scaled_correct = torch.matmul(grad_routed_correct, router_weight_float)
    grad_scaled_correct.mul_(router_scale)
    grad_normed_correct = grad_scaled_correct
    
    grad_normalized_correct = grad_normed_correct * norm_weight_float
    mean_grad_normalized_x_correct = (
        grad_normalized_correct * normalized_correct
    ).mean(-1, keepdim=True)
    grad_activated_from_router = rstd_correct * (
        grad_normalized_correct
        - normalized_correct * mean_grad_normalized_x_correct
    )
    grad_activated.add_(grad_activated_from_router)
    
    # ========== BACKWARD PASS FOR PREDICT STEP ==========
    # Gradient through residual connection
    grad_hidden_states = grad_predictions
    
    # Gradient through permute back
    grad_predictions_permuted = grad_predictions.permute(1, 2, 3, 0)
    
    # Gradient through matmul
    # predictions = h_permuted @ all_coefs
    grad_h_permuted = torch.matmul(
        grad_predictions_permuted,
        all_coefs.transpose(-2, -1)
    )
    
    grad_all_coefs_matmul = torch.matmul(
        h_permuted.transpose(-2, -1),
        grad_predictions_permuted
    )
    
    # Gradient through permute of hidden_states
    grad_hidden_states.add_(grad_h_permuted.permute(3, 0, 1, 2))
    
    # Gradient through transpose of all_coefs
    grad_all_coefs_before_transpose = grad_all_coefs_matmul.permute(0, 1, 3, 2)
    
    # Gradient through reshape
    grad_all_coefs_flat = grad_all_coefs_before_transpose.reshape(
        batch_size, seq_len, altup_num_inputs * altup_num_inputs
    )
    
    # Gradient through linear projection for prediction coefficients
    grad_prediction_coef_weight = torch.matmul(
        grad_all_coefs_flat.reshape(-1, altup_num_inputs * altup_num_inputs).t(),
        modalities_predict.reshape(-1, altup_num_inputs)
    )
    
    grad_modalities_predict = torch.matmul(
        grad_all_coefs_flat,
        prediction_coef_weight_float
    )
    
    # Gradient through router modalities for predict step
    tanh_modalities_predict = modalities_predict
    grad_routed_predict = grad_modalities_predict * (
        1 - tanh_modalities_predict * tanh_modalities_predict
    )

    grad_router_weight_predict = torch.matmul(
        grad_routed_predict.reshape(-1, altup_num_inputs).t(),
        scaled_predict.reshape(-1, hidden_size)
    )

    grad_scaled_predict = torch.matmul(grad_routed_predict, router_weight_float)
    grad_scaled_predict.mul_(router_scale)
    grad_normed_predict = grad_scaled_predict
    
    grad_normalized_predict = grad_normed_predict * norm_weight_float
    mean_grad_normalized_x_predict = (
        grad_normalized_predict * normalized_predict
    ).mean(-1, keepdim=True)
    grad_active_input = rstd_predict * (
        grad_normalized_predict
        - normalized_predict * mean_grad_normalized_x_predict
    )
    
    grad_hidden_states[altup_active_idx].add_(grad_active_input)
    
    # Combine gradients from predict and correct steps
    grad_router_weight_predict.add_(grad_router_weight_correct)
    grad_router_weight = grad_router_weight_predict
    grad_norm_weight = (
        grad_normed_predict * normalized_predict
        + grad_normed_correct * normalized_correct
    ).sum(dim=[0, 1])
    
    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_activated.to(torch.bfloat16),
        grad_prediction_coef_weight,
        grad_correction_coef_weight,
        grad_router_weight,
        grad_norm_weight,
    )
