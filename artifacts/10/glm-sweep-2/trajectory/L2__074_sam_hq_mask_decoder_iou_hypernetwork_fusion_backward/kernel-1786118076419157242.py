import torch

@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
def _run(
    grad_iou_scores: torch.Tensor,
    grad_hyper_weights: torch.Tensor,
    iou_token_out: torch.Tensor,
    mask_tokens_out: torch.Tensor,
    iou_proj_in_weight: torch.Tensor,
    iou_proj_in_bias: torch.Tensor,
    iou_hidden_weight: torch.Tensor,
    iou_hidden_bias: torch.Tensor,
    iou_proj_out_weight: torch.Tensor,
    iou_proj_out_bias: torch.Tensor,
    iou_hidden1: torch.Tensor,
    iou_hidden1_relu: torch.Tensor,
    iou_hidden2: torch.Tensor,
    iou_hidden2_relu: torch.Tensor,
    hyper_proj_in_weights: torch.Tensor,
    hyper_proj_in_biases: torch.Tensor,
    hyper_hidden_weights: torch.Tensor,
    hyper_hidden_biases: torch.Tensor,
    hyper_proj_out_weights: torch.Tensor,
    hyper_proj_out_biases: torch.Tensor,
    hyper_hidden1_0: torch.Tensor,
    hyper_hidden1_1: torch.Tensor,
    hyper_hidden1_2: torch.Tensor,
    hyper_hidden1_3: torch.Tensor,
    hyper_hidden1_relu_0: torch.Tensor,
    hyper_hidden1_relu_1: torch.Tensor,
    hyper_hidden1_relu_2: torch.Tensor,
    hyper_hidden1_relu_3: torch.Tensor,
    hyper_hidden2_0: torch.Tensor,
    hyper_hidden2_1: torch.Tensor,
    hyper_hidden2_2: torch.Tensor,
    hyper_hidden2_3: torch.Tensor,
    hyper_hidden2_relu_0: torch.Tensor,
    hyper_hidden2_relu_1: torch.Tensor,
    hyper_hidden2_relu_2: torch.Tensor,
    hyper_hidden2_relu_3: torch.Tensor,
):
    batch_size, point_batch_size, _ = iou_token_out.shape

    hyper_hidden1_list = [hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3]
    hyper_hidden1_relu_list = [hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2, hyper_hidden1_relu_3]
    hyper_hidden2_list = [hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3]
    hyper_hidden2_relu_list = [hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3]

    grad_iou_scores_flat = grad_iou_scores.reshape(-1, 4)
    iou_hidden2_relu_flat = iou_hidden2_relu.reshape(-1, 1024)
    grad_iou_proj_out_weight = grad_iou_scores_flat.t() @ iou_hidden2_relu_flat
    grad_iou_proj_out_bias = grad_iou_scores_flat.sum(dim=0)
    grad_iou_hidden2_relu = (grad_iou_scores_flat @ iou_proj_out_weight).reshape(batch_size, point_batch_size, 1024)
    grad_iou_hidden2 = grad_iou_hidden2_relu * (iou_hidden2 > 0).float()
    grad_iou_hidden2_flat = grad_iou_hidden2.reshape(-1, 1024)
    iou_hidden1_relu_flat = iou_hidden1_relu.reshape(-1, 1024)
    grad_iou_hidden_weight = grad_iou_hidden2_flat.t() @ iou_hidden1_relu_flat
    grad_iou_hidden_bias = grad_iou_hidden2_flat.sum(dim=0)
    grad_iou_hidden1_relu = (grad_iou_hidden2_flat @ iou_hidden_weight).reshape(batch_size, point_batch_size, 1024)
    grad_iou_hidden1 = grad_iou_hidden1_relu * (iou_hidden1 > 0).float()
    grad_iou_hidden1_flat = grad_iou_hidden1.reshape(-1, 1024)
    iou_token_out_flat = iou_token_out.reshape(-1, 256)
    grad_iou_proj_in_weight = grad_iou_hidden1_flat.t() @ iou_token_out_flat
    grad_iou_proj_in_bias = grad_iou_hidden1_flat.sum(dim=0)
    grad_iou_token_out = (grad_iou_hidden1_flat @ iou_proj_in_weight).reshape(batch_size, point_batch_size, 256)

    grad_mask_tokens_out = torch.zeros_like(mask_tokens_out)
    grad_hyper_proj_in_weights = torch.zeros_like(hyper_proj_in_weights)
    grad_hyper_proj_in_biases = torch.zeros_like(hyper_proj_in_biases)
    grad_hyper_hidden_weights = torch.zeros_like(hyper_hidden_weights)
    grad_hyper_hidden_biases = torch.zeros_like(hyper_hidden_biases)
    grad_hyper_proj_out_weights = torch.zeros_like(hyper_proj_out_weights)
    grad_hyper_proj_out_biases = torch.zeros_like(hyper_proj_out_biases)

    for mask_idx in range(4):
        grad_weights = grad_hyper_weights[:, :, mask_idx, :]
        hidden1 = hyper_hidden1_list[mask_idx]
        hidden1_relu = hyper_hidden1_relu_list[mask_idx]
        hidden2 = hyper_hidden2_list[mask_idx]
        hidden2_relu = hyper_hidden2_relu_list[mask_idx]
        token = mask_tokens_out[:, :, mask_idx, :]
        grad_weights_flat = grad_weights.reshape(-1, 32)
        hidden2_relu_flat = hidden2_relu.reshape(-1, 256)
        hidden1_relu_flat = hidden1_relu.reshape(-1, 256)
        token_flat = token.reshape(-1, 256)
        grad_hyper_proj_out_weights[mask_idx] = grad_weights_flat.t() @ hidden2_relu_flat
        grad_hyper_proj_out_biases[mask_idx] = grad_weights_flat.sum(dim=0)
        grad_hidden2_relu = (grad_weights_flat @ hyper_proj_out_weights[mask_idx]).reshape(batch_size, point_batch_size, 256)
        grad_hidden2 = grad_hidden2_relu * (hidden2 > 0).float()
        grad_hidden2_flat = grad_hidden2.reshape(-1, 256)
        grad_hyper_hidden_weights[mask_idx] = grad_hidden2_flat.t() @ hidden1_relu_flat
        grad_hyper_hidden_biases[mask_idx] = grad_hidden2_flat.sum(dim=0)
        grad_hidden1_relu = (grad_hidden2_flat @ hyper_hidden_weights[mask_idx]).reshape(batch_size, point_batch_size, 256)
        grad_hidden1 = grad_hidden1_relu * (hidden1 > 0).float()
        grad_hidden1_flat = grad_hidden1.reshape(-1, 256)
        grad_hyper_proj_in_weights[mask_idx] = grad_hidden1_flat.t() @ token_flat
        grad_hyper_proj_in_biases[mask_idx] = grad_hidden1_flat.sum(dim=0)
        grad_token = (grad_hidden1_flat @ hyper_proj_in_weights[mask_idx]).reshape(batch_size, point_batch_size, 256)
        grad_mask_tokens_out[:, :, mask_idx, :] = grad_token

    return (
        grad_iou_token_out,
        grad_mask_tokens_out,
        grad_iou_proj_in_weight,
        grad_iou_proj_in_bias,
        grad_iou_hidden_weight,
        grad_iou_hidden_bias,
        grad_iou_proj_out_weight,
        grad_iou_proj_out_bias,
        grad_hyper_proj_in_weights,
        grad_hyper_proj_in_biases,
        grad_hyper_hidden_weights,
        grad_hyper_hidden_biases,
        grad_hyper_proj_out_weights,
        grad_hyper_proj_out_biases,
    )


@torch.no_grad()
def run(
    grad_iou_scores: torch.Tensor,
    grad_hyper_weights: torch.Tensor,
    iou_token_out: torch.Tensor,
    mask_tokens_out: torch.Tensor,
    iou_proj_in_weight: torch.Tensor,
    iou_proj_in_bias: torch.Tensor,
    iou_hidden_weight: torch.Tensor,
    iou_hidden_bias: torch.Tensor,
    iou_proj_out_weight: torch.Tensor,
    iou_proj_out_bias: torch.Tensor,
    iou_hidden1: torch.Tensor,
    iou_hidden1_relu: torch.Tensor,
    iou_hidden2: torch.Tensor,
    iou_hidden2_relu: torch.Tensor,
    hyper_proj_in_weights: torch.Tensor,
    hyper_proj_in_biases: torch.Tensor,
    hyper_hidden_weights: torch.Tensor,
    hyper_hidden_biases: torch.Tensor,
    hyper_proj_out_weights: torch.Tensor,
    hyper_proj_out_biases: torch.Tensor,
    hyper_hidden1_0: torch.Tensor,
    hyper_hidden1_1: torch.Tensor,
    hyper_hidden1_2: torch.Tensor,
    hyper_hidden1_3: torch.Tensor,
    hyper_hidden1_relu_0: torch.Tensor,
    hyper_hidden1_relu_1: torch.Tensor,
    hyper_hidden1_relu_2: torch.Tensor,
    hyper_hidden1_relu_3: torch.Tensor,
    hyper_hidden2_0: torch.Tensor,
    hyper_hidden2_1: torch.Tensor,
    hyper_hidden2_2: torch.Tensor,
    hyper_hidden2_3: torch.Tensor,
    hyper_hidden2_relu_0: torch.Tensor,
    hyper_hidden2_relu_1: torch.Tensor,
    hyper_hidden2_relu_2: torch.Tensor,
    hyper_hidden2_relu_3: torch.Tensor,
):
    return _run(
        grad_iou_scores, grad_hyper_weights, iou_token_out, mask_tokens_out,
        iou_proj_in_weight, iou_proj_in_bias, iou_hidden_weight, iou_hidden_bias,
        iou_proj_out_weight, iou_proj_out_bias, iou_hidden1, iou_hidden1_relu,
        iou_hidden2, iou_hidden2_relu, hyper_proj_in_weights, hyper_proj_in_biases,
        hyper_hidden_weights, hyper_hidden_biases, hyper_proj_out_weights,
        hyper_proj_out_biases, hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2,
        hyper_hidden1_3, hyper_hidden1_relu_0, hyper_hidden1_relu_1,
        hyper_hidden1_relu_2, hyper_hidden1_relu_3, hyper_hidden2_0,
        hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3, hyper_hidden2_relu_0,
        hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3,
    )
