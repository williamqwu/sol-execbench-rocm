import torch

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
    batch_size, point_batch_size, _ = iou_token_out.shape
    BP = batch_size * point_batch_size

    # ==================== IoU Path Backward ====================
    grad_iou_scores_flat = grad_iou_scores.reshape(-1, 4)  # (BP, 4)
    iou_hidden2_relu_flat = iou_hidden2_relu.reshape(-1, 1024)  # (BP, 1024)

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

    # ==================== Hypernetwork Path Backward (batched) ====================
    # Stack the 4 saved activations into batch dims for bmm.
    # shapes: each (B,P,256); stack -> (4, BP, 256)
    hyper_hidden1 = torch.stack([hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3], dim=0)  # (4, BP, 256)
    hyper_hidden1_relu = torch.stack([hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2, hyper_hidden1_relu_3], dim=0)
    hyper_hidden2 = torch.stack([hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3], dim=0)
    hyper_hidden2_relu = torch.stack([hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3], dim=0)
    # mask_tokens_out: (B,P,4,256) -> need (4, BP, 256)
    token = mask_tokens_out.permute(2, 0, 1, 3).reshape(4, BP, 256)

    # grad_hyper_weights: (B,P,4,32) -> (4, BP, 32)
    grad_weights_flat = grad_hyper_weights.permute(2, 0, 1, 3).reshape(4, BP, 32)

    # Output layer: Linear(256->32) backward
    # grad_hyper_proj_out_weights: (4, 32, 256) = bmm(grad_weights_flat.t(1,2), hidden2_relu)
    grad_hyper_proj_out_weights = torch.bmm(grad_weights_flat.transpose(1, 2), hyper_hidden2_relu)  # (4,32,256)
    grad_hyper_proj_out_biases = grad_weights_flat.sum(dim=1)  # (4, 32)

    # grad_hidden2_relu: (4, BP, 256) = bmm(grad_weights_flat, hyper_proj_out_weights)
    # hyper_proj_out_weights: (4, 32, 256) -> already (4, 32, 256)
    grad_hidden2_relu = torch.bmm(grad_weights_flat, hyper_proj_out_weights)  # (4, BP, 256)
    grad_hidden2 = grad_hidden2_relu * (hyper_hidden2 > 0).float()

    # Hidden layer: Linear(256->256) backward
    # grad_hyper_hidden_weights: (4, 256, 256) = bmm(grad_hidden2.t(1,2), hidden1_relu)
    grad_hyper_hidden_weights = torch.bmm(grad_hidden2.transpose(1, 2), hyper_hidden1_relu)
    grad_hyper_hidden_biases = grad_hidden2.sum(dim=1)  # (4, 256)

    # grad_hidden1_relu: (4, BP, 256) = bmm(grad_hidden2, hyper_hidden_weights)
    grad_hidden1_relu = torch.bmm(grad_hidden2, hyper_hidden_weights)
    grad_hidden1 = grad_hidden1_relu * (hyper_hidden1 > 0).float()

    # Input layer: Linear(256->256) backward
    # grad_hyper_proj_in_weights: (4, 256, 256) = bmm(grad_hidden1.t(1,2), token)
    grad_hyper_proj_in_weights = torch.bmm(grad_hidden1.transpose(1, 2), token)
    grad_hyper_proj_in_biases = grad_hidden1.sum(dim=1)  # (4, 256)

    # grad_token: (4, BP, 256) = bmm(grad_hidden1, hyper_proj_in_weights)
    grad_token = torch.bmm(grad_hidden1, hyper_proj_in_weights)  # (4, BP, 256)

    # Reshape grad_token back to (B, P, 4, 256)
    grad_mask_tokens_out = grad_token.reshape(4, batch_size, point_batch_size, 256).permute(1, 2, 0, 3).contiguous()

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
