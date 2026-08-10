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
    
    hyper_hidden1_list = [hyper_hidden1_0, hyper_hidden1_1, hyper_hidden1_2, hyper_hidden1_3]
    hyper_hidden1_relu_list = [hyper_hidden1_relu_0, hyper_hidden1_relu_1, hyper_hidden1_relu_2, hyper_hidden1_relu_3]
    hyper_hidden2_list = [hyper_hidden2_0, hyper_hidden2_1, hyper_hidden2_2, hyper_hidden2_3]
    hyper_hidden2_relu_list = [hyper_hidden2_relu_0, hyper_hidden2_relu_1, hyper_hidden2_relu_2, hyper_hidden2_relu_3]
    
    # ==================== IoU Path Backward ====================
    # Gradient through IoU output layer: Linear(1024 -> 4)
    grad_iou_scores_flat = grad_iou_scores.reshape(-1, 4)  # (B*P, 4)
    iou_hidden2_relu_flat = iou_hidden2_relu.reshape(-1, 1024)  # (B*P, 1024)
    
    # Gradient w.r.t. iou_proj_out_weight: (4, 1024)
    grad_iou_proj_out_weight = grad_iou_scores_flat.t() @ iou_hidden2_relu_flat
    
    # Gradient w.r.t. iou_proj_out_bias: (4,)
    grad_iou_proj_out_bias = grad_iou_scores_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_hidden2_relu: (B, P, 1024)
    grad_iou_hidden2_relu = grad_iou_scores_flat @ iou_proj_out_weight  # (B*P, 1024)
    grad_iou_hidden2_relu = grad_iou_hidden2_relu.reshape(batch_size, point_batch_size, 1024)
    
    # Gradient through ReLU
    grad_iou_hidden2 = grad_iou_hidden2_relu * (iou_hidden2 > 0).float()
    
    # Gradient through IoU hidden layer: Linear(1024 -> 1024)
    grad_iou_hidden2_flat = grad_iou_hidden2.reshape(-1, 1024)  # (B*P, 1024)
    iou_hidden1_relu_flat = iou_hidden1_relu.reshape(-1, 1024)  # (B*P, 1024)
    
    # Gradient w.r.t. iou_hidden_weight: (1024, 1024)
    grad_iou_hidden_weight = grad_iou_hidden2_flat.t() @ iou_hidden1_relu_flat
    
    # Gradient w.r.t. iou_hidden_bias: (1024,)
    grad_iou_hidden_bias = grad_iou_hidden2_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_hidden1_relu: (B, P, 1024)
    grad_iou_hidden1_relu = grad_iou_hidden2_flat @ iou_hidden_weight  # (B*P, 1024)
    grad_iou_hidden1_relu = grad_iou_hidden1_relu.reshape(batch_size, point_batch_size, 1024)
    
    # Gradient through ReLU
    grad_iou_hidden1 = grad_iou_hidden1_relu * (iou_hidden1 > 0).float()
    
    # Gradient through IoU input layer: Linear(256 -> 1024)
    grad_iou_hidden1_flat = grad_iou_hidden1.reshape(-1, 1024)  # (B*P, 1024)
    iou_token_out_flat = iou_token_out.reshape(-1, 256)  # (B*P, 256)
    
    # Gradient w.r.t. iou_proj_in_weight: (1024, 256)
    grad_iou_proj_in_weight = grad_iou_hidden1_flat.t() @ iou_token_out_flat
    
    # Gradient w.r.t. iou_proj_in_bias: (1024,)
    grad_iou_proj_in_bias = grad_iou_hidden1_flat.sum(dim=0)
    
    # Gradient w.r.t. iou_token_out: (B, P, 256)
    grad_iou_token_out = grad_iou_hidden1_flat @ iou_proj_in_weight  # (B*P, 256)
    grad_iou_token_out = grad_iou_token_out.reshape(batch_size, point_batch_size, 256)
    
    # Batch the four independent hypernetwork MLPs.  The saved activations are
    # separate inputs, so stack them once into (mask, sample, channel) layout.
    n = batch_size * point_batch_size
    grad_weights = grad_hyper_weights.reshape(n, 4, 32).permute(1, 0, 2)
    hidden1 = torch.stack(hyper_hidden1_list).reshape(4, n, 256)
    hidden1_relu = torch.stack(hyper_hidden1_relu_list).reshape(4, n, 256)
    hidden2 = torch.stack(hyper_hidden2_list).reshape(4, n, 256)
    hidden2_relu = torch.stack(hyper_hidden2_relu_list).reshape(4, n, 256)
    tokens = mask_tokens_out.reshape(n, 4, 256).permute(1, 0, 2)

    grad_hyper_proj_out_weights = torch.bmm(grad_weights.transpose(1, 2), hidden2_relu)
    grad_hyper_proj_out_biases = grad_weights.sum(dim=1)
    grad_hidden2 = torch.bmm(grad_weights, hyper_proj_out_weights)
    grad_hidden2.mul_(hidden2 > 0)

    grad_hyper_hidden_weights = torch.bmm(grad_hidden2.transpose(1, 2), hidden1_relu)
    grad_hyper_hidden_biases = grad_hidden2.sum(dim=1)
    grad_hidden1 = torch.bmm(grad_hidden2, hyper_hidden_weights)
    grad_hidden1.mul_(hidden1 > 0)

    grad_hyper_proj_in_weights = torch.bmm(grad_hidden1.transpose(1, 2), tokens)
    grad_hyper_proj_in_biases = grad_hidden1.sum(dim=1)
    grad_tokens = torch.bmm(grad_hidden1, hyper_proj_in_weights)
    grad_mask_tokens_out = grad_tokens.permute(1, 0, 2).reshape(
        batch_size, point_batch_size, 4, 256
    )
    
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
