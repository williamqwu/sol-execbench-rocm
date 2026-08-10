import torch
import torch.nn.functional as F

linear = torch._C._nn.linear
relu_ = torch.relu_

def run(
    iou_token_out: torch.Tensor,
    mask_tokens_out: torch.Tensor,
    iou_proj_in_weight: torch.Tensor,
    iou_proj_in_bias: torch.Tensor,
    iou_hidden_weight: torch.Tensor,
    iou_hidden_bias: torch.Tensor,
    iou_proj_out_weight: torch.Tensor,
    iou_proj_out_bias: torch.Tensor,
    hyper0_proj_in_weight: torch.Tensor,
    hyper0_proj_in_bias: torch.Tensor,
    hyper0_hidden_weight: torch.Tensor,
    hyper0_hidden_bias: torch.Tensor,
    hyper0_proj_out_weight: torch.Tensor,
    hyper0_proj_out_bias: torch.Tensor,
    hyper1_proj_in_weight: torch.Tensor,
    hyper1_proj_in_bias: torch.Tensor,
    hyper1_hidden_weight: torch.Tensor,
    hyper1_hidden_bias: torch.Tensor,
    hyper1_proj_out_weight: torch.Tensor,
    hyper1_proj_out_bias: torch.Tensor,
    hyper2_proj_in_weight: torch.Tensor,
    hyper2_proj_in_bias: torch.Tensor,
    hyper2_hidden_weight: torch.Tensor,
    hyper2_hidden_bias: torch.Tensor,
    hyper2_proj_out_weight: torch.Tensor,
    hyper2_proj_out_bias: torch.Tensor,
    hyper3_proj_in_weight: torch.Tensor,
    hyper3_proj_in_bias: torch.Tensor,
    hyper3_hidden_weight: torch.Tensor,
    hyper3_hidden_bias: torch.Tensor,
    hyper3_proj_out_weight: torch.Tensor,
    hyper3_proj_out_bias: torch.Tensor,
):
    """Fused IoU prediction and hypernetwork weight generation for SAM-HQ.
    
    Args:
        iou_token_out: (batch_size, point_batch_size, 256) - IoU token embeddings
        mask_tokens_out: (batch_size, point_batch_size, 4, 256) - Mask token embeddings
        Various weight and bias tensors for IoU head and 4 hypernetwork MLPs
    
    Returns:
        iou_scores: (batch_size, point_batch_size, 4) - Predicted IoU scores
        hyper_weights: (batch_size, point_batch_size, 4, 32) - Dynamic convolution weights
    """
    # IoU prediction path (3-layer MLP with ReLU)
    # Layer 1: 256 -> 1024
    iou_h1 = linear(iou_token_out, iou_proj_in_weight, iou_proj_in_bias)
    relu_(iou_h1)

    # Layer 2: 1024 -> 1024
    iou_h2 = linear(iou_h1, iou_hidden_weight, iou_hidden_bias)
    relu_(iou_h2)

    # Layer 3: 1024 -> 4
    iou_scores = linear(iou_h2, iou_proj_out_weight, iou_proj_out_bias)
    
    # Hypernetwork weight generation (4 parallel MLPs with ReLU)
    token0, token1, token2, token3 = mask_tokens_out.unbind(dim=2)
    
    # Hypernetwork 0
    h0_1 = linear(token0, hyper0_proj_in_weight, hyper0_proj_in_bias)
    relu_(h0_1)
    h0_2 = linear(h0_1, hyper0_hidden_weight, hyper0_hidden_bias)
    relu_(h0_2)
    w0 = linear(h0_2, hyper0_proj_out_weight, hyper0_proj_out_bias)
    
    # Hypernetwork 1
    h1_1 = linear(token1, hyper1_proj_in_weight, hyper1_proj_in_bias)
    relu_(h1_1)
    h1_2 = linear(h1_1, hyper1_hidden_weight, hyper1_hidden_bias)
    relu_(h1_2)
    w1 = linear(h1_2, hyper1_proj_out_weight, hyper1_proj_out_bias)
    
    # Hypernetwork 2
    h2_1 = linear(token2, hyper2_proj_in_weight, hyper2_proj_in_bias)
    relu_(h2_1)
    h2_2 = linear(h2_1, hyper2_hidden_weight, hyper2_hidden_bias)
    relu_(h2_2)
    w2 = linear(h2_2, hyper2_proj_out_weight, hyper2_proj_out_bias)
    
    # Hypernetwork 3
    h3_1 = linear(token3, hyper3_proj_in_weight, hyper3_proj_in_bias)
    relu_(h3_1)
    h3_2 = linear(h3_1, hyper3_hidden_weight, hyper3_hidden_bias)
    relu_(h3_2)
    w3 = linear(h3_2, hyper3_proj_out_weight, hyper3_proj_out_bias)
    
    # Stack all hypernetwork weights: (batch_size, point_batch_size, 4, 32)
    hyper_weights = torch.stack((w0, w1, w2, w3), dim=2)
    
    return iou_scores, hyper_weights
