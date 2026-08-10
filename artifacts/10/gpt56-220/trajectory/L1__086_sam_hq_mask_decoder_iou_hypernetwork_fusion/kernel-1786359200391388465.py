import torch
import torch.nn.functional as F

@torch.no_grad()
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
    iou_h1 = F.linear(iou_token_out, iou_proj_in_weight, iou_proj_in_bias)
    iou_h1 = F.relu(iou_h1, inplace=True)
    
    # Layer 2: 1024 -> 1024
    iou_h2 = F.linear(iou_h1, iou_hidden_weight, iou_hidden_bias)
    iou_h2 = F.relu(iou_h2, inplace=True)
    
    # Layer 3: 1024 -> 4
    iou_scores = F.linear(iou_h2, iou_proj_out_weight, iou_proj_out_bias)
    
    # Hypernetwork weight generation (4 parallel MLPs with ReLU)
    hyper_weights_list = []
    batch_size, point_batch_size = mask_tokens_out.shape[:2]
    hyper_weights_storage = torch.empty(
        (4, batch_size, point_batch_size, 32),
        device=mask_tokens_out.device, dtype=mask_tokens_out.dtype)
    
    # Hypernetwork 0
    token0 = mask_tokens_out[:, :, 0, :]
    h0_1 = F.relu(F.linear(token0, hyper0_proj_in_weight, hyper0_proj_in_bias), inplace=True)
    h0_2 = F.relu(F.linear(h0_1, hyper0_hidden_weight, hyper0_hidden_bias), inplace=True)
    torch.addmm(hyper0_proj_out_bias, h0_2.reshape(-1, 256),
                hyper0_proj_out_weight.t(),
                out=hyper_weights_storage[0].reshape(-1, 32))
    
    # Hypernetwork 1
    token1 = mask_tokens_out[:, :, 1, :]
    h1_1 = F.relu(F.linear(token1, hyper1_proj_in_weight, hyper1_proj_in_bias), inplace=True)
    h1_2 = F.relu(F.linear(h1_1, hyper1_hidden_weight, hyper1_hidden_bias), inplace=True)
    torch.addmm(hyper1_proj_out_bias, h1_2.reshape(-1, 256),
                hyper1_proj_out_weight.t(),
                out=hyper_weights_storage[1].reshape(-1, 32))
    
    # Hypernetwork 2
    token2 = mask_tokens_out[:, :, 2, :]
    h2_1 = F.relu(F.linear(token2, hyper2_proj_in_weight, hyper2_proj_in_bias), inplace=True)
    h2_2 = F.relu(F.linear(h2_1, hyper2_hidden_weight, hyper2_hidden_bias), inplace=True)
    torch.addmm(hyper2_proj_out_bias, h2_2.reshape(-1, 256),
                hyper2_proj_out_weight.t(),
                out=hyper_weights_storage[2].reshape(-1, 32))
    
    # Hypernetwork 3
    token3 = mask_tokens_out[:, :, 3, :]
    h3_1 = F.relu(F.linear(token3, hyper3_proj_in_weight, hyper3_proj_in_bias), inplace=True)
    h3_2 = F.relu(F.linear(h3_1, hyper3_hidden_weight, hyper3_hidden_bias), inplace=True)
    torch.addmm(hyper3_proj_out_bias, h3_2.reshape(-1, 256),
                hyper3_proj_out_weight.t(),
                out=hyper_weights_storage[3].reshape(-1, 32))
    
    # Stack all hypernetwork weights: (batch_size, point_batch_size, 4, 32)
    hyper_weights = hyper_weights_storage.permute(1, 2, 0, 3)
    
    return iou_scores, hyper_weights
