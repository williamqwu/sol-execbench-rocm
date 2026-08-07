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
    iou_h1 = F.relu(F.linear(iou_token_out, iou_proj_in_weight, iou_proj_in_bias))
    iou_h2 = F.relu(F.linear(iou_h1, iou_hidden_weight, iou_hidden_bias))
    iou_scores = F.linear(iou_h2, iou_proj_out_weight, iou_proj_out_bias)

    outs = []
    for i, (wi, bi, wh, bh, wo, bo) in enumerate(
        [
            (hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight,
             hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias),
            (hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight,
             hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias),
            (hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight,
             hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias),
            (hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight,
             hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias),
        ]
    ):
        t = mask_tokens_out[:, :, i, :]
        h = F.relu(F.linear(t, wi, bi))
        h = F.relu(F.linear(h, wh, bh))
        outs.append(F.linear(h, wo, bo))

    hyper_weights = torch.stack(outs, dim=2)
    return iou_scores, hyper_weights
