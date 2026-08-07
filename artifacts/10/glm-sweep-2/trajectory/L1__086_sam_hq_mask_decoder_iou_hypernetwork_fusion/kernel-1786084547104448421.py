import torch
import torch.nn.functional as F


def _run_impl(
    iou_token_out,
    mask_tokens_out,
    iou_proj_in_weight,
    iou_proj_in_bias,
    iou_hidden_weight,
    iou_hidden_bias,
    iou_proj_out_weight,
    iou_proj_out_bias,
    hyper0_proj_in_weight,
    hyper0_proj_in_bias,
    hyper0_hidden_weight,
    hyper0_hidden_bias,
    hyper0_proj_out_weight,
    hyper0_proj_out_bias,
    hyper1_proj_in_weight,
    hyper1_proj_in_bias,
    hyper1_hidden_weight,
    hyper1_hidden_bias,
    hyper1_proj_out_weight,
    hyper1_proj_out_bias,
    hyper2_proj_in_weight,
    hyper2_proj_in_bias,
    hyper2_hidden_weight,
    hyper2_hidden_bias,
    hyper2_proj_out_weight,
    hyper2_proj_out_bias,
    hyper3_proj_in_weight,
    hyper3_proj_in_bias,
    hyper3_hidden_weight,
    hyper3_hidden_bias,
    hyper3_proj_out_weight,
    hyper3_proj_out_bias,
):
    # IoU prediction path (3-layer MLP with ReLU)
    iou_h1 = F.relu(F.linear(iou_token_out, iou_proj_in_weight, iou_proj_in_bias))
    iou_h2 = F.relu(F.linear(iou_h1, iou_hidden_weight, iou_hidden_bias))
    iou_scores = F.linear(iou_h2, iou_proj_out_weight, iou_proj_out_bias)

    # Hypernetwork weight generation (4 parallel MLPs with ReLU)
    hw = [
        hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight, hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias,
        hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight, hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias,
        hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight, hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias,
        hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight, hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias,
    ]
    hyper_weights_list = []
    for i in range(4):
        token = mask_tokens_out[:, :, i, :]
        h1 = F.relu(F.linear(token, hw[i * 6 + 0], hw[i * 6 + 1]))
        h2 = F.relu(F.linear(h1, hw[i * 6 + 2], hw[i * 6 + 3]))
        w = F.linear(h2, hw[i * 6 + 4], hw[i * 6 + 5])
        hyper_weights_list.append(w)
    hyper_weights = torch.stack(hyper_weights_list, dim=2)
    return iou_scores, hyper_weights


_compiled = torch.compile(_run_impl, mode="reduce-overhead", fullgraph=True)


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
    return _compiled(
        iou_token_out, mask_tokens_out,
        iou_proj_in_weight, iou_proj_in_bias, iou_hidden_weight, iou_hidden_bias, iou_proj_out_weight, iou_proj_out_bias,
        hyper0_proj_in_weight, hyper0_proj_in_bias, hyper0_hidden_weight, hyper0_hidden_bias, hyper0_proj_out_weight, hyper0_proj_out_bias,
        hyper1_proj_in_weight, hyper1_proj_in_bias, hyper1_hidden_weight, hyper1_hidden_bias, hyper1_proj_out_weight, hyper1_proj_out_bias,
        hyper2_proj_in_weight, hyper2_proj_in_bias, hyper2_hidden_weight, hyper2_hidden_bias, hyper2_proj_out_weight, hyper2_proj_out_bias,
        hyper3_proj_in_weight, hyper3_proj_in_bias, hyper3_hidden_weight, hyper3_hidden_bias, hyper3_proj_out_weight, hyper3_proj_out_bias,
    )
