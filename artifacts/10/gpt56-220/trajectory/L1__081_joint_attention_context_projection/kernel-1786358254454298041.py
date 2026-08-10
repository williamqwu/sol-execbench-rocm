import torch


@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    image_features = torch.cat((image_attention_output, image_attention_output), dim=-1)
    context_features = torch.cat((context_attention_output, context_attention_output), dim=-1)
    weight_t = to_out_weight.t()
    image = torch.matmul(image_features, weight_t) + to_out_bias
    context = torch.matmul(context_features, weight_t) + to_out_bias
    return image, context
