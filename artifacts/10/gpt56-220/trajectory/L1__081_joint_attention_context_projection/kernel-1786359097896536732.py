import torch
import torch.nn.functional as F


@torch.compile
def _project(image_attention_output, context_attention_output, to_out_weight, to_out_bias):
    image_seq_len = image_attention_output.shape[1]
    combined = torch.cat((image_attention_output, context_attention_output), dim=1)
    features = torch.cat((combined, combined), dim=-1)
    output = F.linear(features, to_out_weight, to_out_bias)
    return torch.split(
        output,
        (image_seq_len, context_attention_output.shape[1]),
        dim=1,
    )


@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    return _project(image_attention_output, context_attention_output, to_out_weight, to_out_bias)
