import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    image_seq_len = image_attention_output.shape[1]
    x = torch.cat((image_attention_output, context_attention_output), dim=1)
    # [x, x] @ [W0, W1]^T == x @ (W0 + W1)^T
    half = image_attention_output.shape[-1]
    weight = to_out_weight[:, :half] + to_out_weight[:, half:]
    y = F.linear(x, weight, to_out_bias)
    return y[:, :image_seq_len, :], y[:, image_seq_len:, :]
