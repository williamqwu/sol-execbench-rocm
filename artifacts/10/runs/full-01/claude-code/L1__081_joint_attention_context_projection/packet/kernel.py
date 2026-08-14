import torch


@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    D = image_attention_output.shape[-1]
    # [X|X] @ W^T == X @ (W1 + W2)^T
    weff = to_out_weight[:, :D] + to_out_weight[:, D:]
    out_img = torch.nn.functional.linear(image_attention_output, weff, to_out_bias)
    out_ctx = torch.nn.functional.linear(context_attention_output, weff, to_out_bias)
    return out_img, out_ctx
