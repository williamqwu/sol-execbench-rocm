import torch

_compiled = None

def _fn(image_attention_output, context_attention_output, to_out_weight, to_out_bias):
    image_seq_len = image_attention_output.shape[1]
    combined_attention = torch.cat(
        [image_attention_output, context_attention_output],
        dim=1
    )
    combined_features = torch.cat(
        [combined_attention, combined_attention],
        dim=-1
    )
    projected_output = torch.matmul(combined_features, to_out_weight.t()) + to_out_bias
    projected_image = projected_output[:, :image_seq_len, :]
    projected_context = projected_output[:, image_seq_len:, :]
    return projected_image, projected_context

@torch.no_grad()
def run(
    image_attention_output: torch.Tensor,
    context_attention_output: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_fn, fullgraph=True, dynamic=True)
    return _compiled(image_attention_output, context_attention_output, to_out_weight, to_out_bias)
