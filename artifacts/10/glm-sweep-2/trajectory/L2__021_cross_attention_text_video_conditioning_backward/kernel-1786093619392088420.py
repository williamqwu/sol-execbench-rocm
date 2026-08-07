import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_weight: torch.Tensor,
    query_bias: torch.Tensor,
    key_weight: torch.Tensor,
    key_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    scale: float,
):
    batch_size, num_video_tokens, hidden_size = video_latents.shape
    num_text_tokens = text_embeddings.shape[1]
    num_heads = 16
    head_dim = 64

    with torch.enable_grad():
        vl = video_latents.detach().requires_grad_(True)
        te = text_embeddings.detach().requires_grad_(True)
        qw = query_weight.detach().requires_grad_(True)
        qb = query_bias.detach().requires_grad_(True)
        kw = key_weight.detach().requires_grad_(True)
        kb = key_bias.detach().requires_grad_(True)
        vw = value_weight.detach().requires_grad_(True)
        vb = value_bias.detach().requires_grad_(True)
        ow = output_weight.detach().requires_grad_(True)
        ob = output_bias.detach().requires_grad_(True)

        queries = F.linear(vl, qw, qb).view(batch_size, num_video_tokens, num_heads, head_dim).transpose(1, 2)
        keys = F.linear(te, kw, kb).view(batch_size, num_text_tokens, num_heads, head_dim).transpose(1, 2)
        values = F.linear(te, vw, vb).view(batch_size, num_text_tokens, num_heads, head_dim).transpose(1, 2)

        context = F.scaled_dot_product_attention(queries, keys, values, scale=scale)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_video_tokens, hidden_size)

        output = F.linear(context, ow, ob)
        output.backward(grad_output)

    return (
        vl.grad,
        te.grad,
        qw.grad,
        qb.grad,
        kw.grad,
        kb.grad,
        vw.grad,
        vb.grad,
        ow.grad,
        ob.grad,
    )
