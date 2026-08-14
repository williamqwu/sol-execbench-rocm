import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    grad_output,
    video_latents,
    text_embeddings,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    scale,
):
    batch_size, num_video_tokens, hidden_size = video_latents.shape
    num_text_tokens = text_embeddings.shape[1]
    num_heads = 16
    head_dim = 64

    queries = F.linear(video_latents, query_weight, query_bias)
    queries = queries.view(batch_size, num_video_tokens, num_heads, head_dim).transpose(1, 2)
    keys = F.linear(text_embeddings, key_weight, key_bias)
    keys = keys.view(batch_size, num_text_tokens, num_heads, head_dim).transpose(1, 2)
    values = F.linear(text_embeddings, value_weight, value_bias)
    values = values.view(batch_size, num_text_tokens, num_heads, head_dim).transpose(1, 2)

    attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
    attention_scores.mul_(scale)
    attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
    context = torch.matmul(attention_probs, values)
    context = context.transpose(1, 2).contiguous().view(batch_size, num_video_tokens, hidden_size)

    grad_context = torch.matmul(grad_output, output_weight)
    grad_output_weight = torch.einsum("bnd,bnk->dk", grad_output, context)
    grad_output_bias = grad_output.sum(dim=(0, 1))

    grad_context_heads = grad_context.view(
        batch_size, num_video_tokens, num_heads, head_dim
    ).transpose(1, 2)
    grad_attention_probs = torch.matmul(grad_context_heads, values.transpose(-2, -1))
    grad_values = torch.matmul(attention_probs.transpose(-2, -1), grad_context_heads)
    sum_grad_probs = (grad_attention_probs * attention_probs).sum(dim=-1, keepdim=True)
    grad_attention_probs.sub_(sum_grad_probs)
    grad_attention_probs.mul_(attention_probs)
    grad_attention_probs.mul_(scale)
    grad_attention_scores = grad_attention_probs
    grad_queries = torch.matmul(grad_attention_scores, keys)
    grad_keys = torch.matmul(grad_attention_scores.transpose(-2, -1), queries)

    grad_queries = grad_queries.transpose(1, 2).contiguous().view(
        batch_size, num_video_tokens, hidden_size
    )
    grad_keys = grad_keys.transpose(1, 2).contiguous().view(
        batch_size, num_text_tokens, hidden_size
    )
    grad_values = grad_values.transpose(1, 2).contiguous().view(
        batch_size, num_text_tokens, hidden_size
    )

    grad_video_latents = torch.matmul(grad_queries, query_weight)
    grad_query_weight = torch.einsum("bnd,bnk->dk", grad_queries, video_latents)
    grad_query_bias = grad_queries.sum(dim=(0, 1))
    grad_text_from_keys = torch.matmul(grad_keys, key_weight)
    grad_key_weight = torch.einsum("bnd,bnk->dk", grad_keys, text_embeddings)
    grad_key_bias = grad_keys.sum(dim=(0, 1))
    grad_text_from_values = torch.matmul(grad_values, value_weight)
    grad_value_weight = torch.einsum("bnd,bnk->dk", grad_values, text_embeddings)
    grad_value_bias = grad_values.sum(dim=(0, 1))
    grad_text_embeddings = grad_text_from_keys.add_(grad_text_from_values)

    return (
        grad_video_latents,
        grad_text_embeddings,
        grad_query_weight,
        grad_query_bias,
        grad_key_weight,
        grad_key_bias,
        grad_value_weight,
        grad_value_bias,
        grad_output_weight,
        grad_output_bias,
    )
