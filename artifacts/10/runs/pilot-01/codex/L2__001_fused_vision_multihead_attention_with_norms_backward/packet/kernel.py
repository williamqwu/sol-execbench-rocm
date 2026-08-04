import torch


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_mean: torch.Tensor,
    x_var: torch.Tensor,
    x_norm: torch.Tensor,
    ln_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    out_weight: torch.Tensor,
    scale: float,
    norm_eps: float,
):
    batch_size, seq_len, embed_dim = x.shape
    num_heads = 16
    head_dim = 64

    grad_output_flat = grad_output.reshape(-1, embed_dim)
    grad_attn_output = torch.matmul(grad_output_flat, out_weight).reshape(
        batch_size, seq_len, embed_dim
    )
    grad_out_weight = torch.matmul(grad_output_flat.t(), attn_output.reshape(-1, embed_dim))
    grad_out_bias = grad_output.sum(dim=(0, 1))

    grad_attn_output_heads = grad_attn_output.view(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)

    grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_attn_output_heads)
    grad_attn_weights = torch.matmul(grad_attn_output_heads, v.transpose(-2, -1))

    sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)

    grad_q = torch.matmul(grad_attn_scores, k)
    grad_q.mul_(scale)
    grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), q * scale)

    grad_q = grad_q.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_k = grad_k.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_v = grad_v.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_qkv = torch.cat([grad_q, grad_k, grad_v], dim=-1)

    grad_qkv_flat = grad_qkv.reshape(-1, 3 * embed_dim)
    grad_x_norm = torch.matmul(grad_qkv_flat, qkv_weight).reshape(
        batch_size, seq_len, embed_dim
    )
    grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), x_norm.reshape(-1, embed_dim))
    grad_qkv_bias = grad_qkv.sum(dim=(0, 1))

    std = torch.sqrt(x_var + norm_eps)
    x_centered = x - x_mean
    x_normalized = x_centered / std
    x_normalized.mul_(grad_x_norm)
    grad_ln_weight = x_normalized.sum(dim=(0, 1))
    grad_ln_bias = grad_x_norm.sum(dim=(0, 1))

    grad_x_normalized = grad_x_norm * ln_weight
    grad_x_from_norm = grad_x_normalized / std
    mean_grad = grad_x_from_norm.mean(dim=-1, keepdim=True)
    grad_x_normalized.mul_(x_centered)
    mean_grad_x_centered = grad_x_normalized.mean(dim=-1, keepdim=True) / (std ** 2)
    grad_x_from_norm.sub_(mean_grad)
    x_centered.mul_(mean_grad_x_centered)
    grad_x_from_norm.sub_(x_centered)
    grad_x_from_norm.add_(grad_output)
    grad_x = grad_x_from_norm

    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
