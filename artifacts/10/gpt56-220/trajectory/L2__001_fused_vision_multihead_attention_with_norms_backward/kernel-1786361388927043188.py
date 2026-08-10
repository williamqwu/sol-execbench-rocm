import torch
import torch.nn.functional as F


@torch.compile(dynamic=True)
def _softmax_backward_pointwise(grad, weights, summed):
    return weights * (grad - summed)


@torch.compile(dynamic=True)
def _layer_norm_fused(x, mean, var, eps, grad, weight, residual):
    std = torch.sqrt(var + eps)
    centered = x - mean
    normalized = centered / std
    grad_normalized = grad * weight
    grad_div_std = grad_normalized / std
    mean_grad = grad_div_std.mean(dim=-1, keepdim=True)
    mean_centered = (grad_normalized * centered).mean(
        dim=-1, keepdim=True
    ) / (std ** 2)
    grad_x = residual + grad_div_std - mean_grad - centered * mean_centered
    return grad * normalized, grad_x


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
    """
    Backward pass for fused vision multi-head attention with norms.
    
    Computes gradients through:
    1. Residual connection
    2. Output projection
    3. Attention mechanism (softmax, matmul)
    4. Head split/merge
    5. QKV projection
    6. LayerNorm
    """
    batch_size, seq_len, embed_dim = x.shape
    num_heads = 16
    head_dim = 64
    
    # Gradient through residual connection
    # output = output_before_residual + residual
    grad_output_before_residual = grad_output
    grad_residual = grad_output
    
    # Gradient through output projection
    # output = F.linear(attn_output, out_weight, out_bias)
    grad_attn_output = torch.mm(
        grad_output_before_residual.reshape(-1, embed_dim), out_weight
    ).view(batch_size, seq_len, embed_dim)
    grad_out_weight = torch.matmul(
        grad_output_before_residual.reshape(-1, embed_dim).t(),
        attn_output.reshape(-1, embed_dim)
    )
    grad_out_bias = grad_output_before_residual.reshape(-1, embed_dim).sum(dim=0)
    
    # Gradient through head merging
    grad_attn_output_heads = grad_attn_output.view(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    
    # Gradient through attention application: attn_output = attn_weights @ v
    bh = batch_size * num_heads
    grad_heads_3d = grad_attn_output_heads.reshape(bh, seq_len, head_dim)
    weights_3d = attn_weights.reshape(bh, seq_len, seq_len)
    v_3d = v.reshape(bh, seq_len, head_dim)
    q_3d = q.reshape(bh, seq_len, head_dim)
    k_3d = k.reshape(bh, seq_len, head_dim)

    grad_v = torch.bmm(weights_3d.transpose(1, 2), grad_heads_3d).view(
        batch_size, num_heads, seq_len, head_dim
    )
    
    grad_attn_weights = torch.bmm(grad_heads_3d, v_3d.transpose(1, 2)).view(
        batch_size, num_heads, seq_len, seq_len
    )
    
    # Gradient through softmax
    sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
    grad_attn_scores = _softmax_backward_pointwise(
        grad_attn_weights, attn_weights, sum_grad
    )
    
    # Gradient through scaled dot-product: attn_scores = (q * scale) @ k^T
    scores_3d = grad_attn_scores.reshape(bh, seq_len, seq_len)
    grad_q = torch.bmm(scores_3d, k_3d).view(
        batch_size, num_heads, seq_len, head_dim
    ) * scale
    grad_k = torch.bmm(scores_3d.transpose(1, 2), q_3d * scale).view(
        batch_size, num_heads, seq_len, head_dim
    )
    
    # Gradient through head splitting
    grad_q = grad_q.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_k = grad_k.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    grad_v = grad_v.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    
    # Gradient through QKV chunking
    grad_qkv = torch.cat([grad_q, grad_k, grad_v], dim=-1)
    
    # Gradient through QKV projection
    grad_x_norm = torch.mm(
        grad_qkv.reshape(-1, 3 * embed_dim), qkv_weight
    ).view(batch_size, seq_len, embed_dim)
    grad_qkv_weight = torch.matmul(
        grad_qkv.reshape(-1, 3 * embed_dim).t(),
        x_norm.reshape(-1, embed_dim)
    )
    grad_qkv_bias = grad_qkv.reshape(-1, 3 * embed_dim).sum(dim=0)
    
    # Gradient through LayerNorm
    ln_weight_product, grad_x = _layer_norm_fused(
            x, x_mean, x_var, norm_eps, grad_x_norm, ln_weight, grad_residual
        )
    grad_ln_weight = ln_weight_product.reshape(-1, embed_dim).sum(dim=0)
    grad_ln_bias = grad_x_norm.reshape(-1, embed_dim).sum(dim=0)
    
    return (
        grad_x,
        grad_qkv_weight,
        grad_qkv_bias,
        grad_out_weight,
        grad_out_bias,
        grad_ln_weight,
        grad_ln_bias,
    )
