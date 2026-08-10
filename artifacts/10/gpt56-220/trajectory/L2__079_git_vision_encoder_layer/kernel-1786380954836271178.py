import torch
import torch.nn.functional as F

@torch.compile(fullgraph=True, dynamic=True)
def _quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

@torch.compile(fullgraph=True, dynamic=True)
def _norm_affine(x, mean, var, weight, bias, eps):
    return (x - mean) / torch.sqrt(var + eps) * weight + bias

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 12
    head_dim = hidden_size // num_heads
    scale = head_dim ** -0.5
    
    # ===== Attention Block =====
    residual = hidden_states
    
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    if batch_size * seq_len >= 1024:
        hidden_states = _norm_affine(hidden_states, mean, var, layer_norm1_weight, layer_norm1_bias, layer_norm_eps)
    else:
        hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
        hidden_states = hidden_states * layer_norm1_weight + layer_norm1_bias
    
    # Q, K, V projections
    queries = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    keys = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    values = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    
    # Reshape for multi-head attention
    queries = queries.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Attention computation
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
    
    # Output projection
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    hidden_states = attn_output.add_(residual)
    
    # ===== MLP Block =====
    residual = hidden_states
    
    mean = hidden_states.mean(dim=-1, keepdim=True)
    var = ((hidden_states - mean) ** 2).mean(dim=-1, keepdim=True)
    if batch_size * seq_len >= 1024:
        hidden_states = _norm_affine(hidden_states, mean, var, layer_norm2_weight, layer_norm2_bias, layer_norm_eps)
    else:
        hidden_states = (hidden_states - mean) / torch.sqrt(var + layer_norm_eps)
        hidden_states = hidden_states * layer_norm2_weight + layer_norm2_bias
    
    # MLP: fc1 -> quick_gelu -> fc2
    hidden_states = F.linear(hidden_states, fc1_weight, fc1_bias)
    # Quick GELU: x * sigmoid(1.702 * x)
    if batch_size * seq_len >= 1024:
        hidden_states = _quick_gelu(hidden_states)
    else:
        hidden_states = hidden_states * torch.sigmoid(1.702 * hidden_states)
    hidden_states = F.linear(hidden_states, fc2_weight, fc2_bias)
    
    # Residual connection
    output = hidden_states.add_(residual)
    
    return output
