import torch
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 32


@torch.compile(fullgraph=True, dynamic=False)
def _compiled_run(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, eps):
    batch_size, seq_len, _ = encoder_hidden_states.shape
    hidden_states = encoder_hidden_states.float()
    variance = hidden_states.square().mean(-1, keepdim=True)
    normalized = (norm_weight * hidden_states * torch.rsqrt(variance + eps)).half()
    keys = F.linear(normalized, k_proj_weight)
    values = F.linear(normalized, v_proj_weight)
    keys = keys.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    values = values.view(batch_size, seq_len, 2, 64).transpose(1, 2)
    return keys, values

@torch.no_grad()
def run(
    encoder_hidden_states: torch.Tensor,
    norm_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    eps: float,
):
    """
    Fused encoder final RMSNorm with cross-attention K/V projection.
    
    Args:
        encoder_hidden_states: (batch_size, encoder_seq_len, 1024)
        norm_weight: (1024,) RMSNorm weight
        k_proj_weight: (128, 1024) Key projection weight
        v_proj_weight: (128, 1024) Value projection weight
        eps: RMSNorm epsilon
    
    Returns:
        keys: (batch_size, 2, encoder_seq_len, 64)
        values: (batch_size, 2, encoder_seq_len, 64)
    """
    return _compiled_run(encoder_hidden_states, norm_weight, k_proj_weight, v_proj_weight, eps)

    batch_size, seq_len, hidden_size = encoder_hidden_states.shape
    num_kv_heads = 2
    head_dim = 64
    
    # RMSNorm computation
    input_dtype = encoder_hidden_states.dtype
    hidden_states = encoder_hidden_states.to(torch.float32)
    
    # Compute variance and normalize
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    normalized = (norm_weight * hidden_states).to(input_dtype)
    
    # K/V projections
    # normalized: (batch, seq_len, 1024)
    # weights: (128, 1024)
    # output: (batch, seq_len, 128)
    kv_weight = torch.cat((k_proj_weight, v_proj_weight), dim=0)
    kv_flat = F.linear(normalized, kv_weight, bias=None)
    
    # Reshape to multi-head format
    # (batch, seq_len, 128) -> (batch, seq_len, 2, 64) -> (batch, 2, seq_len, 64)
    kv = kv_flat.view(batch_size, seq_len, 2 * num_kv_heads, head_dim)
    kv = kv.transpose(1, 2).contiguous()
    keys = kv[:, :num_kv_heads]
    values = kv[:, num_kv_heads:]
    
    return keys, values
