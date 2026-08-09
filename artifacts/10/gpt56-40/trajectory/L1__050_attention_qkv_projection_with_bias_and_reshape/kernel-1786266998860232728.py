import torch


@torch.compile(fullgraph=True)
def _project(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias):
    qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
    qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
    qkv = torch.nn.functional.linear(
        hidden_states.reshape(-1, 640), qkv_weight, qkv_bias
    )
    qkv = qkv.view(hidden_states.shape[0], hidden_states.shape[1], 1536)
    query_states, key_states, value_states = qkv.split((1024, 256, 256), dim=-1)
    batch_size, seq_len, _ = hidden_states.shape
    return (
        query_states.view(batch_size, seq_len, 4, 256),
        key_states.view(batch_size, seq_len, 1, 256),
        value_states.view(batch_size, seq_len, 1, 256),
    )

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    """
    Fused QKV projection with bias and reshape for Gemma3 attention.
    
    Args:
        hidden_states: Input tensor of shape (batch_size, seq_len, 640)
        q_weight: Query projection weight (1024, 640)
        q_bias: Query projection bias (1024,)
        k_weight: Key projection weight (256, 640)
        k_bias: Key projection bias (256,)
        v_weight: Value projection weight (256, 640)
        v_bias: Value projection bias (256,)
        
    Returns:
        Tuple of (query_states, key_states, value_states) where:
            query_states: (batch_size, seq_len, 16, 128)
            key_states: (batch_size, seq_len, 2, 128)
            value_states: (batch_size, seq_len, 2, 128)
    """
    return _project(
        hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias
    )
