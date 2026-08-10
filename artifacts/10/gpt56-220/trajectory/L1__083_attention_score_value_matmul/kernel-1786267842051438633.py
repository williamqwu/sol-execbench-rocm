import torch

@torch.compile(fullgraph=True, dynamic=True, mode="reduce-overhead")
def _compiled(attention_weights: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch_size, _, seq_len_q, _ = attention_weights.shape
    x = torch.bmm(
        attention_weights.flatten(0, 1),
        value.flatten(0, 1),
    )
    return x.unflatten(0, (batch_size, 20)).transpose(1, 2).contiguous().reshape(batch_size, seq_len_q, 1280)

@torch.no_grad()
def run(
    attention_weights: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fused attention output: attention_weights @ value with transpose and reshape.
    
    Args:
        attention_weights: Softmax-normalized attention scores [B, H, Q, K]
        value: Value matrix [B, H, K, D]
        
    Returns:
        output: Attention output [B, Q, H*D] ready for output projection
    """
    return _compiled(attention_weights, value)
