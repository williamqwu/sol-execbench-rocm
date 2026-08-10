import torch

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
    batch_size, _, seq_len_q, _ = attention_weights.shape
    return torch.einsum("bhqk,bhkd->bqhd", attention_weights, value).reshape(batch_size, seq_len_q, 1280)
