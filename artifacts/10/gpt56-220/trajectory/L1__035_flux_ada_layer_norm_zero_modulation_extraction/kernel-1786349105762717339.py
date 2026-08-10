import torch

def run(emb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    AdaLayerNormZero modulation parameter extraction.
    
    Performs a linear projection followed by chunking into 6 modulation parameters.
    
    Args:
        emb: Timestep embeddings of shape [batch_size, inner_dim]
        weight: Linear weight matrix of shape [6 * inner_dim, inner_dim]
        bias: Linear bias of shape [6 * inner_dim]
        
    Returns:
        Tuple of 6 tensors, each of shape [batch_size, inner_dim]:
        - shift_msa, scale_msa, gate_msa: for attention path
        - shift_mlp, scale_mlp, gate_mlp: for MLP path
    """
    # Linear projection: [batch_size, inner_dim] @ [inner_dim, 6*inner_dim] + bias
    # = [batch_size, 6*inner_dim]
    emb_out = torch._C._nn.linear(emb, weight, bias)
    
    # Chunk into 6 equal parts along the last dimension
    # Each chunk has shape [batch_size, inner_dim]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.ops.aten.split_with_sizes.default(
        emb_out, [3072, 3072, 3072, 3072, 3072, 3072], 1
    )
    
    return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
