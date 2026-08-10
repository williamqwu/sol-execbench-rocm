import torch

def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Generate inputs for backward pass testing."""
    total_patches = axes_and_scalars['total_patches']
    head_dim = axes_and_scalars['head_dim']
    head_dim_quarter = axes_and_scalars['head_dim_quarter']
    max_grid_size = axes_and_scalars['max_grid_size']
    
    # Gradients from upstream
    grad_cos = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    grad_sin = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    # Position IDs - random valid indices into frequency table
    pos_ids = torch.randint(0, max_grid_size, (total_patches, 2), device=device, dtype=torch.int64)
    
    # Inverse frequencies (learnable parameter)
    inv_freq = torch.randn(head_dim_quarter, device=device, dtype=torch.float32)
    
    # Saved embedding from forward pass (random values representing actual embeddings)
    emb = torch.randn(total_patches, head_dim, device=device, dtype=torch.float32)
    
    return {
        'grad_cos': grad_cos,
        'grad_sin': grad_sin,
        'pos_ids': pos_ids,
        'inv_freq': inv_freq,
        'emb': emb
    }

@torch.no_grad()
def run(
    grad_cos: torch.Tensor,
    grad_sin: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    emb: torch.Tensor
) -> torch.Tensor:
    """Backward pass for vision rotary position embedding generation.
    
    Computes gradient w.r.t. inverse frequencies through:
    1. Gradient through cos/sin: grad_emb = -grad_cos * sin(emb) + grad_sin * cos(emb)
    2. Gradient through concatenation: split and sum
    3. Gradient through indexing: scatter via index_add_
    4. Gradient through outer product: matrix multiply with seq
    
    Args:
        grad_cos: [total_patches, head_dim] gradient w.r.t. cosine output
        grad_sin: [total_patches, head_dim] gradient w.r.t. sine output
        pos_ids: [total_patches, 2] position indices for h and w
        inv_freq: [head_dim//4] inverse frequency tensor
        emb: [total_patches, head_dim] saved embedding from forward
        
    Returns:
        grad_inv_freq: [head_dim//4] gradient w.r.t. inverse frequencies
    """
    grad_emb = -grad_cos * emb.sin() + grad_sin * emb.cos()
    grad_rotary = grad_emb[:, :36] + grad_emb[:, 36:]
    grad_rotary = grad_rotary.reshape(-1, 2, 18)
    weights = pos_ids.to(torch.float32).unsqueeze(-1)
    return (grad_rotary * weights).sum(dim=(0, 1))
