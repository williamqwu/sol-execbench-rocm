import torch
import triton
import triton.language as tl

@triton.jit
def _backward_kernel(gc, gs, pos, emb, out, n: tl.constexpr, BLOCK: tl.constexpr,
                     DIRECT: tl.constexpr):
    d = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = p < n
    base = p * 72 + d
    h0 = tl.load(emb + base, mask=mask, other=0.0)
    h1 = tl.load(emb + base + 36, mask=mask, other=0.0)
    w0 = tl.load(emb + base + 18, mask=mask, other=0.0)
    w1 = tl.load(emb + base + 54, mask=mask, other=0.0)
    gh = (-tl.load(gc + base, mask=mask, other=0.0) * tl.sin(h0)
          + tl.load(gs + base, mask=mask, other=0.0) * tl.cos(h0)
          - tl.load(gc + base + 36, mask=mask, other=0.0) * tl.sin(h1)
          + tl.load(gs + base + 36, mask=mask, other=0.0) * tl.cos(h1))
    gw = (-tl.load(gc + base + 18, mask=mask, other=0.0) * tl.sin(w0)
          + tl.load(gs + base + 18, mask=mask, other=0.0) * tl.cos(w0)
          - tl.load(gc + base + 54, mask=mask, other=0.0) * tl.sin(w1)
          + tl.load(gs + base + 54, mask=mask, other=0.0) * tl.cos(w1))
    ph = tl.load(pos + p * 2, mask=mask, other=0).to(tl.float32)
    pw = tl.load(pos + p * 2 + 1, mask=mask, other=0).to(tl.float32)
    result = tl.sum(gh * ph + gw * pw, axis=0)
    if DIRECT:
        tl.store(out + d, result)
    else:
        tl.atomic_add(out + d, result)

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
    n = grad_cos.shape[0]
    if n <= 256:
        block = 256
    elif n <= 512:
        block = 512
    elif n <= 1024:
        block = 128
    elif n >= 4096:
        block = 512
    else:
        block = 256
    direct = n <= block
    warps = 2 if block == 128 else (8 if block == 512 else 4)
    out = (torch.empty if direct else torch.zeros)(
        (18,), device=grad_cos.device, dtype=torch.float32)
    _backward_kernel[(18, triton.cdiv(n, block))](
        grad_cos, grad_sin, pos_ids, emb, out, n=n, BLOCK=block,
        DIRECT=direct, num_warps=warps)
    return out
