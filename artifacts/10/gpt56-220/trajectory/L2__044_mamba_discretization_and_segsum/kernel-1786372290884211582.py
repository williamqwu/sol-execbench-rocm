import torch
import torch.nn.functional as F


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Stable segment sum using cumulative sums and masking.
    
    Args:
        input_tensor: [..., chunk_size] tensor
    
    Returns:
        tensor_segsum: [..., chunk_size, chunk_size] cumulative sum matrix
    """
    chunk_size = input_tensor.size(-1)
    device = input_tensor.device
    
    # Expand to [..., chunk_size, chunk_size] and create lower triangular structure
    input_expanded = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    
    # Create lower triangular mask (diagonal=-1 excludes diagonal)
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=-1
    )
    input_masked = input_expanded.masked_fill(~mask, 0)
    
    # Compute cumulative sum along second-to-last dimension
    tensor_segsum = torch.cumsum(input_masked, dim=-2)
    
    # Apply mask to keep only lower triangular part (including diagonal this time)
    mask_final = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=0
    )
    tensor_segsum = tensor_segsum.masked_fill(~mask_final, float('-inf'))
    
    return tensor_segsum


@torch.compile(fullgraph=True)
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
):
    """
    Mamba SSM parameter discretization and segment sum computation.
    
    Args:
        hidden_states: [batch, chunks, chunk_size, num_heads, head_dim]
        dt: [batch, chunks, chunk_size, num_heads]
        A_log: [num_heads] - log-scale state transition parameter
        B: [batch, chunks, chunk_size, n_groups, d_state]
        C: [batch, chunks, chunk_size, n_groups, d_state]
    
    Returns:
        L: Decay matrix for intra-chunk computation
        G: Attention-like weights from B and C contraction
        M: Masked attention weights
        A_cumsum: Cumulative decay factors
        decay_states: State decay factors for inter-chunk recurrence
    """
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    n_groups = B.shape[3]
    
    # 1. Discretize A parameter
    # A is in log scale, so exp(-A) gives the continuous-time decay
    A = -torch.exp(A_log.float())  # [num_heads]
    
    # Apply softplus to dt for numerical stability
    dt_soft = F.softplus(dt.float())  # [batch, chunks, chunk_size, num_heads]
    
    # Discretize: A_discrete = A * dt
    # Rearrange dt to [batch, num_heads, chunks, chunk_size]
    dt_permuted = dt_soft.permute(0, 3, 1, 2)  # [batch, num_heads, chunks, chunk_size]
    A_discrete = A[None, :, None, None] * dt_permuted  # [batch, num_heads, chunks, chunk_size]
    
    # 2. Compute cumulative sum of A (for decay factors)
    A_cumsum = torch.cumsum(A_discrete, dim=-1)  # [batch, num_heads, chunks, chunk_size]
    
    # 3. Prefix differences give the segment sums directly.  The reference's
    # expanded scan computes prefix[i] - prefix[j] in its lower triangle.
    seg = A_cumsum[..., :, None] - A_cumsum[..., None, :]
    lower = torch.ones(
        chunk_size, chunk_size, device=dt.device, dtype=torch.bool
    ).tril_()
    L = torch.where(lower, torch.exp(seg), 0.0)
    
    # There is one state group, shared by all heads.  Contract it once rather
    # than physically repeating B/C and asking GEMM for the same result 16x.
    G_group = torch.matmul(
        C[..., 0, :], B[..., 0, :].transpose(-1, -2)
    )
    G = G_group[..., None].expand(-1, -1, -1, -1, num_heads)
    
    # 6. Compute M matrix (masked attention weights)
    # M = G * L (element-wise, with proper broadcasting)
    # L: [batch, num_heads, chunks, chunk_size, chunk_size]
    # G: [batch, chunks, chunk_size, chunk_size, num_heads]
    # Need to align dimensions
    L_bf16 = L.to(torch.bfloat16)
    L_permuted = L_bf16.permute(0, 2, 3, 4, 1)
    M = G * L_permuted
    
    # 7. Compute decay factors for state updates
    # decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    decay_states = torch.exp(
        A_cumsum[:, :, :, -1:] - A_cumsum
    )  # [batch, num_heads, chunks, chunk_size]
    
    return (
        L_bf16,
        G,
        M,
        A_cumsum,
        decay_states
    )
