import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')

_compiled_compute = None

def _compute(
    query, key, value, g, beta, scale,
    batch_size, num_k_heads, sequence_length, k_head_dim,
    num_v_heads, v_head_dim, chunk_size, num_v_per_k,
):
    # Convert to float32 for numerical stability
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    beta = beta.to(torch.float32)
    g = g.to(torch.float32)

    # Apply scaling to queries
    query = query * scale

    # Pad sequence to multiple of chunk_size
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    num_chunks = total_sequence_length // chunk_size

    # Compute beta-weighted values and keys
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key.unsqueeze(1).expand(-1, num_v_per_k, -1, -1, -1)
    k_beta = k_beta.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    k_beta = k_beta * beta.unsqueeze(-1)

    # Reshape to chunks
    query_chunks = query.unsqueeze(1).expand(-1, num_v_per_k, -1, -1, -1)
    query_chunks = query_chunks.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    query_chunks = query_chunks.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)

    key_chunks = key.unsqueeze(1).expand(-1, num_v_per_k, -1, -1, -1)
    key_chunks = key_chunks.reshape(batch_size, num_v_heads, total_sequence_length, k_head_dim)
    key_chunks = key_chunks.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)

    value_chunks = value.reshape(batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim)
    k_beta_chunks = k_beta.reshape(batch_size, num_v_heads, num_chunks, chunk_size, k_head_dim)
    v_beta_chunks = v_beta.reshape(batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim)
    g_chunks = g.reshape(batch_size, num_v_heads, num_chunks, chunk_size)

    # Compute cumulative decay within chunks
    g_cumsum = g_chunks.cumsum(dim=-1)

    # Create decay mask (lower-triangular)
    decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp()
    decay_mask = decay_mask.tril()

    # Precompute exp(g) once
    g_exp = g_chunks.exp()

    # Create causal mask for attention
    causal_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0
    )

    # Compute intra-chunk attention: A = -k_beta @ key^T (strictly lower after mask)
    attn = -(k_beta_chunks @ key_chunks.transpose(-1, -2)) * decay_mask
    attn = attn.masked_fill(causal_mask, 0)

    # (I - A)^{-1} @ X == solve_triangular(I - A, X, upper=False, unitriangular=True)
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    M = eye - attn

    value_transformed = torch.linalg.solve_triangular(M, v_beta_chunks, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(M, k_beta_chunks * g_exp.unsqueeze(-1), upper=False, unitriangular=True)

    # Precompute per-chunk quantities for the recurrent loop
    # chunk_decay: [b, h, num_chunks, 1, 1]
    g_chunk_last = g_chunks[:, :, :, -1]  # [b, h, num_chunks]
    chunk_decay_all = g_chunk_last.exp().unsqueeze(-1).unsqueeze(-1)  # [b, h, num_chunks, 1, 1]
    # token_decay: [b, h, num_chunks, chunk_size, 1]
    token_decay_all = (g_chunk_last.unsqueeze(-1) - g_chunks).exp().unsqueeze(-1)
    # q_scaled for inter-chunk: q_chunk * g_exp
    q_gexp = query_chunks * g_exp.unsqueeze(-1)  # [b, h, num_chunks, cs, kd]
    # k * token_decay for state update
    k_tokdecay = key_chunks * token_decay_all  # [b, h, num_chunks, cs, kd]

    # Initialize recurrent state
    recurrent_state = torch.zeros(
        batch_size, num_v_heads, k_head_dim, v_head_dim,
        dtype=torch.float32, device=value.device
    )

    # Output tensor
    output = torch.zeros(
        batch_size, num_v_heads, num_chunks, chunk_size, v_head_dim,
        dtype=torch.float32, device=value.device
    )

    # Create upper triangular mask
    upper_mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1
    )

    for chunk_idx in range(num_chunks):
        q_chunk = query_chunks[:, :, chunk_idx]
        k_chunk = key_chunks[:, :, chunk_idx]
        v_transformed_chunk = value_transformed[:, :, chunk_idx]
        k_cumdecay_chunk = k_cumdecay[:, :, chunk_idx]
        decay_mask_chunk = decay_mask[:, :, chunk_idx]
        q_gexp_chunk = q_gexp[:, :, chunk_idx]
        k_tokdecay_chunk = k_tokdecay[:, :, chunk_idx]

        attn_intra = (q_chunk @ k_chunk.transpose(-1, -2)) * decay_mask_chunk
        attn_intra = attn_intra.masked_fill_(upper_mask, 0)

        v_prime = k_cumdecay_chunk @ recurrent_state
        v_new = v_transformed_chunk - v_prime

        attn_inter = q_gexp_chunk @ recurrent_state

        output[:, :, chunk_idx] = attn_inter + attn_intra @ v_new

        recurrent_state = recurrent_state * chunk_decay_all[:, :, chunk_idx]
        recurrent_state = recurrent_state + k_tokdecay_chunk.transpose(-1, -2) @ v_new

    # Reshape output and remove padding
    output = output.reshape(batch_size, num_v_heads, total_sequence_length, v_head_dim)
    output = output[:, :, :sequence_length]

    return output


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
):
    """
    Chunk-based Gated Delta Rule linear attention.
    """
    global _compiled_compute
    initial_dtype = query.dtype
    batch_size, num_k_heads, sequence_length, k_head_dim = key.shape
    num_v_heads = value.shape[1]
    v_head_dim = value.shape[-1]
    chunk_size = 64
    num_v_per_k = num_v_heads // num_k_heads

    # L2 normalization
    def l2norm(x, dim=-1, eps=1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    query = l2norm(query, dim=-1, eps=1e-6)
    key = l2norm(key, dim=-1, eps=1e-6)

    if _compiled_compute is None:
        _compiled_compute = torch.compile(_compute, dynamic=False)

    output = _compiled_compute(query, key, value, g, beta, scale,
                      batch_size, num_k_heads, sequence_length, k_head_dim,
                      num_v_heads, v_head_dim, chunk_size, num_v_per_k)

    # Transpose to [batch, seq_len, num_v_heads, head_v_dim]
    output = output.transpose(1, 2).contiguous().to(initial_dtype)

    return output
