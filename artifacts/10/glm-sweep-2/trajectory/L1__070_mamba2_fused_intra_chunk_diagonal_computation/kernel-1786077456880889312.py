import torch

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    CHUNK_SIZE = 128
    NUM_HEADS = 32
    N_GROUPS = 8

    batch_size, num_chunks, _, _, _ = hidden_states.shape
    state_size = B.shape[-1]

    # --- Build causal decay mask L[i,j] = exp(A_cumsum[i] - A_cumsum[j]) for i >= j, else 0 ---
    # A_cumsum: [batch, num_heads, num_chunks, chunk_size]  (bf16)
    A = A_cumsum.to(torch.float32)  # [b, h, c, s]
    # Compute differences: diff[i,j] = A[i] - A[j]
    # [b, h, c, s, 1] - [b, h, c, 1, s] -> [b, h, c, s, s]
    L = torch.exp(A[..., :, None] - A[..., None, :])
    # Zero out strictly upper triangle (j > i): keep i >= j
    tril_mask = torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool).tril()
    L = L * tril_mask

    # --- G[i,j] = sum_n C[i,n] * B[j,n], grouped (n_groups -> num_heads via repeat_interleave) ---
    # B, C: [batch, num_chunks, chunk_size, n_groups, state_size] (bf16)
    Bf = B.to(torch.float32)  # [b, c, s, g, n]
    Cf = C.to(torch.float32)
    # Expand groups to heads: repeat_interleave on dim=3, factor = NUM_HEADS//N_GROUPS = 4
    # [b, c, s, h, n]
    Bh = Bf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    Ch = Cf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)

    # Reshape for batched matmul over (state_size) contraction.
    # We want, per (b, chunk, head): G[i,j] = sum_n Ch[i,n] * Bh[j,n]
    # That's Ch @ Bh.T  -> [b*c*h, s_i, s_j]
    # Ch: [b, c, s, h, n] -> permute to [b, c, h, s, n] -> reshape [b*c*h, s, n]
    Ch = Ch.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    Bh = Bh.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    G = torch.bmm(Ch, Bh.transpose(1, 2))  # [b*c*h, s_i, s_j]

    # --- Apply L: M = G * L ---
    # L: [b, h, c, s, s] -> need [b*c*h, s, s] to match G
    # permute L to [b, c, h, s, s] then reshape
    L = L.permute(0, 2, 1, 3, 4).reshape(-1, CHUNK_SIZE, CHUNK_SIZE)
    M = G * L  # [b*c*h, s_i, s_j]

    # --- Y_diag[i] = sum_j M[i,j] * hidden_states[j] ---
    # hidden_states: [b, c, s, h, d] (bf16) -> [b, c, h, s, d] -> [b*c*h, s, d]
    Hf = hidden_states.to(torch.float32)
    Hf = Hf.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, hidden_states.shape[-1])
    Y = torch.bmm(M, Hf)  # [b*c*h, s_i, d]

    # Reshape back: [b, c, s, h, d]
    head_dim = hidden_states.shape[-1]
    Y = Y.reshape(batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE, head_dim)
    Y = Y.permute(0, 1, 3, 2, 4).contiguous()
    return Y.to(torch.bfloat16)
