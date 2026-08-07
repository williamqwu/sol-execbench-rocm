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
    head_dim = hidden_states.shape[-1]

    # L[i,j] = exp(sum_{k=j+1}^{i} A_cumsum[k]) = exp(S[i]-S[j]) for i>=j, else 0.
    # S = inclusive prefix sum of A_cumsum along chunk dim.
    A = A_cumsum.to(torch.float32)  # [b, h, c, s]
    S = torch.cumsum(A, dim=-1)
    diff = S[..., :, None] - S[..., None, :]  # [b, h, c, s, s]
    tril_mask = torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool).tril()
    # Set upper triangle to -inf so exp -> 0 (avoids inf*0 = NaN)
    diff = diff.masked_fill(~tril_mask, float("-inf"))
    L = torch.exp(diff)  # [b, h, c, s, s]

    # G[i,j] = sum_n C[i,n]*B[j,n], grouped (n_groups -> num_heads)
    Bf = B.to(torch.float32)
    Cf = C.to(torch.float32)
    Bh = Bf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [b, c, s, h, n]
    Ch = Cf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    Ch = Ch.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)  # [b*c*h, s, n]
    Bh = Bh.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    G = torch.bmm(Ch, Bh.transpose(1, 2))  # [b*c*h, s_i, s_j]

    # M = G * L
    L = L.permute(0, 2, 1, 3, 4).reshape(-1, CHUNK_SIZE, CHUNK_SIZE)  # [b*c*h, s, s]
    M = G * L

    # Y_diag[i] = sum_j M[i,j]*hidden_states[j]
    Hf = hidden_states.to(torch.float32)
    Hf = Hf.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, head_dim)  # [b*c*h, s, d]
    Y = torch.bmm(M, Hf)  # [b*c*h, s_i, d]

    Y = Y.reshape(batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE, head_dim)
    Y = Y.permute(0, 1, 3, 2, 4).contiguous()
    return Y.to(torch.bfloat16)
