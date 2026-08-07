import torch

def _run_impl(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    CHUNK_SIZE = 128
    NUM_HEADS = 32
    N_GROUPS = 8
    GROUP_REPEAT = NUM_HEADS // N_GROUPS  # 4

    batch_size, num_chunks, _, _, _ = hidden_states.shape
    state_size = B.shape[-1]
    head_dim = hidden_states.shape[-1]

    # L[i,j] = exp(sum_{k=j+1}^{i} A[k]) = exp(S[i]-S[j]), i>=j else 0
    A = A_cumsum.to(torch.float32)  # [b, h, c, s]
    S = torch.cumsum(A, dim=-1)
    diff = S[..., :, None] - S[..., None, :]  # [b, h, c, s, s]
    tril_mask = torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool).tril()
    diff = diff.masked_fill(~tril_mask, float("-inf"))
    L = torch.exp(diff)  # [b, h, c, s, s]

    # G per group: Gg[i,j,g] = sum_n C[i,g,n]*B[j,g,n]
    # B,C: [b, c, s, g, n] -> per-group bmm [b*c*g, s, n]
    Bf = B.to(torch.float32)
    Cf = C.to(torch.float32)
    Cg = Cf.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)  # [b*c*g, s, n]
    Bg = Bf.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    Gg = torch.bmm(Cg, Bg.transpose(1, 2))  # [b*c*g, s_i, s_j]
    # Expand groups -> heads: [b, c, g, s, s] -> repeat_interleave g->h -> [b, c, h, s, s]
    Gg = Gg.reshape(batch_size, num_chunks, N_GROUPS, CHUNK_SIZE, CHUNK_SIZE)
    Gh = Gg.repeat_interleave(GROUP_REPEAT, dim=2)  # [b, c, h, s, s]

    # M = G * L. L: [b, h, c, s, s] -> permute to [b, c, h, s, s]
    L = L.permute(0, 2, 1, 3, 4)  # [b, c, h, s, s]
    M = Gh * L  # [b, c, h, s, s]

    # Y[i] = sum_j M[i,j] * H[j]
    # M: [b, c, h, s_i, s_j], H: [b, c, s_j, h, d] -> need [b, c, h, s_j, d]
    Hf = hidden_states.to(torch.float32)
    Hf = Hf.permute(0, 1, 3, 2, 4)  # [b, c, h, s, d]
    # bmm over [b*c*h, s, s] x [b*c*h, s, d]
    M_b = M.reshape(-1, CHUNK_SIZE, CHUNK_SIZE)
    Hf_b = Hf.reshape(-1, CHUNK_SIZE, head_dim)
    Y = torch.bmm(M_b, Hf_b)  # [b*c*h, s_i, d]

    Y = Y.reshape(batch_size, num_chunks, NUM_HEADS, CHUNK_SIZE, head_dim)
    Y = Y.permute(0, 1, 3, 2, 4).contiguous()
    return Y.to(torch.bfloat16)


@torch.compile(dynamic=True)
def _compiled_run(hidden_states, A_cumsum, B, C):
    return _run_impl(hidden_states, A_cumsum, B, C)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
) -> torch.Tensor:
    return _compiled_run(hidden_states, A_cumsum, B, C)
