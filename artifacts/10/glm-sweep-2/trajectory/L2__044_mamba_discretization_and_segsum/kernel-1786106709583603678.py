import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor
):
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
    n_groups = B.shape[3]
    device = hidden_states.device

    # 1. Discretize A parameter
    A = -torch.exp(A_log.float())  # [num_heads]
    dt_soft = F.softplus(dt.float())  # [batch, chunks, chunk_size, num_heads]
    dt_permuted = dt_soft.permute(0, 3, 1, 2)  # [batch, num_heads, chunks, chunk_size]
    A_discrete = A[None, :, None, None] * dt_permuted  # [batch, num_heads, chunks, chunk_size]

    # 2. Cumulative sum of A (inclusive)
    A_cumsum = torch.cumsum(A_discrete, dim=-1)  # [batch, num_heads, chunks, chunk_size]

    # 3. L matrix via outer difference of A_cumsum.
    #    seg[i,j] = A_cumsum[i] - A_cumsum[j] for i >= j, -inf for i < j.
    #    L = exp(seg): lower-tri (incl. diag) = exp(diff), upper-tri = 0.
    diff = A_cumsum.unsqueeze(-1) - A_cumsum.unsqueeze(-2)  # [b,h,c,s,s]
    mask = torch.tril(
        torch.ones(chunk_size, chunk_size, device=device, dtype=torch.bool),
        diagonal=0,
    )
    diff = diff.masked_fill(~mask, float('-inf'))
    L = torch.exp(diff)  # [b,h,c,s,s], upper-tri is 0
    L_bf16 = L.to(torch.bfloat16)

    # 4-5. G matrix. With n_groups=1 all heads share the same B,C, so the
    #      contraction G[i,j,h] = sum_d C[i,d]*B[j,d] is independent of h.
    repeats = num_heads // n_groups
    if repeats == 1:
        Cf = C.float().squeeze(3)  # [b,c,s,d]
        Bf = B.float().squeeze(3)
        G1 = torch.einsum('bcsd,bctd->bcst', Cf, Bf)  # [b,c,s,s] fp32
        G_bf16 = G1.to(torch.bfloat16).unsqueeze(-1).expand(
            batch_size, num_chunks, chunk_size, chunk_size, num_heads
        )
    else:
        Be = B.repeat_interleave(repeats, dim=3)
        Ce = C.repeat_interleave(repeats, dim=3)
        G1_full = torch.einsum('bcihd,bcjhd->bcijh', Ce.float(), Be.float())
        G_bf16 = G1_full.to(torch.bfloat16)

    # 6. M matrix (masked attention weights): M = G * L
    #    L_permuted: [b,h,c,s,s] -> [b,c,s,s,h]
    L_permuted = L.permute(0, 2, 3, 4, 1)  # [b,c,s,s,h] fp32
    if repeats == 1:
        # G1 [b,c,s,s] -> [b,c,s,s,1] broadcast against L_permuted [b,c,s,s,h]
        M = G1.unsqueeze(-1) * L_permuted
    else:
        M = G1_full * L_permuted
    M_bf16 = M.to(torch.bfloat16)

    # 7. decay factors for state updates
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)

    return (
        L_bf16,
        G_bf16,
        M_bf16,
        A_cumsum,
        decay_states,
    )
