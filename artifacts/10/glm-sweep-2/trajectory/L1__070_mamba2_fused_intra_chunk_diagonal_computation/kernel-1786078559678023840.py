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

    # G per group via bmm in bf16 (fp32 accumulate)
    Cg = C.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)  # [b*c*g, s, n] bf16
    Bg = B.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    Gg = torch.bmm(Cg, Bg.transpose(1, 2))  # [b*c*g, s_i, s_j] bf16
    Gg = Gg.reshape(batch_size, num_chunks, N_GROUPS, CHUNK_SIZE, CHUNK_SIZE).float()
    Gh = Gg.repeat_interleave(GROUP_REPEAT, dim=2)  # [b, c, h, s, s] f32

    L = L.permute(0, 2, 1, 3, 4)  # [b, c, h, s, s]
    M = Gh * L  # [b, c, h, s, s]
    Mb = M.to(torch.bfloat16).reshape(-1, CHUNK_SIZE, CHUNK_SIZE)

    Hf = hidden_states.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, head_dim)  # bf16
    Y = torch.bmm(Mb, Hf)  # [b*c*h, s_i, d] bf16

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
