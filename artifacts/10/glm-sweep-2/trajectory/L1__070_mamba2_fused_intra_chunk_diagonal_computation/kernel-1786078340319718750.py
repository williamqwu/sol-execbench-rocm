import torch

_RUN_COMPILED = None

def _run_impl(
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

    A = A_cumsum.to(torch.float32)
    S = torch.cumsum(A, dim=-1)
    diff = S[..., :, None] - S[..., None, :]
    tril_mask = torch.ones(CHUNK_SIZE, CHUNK_SIZE, device=A_cumsum.device, dtype=torch.bool).tril()
    diff = diff.masked_fill(~tril_mask, float("-inf"))
    L = torch.exp(diff)

    Bf = B.to(torch.float32)
    Cf = C.to(torch.float32)
    Bh = Bf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    Ch = Cf.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
    Ch = Ch.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    Bh = Bh.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, state_size)
    G = torch.bmm(Ch, Bh.transpose(1, 2))

    L = L.permute(0, 2, 1, 3, 4).reshape(-1, CHUNK_SIZE, CHUNK_SIZE)
    M = G * L

    Hf = hidden_states.to(torch.float32)
    Hf = Hf.permute(0, 1, 3, 2, 4).reshape(-1, CHUNK_SIZE, head_dim)
    Y = torch.bmm(M, Hf)

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
    global _RUN_COMPILED
    try:
        if _RUN_COMPILED is None:
            _RUN_COMPILED = _compiled_run
        return _RUN_COMPILED(hidden_states, A_cumsum, B, C)
    except Exception:
        return _run_impl(hidden_states, A_cumsum, B, C)
