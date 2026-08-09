import torch
import triton
import triton.language as tl


@triton.jit
def _projection_kernel(
    a, b, c, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ko in range(0, k, BLOCK_K):
        av = tl.load(a + offs_m[:, None] * k + ko + offs_k[None, :],
                     mask=offs_m[:, None] < m, other=0.0)
        bv = tl.load(b + offs_n[None, :] * k + ko + offs_k[:, None],
                     mask=offs_n[None, :] < n, other=0.0)
        acc = tl.dot(av, bv, acc)
    tl.store(c + offs_m[:, None] * n + offs_n[None, :], acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Whisper decoder output projection: projects hidden states to vocabulary logits.
    
    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, d_model=1280)
        weight: Tensor of shape (vocab_size=51866, d_model=1280)
    
    Returns:
        logits: Tensor of shape (batch_size, seq_len, vocab_size=51866)
    """
    # Linear projection without bias: output = input @ weight.T
    # hidden_states: (batch_size, seq_len, d_model)
    # weight: (vocab_size, d_model)
    # logits: (batch_size, seq_len, vocab_size)
    m = hidden_states.numel() // hidden_states.shape[-1]
    n = weight.shape[0]
    k = weight.shape[1]
    out = torch.empty((*hidden_states.shape[:-1], n), device=hidden_states.device,
                      dtype=hidden_states.dtype)
    _projection_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 128))](
        hidden_states, weight, out, m=m, n=n, k=k,
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=32, num_warps=8,
    )
    return out
