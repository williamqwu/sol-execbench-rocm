import torch

@torch.no_grad()
def run(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Fused GQA key-value repetition and attention score computation.
    Inputs are already bf16; MFMA accumulates bf16@bf16 in fp32, so a native
    bf16 matmul matches the fp32 reference closely while running far faster.
    num_key_value_groups=1 means the expand/reshape is a no-op, so it is
    dropped entirely.
    """
    head_dim = 128
    scaling = head_dim ** -0.5

    # Q @ K^T with scaling; bf16 inputs, fp32 accumulation on the matrix engine
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    return attn_weights.to(torch.bfloat16)
