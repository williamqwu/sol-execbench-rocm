import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def _triton_gemm(a, b_t):
    """a: [M,K] bf16, b_t: [K,N] bf16 -> [M,N] fp32."""
    M, K = a.shape
    K2, N = b_t.shape
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, b_t, c, M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    num_experts = 256
    top_k = 8
    n_group = 8
    topk_group = 4
    experts_per_group = num_experts // n_group  # 32

    num_tokens = hidden_states.shape[0]

    # bf16 GEMM with fp32 accumulation & fp32 output via Triton
    logits = _triton_gemm(hidden_states, weight.t().contiguous())

    scores = torch.sigmoid(logits)  # [num_tokens, 256]
    scores_for_routing = scores + expert_bias.to(torch.float32)

    group_scores_reshaped = scores_for_routing.view(num_tokens, n_group, experts_per_group)
    top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=-1)

    _, group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, n_group, experts_per_group)
        .reshape(num_tokens, num_experts)
    )

    neg_inf = torch.finfo(torch.float32).min
    masked_scores = scores_for_routing.masked_fill(score_mask == 0, neg_inf)

    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)

    selected_scores = torch.gather(scores, dim=1, index=topk_idx)

    topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    return topk_idx, topk_weight
