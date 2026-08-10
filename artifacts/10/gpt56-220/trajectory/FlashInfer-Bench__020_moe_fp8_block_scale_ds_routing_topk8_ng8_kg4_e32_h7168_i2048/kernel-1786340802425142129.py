import torch


@torch.compile(fullgraph=True)
def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor):
    return (weight.float().view(scale.shape[0], 128, scale.shape[1], 128) *
            scale[:, None, :, None]).reshape(weight.shape)


@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    """
    • FP8 block-scale dequantization: float ≈ fp8 * scale
    • DeepSeek-V3 no-aux routing:
        s = sigmoid(logits)
        s_with_bias = s + bias
        group by n_group=8; per group take top-2 sum → pick topk_group=4 groups
        on the kept groups, take global top_k=8 experts
        combine with weights derived from s (without bias), normalized and
        scaled by routed_scaling_factor
    • Local computation:
        only experts in [local_expert_offset, local_expert_offset + E_local) are
        computed on this rank (GEMM1 → SwiGLU → GEMM2), then per-token weighted
        accumulation.
    """

    # Fixed DeepSeek-V3/R1 geometry
    H = 7168
    I = 2048
    E_local = gemm1_weights.shape[0]
    
    BLOCK = 128
    E_global = routing_logits.shape[1]
    T = routing_logits.shape[0]

    assert H == 7168, "hidden_size must be 7168" 
    assert I == 2048, "intermediate_size must be 2048"
    assert E_global == 256, "num_experts must be 256"
    assert E_local == 32, "num_local_experts must be 32"

    # Routing constants
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4

    # Block counts
    num_hidden_blocks = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK    # 16
    num_gemm1_out_blocks = (2 * I) // BLOCK # 32

    # Shape checks
    assert hidden_states.shape == (T, H)
    assert hidden_states_scale.shape == (num_hidden_blocks, T)
    assert gemm1_weights.shape == (E_local, 2 * I, H)
    assert gemm1_weights_scale.shape == (E_local, num_gemm1_out_blocks, num_hidden_blocks)
    assert gemm2_weights.shape == (E_local, H, I)
    assert gemm2_weights_scale.shape == (E_local, num_hidden_blocks, num_intermediate_blocks)
    assert routing_bias.shape[-1] == E_global

    device = hidden_states.device

    # 1) FP8 block-scale dequantization of the token activations
    A_scale_TH = hidden_states_scale.permute(1, 0)
    A = hidden_states.to(torch.float32).view(T, num_hidden_blocks, BLOCK)
    A.mul_(A_scale_TH.unsqueeze(-1))
    A = A.reshape(T, H)

    # 2) No-aux routing
    logits = routing_logits.to(torch.float32)                      # [T, E_global]
    bias = routing_bias.to(torch.float32).reshape(-1)              # [E_global]

    # Sigmoid
    s = 1.0 / (1.0 + torch.exp(-logits))                       # [T, E]
    s_with_bias = s + bias                                     # [T, E] (broadcast)

    # Grouping
    group_size = E_global // N_GROUP # 32
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)    # [T, 8, 32]

    # Group scores = sum of top-2 values within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  # [T, 8, 2]
    group_scores = top2_vals.sum(dim=2)                        # [T, 8]

    # Select topk_group groups → group mask
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)  # [T, 4]
    group_mask = torch.zeros_like(group_scores)                # [T, 8]
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)  # [T, E]

    # Global top-k (within kept groups), based on s_with_bias
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)                  # [T, E]
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)  # [T, 8]

    # Combination weights: use s (without bias) for normalization
    weights = torch.zeros_like(s)
    weights.scatter_(1, topk_idx, s.gather(1, topk_idx))
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights.div_(weights_sum).mul_(routed_scaling_factor)

    # 3) Local expert compute and accumulation
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    local_start = int(local_expert_offset)

    # Discover active local experts with a single device synchronization rather
    # than synchronizing once for every one of the 32 `.any()` checks.
    active_ges = torch.unique(topk_idx).tolist()
    active_les = [ge - local_start for ge in active_ges
                  if local_start <= ge < local_start + E_local]

    # For each active local expert: GEMM1→SwiGLU→GEMM2, then accumulate.
    for le in active_les:
        ge = local_start + le
        token_idx = torch.nonzero(topk_idx == ge, as_tuple=False)[:, 0]
        Tk = token_idx.numel()

        # Gather inputs and weights for this expert
        A_e = A.index_select(0, token_idx)
        W13_e = _dequant_weight(gemm1_weights[le], gemm1_weights_scale[le])
        G1 = A_e.matmul(W13_e.t())

        # SwiGLU: split and apply silu(x) = x / (1 + exp(-x))
        X1 = G1[:, :I]                                         # [Tk, I]
        X2 = G1[:, I:]                                         # [Tk, I]
        C = torch.nn.functional.silu(X2) * X1                  # [Tk, I]

        # Materialize the second weight immediately before it is consumed,
        # keeping the two large dequantized matrices from overlapping.
        W2_e = _dequant_weight(gemm2_weights[le], gemm2_weights_scale[le])

        # GEMM2: [Tk, I] @ [I, H] = [Tk, H]
        O = C.matmul(W2_e.t())                                 # [Tk, H]

        # Accumulate with per-token routing weights for this expert
        w_tok = weights.index_select(0, token_idx)[:, ge]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)
