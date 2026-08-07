import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _grouped_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    tile_expert_ptr,  # [num_m_tiles] expert id per m-tile
    K_in, K_out,
    total_tokens,
    stride_a_m, stride_a_k,
    stride_b_e, stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(K_out, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    expert = tl.load(tile_expert_ptr + pid_m)
    m_start = pid_m * BLOCK_M

    rows = m_start + tl.arange(0, BLOCK_M)
    row_mask = rows < total_tokens
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < K_out

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K_in, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_in
        a = tl.load(A_ptr + rows[:, None] * stride_a_m + k_offs[None, :] * stride_a_k,
                    mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(B_ptr + expert * stride_b_e + k_offs[:, None] * stride_b_k + cols[None, :] * stride_b_n,
                    mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc += tl.dot(a, b)

    tl.store(C_ptr + rows[:, None] * stride_c_m + cols[None, :] * stride_c_n,
             acc.to(C_ptr.dtype.element_ty),
             mask=row_mask[:, None] & col_mask[None, :])


def grouped_gemm(x_sorted, weights, offsets_cpu, total_tokens, K_in, K_out,
                 BLOCK_M=64, BLOCK_N=128, BLOCK_K=64):
    # x_sorted: [total_tokens, K_in], row-major (stride_a_k=1)
    # weights: [E, K_out, K_in] (F.linear: y = x @ W^T)
    # offsets_cpu: [E+1] on CPU, int64
    E = weights.shape[0]
    num_m_tiles = (total_tokens + BLOCK_M - 1) // BLOCK_M
    # Build tile -> expert mapping on CPU.
    tile_expert = torch.empty(num_m_tiles, dtype=torch.int32)
    for t in range(num_m_tiles):
        m_start = t * BLOCK_M
        # find expert e such that offsets_cpu[e] <= m_start < offsets_cpu[e+1]
        # offsets is non-decreasing; use searchsorted
        e = torch.searchsorted(offsets_cpu, m_start, right=True).item() - 1
        if e < 0:
            e = 0
        tile_expert[t] = e
    tile_expert = tile_expert.to(x_sorted.device, non_blocking=True)

    C = torch.empty(total_tokens, K_out, device=x_sorted.device, dtype=x_sorted.dtype)
    num_n_tiles = (K_out + BLOCK_N - 1) // BLOCK_N
    grid = (num_m_tiles * num_n_tiles,)
    _grouped_gemm_kernel[grid](
        x_sorted, weights, C, tile_expert, K_in, K_out, total_tokens,
        x_sorted.stride(0), x_sorted.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


@torch.no_grad()
def run(
    hidden_states, topk_idx, topk_weight,
    expert_gate_projs, expert_up_projs, expert_down_projs,
):
    batch_seq_len, hidden_size = hidden_states.shape
    num_experts = expert_gate_projs.shape[0]
    moe_intermediate_size = expert_gate_projs.shape[1]
    K = topk_idx.shape[1]
    N = batch_seq_len
    M = moe_intermediate_size
    H = hidden_size
    NK = N * K

    hidden_states_repeated = hidden_states.repeat_interleave(K, dim=0)  # [NK, H]
    flat_topk_idx = topk_idx.reshape(-1)  # [NK]

    sorted_expert_idx = torch.argsort(flat_topk_idx, stable=True)
    sorted_experts = flat_topk_idx[sorted_expert_idx]
    x_sorted = hidden_states_repeated.index_select(0, sorted_expert_idx)  # [NK, H]

    counts = torch.bincount(sorted_experts, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, device=hidden_states.device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    offsets_cpu = offsets.cpu()  # one sync

    gate_out = grouped_gemm(x_sorted, expert_gate_projs, offsets_cpu, NK, H, M)  # [NK, M]
    up_out = grouped_gemm(x_sorted, expert_up_projs, offsets_cpu, NK, H, M)      # [NK, M]
    inter = F.silu(gate_out) * up_out                                             # [NK, M]
    y_sorted = grouped_gemm(inter, expert_down_projs, offsets_cpu, NK, M, H)     # [NK, H]

    y = torch.empty(NK, H, device=hidden_states.device, dtype=hidden_states.dtype)
    y.index_copy_(0, sorted_expert_idx, y_sorted)

    y = y.view(N, K, H)
    output = (y * topk_weight.unsqueeze(-1)).sum(dim=1)
    return output
