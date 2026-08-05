import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_kernel(
    X_ptr, GW_ptr, UW_ptr, GY_ptr, UY_ptr,
    token_ptr,
    expert_start_ptr, expert_count_ptr,
    K_DIM, N_DIM,
    stride_x_tok, stride_x_dim,
    stride_gw_exp, stride_gw_out, stride_gw_in,
    stride_uw_exp, stride_uw_out, stride_uw_in,
    stride_y_tok, stride_y_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    start = tl.load(expert_start_ptr + expert_id)
    count = tl.load(expert_count_ptr + expert_id)
    if count == 0:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < count
    tok_ids = tl.load(token_ptr + start + m_offs, mask=m_mask, other=0)

    k_offs = tl.arange(0, BLOCK_K)
    x_ptrs = X_ptr + tok_ids[:, None] * stride_x_tok + k_offs[None, :] * stride_x_dim

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_DIM
    gw_ptrs = GW_ptr + expert_id * stride_gw_exp + n_offs[:, None] * stride_gw_out + k_offs[None, :] * stride_gw_in
    uw_ptrs = UW_ptr + expert_id * stride_uw_exp + n_offs[:, None] * stride_uw_out + k_offs[None, :] * stride_uw_in

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K_DIM, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        gw = tl.load(gw_ptrs)
        uw = tl.load(uw_ptrs)
        acc_g += tl.dot(x, gw.T)
        acc_u += tl.dot(x, uw.T)
        x_ptrs += BLOCK_K * stride_x_dim
        gw_ptrs += BLOCK_K * stride_gw_in
        uw_ptrs += BLOCK_K * stride_uw_in

    gy_ptrs = GY_ptr + (start + m_offs)[:, None] * stride_y_tok + n_offs[None, :] * stride_y_dim
    uy_ptrs = UY_ptr + (start + m_offs)[:, None] * stride_y_tok + n_offs[None, :] * stride_y_dim
    tl.store(gy_ptrs, acc_g.to(GY_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    tl.store(uy_ptrs, acc_u.to(UY_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _grouped_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    expert_start_ptr, expert_count_ptr,
    K_DIM, N_DIM,
    stride_x_tok, stride_x_dim,
    stride_w_exp, stride_w_out, stride_w_in,
    stride_y_tok, stride_y_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert_id = tl.program_id(2)

    start = tl.load(expert_start_ptr + expert_id)
    count = tl.load(expert_count_ptr + expert_id)
    if count == 0:
        return

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < count
    row_base = (start + m_offs) * stride_x_tok

    k_offs = tl.arange(0, BLOCK_K)
    x_ptrs = X_ptr + row_base[:, None] + k_offs[None, :] * stride_x_dim

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N_DIM
    w_ptrs = W_ptr + expert_id * stride_w_exp + n_offs[:, None] * stride_w_out + k_offs[None, :] * stride_w_in

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K_DIM, BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w.T)
        x_ptrs += BLOCK_K * stride_x_dim
        w_ptrs += BLOCK_K * stride_w_in

    y_ptrs = Y_ptr + (start + m_offs)[:, None] * stride_y_tok + n_offs[None, :] * stride_y_dim
    tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def _fused_gate_up(x, gw, uw, expert_starts, expert_counts, n_experts, sorted_tokens, gy, uy, bm, bk, bn):
    K_DIM = x.shape[1]
    N_DIM = gw.shape[1]
    max_count = expert_counts.max().item()
    grid = (triton.cdiv(max_count, bm), triton.cdiv(N_DIM, bn), n_experts)
    _fused_gate_up_kernel[grid](
        x, gw, uw, gy, uy,
        sorted_tokens,
        expert_starts, expert_counts,
        K_DIM, N_DIM,
        x.stride(0), x.stride(1),
        gw.stride(0), gw.stride(1), gw.stride(2),
        uw.stride(0), uw.stride(1), uw.stride(2),
        gy.stride(0), gy.stride(1),
        BLOCK_M=bm, BLOCK_K=bk, BLOCK_N=bn,
        num_warps=8, num_stages=2,
    )


def _grouped_gemm(x, w, expert_starts, expert_counts, n_experts, out, bm, bk, bn):
    K_DIM = x.shape[1]
    N_DIM = w.shape[1]
    max_count = expert_counts.max().item()
    grid = (triton.cdiv(max_count, bm), triton.cdiv(N_DIM, bn), n_experts)
    _grouped_gemm_kernel[grid](
        x, w, out,
        expert_starts, expert_counts,
        K_DIM, N_DIM,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_K=bk, BLOCK_N=bn,
        num_warps=8, num_stages=2,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_proj_weights.shape[0]
    num_experts_per_tok = topk_indices.shape[1]
    device = hidden_states.device

    total_pairs = num_tokens * num_experts_per_tok

    token_idx_flat = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_experts_per_tok).reshape(-1)
    weight_idx_flat = torch.arange(num_experts_per_tok, device=device).unsqueeze(0).expand(num_tokens, -1).reshape(-1)
    expert_idx_flat = topk_indices.reshape(-1)

    order = torch.argsort(expert_idx_flat, stable=True)
    sorted_tokens = token_idx_flat[order].to(torch.int32)
    sorted_weights_idx = weight_idx_flat[order].to(torch.int32)

    expert_counts = torch.bincount(expert_idx_flat, minlength=n_routed_experts).to(torch.int32)
    expert_starts = torch.cumsum(expert_counts, dim=0, dtype=torch.int32)
    expert_starts = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), expert_starts[:-1]])

    sorted_route_weights = topk_weights[token_idx_flat[order], sorted_weights_idx].to(torch.float32)

    inter_size = gate_proj_weights.shape[1]

    # Tuned block sizes (BM=128 universally best on MI350X)
    f_bm, f_bk, f_bn = 128, 128, 128
    d_bm, d_bk, d_bn = 128, 128, 128

    gate_out = torch.empty(total_pairs, inter_size, dtype=torch.bfloat16, device=device)
    up_out = torch.empty(total_pairs, inter_size, dtype=torch.bfloat16, device=device)
    _fused_gate_up(hidden_states, gate_proj_weights, up_proj_weights, expert_starts, expert_counts, n_routed_experts, sorted_tokens, gate_out, up_out, f_bm, f_bk, f_bn)

    intermediate = (F.silu(gate_out.float()) * up_out.float()).to(torch.bfloat16)

    expert_out = torch.empty(total_pairs, hidden_size, dtype=torch.bfloat16, device=device)
    _grouped_gemm(intermediate, down_proj_weights, expert_starts, expert_counts, n_routed_experts, expert_out, d_bm, d_bk, d_bn)

    weighted = expert_out.to(torch.float32) * sorted_route_weights.unsqueeze(-1)
    final_hidden_states = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)
    final_hidden_states.index_add_(0, token_idx_flat[order], weighted)

    return final_hidden_states.to(torch.bfloat16)
