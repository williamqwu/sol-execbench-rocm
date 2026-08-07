import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    in_proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    D: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    time_step_limit_min: float,
    time_step_limit_max: float,
    layer_norm_epsilon: float,
):
    # Constants
    hidden_size = 8192
    num_heads = 256
    head_dim = 64
    intermediate_size = 16384
    ssm_state_size = 256
    conv_kernel_size = 4
    n_groups = 8
    chunk_size = 128
    groups_time_state_size = n_groups * ssm_state_size
    conv_dim = intermediate_size + 2 * groups_time_state_size
    heads_per_group = num_heads // n_groups

    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    # 1. Input projection
    projected = torch.matmul(hidden_states, in_proj_weight.t())

    # Split projections
    gate_start = projected.shape[-1] - intermediate_size - conv_dim - num_heads
    gate = projected[..., gate_start:gate_start + intermediate_size]
    hidden_B_C = projected[..., gate_start + intermediate_size:gate_start + intermediate_size + conv_dim]
    dt = projected[..., -num_heads:]

    # 2. Causal Convolution
    hidden_B_C_t = hidden_B_C.transpose(1, 2)
    conv_out = F.conv1d(
        hidden_B_C_t,
        conv1d_weight,
        conv1d_bias,
        padding=conv_kernel_size - 1,
        groups=conv_dim
    )[..., :seq_len]
    hidden_B_C = (conv_out * torch.sigmoid(conv_out)).transpose(1, 2)  # silu

    # Split into hidden_states, B, C
    hidden_states_ssm = hidden_B_C[..., :intermediate_size]
    B = hidden_B_C[..., intermediate_size:intermediate_size + groups_time_state_size]
    C = hidden_B_C[..., intermediate_size + groups_time_state_size:]

    # 3. Selective State Space Model with Chunking

    # Discretize time step
    dt = F.softplus(dt + dt_bias)
    dt = torch.clamp(dt, time_step_limit_min, time_step_limit_max)

    # Reshape for SSM computation (keep B, C at GROUP resolution; do not repeat)
    hidden_states_ssm = hidden_states_ssm.view(batch_size, seq_len, num_heads, head_dim).float()
    B = B.view(batch_size, seq_len, n_groups, ssm_state_size).float()
    C = C.view(batch_size, seq_len, n_groups, ssm_state_size).float()

    # Pad to chunk size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    def pad_tensor_4d(x, pad_size):
        if pad_size > 0:
            return F.pad(x, (0, 0, 0, 0, 0, pad_size, 0, 0))
        return x

    def pad_tensor_3d(x, pad_size):
        if pad_size > 0:
            return F.pad(x, (0, 0, 0, pad_size, 0, 0))
        return x

    # D residual (skip connection). head i -> group i % n_groups (repeat tiling).
    D_residual = D.float()[..., None] * pad_tensor_4d(hidden_states_ssm, pad_size)

    # Discretize x and A
    hidden_states_ssm = hidden_states_ssm * dt[..., None]
    A = -torch.exp(A_log.float()) * dt

    # Pad tensors
    hidden_states_ssm_padded = pad_tensor_4d(hidden_states_ssm, pad_size)
    A_padded = pad_tensor_3d(A, pad_size)
    B_padded = pad_tensor_4d(B, pad_size)
    C_padded = pad_tensor_4d(C, pad_size)

    padded_seq_len = hidden_states_ssm_padded.shape[1]
    num_chunks = padded_seq_len // chunk_size

    # Reshape into chunks
    # x: (batch, chunks, cs, num_heads, head_dim)
    x_chunked = hidden_states_ssm_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)
    A_chunked = A_padded.reshape(batch_size, num_chunks, chunk_size, num_heads)
    # B_g, C_g: (batch, chunks, cs, n_groups, state)
    B_chunked = B_padded.reshape(batch_size, num_chunks, chunk_size, n_groups, ssm_state_size)
    C_chunked = C_padded.reshape(batch_size, num_chunks, chunk_size, n_groups, ssm_state_size)

    # Permute A for computation: (batch, num_heads, num_chunks, chunk_size)
    A_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_perm, dim=-1)  # (batch, heads, chunks, cs)

    # 3a. Intra-chunk computation (diagonal blocks)
    # Segment sum via cumsum-difference (avoids materializing cs x cs mask via expand).
    # segsum[i, j] = sum_{k=i}^{j-1} x[k] for i < j, 0 for i == j, -inf for i > j.
    def segment_sum_fast(x):
        # segsum[i, j] = sum_{k=j+1}^{i} x[k] for i>j; 0 for i==j; -inf for i<j
        csm = x.size(-1)
        cu = torch.cumsum(x, dim=-1)  # inclusive prefix sum
        segsum = cu.unsqueeze(-1) - cu.unsqueeze(-2)  # [i, j] = cu[i] - cu[j]
        i_idx = torch.arange(csm, device=device, dtype=torch.long).view(-1, 1)
        j_idx = torch.arange(csm, device=device, dtype=torch.long).view(1, -1)
        mask = i_idx > j_idx
        segsum = torch.where(mask, segsum, torch.where(i_idx == j_idx, torch.zeros_like(segsum), torch.full_like(segsum, -torch.inf)))
        return segsum

    L = torch.exp(segment_sum_fast(A_perm))  # (batch, heads, chunks, cs, cs)

    # Attention-like weights at GROUP resolution: G_g[l, s, g] = sum_k C_g[l, g, k] B_g[s, g, k]
    # G_g: (batch, chunks, cs_l, cs_s, n_groups)
    G_group = torch.einsum('bclgn,bcsgn->bclsg', C_chunked, B_chunked)

    # L: (batch, heads, chunks, cs_l, cs_s). head i -> group i % n_groups.
    # Permute to (batch, chunks, cs_l, cs_s, heads) then group heads.
    L_perm = L.permute(0, 2, 3, 4, 1)  # (batch, chunks, cs_l, cs_s, heads)
    # Group heads: heads dim -> (heads_per_group, n_groups) with head = j * n_groups + g
    L_grouped = L_perm.reshape(batch_size, num_chunks, chunk_size, chunk_size, heads_per_group, n_groups)
    # M_group: (batch, chunks, cs_l, cs_s, heads_per_group, n_groups)
    M_group = G_group.unsqueeze(-2) * L_grouped

    # Y_diag: y[l, h, d] = sum_s M[l, s, h] x[s, h, d], grouped over heads.
    # x_grouped: (batch, chunks, cs, heads_per_group, n_groups, head_dim)
    x_grouped = x_chunked.reshape(batch_size, num_chunks, chunk_size, heads_per_group, n_groups, head_dim)
    # Y_diag_group: (batch, chunks, cs_l, heads_per_group, n_groups, head_dim)
    Y_diag_group = torch.einsum('bclsjg,bcsjgd->bcljgd', M_group, x_grouped)
    Y_diag = Y_diag_group.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)

    # 3b. Compute states at chunk boundaries
    # decay_states: exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (batch, heads, chunks, cs)
    decay_states_perm = decay_states.permute(0, 2, 3, 1)  # (batch, chunks, cs, heads)

    # states = einsum(x_chunked, B_decay). B is at group resolution, so expand to heads.
    # B_decay: (batch, chunks, cs, hpg, n_groups, state) via group->head mapping (head i -> group i%n_groups)
    decay_states_g = decay_states_perm.reshape(batch_size, num_chunks, chunk_size, heads_per_group, n_groups)
    B_decay_group = B_chunked.unsqueeze(3) * decay_states_g.unsqueeze(-1)  # (batch, chunks, cs, hpg, n_groups, state)
    x_g = x_grouped  # (batch, chunks, cs, hpg, n_groups, head_dim)
    # states_group: (batch, chunks, hpg, n_groups, head_dim, state) = sum_s x[s,j,g,d] * B_decay[s,j,g,n]
    # Use bmm over the cs contraction for efficient GEMM dispatch:
    # batch = (b, chunks, hpg, n_groups); M=head_dim, K=cs, N=state
    _Bm = batch_size * num_chunks * heads_per_group * n_groups
    _x_bmm = x_g.permute(0, 1, 3, 4, 2, 5).reshape(_Bm, chunk_size, head_dim).transpose(1, 2)  # (B, head_dim, cs)
    _Bd_bmm = B_decay_group.permute(0, 1, 3, 4, 2, 5).reshape(_Bm, chunk_size, ssm_state_size)  # (B, cs, state)
    states_group = torch.bmm(_x_bmm, _Bd_bmm).reshape(batch_size, num_chunks, heads_per_group, n_groups, head_dim, ssm_state_size)
    # states: (batch, chunks, heads, head_dim, state) with head = j*n_groups + g
    states = states_group.reshape(batch_size, num_chunks, num_heads, head_dim, ssm_state_size)

    # 3c. Inter-chunk recurrence
    previous_states = torch.zeros_like(states[:, :1])  # (batch, 1, heads, head_dim, state)
    states_with_prev = torch.cat([previous_states, states], dim=1)  # (batch, chunks+1, heads, head_dim, state)

    # decay_chunk computation
    A_chunk_ends = A_cumsum[:, :, :, -1]  # (batch, heads, chunks)
    A_chunk_ends_padded = F.pad(A_chunk_ends, (1, 0))  # (batch, heads, chunks+1)

    decay_chunk = torch.exp(segment_sum_fast(A_chunk_ends_padded))  # (batch, heads, chunks+1, chunks+1)

    # new_states = einsum(decay_chunk, states_with_prev)
    states_with_prev_perm = states_with_prev.permute(0, 2, 1, 3, 4)  # (batch, heads, chunks+1, head_dim, state)
    new_states = torch.einsum('bhcd,bhdin->bhcin', decay_chunk, states_with_prev_perm)
    new_states = new_states[:, :, :-1, :, :]  # (batch, heads, chunks, head_dim, state)
    states_final = new_states.permute(0, 2, 1, 3, 4)  # (batch, chunks, heads, head_dim, state)

    # 3d. State to output (off-diagonal blocks)
    state_decay_out = torch.exp(A_cumsum)  # (batch, heads, chunks, cs)
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)  # (batch, chunks, cs, heads)

    # Y_off = einsum(C_chunked, states_final, state_decay_out_perm)
    # C at group resolution: expand to heads for the contraction with states_final (per-head).
    # states_final: (batch, chunks, heads, head_dim, state); C_group: (batch, chunks, cs, n_groups, state)
    # head i -> group i%n_groups. Expand C over heads_per_group.
    C_grouped = C_chunked  # (batch, chunks, cs, n_groups, state)
    # Build C per-head by tiling: (batch, chunks, cs, hpg, n_groups, state)
    C_per_head = C_grouped.unsqueeze(3).expand(batch_size, num_chunks, chunk_size, heads_per_group, n_groups, ssm_state_size)
    states_final_grouped = states_final.reshape(batch_size, num_chunks, heads_per_group, n_groups, head_dim, ssm_state_size)
    state_decay_g = state_decay_out_perm.reshape(batch_size, num_chunks, chunk_size, heads_per_group, n_groups)
    # Y_off_group: (batch, chunks, cs, hpg, n_groups, head_dim) = sum_n C[s,j,g,n] * states_final[j,g,d,n] * decay[s,j,g]
    Y_off_group = torch.einsum('bcsjgn,bcjgdn,bcsjg->bcsjgd', C_per_head, states_final_grouped, state_decay_g)
    Y_off = Y_off_group.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)

    # Combine intra and inter chunk outputs
    y = Y_diag + Y_off  # (batch, chunks, cs, heads, head_dim)
    y = y.reshape(batch_size, padded_seq_len, num_heads, head_dim)

    # Add skip connection
    y = y + D_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len]

    # Reshape and convert back to original dtype
    y = y.reshape(batch_size, seq_len, intermediate_size).to(dtype)

    # 4. Gated normalization
    group_size = intermediate_size // n_groups
    y_grouped = y.view(batch_size, seq_len, n_groups, group_size)
    variance = y_grouped.float().pow(2).mean(dim=-1, keepdim=True)
    y_normed = y_grouped * torch.rsqrt(variance + layer_norm_epsilon)
    y_normed = y_normed.view(batch_size, seq_len, intermediate_size).to(dtype)
    y_normed = y_normed * norm_weight
    y = y_normed * (gate * torch.sigmoid(gate))  # silu gate

    # 5. Output projection
    output = torch.matmul(y, out_proj_weight.t())

    return output
