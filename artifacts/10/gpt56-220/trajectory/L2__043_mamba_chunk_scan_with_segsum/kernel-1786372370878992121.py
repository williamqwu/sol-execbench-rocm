import torch
import torch.nn.functional as F


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """Compute segment sum (cumulative sum with lower triangular masking)."""
    chunk_size = input_tensor.size(-1)
    prefix = torch.cumsum(input_tensor, dim=-1)
    tensor_segsum = prefix[..., :, None] - prefix[..., None, :]
    indices = torch.arange(chunk_size, device=input_tensor.device)
    mask = indices[:, None] >= indices[None, :]
    tensor_segsum = tensor_segsum.masked_fill(~mask, float('-inf'))
    return tensor_segsum


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pad tensor on seq_len dimension."""
    if pad_size == 0:
        return input_tensor
    if len(input_tensor.shape) == 4:
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0)
    else:
        pad_shape = (0, 0, 0, pad_size, 0, 0)
    return F.pad(input_tensor, pad_shape, mode='constant', value=0)


def reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """Reshape tensor into chunks after padding."""
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_states: torch.Tensor,
):
    """Mamba-2 chunk-based parallel scan with segment sum."""
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    state_size = 256
    n_groups = 1
    chunk_size = 256
    
    # Compute padding size to make seq_len multiple of chunk_size
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    
    # Convert to float32 for numerical stability
    hidden_states_f = hidden_states.to(torch.float32)
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C.to(torch.float32)
    D_f = D.to(torch.float32)
    initial_states_f = initial_states.to(torch.float32)
    
    # Apply D residual (before chunking)
    # Reshape into chunks
    hidden_states_chunked = reshape_into_chunks(hidden_states_f, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    A_transposed = A_f.transpose(1, 2)  # [batch, seq_len, num_heads]
    A_chunked = reshape_into_chunks(A_transposed, pad_size, chunk_size)
    # [batch, num_chunks, chunk_size, num_heads]
    
    B_chunked = reshape_into_chunks(B_f, pad_size, chunk_size).squeeze(3)
    C_chunked = reshape_into_chunks(C_f, pad_size, chunk_size).squeeze(3)
    
    num_chunks = A_chunked.shape[1]
    
    # Permute A for cumsum: [batch, num_chunks, chunk_size, num_heads] -> [batch, num_heads, num_chunks, chunk_size]
    A_chunked_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_chunked_perm, dim=-1)
    # [batch, num_heads, num_chunks, chunk_size]
    
    # 1. Compute intra-chunk outputs (diagonal blocks)
    # L matrix: exponential of segment sum of A
    L = torch.exp(segment_sum(A_chunked_perm))  # [batch, num_heads, num_chunks, chunk_size, chunk_size]
    
    # Compute G: contraction of C and B over state_size
    # C_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # B_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # G[b, nc, i, j, h] = sum_s C[b, nc, i, h, s] * B[b, nc, j, h, s]
    G = torch.einsum('bcis,bcjs->bcij', C_chunked, B_chunked)
    # [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Compute M: apply L (attention-like pattern) to G
    # L: [batch, num_heads, num_chunks, chunk_size, chunk_size]
    # G: [batch, num_chunks, chunk_size, chunk_size, num_heads]
    # Permute L to [batch, num_chunks, chunk_size, chunk_size, num_heads]
    L_perm = L.permute(0, 2, 3, 4, 1)  # [batch, num_chunks, chunk_size, chunk_size, num_heads]
    M = G[..., None] * L_perm  # element-wise, [batch, num_chunks, chunk_size, chunk_size, num_heads]
    
    # Apply M to hidden_states (like attention to values)
    # M: [batch, num_chunks, chunk_size_i, chunk_size_j, num_heads]
    # hidden_states_chunked: [batch, num_chunks, chunk_size_j, num_heads, head_dim]
    # Y_diag[b, nc, i, h, d] = sum_j M[b, nc, i, j, h] * hidden_states[b, nc, j, h, d]
    Y_diag = torch.matmul(
        M.permute(0, 1, 4, 2, 3),
        hidden_states_chunked.permute(0, 1, 3, 2, 4),
    ).permute(0, 1, 3, 2, 4)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # 2. Compute states for each chunk (right term of factorization)
    # decay_states: exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    # A_cumsum: [batch, num_heads, num_chunks, chunk_size]
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]
    # Permute to [batch, num_chunks, chunk_size, num_heads]
    decay_states_perm = decay_states.permute(0, 2, 3, 1)  # [batch, num_chunks, chunk_size, num_heads]
    
    # B_decay: [batch, num_chunks, chunk_size, num_heads, state_size]
    B_decay = B_chunked[:, :, :, None, :] * decay_states_perm[..., None]

    # Compute states: sum over chunk_size dimension
    # states[b, nc, h, d, s] = sum_t B_decay[b, nc, t, h, s] * hidden_states[b, nc, t, h, d]
    states = torch.einsum('bcths,bcthd->bchds', B_decay, hidden_states_chunked)
    # [batch, num_chunks, num_heads, head_dim, state_size]
    
    # 3. Compute inter-chunk recurrence (middle term)
    # Prepend initial state
    # initial_states_f: [batch, num_heads, head_dim, state_size]
    # states: [batch, num_chunks, num_heads, head_dim, state_size]
    # Need to transpose states to [batch, num_chunks, num_heads, head_dim, state_size] -> already in this form
    # Prepend: [batch, 1, num_heads, head_dim, state_size]
    current_state = initial_states_f
    states_out_list = []
    for chunk in range(num_chunks):
        states_out_list.append(current_state)
        chunk_decay = torch.exp(A_cumsum[:, :, chunk, -1])[:, :, None, None]
        current_state = current_state * chunk_decay + states[:, chunk]
    states_out = torch.stack(states_out_list, dim=1)
    final_state = current_state
    
    # 4. Compute state -> output conversion (left term)
    state_decay_out = torch.exp(A_cumsum)  # [batch, num_heads, num_chunks, chunk_size]
    state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)  # [batch, num_chunks, chunk_size, num_heads]
    
    # C_times_states: contract C with states over state_size
    # C_chunked: [batch, num_chunks, chunk_size, num_heads, state_size]
    # states_out: [batch, num_chunks, num_heads, head_dim, state_size]
    # Y_off[b, nc, t, h, d] = sum_s C[b, nc, t, h, s] * states[b, nc, h, d, s] * decay[b, nc, t, h]
    C_times_states = torch.einsum('bcts,bchds->bcthd', C_chunked, states_out)
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    Y_off = C_times_states * state_decay_out_perm[..., None]
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # 5. Combine intra-chunk and inter-chunk outputs
    y = Y_diag + Y_off
    # [batch, num_chunks, chunk_size, num_heads, head_dim]
    
    # Reshape back: [batch, num_chunks, chunk_size, num_heads, head_dim] -> [batch, seq_len_padded, num_heads, head_dim]
    y = y.reshape(batch_size, -1, num_heads, head_dim)
    
    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    # Add the residual only for real (unpadded) sequence elements.
    y = y + D_f[None, None, :, None] * hidden_states_f

    # Reshape to [batch, seq_len, num_heads * head_dim]
    output = y.reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)
    final_state = final_state.to(torch.bfloat16)
    
    return output, final_state
