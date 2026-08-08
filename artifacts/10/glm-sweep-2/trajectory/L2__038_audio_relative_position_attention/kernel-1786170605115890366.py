import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    per_dim_scale: torch.Tensor,
    inv_timescales: torch.Tensor,
    attention_logits_soft_cap: float,
):
    NUM_HEADS = 8
    HEAD_DIM = 64
    CHUNK_SIZE = 32
    MAX_BACKWARD = 127
    MAX_FORWARD = 0
    CONTEXT_SIZE = 160

    batch_size, q_time, hidden_size = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Query scaling factor - must match reference exactly
    q_scale_base = HEAD_DIM ** -0.5
    r_softplus_0 = 1.0 / F.softplus(torch.tensor(0.0, device=device))
    q_scale = q_scale_base * r_softplus_0

    # Build local causal mask
    lower_causal_mask = torch.tril(
        torch.ones((CONTEXT_SIZE, CHUNK_SIZE), dtype=torch.bool, device=device),
        diagonal=0,
    ).T
    upper_causal_mask = torch.tril(
        torch.ones((CHUNK_SIZE, CONTEXT_SIZE), dtype=torch.bool, device=device),
        diagonal=MAX_BACKWARD + MAX_FORWARD,
    )
    local_causal_valid_mask = torch.ones((CHUNK_SIZE, CONTEXT_SIZE), dtype=torch.bool, device=device)
    local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask

    # Project to Q, K, V
    query_states = torch.matmul(hidden_states, q_proj_weight.T)
    key_states = torch.matmul(hidden_states, k_proj_weight.T)
    value_states = torch.matmul(hidden_states, v_proj_weight.T)

    # Reshape to multi-head format
    qkv_shape = (batch_size, q_time, NUM_HEADS, HEAD_DIM)
    query_states = query_states.reshape(qkv_shape)
    key_states = key_states.reshape(qkv_shape)
    value_states = value_states.reshape(qkv_shape)

    # Apply per-dimension scaling to queries
    per_dim_scale_sp = F.softplus(per_dim_scale)
    per_dim_scale_sp_broadcast = per_dim_scale_sp.view(1, 1, 1, HEAD_DIM)
    query_states = query_states * q_scale * per_dim_scale_sp_broadcast

    # --- Block conversion ---
    num_blocks = (q_time + CHUNK_SIZE - 1) // CHUNK_SIZE
    padding_len = num_blocks * CHUNK_SIZE - q_time

    # Pad query along time
    if padding_len > 0:
        query_states = F.pad(query_states, (0, 0, 0, 0, 0, padding_len))
    query_blocks = query_states.reshape(batch_size, num_blocks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)

    # --- Extract block context for K, V ---
    key_states = F.pad(key_states, (0, 0, 0, 0, MAX_BACKWARD, CHUNK_SIZE - 1))
    value_states = F.pad(value_states, (0, 0, 0, 0, MAX_BACKWARD, CHUNK_SIZE - 1))

    key_blocks = key_states.unfold(1, CONTEXT_SIZE, CHUNK_SIZE)
    value_blocks = value_states.unfold(1, CONTEXT_SIZE, CHUNK_SIZE)
    key_blocks = key_blocks.movedim(-1, 2)
    value_blocks = value_blocks.movedim(-1, 2)

    # --- Mask blocks ---
    original_valid_mask = ~mask
    original_valid_mask = F.pad(original_valid_mask, (MAX_BACKWARD, CHUNK_SIZE - 1))
    extracted_valid_mask_blocks = original_valid_mask.unfold(1, CONTEXT_SIZE, CHUNK_SIZE)
    extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(batch_size, num_blocks, CONTEXT_SIZE)

    condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)
    condition_from_causality = local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    final_condition_for_where = torch.logical_and(condition_from_input_validity, condition_from_causality)

    # --- Position embeddings ---
    pos_indices = torch.arange(MAX_BACKWARD, -MAX_FORWARD - 1, -1, device=device).unsqueeze(0)
    max_span_plus_1 = pos_indices.shape[1]

    position = pos_indices.float().unsqueeze(-1)
    scaled_time = position * inv_timescales.unsqueeze(0).unsqueeze(0)
    timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
    timing_signal = timing_signal.to(dtype)

    projected_sin_emb = torch.matmul(timing_signal, pos_proj_weight.T)
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, NUM_HEADS, HEAD_DIM).squeeze(0)

    # --- Attention logits ---
    queries_p = query_blocks.permute(0, 3, 1, 2, 4)
    keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
    term_ac = torch.matmul(queries_p, keys_p_t)

    s_permuted = sin_emb.permute(1, 2, 0)
    q_reshaped = queries_p.reshape(batch_size, NUM_HEADS, num_blocks * CHUNK_SIZE, HEAD_DIM)
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size, NUM_HEADS, num_blocks, CHUNK_SIZE, max_span_plus_1
    )

    # relative_shift
    pad_amount_last_dim = (CONTEXT_SIZE + 1) - max_span_plus_1
    term_bd_padded = F.pad(term_bd_unshifed, (0, pad_amount_last_dim))
    term_bd_reshaped = term_bd_padded.reshape(batch_size, NUM_HEADS, num_blocks, CHUNK_SIZE * (CONTEXT_SIZE + 1))
    term_bd_sliced = term_bd_reshaped[:, :, :, :CHUNK_SIZE * CONTEXT_SIZE]
    term_bd_shifted = term_bd_sliced.reshape(batch_size, NUM_HEADS, num_blocks, CHUNK_SIZE, CONTEXT_SIZE)

    logits = term_ac + term_bd_shifted

    # Soft-capping - must match reference exactly
    softcap_val = torch.tensor(attention_logits_soft_cap, device=device, dtype=torch.float32)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val

    # Apply mask
    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)

    # Softmax
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=value_blocks.dtype)

    # Context vectors
    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = HEAD_DIM
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
    result_bmm = torch.bmm(prob_bun, v_bun)
    context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)

    context_vectors = context_vectors.reshape(batch_size, num_blocks * CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    context_vectors = context_vectors[:, :q_time]

    return context_vectors
