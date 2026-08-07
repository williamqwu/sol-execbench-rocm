import torch
import torch.nn.functional as F
import math


# Precomputed static causal mask. Depends only on constants, not on inputs.
# Cached per-device to avoid rebuilding every call.
_LOCAL_CAUSAL_MASK_CACHE: dict = {}


def _get_local_causal_mask(device):
    cached = _LOCAL_CAUSAL_MASK_CACHE.get(device)
    if cached is not None:
        return cached
    CONTEXT_SIZE = 160
    CHUNK_SIZE = 32
    MAX_BACKWARD = 127
    MAX_FORWARD = 0
    lower_causal_mask = torch.tril(
        torch.ones((CONTEXT_SIZE, CHUNK_SIZE), dtype=torch.bool, device=device),
        diagonal=0,
    ).T
    upper_causal_mask = torch.tril(
        torch.ones((CHUNK_SIZE, CONTEXT_SIZE), dtype=torch.bool, device=device),
        diagonal=MAX_BACKWARD + MAX_FORWARD,
    )
    local_causal_valid_mask = torch.ones(
        (CHUNK_SIZE, CONTEXT_SIZE), dtype=torch.bool, device=device
    )
    local_causal_valid_mask = local_causal_valid_mask * lower_causal_mask * upper_causal_mask
    _LOCAL_CAUSAL_MASK_CACHE[device] = local_causal_valid_mask
    return local_causal_valid_mask


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for audio relative position attention."""
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]
    head_dim = axes_and_scalars["head_dim"]
    num_timescales = axes_and_scalars["num_timescales"]

    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
    mask = torch.rand(batch_size, seq_len, device=device) < 0.1

    q_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    k_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    v_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02
    pos_proj_weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32) * 0.02

    per_dim_scale = torch.zeros(head_dim, device=device, dtype=torch.float32)

    min_timescale = 1.0
    max_timescale = 1.0e4
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32, device=device) * -log_timescale_increment
    )

    attention_logits_soft_cap = 50.0

    return {
        "hidden_states": hidden_states,
        "mask": mask,
        "q_proj_weight": q_proj_weight,
        "k_proj_weight": k_proj_weight,
        "v_proj_weight": v_proj_weight,
        "pos_proj_weight": pos_proj_weight,
        "per_dim_scale": per_dim_scale,
        "inv_timescales": inv_timescales,
        "attention_logits_soft_cap": attention_logits_soft_cap,
    }


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

    q_scale_base = HEAD_DIM ** -0.5
    r_softplus_0 = 1.0 / F.softplus(torch.tensor(0.0, device=device))
    q_scale = q_scale_base * r_softplus_0

    local_causal_valid_mask = _get_local_causal_mask(device)

    # Fuse Q/K/V projections into a single matmul.
    qkv_weight = torch.cat(
        [q_proj_weight.T, k_proj_weight.T, v_proj_weight.T], dim=1
    )  # [hidden_size, 3*hidden_size]
    qkv = torch.matmul(hidden_states, qkv_weight)  # [B, T, 3*hidden_size]
    qkv = qkv.view(batch_size, q_time, 3, hidden_size)
    query_states, key_states, value_states = qkv.unbind(dim=2)

    qkv_shape = (batch_size, q_time, NUM_HEADS, HEAD_DIM)
    query_states = query_states.reshape(qkv_shape).contiguous()
    key_states = key_states.reshape(qkv_shape).contiguous()
    value_states = value_states.reshape(qkv_shape).contiguous()

    per_dim_scale_sp = F.softplus(per_dim_scale)
    per_dim_scale_sp_broadcast = per_dim_scale_sp.view(1, 1, 1, HEAD_DIM)
    query_states = query_states * q_scale * per_dim_scale_sp_broadcast

    def pad_dim1(x, pad_left, pad_right):
        b, _, *tail_shape = x.shape
        left = x.new_zeros((b, pad_left, *tail_shape))
        right = x.new_zeros((b, pad_right, *tail_shape))
        return torch.cat([left, x, right], dim=1)

    def convert_to_block(hs):
        b, t = hs.shape[:2]
        num_blocks = (t + CHUNK_SIZE - 1) // CHUNK_SIZE
        padding_len = num_blocks * CHUNK_SIZE - t
        if padding_len > 0:
            hs = pad_dim1(hs, 0, padding_len)
        permute_dims = (b, num_blocks, CHUNK_SIZE) + hs.shape[2:]
        return hs.reshape(permute_dims).contiguous()

    def extract_block_context(hs):
        pad_left = MAX_BACKWARD
        pad_right = MAX_FORWARD + CHUNK_SIZE - 1
        hs = pad_dim1(hs, pad_left, pad_right)
        x_unfolded = hs.unfold(dimension=1, size=CONTEXT_SIZE, step=CHUNK_SIZE)
        if hs.ndim > 2 and x_unfolded.ndim > 3:
            x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
        return x_unfolded.contiguous()

    def relative_shift(term_bd_before_shift, bs, nh, nqb, qbs, kcs, msp1):
        pad_amount_last_dim = (kcs + 1) - msp1
        padding_tuple = (0, pad_amount_last_dim)
        term_bd_padded = F.pad(term_bd_before_shift, padding_tuple)
        term_bd_reshaped = term_bd_padded.reshape(bs, nh, nqb, qbs * (kcs + 1))
        term_bd_sliced = term_bd_reshaped[:, :, :, :qbs * kcs]
        term_bd_shifted = term_bd_sliced.reshape(bs, nh, nqb, qbs, kcs)
        return term_bd_shifted

    def get_timing_signal_1d_pos(position):
        position = position.float().unsqueeze(-1)
        scaled_time = position * inv_timescales.unsqueeze(0).unsqueeze(0)
        timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return timing_signal.to(dtype)

    query_blocks = convert_to_block(query_states)
    key_blocks = extract_block_context(key_states)
    value_blocks = extract_block_context(value_states)
    num_query_blocks = query_blocks.shape[1]

    original_valid_mask = ~mask
    extracted_valid_mask_blocks = extract_block_context(original_valid_mask)
    if extracted_valid_mask_blocks.ndim == 4:
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            batch_size, num_query_blocks, CONTEXT_SIZE
        )

    condition_from_input_validity = extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2)
    condition_from_causality = local_causal_valid_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    final_condition_for_where = torch.logical_and(
        condition_from_input_validity,
        condition_from_causality
    )

    pos_indices = torch.arange(MAX_BACKWARD, -MAX_FORWARD - 1, -1, device=device).unsqueeze(0)
    max_span_plus_1 = pos_indices.shape[1]

    sin_emb_timing_signal = get_timing_signal_1d_pos(pos_indices)
    projected_sin_emb = torch.matmul(sin_emb_timing_signal, pos_proj_weight.T)
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, NUM_HEADS, HEAD_DIM).squeeze(0)

    queries_p = query_blocks.permute(0, 3, 1, 2, 4)  # [B, N, U, W, H]
    keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)     # [B, N, U, H, C]
    term_ac = torch.matmul(queries_p, keys_p_t)      # [B, N, U, W, C]

    q_permuted = query_blocks.permute(0, 3, 1, 2, 4)
    s_permuted = sin_emb.permute(1, 2, 0)
    q_reshaped = q_permuted.reshape(batch_size, NUM_HEADS, num_query_blocks * CHUNK_SIZE, HEAD_DIM)
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, s_permuted)
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size, NUM_HEADS, num_query_blocks, CHUNK_SIZE, max_span_plus_1
    )

    term_bd_shifted = relative_shift(
        term_bd_unshifed, batch_size, NUM_HEADS, num_query_blocks,
        CHUNK_SIZE, CONTEXT_SIZE, max_span_plus_1
    )

    logits = term_ac + term_bd_shifted

    softcap_val = torch.tensor(attention_logits_soft_cap, device=device, dtype=torch.float32)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val

    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32).to(dtype=value_blocks.dtype)

    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = value_blocks.shape[-1]
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
    result_bmm = torch.bmm(prob_bun, v_bun)
    context_vectors = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)

    context_vectors = context_vectors.reshape(
        batch_size, num_query_blocks * CHUNK_SIZE, NUM_HEADS, HEAD_DIM
    )
    context_vectors = context_vectors[:, :q_time]

    return context_vectors
