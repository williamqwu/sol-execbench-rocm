import torch

NUM_HEADS = 16
NUM_KV_HEADS = 4
NUM_KV_GROUPS = 4
HEAD_DIM = 256

_RUN = None


@torch.no_grad()
def _run_impl(
    grad_query_states,
    grad_key_states,
    grad_value_states,
    decoder_hidden_states,
    encoder_hidden_states,
    q_weight,
    k_weight,
    v_weight,
):
    batch_size = decoder_hidden_states.shape[0]
    seq_len_dec = decoder_hidden_states.shape[1]
    seq_len_enc = encoder_hidden_states.shape[1]

    # ===== Query path =====
    grad_query_proj = grad_query_states.transpose(1, 2).reshape(
        batch_size, seq_len_dec, NUM_HEADS * HEAD_DIM
    )
    grad_decoder_hidden_states = torch.matmul(grad_query_proj, q_weight)
    decoder_flat = decoder_hidden_states.reshape(-1, decoder_hidden_states.shape[-1])
    grad_query_flat = grad_query_proj.reshape(-1, grad_query_proj.shape[-1])
    grad_q_weight = torch.matmul(grad_query_flat.t(), decoder_flat)

    # ===== Key path =====
    grad_key_proj = (
        grad_key_states.view(batch_size, NUM_KV_HEADS, NUM_KV_GROUPS, seq_len_enc, HEAD_DIM)
        .sum(dim=2)
        .transpose(1, 2)
        .reshape(batch_size, seq_len_enc, NUM_KV_HEADS * HEAD_DIM)
    )
    grad_encoder_hidden_states = torch.matmul(grad_key_proj, k_weight)
    encoder_flat = encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1])
    grad_key_flat = grad_key_proj.reshape(-1, grad_key_proj.shape[-1])
    grad_k_weight = torch.matmul(grad_key_flat.t(), encoder_flat)

    # ===== Value path =====
    grad_value_proj = (
        grad_value_states.view(batch_size, NUM_KV_HEADS, NUM_KV_GROUPS, seq_len_enc, HEAD_DIM)
        .sum(dim=2)
        .transpose(1, 2)
        .reshape(batch_size, seq_len_enc, NUM_KV_HEADS * HEAD_DIM)
    )
    grad_from_value = torch.matmul(grad_value_proj, v_weight)
    grad_encoder_hidden_states = grad_encoder_hidden_states + grad_from_value
    grad_value_flat = grad_value_proj.reshape(-1, grad_value_proj.shape[-1])
    grad_v_weight = torch.matmul(grad_value_flat.t(), encoder_flat)

    return (
        grad_decoder_hidden_states,
        grad_encoder_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
    )


@torch.no_grad()
def run(
    grad_query_states,
    grad_key_states,
    grad_value_states,
    decoder_hidden_states,
    encoder_hidden_states,
    q_weight,
    k_weight,
    v_weight,
):
    global _RUN
    if _RUN is None:
        _RUN = torch.compile(_run_impl, mode="max-autotune")
    return _RUN(
        grad_query_states,
        grad_key_states,
        grad_value_states,
        decoder_hidden_states,
        encoder_hidden_states,
        q_weight,
        k_weight,
        v_weight,
    )
