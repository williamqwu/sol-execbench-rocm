import torch

NUM_HEADS = 16
NUM_KV_HEADS = 4
NUM_KV_GROUPS = 4
HEAD_DIM = 256
Q_OUT_DIM = NUM_HEADS * HEAD_DIM       # 4096
KV_OUT_DIM = NUM_KV_HEADS * HEAD_DIM   # 1024


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
    batch_size = decoder_hidden_states.shape[0]
    seq_len_dec = decoder_hidden_states.shape[1]
    seq_len_enc = encoder_hidden_states.shape[1]

    # ===== Query path =====
    # (B, 16, S_dec, 256) -> (B, S_dec, 4096)
    grad_query_proj = grad_query_states.transpose(1, 2).reshape(
        batch_size, seq_len_dec, Q_OUT_DIM
    )
    grad_decoder_hidden_states = torch.matmul(grad_query_proj, q_weight)

    decoder_flat = decoder_hidden_states.reshape(-1, decoder_hidden_states.shape[-1])
    grad_query_flat = grad_query_proj.reshape(-1, Q_OUT_DIM)
    grad_q_weight = torch.matmul(grad_query_flat.t(), decoder_flat)

    # ===== Key + Value path (fused where numerically safe) =====
    # GQA un-repeat: (B, 16, S_enc, 256) -> (B, 4, S_enc, 256) via sum over groups
    grad_key_proj = (
        grad_key_states.view(batch_size, NUM_KV_HEADS, NUM_KV_GROUPS, seq_len_enc, HEAD_DIM)
        .sum(dim=2)
        .transpose(1, 2)
        .reshape(batch_size, seq_len_enc, KV_OUT_DIM)
    )
    grad_value_proj = (
        grad_value_states.view(batch_size, NUM_KV_HEADS, NUM_KV_GROUPS, seq_len_enc, HEAD_DIM)
        .sum(dim=2)
        .transpose(1, 2)
        .reshape(batch_size, seq_len_enc, KV_OUT_DIM)
    )

    # Input grads: compute V contribution first, then fuse K via addmm
    # (addition is exactly commutative in IEEE754, so result is identical)
    grad_from_value = torch.matmul(grad_value_proj, v_weight)
    grad_encoder_hidden_states = torch.addmm(grad_from_value, grad_key_proj, k_weight)

    # Weight grads: both K and V use the same encoder_flat.
    # Stack [grad_key_flat.t(); grad_value_flat.t()] -> [2048, B*S_enc]
    # One GEMM with encoder_flat -> [2048, 1024], split into k/v.
    # Each output row is independent => numerically identical to separate GEMMs.
    encoder_flat = encoder_hidden_states.reshape(-1, encoder_hidden_states.shape[-1])
    grad_key_flat = grad_key_proj.reshape(-1, KV_OUT_DIM)
    grad_value_flat = grad_value_proj.reshape(-1, KV_OUT_DIM)
    grad_kv = torch.matmul(
        torch.cat((grad_key_flat.t(), grad_value_flat.t()), dim=0), encoder_flat
    )
    grad_k_weight = grad_kv[:KV_OUT_DIM]
    grad_v_weight = grad_kv[KV_OUT_DIM:]

    return (
        grad_decoder_hidden_states,
        grad_encoder_hidden_states,
        grad_q_weight,
        grad_k_weight,
        grad_v_weight,
    )
