import torch
import torch.nn.functional as F
import math


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    lm_embeddings: torch.Tensor,
    audio_token_positions: torch.Tensor,
    encoder_input_weight: torch.Tensor,
    encoder_input_bias: torch.Tensor,
    encoder_out_weight: torch.Tensor,
    encoder_out_bias: torch.Tensor,
    encoder_out_mid_weight: torch.Tensor,
    encoder_out_mid_bias: torch.Tensor,
    learnable_queries: torch.Tensor,
    qformer_q_proj_weight: torch.Tensor,
    qformer_q_proj_bias: torch.Tensor,
    qformer_k_proj_weight: torch.Tensor,
    qformer_k_proj_bias: torch.Tensor,
    qformer_v_proj_weight: torch.Tensor,
    qformer_v_proj_bias: torch.Tensor,
    qformer_out_proj_weight: torch.Tensor,
    qformer_out_proj_bias: torch.Tensor,
    projector_weight: torch.Tensor,
    projector_bias: torch.Tensor,
):
    batch_size = input_features.shape[0]
    num_audio_tokens = audio_token_positions.shape[1]

    encoder_hidden_dim = 512
    projector_hidden_size = 1024
    text_hidden_size = 4096
    window_size = 15
    num_queries = 40
    qformer_num_heads = 16
    head_dim = projector_hidden_size // qformer_num_heads

    # Stage 1: Encoder (bf16 compute)
    hidden_states = F.linear(input_features, encoder_input_weight, encoder_input_bias)
    hidden_states_mid = F.linear(hidden_states, encoder_out_weight, encoder_out_bias)
    softmax_out = F.softmax(hidden_states_mid.float(), dim=-1).to(torch.bfloat16)
    feedback = F.linear(softmax_out, encoder_out_mid_weight, encoder_out_mid_bias)
    hidden_states = hidden_states + feedback
    encoder_output = hidden_states

    # Stage 2: Windowed Q-Former downsampling
    seq_len = encoder_output.shape[1]
    nblocks = math.ceil(seq_len / window_size)
    pad = nblocks * window_size - seq_len

    if pad > 0:
        encoder_output_padded = F.pad(encoder_output, (0, 0, 0, pad), "constant", 0)
    else:
        encoder_output_padded = encoder_output

    encoder_output_windowed = encoder_output_padded.view(
        batch_size * nblocks, window_size, encoder_hidden_dim
    )

    queries = learnable_queries.expand(batch_size * nblocks, -1, -1)

    Q = F.linear(queries, qformer_q_proj_weight, qformer_q_proj_bias)
    K = F.linear(encoder_output_windowed, qformer_k_proj_weight, qformer_k_proj_bias)
    V = F.linear(encoder_output_windowed, qformer_v_proj_weight, qformer_v_proj_bias)

    bsz_nblocks = batch_size * nblocks
    Q = Q.view(bsz_nblocks, num_queries, qformer_num_heads, head_dim).transpose(1, 2)
    K = K.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)
    V = V.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(Q, K, V)

    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bsz_nblocks, num_queries, projector_hidden_size
    )

    query_output = F.linear(attn_output, qformer_out_proj_weight, qformer_out_proj_bias)

    query_output = query_output.view(
        batch_size, nblocks * num_queries, projector_hidden_size
    )

    audio_embeddings = F.linear(query_output, projector_weight, projector_bias)

    fused_embeddings = lm_embeddings.clone()
    audio_emb_bf16 = audio_embeddings[:, :num_audio_tokens, :]
    idx = audio_token_positions.unsqueeze(-1).expand(-1, -1, text_hidden_size)
    fused_embeddings.scatter_(1, idx, audio_emb_bf16)
    return fused_embeddings


run = torch.compile(run, dynamic=True)
