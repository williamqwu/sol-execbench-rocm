import torch
import torch.nn.functional as F
import math

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


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

    # Stage 1: Encoder (bf16 compute) - runs fully, feeds the windows we need.
    hidden_states = F.linear(input_features, encoder_input_weight, encoder_input_bias)
    hidden_states_mid = F.linear(hidden_states, encoder_out_weight, encoder_out_bias)
    softmax_out = F.softmax(hidden_states_mid, dim=-1)
    feedback = F.linear(softmax_out, encoder_out_mid_weight, encoder_out_mid_bias)
    hidden_states = hidden_states + feedback
    encoder_output = hidden_states  # [batch_size, audio_seq_len, 512]

    # Stage 2: Windowed Q-Former downsampling.
    # Only the first num_audio_tokens audio embeddings are used downstream, and
    # Q-Former cross-attention produces per-query outputs that are independent
    # across queries (no query-query mixing). The flat layout maps the first
    # num_audio_tokens entries to the first ceil(num_audio_tokens/num_queries)
    # windows, so we only need to run Q-Former + projector on those windows.
    seq_len = encoder_output.shape[1]
    nblocks_total = math.ceil(seq_len / window_size)
    need_win = min(nblocks_total, math.ceil(num_audio_tokens / num_queries))

    # Slice encoder output to only the frames the needed windows cover.
    need_frames = need_win * window_size
    encoder_output_sliced = encoder_output[:, :need_frames, :]

    # Reshape into windows (no padding needed: need_frames is a multiple of window_size).
    encoder_output_windowed = encoder_output_sliced.reshape(
        batch_size * need_win, window_size, encoder_hidden_dim
    )

    queries = learnable_queries.expand(batch_size * need_win, -1, -1)
    Q = F.linear(queries, qformer_q_proj_weight, qformer_q_proj_bias)
    K = F.linear(encoder_output_windowed, qformer_k_proj_weight, qformer_k_proj_bias)
    V = F.linear(encoder_output_windowed, qformer_v_proj_weight, qformer_v_proj_bias)

    bsz_nblocks = batch_size * need_win
    Q = Q.view(bsz_nblocks, num_queries, qformer_num_heads, head_dim).transpose(1, 2)
    K = K.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)
    V = V.view(bsz_nblocks, window_size, qformer_num_heads, head_dim).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(Q, K, V)

    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bsz_nblocks, num_queries, projector_hidden_size
    )

    query_output = F.linear(attn_output, qformer_out_proj_weight, qformer_out_proj_bias)

    query_output = query_output.view(
        batch_size, need_win * num_queries, projector_hidden_size
    )

    # Only the first num_audio_tokens rows are used; trim any excess from the
    # last partial window of queries.
    audio_embeddings = F.linear(
        query_output[:, :num_audio_tokens, :], projector_weight, projector_bias
    )

    # Stage 3: Scatter audio embeddings into LM embeddings (bf16, no fp32 round-trip).
    fused_embeddings = lm_embeddings.clone()
    idx = audio_token_positions.unsqueeze(-1).expand(-1, -1, text_hidden_size)
    fused_embeddings.scatter_(1, idx, audio_embeddings)
    return fused_embeddings

run = torch.compile(run, dynamic=True)
