import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs with valid cu_seqlens for variable-length attention backward."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_chunks = axes_and_scalars["num_chunks"]
    d_model = axes_and_scalars["d_model"]
    num_heads = axes_and_scalars["num_heads"]
    head_dim = axes_and_scalars["head_dim"]
    qkv_dim = num_heads * head_dim
    scaling = head_dim ** -0.5

    if num_chunks == 1:
        chunk_lengths = [total_seq_len]
    else:
        base_len = total_seq_len // num_chunks
        remainder = total_seq_len % num_chunks
        chunk_lengths = [base_len + (1 if i < remainder else 0) for i in range(num_chunks)]

    cu_seqlens = torch.zeros(num_chunks + 1, dtype=torch.int32, device=device)
    for i in range(num_chunks):
        cu_seqlens[i + 1] = cu_seqlens[i] + chunk_lengths[i]

    grad_output = torch.randn(total_seq_len, d_model, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(total_seq_len, d_model, dtype=torch.bfloat16, device=device)
    query_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    key_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    value_states = torch.randn(1, num_heads, total_seq_len, head_dim, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    v_weight = torch.randn(qkv_dim, d_model, dtype=torch.bfloat16, device=device)
    out_weight = torch.randn(d_model, qkv_dim, dtype=torch.bfloat16, device=device)

    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "query_states": query_states,
        "key_states": key_states,
        "value_states": value_states,
        "cu_seqlens": cu_seqlens,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "v_weight": v_weight,
        "out_weight": out_weight,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    out_weight: torch.Tensor,
):
    total_seq_len = hidden_states.shape[0]
    d_model = hidden_states.shape[1]
    num_heads = query_states.shape[1]
    head_dim = query_states.shape[3]
    scaling = head_dim ** -0.5

    # Keep bf16 for matmul inputs (MFMA does fp32 accumulation); softmax in fp32.
    # Step 1: grad_attn_output = grad_output @ out_weight  (total_seq_len, qkv_dim)
    grad_attn_output = torch.matmul(grad_output, out_weight)  # bf16 matmul, fp32 accum

    # Reshape: (total_seq_len, qkv_dim) -> (1, num_heads, total_seq_len, head_dim)
    grad_attn_output = grad_attn_output.reshape(total_seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0).contiguous()

    # Use fp32 for attention softmax stability; q/k/v stay bf16 for matmul speed
    query_bf = query_states  # (1, num_heads, total_seq_len, head_dim) bf16
    key_bf = key_states
    value_bf = value_states

    cu_seqlens_cpu = cu_seqlens.cpu()
    chunk_lengths = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).tolist()

    query_chunks = query_bf.split(chunk_lengths, dim=2)
    key_chunks = key_bf.split(chunk_lengths, dim=2)
    value_chunks = value_bf.split(chunk_lengths, dim=2)
    grad_attn_chunks = grad_attn_output.split(chunk_lengths, dim=2)

    attn_outputs_recompute = []
    grad_query_chunks = []
    grad_key_chunks = []
    grad_value_chunks = []

    for grad_attn, q, k, v in zip(grad_attn_chunks, query_chunks, key_chunks, value_chunks):
        # QK^T in bf16, then softmax in fp32
        scores = torch.matmul(q, k.transpose(-2, -1)) * scaling  # bf16
        attn_weights = F.softmax(scores.to(torch.float32), dim=-1, dtype=torch.float32)
        attn_weights_bf = attn_weights.to(torch.bfloat16)

        # Attention output (for grad_out_weight): bf16 matmul
        attn_outputs_recompute.append(torch.matmul(attn_weights_bf, v))

        # grad_v = attn_weights^T @ grad_attn  (bf16)
        grad_v = torch.matmul(attn_weights_bf.transpose(-2, -1), grad_attn.to(torch.bfloat16))

        # grad_attn_weights = grad_attn @ V^T  (bf16 -> fp32 for softmax bwd)
        grad_attn_weights = torch.matmul(grad_attn.to(torch.bfloat16), v.transpose(-2, -1)).to(torch.float32)

        # Backprop through softmax (fp32)
        aw_f32 = attn_weights
        grad_attn_weights = aw_f32 * (grad_attn_weights - (aw_f32 * grad_attn_weights).sum(dim=-1, keepdim=True))
        grad_attn_weights_bf = grad_attn_weights.to(torch.bfloat16)

        # grad_q, grad_k (bf16)
        grad_q = torch.matmul(grad_attn_weights_bf, k) * scaling
        grad_k = torch.matmul(grad_attn_weights_bf.transpose(-2, -1), q) * scaling

        grad_query_chunks.append(grad_q)
        grad_key_chunks.append(grad_k)
        grad_value_chunks.append(grad_v)

    attn_output_recompute = torch.cat(attn_outputs_recompute, dim=2)
    attn_output_recompute = attn_output_recompute.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()

    grad_out_weight = torch.matmul(grad_output.t(), attn_output_recompute)
    grad_out_bias = grad_output.sum(dim=0)

    grad_query = torch.cat(grad_query_chunks, dim=2).squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_key = torch.cat(grad_key_chunks, dim=2).squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_value = torch.cat(grad_value_chunks, dim=2).squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()

    grad_q_weight = torch.matmul(grad_query.t(), hidden_states)
    grad_q_bias = grad_query.sum(dim=0)
    grad_k_weight = torch.matmul(grad_key.t(), hidden_states)
    grad_k_bias = grad_key.sum(dim=0)
    grad_v_weight = torch.matmul(grad_value.t(), hidden_states)
    grad_v_bias = grad_value.sum(dim=0)

    grad_hidden_states = torch.matmul(grad_query, q_weight) + \
                        torch.matmul(grad_key, k_weight) + \
                        torch.matmul(grad_value, v_weight)

    return (
        grad_hidden_states.to(torch.bfloat16),
        grad_q_weight.to(torch.bfloat16),
        grad_q_bias.to(torch.bfloat16),
        grad_k_weight.to(torch.bfloat16),
        grad_k_bias.to(torch.bfloat16),
        grad_v_weight.to(torch.bfloat16),
        grad_v_bias.to(torch.bfloat16),
        grad_out_weight.to(torch.bfloat16),
        grad_out_bias.to(torch.bfloat16),
    )
