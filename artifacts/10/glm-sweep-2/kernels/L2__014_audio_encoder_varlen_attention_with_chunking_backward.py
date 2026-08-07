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

    grad_output_f32 = grad_output.to(torch.float32)
    hidden_states_f32 = hidden_states.to(torch.float32)
    qf = query_states.to(torch.float32)
    kf = key_states.to(torch.float32)
    vf = value_states.to(torch.float32)
    q_weight_f32 = q_weight.to(torch.float32)
    k_weight_f32 = k_weight.to(torch.float32)
    v_weight_f32 = v_weight.to(torch.float32)

    # grad_attn_output = grad_output @ out_weight
    grad_attn_output = torch.matmul(grad_output_f32, out_weight.to(torch.float32))
    # (total_seq_len, qkv_dim) -> (1, num_heads, total_seq_len, head_dim)
    grad_attn_output = grad_attn_output.reshape(total_seq_len, num_heads, head_dim).permute(1, 0, 2).unsqueeze(0).contiguous()

    # Block-diagonal attention mask: positions attend only within their chunk.
    cu = cu_seqlens.long()
    chunk_ids = torch.zeros(total_seq_len, dtype=torch.long, device=hidden_states.device)
    for ci in range(len(cu) - 1):
        chunk_ids[cu[ci]:cu[ci + 1]] = ci
    same = chunk_ids.unsqueeze(0) == chunk_ids.unsqueeze(1)  # (T, T) bool
    mask = torch.zeros(total_seq_len, total_seq_len, device=hidden_states.device)
    mask[~same] = float('-inf')

    # Attention scores + softmax (full sequence, masked)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scaling + mask  # (1, H, T, T)
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)  # (1, H, T, T)

    attn_out = torch.matmul(attn_weights, vf)  # (1, H, T, D)
    attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()

    grad_out_weight = torch.matmul(grad_output.t(), attn_out.to(torch.bfloat16))
    grad_out_bias = grad_output_f32.sum(dim=0)

    # Attention backward
    grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_attn_output)  # (1, H, T, D)
    grad_attn_weights = torch.matmul(grad_attn_output, vf.transpose(-2, -1))  # (1, H, T, T)
    grad_attn_weights = attn_weights * (grad_attn_weights - (attn_weights * grad_attn_weights).sum(dim=-1, keepdim=True))

    grad_q = torch.matmul(grad_attn_weights, kf) * scaling  # (1, H, T, D)
    grad_k = torch.matmul(grad_attn_weights.transpose(-2, -1), qf) * scaling

    grad_q = grad_q.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_k = grad_k.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()
    grad_v = grad_v.squeeze(0).transpose(0, 1).reshape(total_seq_len, -1).contiguous()

    grad_q_weight = torch.matmul(grad_q.to(torch.bfloat16).t(), hidden_states)
    grad_q_bias = grad_q.sum(dim=0)
    grad_k_weight = torch.matmul(grad_k.to(torch.bfloat16).t(), hidden_states)
    grad_k_bias = grad_k.sum(dim=0)
    grad_v_weight = torch.matmul(grad_v.to(torch.bfloat16).t(), hidden_states)
    grad_v_bias = grad_v.sum(dim=0)

    grad_hidden_states = torch.matmul(grad_q, q_weight_f32) + \
                        torch.matmul(grad_k, k_weight_f32) + \
                        torch.matmul(grad_v, v_weight_f32)

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
