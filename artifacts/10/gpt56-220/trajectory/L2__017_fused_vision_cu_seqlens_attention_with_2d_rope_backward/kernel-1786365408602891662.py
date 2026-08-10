import torch
import torch.nn.functional as F

_aux_stream = None


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Generate inputs for backward pass testing."""
    total_seq_len = axes_and_scalars["total_seq_len"]
    num_sequences = axes_and_scalars["num_sequences"]
    hidden_size = 3584
    num_heads = 28
    head_dim = 128
    qkv_out_size = hidden_size * 3
    
    # Generate grad_output
    grad_output = torch.randn(total_seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Generate hidden_states
    hidden_states = torch.randn(total_seq_len, hidden_size, dtype=torch.float32, device=device)
    
    # Generate QKV weights and biases
    qkv_weight = torch.randn(qkv_out_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    qkv_bias = torch.randn(qkv_out_size, dtype=torch.float32, device=device) * 0.01
    
    # Generate projection weights and biases
    proj_weight = torch.randn(hidden_size, hidden_size, dtype=torch.float32, device=device) * 0.02
    proj_bias = torch.randn(hidden_size, dtype=torch.float32, device=device) * 0.01
    
    # Generate RoPE embeddings
    cos = torch.randn(total_seq_len, head_dim, dtype=torch.float32, device=device)
    sin = torch.randn(total_seq_len, head_dim, dtype=torch.float32, device=device)
    
    # Generate cu_seqlens with shape (num_sequences + 1,)
    # Distribute total_seq_len across num_sequences
    if num_sequences <= 1:
        cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32, device=device)
    else:
        base_len = total_seq_len // num_sequences
        remainder = total_seq_len % num_sequences
        lengths = [base_len] * num_sequences
        for i in range(remainder):
            lengths[i] += 1
        cu_seqlens_list = [0]
        for l in lengths:
            cu_seqlens_list.append(cu_seqlens_list[-1] + l)
        cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    
    return {
        "grad_output": grad_output,
        "hidden_states": hidden_states,
        "qkv_weight": qkv_weight,
        "qkv_bias": qkv_bias,
        "proj_weight": proj_weight,
        "proj_bias": proj_bias,
        "cos": cos,
        "sin": sin,
        "cu_seqlens": cu_seqlens,
        "attention_dropout": 0.0,
        "scaling": head_dim ** -0.5,
    }


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    hidden_states: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    attention_dropout: float,
    scaling: float,
):
    """Backward pass for fused vision attention with cu_seqlens and 2D RoPE.
    
    Computes gradients for:
    - hidden_states
    - qkv_weight, qkv_bias
    - proj_weight, proj_bias
    """
    seq_length = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    num_heads = 28
    head_dim = 128
    global _aux_stream
    if _aux_stream is None:
        _aux_stream = torch.cuda.Stream(priority=-1)
    
    # ============ Forward recomputation ============
    # QKV projection
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)
    
    # Apply 2D RoPE
    cos_expanded = cos.unsqueeze(-2).float()
    sin_expanded = sin.unsqueeze(-2).float()
    
    query_states_float = query_states.float()
    key_states_float = key_states.float()

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_half_inverse(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((x2, -x1), dim=-1)

    query_states_rope = query_states_float * cos_expanded + rotate_half(query_states_float) * sin_expanded
    key_states_rope = key_states_float * cos_expanded + rotate_half(key_states_float) * sin_expanded
    
    # The generated variable-length batches contain at most two distinct
    # lengths.  Batch equal-length sequences so each attention operation is a
    # large strided batched GEMM instead of one launch per sequence.
    nseq = cu_seqlens.numel() - 1
    short_len = seq_length // nseq
    nlong = seq_length % nseq
    groups = []
    offset = 0
    attn_output_parts = []
    for total_count, length in ((nlong, short_len + 1), (nseq - nlong, short_len)):
        while total_count:
            # hipBLAS algorithm-selection thresholds vary with GEMM size.
            max_batch = 5 if length >= 256 else 15
            count = min(total_count, max_batch)
            end = offset + count * length
            q = query_states_rope[offset:end].reshape(count, length, num_heads, head_dim).transpose(1, 2)
            k = key_states_rope[offset:end].reshape(count, length, num_heads, head_dim).transpose(1, 2)
            v = value_states[offset:end].reshape(count, length, num_heads, head_dim).transpose(1, 2)
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            out = torch.matmul(attn_weights, v)
            attn_output_parts.append(out.transpose(1, 2).reshape(count * length, hidden_size))
            groups.append((offset, end, count, length, q, k, v, attn_weights))
            offset = end
            total_count -= count

    attn_output_reshaped = (attn_output_parts[0] if len(attn_output_parts) == 1
                            else torch.cat(attn_output_parts, dim=0))
    
    # ============ Backward computation ============
    
    # Gradient through output projection
    _aux_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_aux_stream):
        grad_proj_bias = grad_output.sum(dim=0)
    grad_proj_weight = grad_output.t() @ attn_output_reshaped
    grad_attn_output = grad_output @ proj_weight
    
    grad_query_splits = []
    grad_key_splits = []
    grad_value_splits = []

    for offset, end, count, length, q, k, v, attn_weights in groups:
        grad_attn_out = grad_attn_output[offset:end].reshape(
            count, length, num_heads, head_dim
        ).transpose(1, 2)
        # Gradient through attention output
        grad_attn_weights = torch.matmul(grad_attn_out, v.transpose(2, 3))
        grad_v = torch.matmul(attn_weights.transpose(2, 3), grad_attn_out)
        
        # Gradient through softmax
        sum_grad = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
        grad_attn_scores = attn_weights * (grad_attn_weights - sum_grad)
        grad_attn_scores = grad_attn_scores * scaling
        
        # Gradient through Q @ K^T
        grad_q = torch.matmul(grad_attn_scores, k)
        grad_k = torch.matmul(grad_attn_scores.transpose(2, 3), q)
        
        grad_query_splits.append(grad_q.transpose(1, 2).reshape(count * length, num_heads, head_dim))
        grad_key_splits.append(grad_k.transpose(1, 2).reshape(count * length, num_heads, head_dim))
        grad_value_splits.append(grad_v.transpose(1, 2).reshape(count * length, num_heads, head_dim))

    if len(grad_query_splits) == 1:
        grad_query_states = grad_query_splits[0]
        grad_key_states = grad_key_splits[0]
        grad_value_states = grad_value_splits[0]
    else:
        grad_query_states = torch.cat(grad_query_splits, dim=0)
        grad_key_states = torch.cat(grad_key_splits, dim=0)
        grad_value_states = torch.cat(grad_value_splits, dim=0)
    
    # Gradient through 2D RoPE
    grad_query_states_float = grad_query_states.float()
    grad_key_states_float = grad_key_states.float()
    
    grad_query_pre_rope = (
        grad_query_states_float * cos_expanded +
        rotate_half_inverse(grad_query_states_float * sin_expanded)
    )

    grad_key_pre_rope = (
        grad_key_states_float * cos_expanded +
        rotate_half_inverse(grad_key_states_float * sin_expanded)
    )
    
    # Stack gradients
    grad_qkv = torch.stack([grad_query_pre_rope, grad_key_pre_rope, grad_value_states], dim=1)
    grad_qkv = grad_qkv.reshape(seq_length, 3 * hidden_size)
    
    # Gradient through QKV projection
    _aux_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(_aux_stream):
        grad_qkv_bias = grad_qkv.sum(dim=0)
    grad_hidden_states = grad_qkv @ qkv_weight
    grad_qkv_weight = grad_qkv.t() @ hidden_states
    torch.cuda.current_stream().wait_stream(_aux_stream)
    
    return grad_hidden_states, grad_qkv_weight, grad_qkv_bias, grad_proj_weight, grad_proj_bias
