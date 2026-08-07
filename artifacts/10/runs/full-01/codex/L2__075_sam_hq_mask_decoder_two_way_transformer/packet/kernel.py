import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _final_residual_layer_norm(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * BLOCK + cols
    x = tl.load(x_ptr + offsets) + tl.load(residual_ptr + offsets)
    mean = tl.sum(x, axis=0) * (1.0 / BLOCK)
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / BLOCK)
    out = centered / tl.sqrt(var + 1.0e-6)
    out = out * tl.load(weight_ptr + cols)
    tl.store(out_ptr + offsets, out)
    tl.debug_barrier()
    out = tl.load(out_ptr + offsets)
    out = out + tl.load(bias_ptr + cols)
    tl.store(out_ptr + offsets, out)


@torch.no_grad()
def run(
    queries,
    keys,
    query_point_embedding,
    key_point_embedding,
    self_attn_q_weight,
    self_attn_q_bias,
    self_attn_k_weight,
    self_attn_k_bias,
    self_attn_v_weight,
    self_attn_v_bias,
    self_attn_out_weight,
    self_attn_out_bias,
    layer_norm1_weight,
    layer_norm1_bias,
    cross_t2i_q_weight,
    cross_t2i_q_bias,
    cross_t2i_k_weight,
    cross_t2i_k_bias,
    cross_t2i_v_weight,
    cross_t2i_v_bias,
    cross_t2i_out_weight,
    cross_t2i_out_bias,
    layer_norm2_weight,
    layer_norm2_bias,
    mlp_lin1_weight,
    mlp_lin1_bias,
    mlp_lin2_weight,
    mlp_lin2_bias,
    layer_norm3_weight,
    layer_norm3_bias,
    cross_i2t_q_weight,
    cross_i2t_q_bias,
    cross_i2t_k_weight,
    cross_i2t_k_bias,
    cross_i2t_v_weight,
    cross_i2t_v_bias,
    cross_i2t_out_weight,
    cross_i2t_out_bias,
    layer_norm4_weight,
    layer_norm4_bias,
    skip_first_layer_pe,
):
    point_batch_size = queries.shape[1]

    def attention(q, k, v, head_dim):
        b, pb, nq, _ = q.shape
        nk = k.shape[2]
        q = q.reshape(b * pb, nq, 8, head_dim).transpose(1, 2)
        k = k.reshape(b * pb, nk, 8, head_dim).transpose(1, 2)
        v = v.reshape(b * pb, nk, 8, head_dim).transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-2, -1))
        weights.mul_(head_dim ** -0.5)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(weights, v)
        return out.transpose(1, 2).reshape(b, point_batch_size, nq, 8 * head_dim)

    def layer_norm(x, weight, bias):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered ** 2).mean(dim=-1, keepdim=True)
        var.add_(1e-6).sqrt_()
        centered.div_(var).mul_(weight).add_(bias)
        return centered

    q_input = queries if skip_first_layer_pe else queries + query_point_embedding
    q_proj = F.linear(q_input, self_attn_q_weight, self_attn_q_bias)
    k_proj = F.linear(q_input, self_attn_k_weight, self_attn_k_bias)
    v_proj = F.linear(queries, self_attn_v_weight, self_attn_v_bias)
    attn_out = attention(q_proj, k_proj, v_proj, 32)
    attn_out = F.linear(attn_out, self_attn_out_weight, self_attn_out_bias)
    attn_out.add_(queries)
    queries = layer_norm(attn_out, layer_norm1_weight, layer_norm1_bias)

    q_input = queries + query_point_embedding
    k_input = keys + key_point_embedding
    q_proj = F.linear(q_input, cross_t2i_q_weight, cross_t2i_q_bias)
    k_proj = F.linear(k_input, cross_t2i_k_weight, cross_t2i_k_bias)
    v_proj = F.linear(keys, cross_t2i_v_weight, cross_t2i_v_bias)
    attn_out = attention(q_proj, k_proj, v_proj, 16)
    attn_out = F.linear(attn_out, cross_t2i_out_weight, cross_t2i_out_bias)
    attn_out.add_(queries)
    queries = layer_norm(attn_out, layer_norm2_weight, layer_norm2_bias)

    mlp_out = F.linear(queries, mlp_lin1_weight, mlp_lin1_bias)
    mlp_out.relu_()
    mlp_out = F.linear(mlp_out, mlp_lin2_weight, mlp_lin2_bias)
    mlp_out.add_(queries)
    queries = layer_norm(mlp_out, layer_norm3_weight, layer_norm3_bias)

    q_input = queries + query_point_embedding
    q_proj = F.linear(k_input, cross_i2t_q_weight, cross_i2t_q_bias)
    k_proj = F.linear(q_input, cross_i2t_k_weight, cross_i2t_k_bias)
    v_proj = F.linear(queries, cross_i2t_v_weight, cross_i2t_v_bias)
    attn_out = attention(q_proj, k_proj, v_proj, 16)
    residual = F.linear(attn_out, cross_i2t_out_weight, cross_i2t_out_bias)
    output_keys = torch.empty_like(keys)
    n_key_rows = keys.numel() // 256
    _final_residual_layer_norm[(n_key_rows,)](
        keys,
        residual,
        layer_norm4_weight,
        layer_norm4_bias,
        output_keys,
        BLOCK=256,
        num_warps=1,
        waves_per_eu=4 if n_key_rows >= 32768 else 0,
    )
    return queries, output_keys
