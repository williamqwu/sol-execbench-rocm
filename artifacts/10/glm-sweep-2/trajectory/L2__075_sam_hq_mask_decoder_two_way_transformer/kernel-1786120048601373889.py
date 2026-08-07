import torch
import torch.nn.functional as F

NUM_HEADS = 8
HIDDEN = 256
INTERNAL = 128
EPS = 1e-6

_GRAPH_CACHE: dict = {}


def _compute(
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
    batch_size, point_batch_size, n_query_tokens, hidden_size = queries.shape
    _, _, n_key_tokens, _ = keys.shape
    head_dim_self = hidden_size // NUM_HEADS
    head_dim_cross = INTERNAL // NUM_HEADS

    def separate_heads(x, num_heads, head_dim):
        b, pb, n, c = x.shape
        x = x.reshape(b * pb, n, num_heads, head_dim)
        return x.transpose(1, 2)

    def recombine_heads(x, point_batch_size):
        bpb, nh, n, hd = x.shape
        b = bpb // point_batch_size
        return x.transpose(1, 2).reshape(b, point_batch_size, n, nh * hd)

    def attention(q, k, v, num_heads, head_dim, point_batch_size):
        scaling = head_dim ** -0.5
        q = separate_heads(q, num_heads, head_dim)
        k = separate_heads(k, num_heads, head_dim)
        v = separate_heads(v, num_heads, head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        return recombine_heads(attn_output, point_batch_size)

    def layer_norm(x, weight, bias):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + EPS) * weight + bias

    if skip_first_layer_pe:
        q_input = queries
    else:
        q_input = queries + query_point_embedding

    q_proj = F.linear(q_input, self_attn_q_weight, self_attn_q_bias)
    k_proj = F.linear(q_input, self_attn_k_weight, self_attn_k_bias)
    v_proj = F.linear(queries, self_attn_v_weight, self_attn_v_bias)

    attn_out = attention(q_proj, k_proj, v_proj, NUM_HEADS, head_dim_self, point_batch_size)
    attn_out = F.linear(attn_out, self_attn_out_weight, self_attn_out_bias)
    queries = queries + attn_out
    queries = layer_norm(queries, layer_norm1_weight, layer_norm1_bias)

    q_input = queries + query_point_embedding
    k_input = keys + key_point_embedding

    q_proj = F.linear(q_input, cross_t2i_q_weight, cross_t2i_q_bias)
    k_proj = F.linear(k_input, cross_t2i_k_weight, cross_t2i_k_bias)
    v_proj = F.linear(keys, cross_t2i_v_weight, cross_t2i_v_bias)

    attn_out = attention(q_proj, k_proj, v_proj, NUM_HEADS, head_dim_cross, point_batch_size)
    attn_out = F.linear(attn_out, cross_t2i_out_weight, cross_t2i_out_bias)
    queries = queries + attn_out
    queries = layer_norm(queries, layer_norm2_weight, layer_norm2_bias)

    mlp_out = F.linear(queries, mlp_lin1_weight, mlp_lin1_bias)
    mlp_out = F.relu(mlp_out)
    mlp_out = F.linear(mlp_out, mlp_lin2_weight, mlp_lin2_bias)
    queries = queries + mlp_out
    queries = layer_norm(queries, layer_norm3_weight, layer_norm3_bias)

    q_input = queries + query_point_embedding
    k_input = keys + key_point_embedding

    q_proj = F.linear(k_input, cross_i2t_q_weight, cross_i2t_q_bias)
    k_proj = F.linear(q_input, cross_i2t_k_weight, cross_i2t_k_bias)
    v_proj = F.linear(queries, cross_i2t_v_weight, cross_i2t_v_bias)

    attn_out = attention(q_proj, k_proj, v_proj, NUM_HEADS, head_dim_cross, point_batch_size)
    attn_out = F.linear(attn_out, cross_i2t_out_weight, cross_i2t_out_bias)
    keys = keys + attn_out
    keys = layer_norm(keys, layer_norm4_weight, layer_norm4_bias)

    return queries, keys


# All tensor args in order (everything except the trailing bool skip_first_layer_pe).
_ARG_NAMES = [
    "queries", "keys", "query_point_embedding", "key_point_embedding",
    "self_attn_q_weight", "self_attn_q_bias", "self_attn_k_weight", "self_attn_k_bias",
    "self_attn_v_weight", "self_attn_v_bias", "self_attn_out_weight", "self_attn_out_bias",
    "layer_norm1_weight", "layer_norm1_bias",
    "cross_t2i_q_weight", "cross_t2i_q_bias", "cross_t2i_k_weight", "cross_t2i_k_bias",
    "cross_t2i_v_weight", "cross_t2i_v_bias", "cross_t2i_out_weight", "cross_t2i_out_bias",
    "layer_norm2_weight", "layer_norm2_bias",
    "mlp_lin1_weight", "mlp_lin1_bias", "mlp_lin2_weight", "mlp_lin2_bias",
    "layer_norm3_weight", "layer_norm3_bias",
    "cross_i2t_q_weight", "cross_i2t_q_bias", "cross_i2t_k_weight", "cross_i2t_k_bias",
    "cross_i2t_v_weight", "cross_i2t_v_bias", "cross_i2t_out_weight", "cross_i2t_out_bias",
    "layer_norm4_weight", "layer_norm4_bias",
]


@torch.no_grad()
def run(
    queries: torch.Tensor,
    keys: torch.Tensor,
    query_point_embedding: torch.Tensor,
    key_point_embedding: torch.Tensor,
    self_attn_q_weight: torch.Tensor,
    self_attn_q_bias: torch.Tensor,
    self_attn_k_weight: torch.Tensor,
    self_attn_k_bias: torch.Tensor,
    self_attn_v_weight: torch.Tensor,
    self_attn_v_bias: torch.Tensor,
    self_attn_out_weight: torch.Tensor,
    self_attn_out_bias: torch.Tensor,
    layer_norm1_weight: torch.Tensor,
    layer_norm1_bias: torch.Tensor,
    cross_t2i_q_weight: torch.Tensor,
    cross_t2i_q_bias: torch.Tensor,
    cross_t2i_k_weight: torch.Tensor,
    cross_t2i_k_bias: torch.Tensor,
    cross_t2i_v_weight: torch.Tensor,
    cross_t2i_v_bias: torch.Tensor,
    cross_t2i_out_weight: torch.Tensor,
    cross_t2i_out_bias: torch.Tensor,
    layer_norm2_weight: torch.Tensor,
    layer_norm2_bias: torch.Tensor,
    mlp_lin1_weight: torch.Tensor,
    mlp_lin1_bias: torch.Tensor,
    mlp_lin2_weight: torch.Tensor,
    mlp_lin2_bias: torch.Tensor,
    layer_norm3_weight: torch.Tensor,
    layer_norm3_bias: torch.Tensor,
    cross_i2t_q_weight: torch.Tensor,
    cross_i2t_q_bias: torch.Tensor,
    cross_i2t_k_weight: torch.Tensor,
    cross_i2t_k_bias: torch.Tensor,
    cross_i2t_v_weight: torch.Tensor,
    cross_i2t_v_bias: torch.Tensor,
    cross_i2t_out_weight: torch.Tensor,
    cross_i2t_out_bias: torch.Tensor,
    layer_norm4_weight: torch.Tensor,
    layer_norm4_bias: torch.Tensor,
    skip_first_layer_pe: bool,
):
    args = [
        queries, keys, query_point_embedding, key_point_embedding,
        self_attn_q_weight, self_attn_q_bias, self_attn_k_weight, self_attn_k_bias,
        self_attn_v_weight, self_attn_v_bias, self_attn_out_weight, self_attn_out_bias,
        layer_norm1_weight, layer_norm1_bias,
        cross_t2i_q_weight, cross_t2i_q_bias, cross_t2i_k_weight, cross_t2i_k_bias,
        cross_t2i_v_weight, cross_t2i_v_bias, cross_t2i_out_weight, cross_t2i_out_bias,
        layer_norm2_weight, layer_norm2_bias,
        mlp_lin1_weight, mlp_lin1_bias, mlp_lin2_weight, mlp_lin2_bias,
        layer_norm3_weight, layer_norm3_bias,
        cross_i2t_q_weight, cross_i2t_q_bias, cross_i2t_k_weight, cross_i2t_k_bias,
        cross_i2t_v_weight, cross_i2t_v_bias, cross_i2t_out_weight, cross_i2t_out_bias,
        layer_norm4_weight, layer_norm4_bias,
    ]
    q_shape = queries.shape
    k_shape = keys.shape
    cache_key = (q_shape, k_shape, bool(skip_first_layer_pe))

    entry = _GRAPH_CACHE.get(cache_key)
    if entry is None:
        # First call for this shape: warm up the allocator/cublas workspace on a
        # side stream, then capture a CUDA graph of the exact eager computation.
        # The graph replays the identical kernels, so output is bit-exact.
        static_inputs = [t.clone() for t in args]

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.default_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                _compute(*static_inputs, skip_first_layer_pe)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_q, out_k = _compute(*static_inputs, skip_first_layer_pe)

        entry = {"graph": graph, "static": static_inputs, "out_q": out_q, "out_k": out_k}
        _GRAPH_CACHE[cache_key] = entry

    static = entry["static"]
    for s, a in zip(static, args):
        s.copy_(a)
    entry["graph"].replay()
    return entry["out_q"], entry["out_k"]
