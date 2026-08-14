import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _accumulate_routes(
    expert_output,
    inverse_route_order,
    slot_order,
    output,
    num_tokens,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    hidden = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    hidden_mask = hidden < HIDDEN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for rank in tl.static_range(8):
        slot = tl.load(slot_order + token * 8 + rank)
        original_route = slot * num_tokens + token
        sorted_route = tl.load(inverse_route_order + original_route)
        value = tl.load(
            expert_output + sorted_route * HIDDEN + hidden, mask=hidden_mask
        )
        acc += value
    tl.store(output + token * HIDDEN + hidden, acc, mask=hidden_mask)


def _rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _repeat_kv(x, n_rep):
    batch, heads, slen, dim = x.shape
    return x[:, :, None, :, :].expand(batch, heads, n_rep, slen, dim).reshape(
        batch, heads * n_rep, slen, dim
    )


@torch.no_grad()
def run(
    hidden_states,
    cos,
    sin,
    attention_mask,
    input_layernorm_weight,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    q_norm_weight,
    k_norm_weight,
    o_proj_weight,
    o_proj_bias,
    post_attention_layernorm_weight,
    router_weight,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
    rms_norm_eps,
):
    batch_size, seq_len, hidden_size = hidden_states.shape

    residual = hidden_states
    x = _rms_norm(hidden_states, input_layernorm_weight, rms_norm_eps)

    q = F.linear(x, q_proj_weight, q_proj_bias).view(batch_size, seq_len, 32, 128)
    k = F.linear(x, k_proj_weight, k_proj_bias).view(batch_size, seq_len, 4, 128)
    v = F.linear(x, v_proj_weight, v_proj_bias).view(batch_size, seq_len, 4, 128)

    q = _rms_norm(q, q_norm_weight, rms_norm_eps).transpose(1, 2)
    k = _rms_norm(k, k_norm_weight, rms_norm_eps).transpose(1, 2)
    v = v.transpose(1, 2)

    rotary_cos = cos.unsqueeze(1)
    rotary_sin = sin.unsqueeze(1)
    q = (q * rotary_cos) + (_rotate_half(q) * rotary_sin)
    k = (k * rotary_cos) + (_rotate_half(k) * rotary_sin)

    k = _repeat_kv(k, 8)
    v = _repeat_kv(v, 8)

    attn_weights = torch.matmul(q, k.transpose(2, 3))
    attn_weights = torch.add(
        attention_mask, attn_weights, alpha=(128 ** -0.5)
    )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        batch_size, seq_len, 4096
    )
    attn_output = F.linear(attn_output, o_proj_weight, o_proj_bias)

    x = residual + attn_output
    residual = x
    x = _rms_norm(x, post_attention_layernorm_weight, rms_norm_eps)
    x = x.view(-1, hidden_size)

    router_logits = F.linear(x, router_weight)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, 8, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    final = torch.zeros_like(x)

    # Group routes once.  Flattening the transposed table gives the same
    # slot-major order as torch.where() on the reference's [8, tokens] mask;
    # the stable sort therefore keeps every expert GEMM's row order unchanged.
    num_tokens = x.shape[0]
    route_experts = selected_experts.T.contiguous().view(-1)
    route_counts = torch.bincount(route_experts, minlength=128).to(torch.int32)
    route_offsets = route_counts.cumsum(0, dtype=torch.int32)
    route_order = torch.argsort(route_experts, stable=True)
    route_slot = torch.div(route_order, num_tokens, rounding_mode="floor")
    route_token = route_order.remainder(num_tokens)

    current = x[route_token]
    gate = torch.ops.aten._grouped_mm(
        current, expert_gate_weights.transpose(1, 2), route_offsets
    )
    up = torch.ops.aten._grouped_mm(
        current, expert_up_weights.transpose(1, 2), route_offsets
    )
    expert_out = F.silu(gate) * up
    expert_out = torch.ops.aten._grouped_mm(
        expert_out, expert_down_weights.transpose(1, 2), route_offsets
    )
    expert_out = expert_out * routing_weights[route_token, route_slot, None]

    # The reference adds a token's eight routes in ascending expert order.
    slot_order = torch.argsort(selected_experts, dim=1, stable=True)
    inverse_route_order = torch.empty_like(route_order)
    inverse_route_order[route_order] = torch.arange(
        route_order.numel(), device=x.device, dtype=route_order.dtype
    )
    _accumulate_routes[(num_tokens, triton.cdiv(hidden_size, 256))](
        expert_out,
        inverse_route_order,
        slot_order,
        final,
        num_tokens,
        HIDDEN=hidden_size,
        BLOCK=256,
        num_warps=4,
    )

    return residual + final.view(batch_size, seq_len, hidden_size)
