import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _top4_attention_kernel(scores, values, output):
    row = tl.program_id(0).to(tl.int64)
    key_offsets = tl.arange(0, 2048)
    row_scores = tl.load(
        scores + row * 1500 + key_offsets,
        mask=key_offsets < 1500,
        other=-float("inf"),
    )

    score0 = tl.max(row_scores, axis=0)
    index0 = tl.min(
        tl.where(row_scores == score0, key_offsets, 2048), axis=0
    )
    row_scores = tl.where(key_offsets == index0, -float("inf"), row_scores)
    score1 = tl.max(row_scores, axis=0)
    index1 = tl.min(
        tl.where(row_scores == score1, key_offsets, 2048), axis=0
    )
    row_scores = tl.where(key_offsets == index1, -float("inf"), row_scores)
    score2 = tl.max(row_scores, axis=0)
    index2 = tl.min(
        tl.where(row_scores == score2, key_offsets, 2048), axis=0
    )
    row_scores = tl.where(key_offsets == index2, -float("inf"), row_scores)
    score3 = tl.max(row_scores, axis=0)
    index3 = tl.min(
        tl.where(row_scores == score3, key_offsets, 2048), axis=0
    )

    score0 = score0.to(tl.float32)
    exp1 = tl.exp(score1.to(tl.float32) - score0)
    exp2 = tl.exp(score2.to(tl.float32) - score0)
    exp3 = tl.exp(score3.to(tl.float32) - score0)
    denominator = 1.0 + exp1 + exp2 + exp3
    probability0 = (1.0 / denominator).to(tl.bfloat16)
    probability1 = (exp1 / denominator).to(tl.bfloat16)
    probability2 = (exp2 / denominator).to(tl.bfloat16)
    probability3 = (exp3 / denominator).to(tl.bfloat16)

    head = row // 1500
    dimensions = tl.arange(0, 256)
    value_base = head * (1500 * 256) + dimensions
    value0 = tl.load(values + value_base + index0 * 256).to(tl.float32)
    value1 = tl.load(values + value_base + index1 * 256).to(tl.float32)
    value2 = tl.load(values + value_base + index2 * 256).to(tl.float32)
    value3 = tl.load(values + value_base + index3 * 256).to(tl.float32)
    result = (
        probability0.to(tl.float32) * value0
        + probability1.to(tl.float32) * value1
        + probability2.to(tl.float32) * value2
        + probability3.to(tl.float32) * value3
    )
    tl.store(output + row * 256 + dimensions, result)


@triton.jit
def _layer_norm_kernel(
    x,
    weight,
    bias,
    output,
    stride_batch,
    stride_sequence,
    stride_feature,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // 1500
    sequence = row - batch * 1500
    features = tl.arange(0, 8192)
    mask = features < 5120
    input_offsets = (
        batch * stride_batch
        + sequence * stride_sequence
        + features * stride_feature
    )
    values = tl.load(x + input_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / 5120.0
    centered = values - mean
    centered = tl.where(mask, centered, 0.0)
    variance = tl.sum(centered * centered, axis=0) / 5120.0
    normalized = centered / tl.sqrt(variance + 1e-5)
    scales = tl.load(weight + features, mask=mask).to(tl.float32)
    shifts = tl.load(bias + features, mask=mask).to(tl.float32)
    result = normalized * scales + shifts
    tl.store(output + row * 5120 + features, result, mask=mask)


def _reference_layer_norm(x, weight, bias):
    x32 = x.float()
    mean = x32.mean(dim=-1, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x32 - mean) / torch.sqrt(var + 1e-5)
    return (normalized * weight.float() + bias.float()).bfloat16()


def _triton_layer_norm(x, weight, bias):
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // 5120
    _layer_norm_kernel[(rows,)](
        x,
        weight,
        bias,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        num_warps=4,
    )
    return output


@torch.no_grad()
def run(
    input_features,
    conv1_weight,
    conv1_bias,
    conv2_weight,
    conv2_bias,
    embed_positions_weight,
    self_attn_layer_norm_weight,
    self_attn_layer_norm_bias,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    v_proj_weight,
    v_proj_bias,
    out_proj_weight,
    out_proj_bias,
    final_layer_norm_weight,
    final_layer_norm_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
):
    bsz = input_features.shape[0]
    x = F.gelu(F.conv1d(input_features, conv1_weight, conv1_bias, padding=1))
    x = F.gelu(F.conv1d(x, conv2_weight, conv2_bias, stride=2, padding=1))
    hidden_states = x.permute(0, 2, 1) + embed_positions_weight

    residual = hidden_states
    hidden_states = _reference_layer_norm(
        hidden_states,
        self_attn_layer_norm_weight,
        self_attn_layer_norm_bias,
    )
    q = F.linear(hidden_states, q_proj_weight, q_proj_bias) * 0.0625
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    q = q.view(bsz, 1500, 20, 256).transpose(1, 2).contiguous()
    k = k.view(bsz, 1500, 20, 256).transpose(1, 2).contiguous()
    v = v.view(bsz, 1500, 20, 256).transpose(1, 2).contiguous()
    attn_scores = torch.matmul(q, k.transpose(2, 3))
    attn_output = torch.empty_like(v)
    _top4_attention_kernel[(bsz * 20 * 1500,)](
        attn_scores, v, attn_output, num_warps=4
    )
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, 1500, 5120)
    hidden_states = residual + F.linear(attn_output, out_proj_weight, out_proj_bias)

    residual = hidden_states
    hidden_states = _triton_layer_norm(
        hidden_states,
        final_layer_norm_weight,
        final_layer_norm_bias,
    )
    hidden_states = F.gelu(F.linear(hidden_states, fc1_weight, fc1_bias))
    hidden_states = F.linear(hidden_states, fc2_weight, fc2_bias)
    return residual + hidden_states
