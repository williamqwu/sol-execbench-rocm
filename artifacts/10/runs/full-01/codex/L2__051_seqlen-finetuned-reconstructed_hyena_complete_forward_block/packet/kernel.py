import math

import torch
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.accumulated_cache_size_limit = 64


@torch.no_grad()
def _impl(
    hidden_states,
    norm1_weight,
    norm1_bias,
    norm2_weight,
    norm2_bias,
    in_proj_weight,
    in_proj_bias,
    short_conv_weight,
    short_conv_bias,
    filter_linear1_weight,
    filter_linear1_bias,
    sin_freq,
    filter_linear2_weight,
    filter_linear2_bias,
    filter_linear3_weight,
    filter_linear3_bias,
    filter_linear_final_weight,
    filter_bias,
    exp_mod_deltas,
    out_proj_weight,
    out_proj_bias,
    mlp_fc1_weight,
    mlp_fc1_bias,
    mlp_fc2_weight,
    mlp_fc2_bias,
    layer_norm_eps,
    exp_mod_shift,
):
    d_model = 256
    inner_width = 768
    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, 32768)

    residual = hidden_states.to(torch.float32)
    mean = residual.mean(dim=-1, keepdim=True)
    var = residual.var(dim=-1, keepdim=True, unbiased=False)
    normed = (residual - mean) / torch.sqrt(var + layer_norm_eps)
    normed = normed * norm1_weight + norm1_bias

    u = F.linear(normed, in_proj_weight, in_proj_bias)
    u_pad = F.pad(u, (0, 0, 2, 0))
    w0 = short_conv_weight[:, 0, 0]
    w1 = short_conv_weight[:, 0, 1]
    w2 = short_conv_weight[:, 0, 2]
    uc = (
        (u_pad[:, :l_filter] * w0 + u_pad[:, 1:l_filter + 1] * w1)
        + u_pad[:, 2:l_filter + 2] * w2
    ) + short_conv_bias
    x0, x1, v = (
        part.transpose(1, 2) for part in uc.split(d_model, dim=2)
    )

    t = torch.linspace(0, 1, l_filter, device=hidden_states.device)[None, :, None]
    t_rescaled = torch.linspace(
        0, l_filter - 1, l_filter, device=hidden_states.device
    )[None, :, None]
    w = 2 * math.pi * t_rescaled / l_filter
    f = torch.linspace(1e-4, 1, 2, device=hidden_states.device)[None, None]
    z = torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], dim=-1)

    h = F.linear(z, filter_linear1_weight, filter_linear1_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear2_weight, filter_linear2_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear3_weight, filter_linear3_bias)
    h = torch.sin(sin_freq * h)
    h = F.linear(h, filter_linear_final_weight, None)
    decay = torch.exp(-t * exp_mod_deltas.abs())
    h = h * (decay + exp_mod_shift)
    h = h + filter_bias.view(1, 1, d_model)
    k = h.transpose(0, 1).reshape(1, d_model, l_filter)

    v = v * x1
    fft_size = 1 << (2 * l_filter - 1).bit_length()
    k_f = torch.fft.rfft(k[0].to(torch.float32), n=fft_size) / fft_size
    v_f = torch.fft.rfft(v.to(torch.float32), n=fft_size)
    y = torch.fft.irfft(v_f * k_f, n=fft_size, norm="forward")[..., :l_filter]
    v = y + v * filter_bias.reshape(1, d_model, 1)
    y = (v * x0).transpose(1, 2)

    if l_filter < seq_len:
        y = F.pad(y, (0, 0, 0, seq_len - l_filter))

    hyena_out = F.linear(y, out_proj_weight, out_proj_bias)
    residual = hyena_out + residual
    residual_float = residual.to(torch.float32)
    mean = residual_float.mean(dim=-1, keepdim=True)
    var = residual_float.var(dim=-1, keepdim=True, unbiased=False)
    normed = (residual_float - mean) / torch.sqrt(var + layer_norm_eps)
    normed = normed * norm2_weight + norm2_bias

    mlp_out = F.linear(normed, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out, approximate="tanh")
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    return mlp_out + residual_float


# Inductor keeps rocBLAS/rocFFT/MIOpen for the heavy operations and folds the
# surrounding elementwise chains.
_compiled_impl = torch.compile(_impl, fullgraph=True, dynamic=False)


@torch.no_grad()
def run(
    hidden_states,
    norm1_weight,
    norm1_bias,
    norm2_weight,
    norm2_bias,
    in_proj_weight,
    in_proj_bias,
    short_conv_weight,
    short_conv_bias,
    filter_linear1_weight,
    filter_linear1_bias,
    sin_freq,
    filter_linear2_weight,
    filter_linear2_bias,
    filter_linear3_weight,
    filter_linear3_bias,
    filter_linear_final_weight,
    filter_bias,
    exp_mod_deltas,
    out_proj_weight,
    out_proj_bias,
    mlp_fc1_weight,
    mlp_fc1_bias,
    mlp_fc2_weight,
    mlp_fc2_bias,
    layer_norm_eps,
    exp_mod_shift,
):
    return _compiled_impl(
        hidden_states,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        in_proj_weight,
        in_proj_bias,
        short_conv_weight,
        short_conv_bias,
        filter_linear1_weight,
        filter_linear1_bias,
        sin_freq,
        filter_linear2_weight,
        filter_linear2_bias,
        filter_linear3_weight,
        filter_linear3_bias,
        filter_linear_final_weight,
        filter_bias,
        exp_mod_deltas,
        out_proj_weight,
        out_proj_bias,
        mlp_fc1_weight,
        mlp_fc1_bias,
        mlp_fc2_weight,
        mlp_fc2_bias,
        layer_norm_eps,
        exp_mod_shift,
    )
