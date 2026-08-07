import torch
import torch.nn.functional as F


def _run(
    hidden_states,
    layer_norm1_weight,
    layer_norm1_bias,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    out_proj_weight,
    out_proj_bias,
    layer_norm2_weight,
    layer_norm2_bias,
    fc1_weight,
    fc1_bias,
    fc2_weight,
    fc2_bias,
    layer_norm_eps: float,
):
    batch_size, seq_len, hidden_size = hidden_states.shape

    residual = hidden_states
    mean = hidden_states.mean(dim=-1, keepdim=True)
    centered = hidden_states - mean
    var = (centered**2).mean(dim=-1, keepdim=True)
    var.add_(layer_norm_eps).sqrt_()
    hidden_states = centered.div_(var)
    torch.addcmul(
        layer_norm1_bias,
        hidden_states,
        layer_norm1_weight,
        out=hidden_states,
    )

    queries = torch.addmm(
        q_proj_bias,
        hidden_states.reshape(-1, hidden_size),
        q_proj_weight.t(),
        alpha=0.125,
        beta=0.125,
    ).reshape(batch_size, seq_len, hidden_size)
    keys = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    values = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    queries = queries.view(batch_size, seq_len, 12, 64).transpose(1, 2)
    keys = keys.view(batch_size, seq_len, 12, 64).transpose(1, 2)
    values = values.view(batch_size, seq_len, 12, 64).transpose(1, 2)
    attn_weights = torch.matmul(queries, keys.transpose(-1, -2))
    torch.softmax(
        attn_weights, dim=-1, dtype=torch.float32, out=attn_weights
    )
    torch.matmul(attn_weights, values, out=queries)
    attn_output = queries.transpose(1, 2).reshape(
        batch_size, seq_len, hidden_size
    )
    projection_out = keys.transpose(1, 2).reshape(
        batch_size, seq_len, hidden_size
    )
    torch.addmm(
        out_proj_bias,
        attn_output.reshape(-1, hidden_size),
        out_proj_weight.t(),
        out=projection_out.reshape(-1, hidden_size),
    )
    hidden_states = projection_out.add_(residual)

    residual = hidden_states
    mean = hidden_states.mean(dim=-1, keepdim=True)
    centered = hidden_states - mean
    var = (centered**2).mean(dim=-1, keepdim=True)
    var.add_(layer_norm_eps).sqrt_()
    hidden_states = centered.div_(var)
    torch.addcmul(
        layer_norm2_bias,
        hidden_states,
        layer_norm2_weight,
        out=hidden_states,
    )
    fc2_out = hidden_states
    hidden_states = F.linear(hidden_states, fc1_weight, fc1_bias)
    gate = hidden_states.mul(1.702).sigmoid_()
    hidden_states.mul_(gate)
    torch.addmm(
        fc2_bias,
        hidden_states.reshape(-1, 3072),
        fc2_weight.t(),
        out=fc2_out.reshape(-1, hidden_size),
    )
    return fc2_out.add_(residual)


run = torch.jit.script(_run)
