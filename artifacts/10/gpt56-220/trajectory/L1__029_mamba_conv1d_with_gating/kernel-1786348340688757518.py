import torch
import torch.nn.functional as F

torch._dynamo.config.cache_size_limit = 32

@torch.compile(dynamic=False, mode="max-autotune-no-cudagraphs")
def _compiled(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
):
    """
    Fused Mamba conv1d with gating.
    
    Args:
        hidden_states: (batch_size, seq_len, 8192)
        attention_mask: (batch_size, seq_len)
        in_proj_weight: (32768, 8192)
        in_proj_bias: (32768,)
        conv1d_weight: (16384, 1, 4) - depthwise conv
        conv1d_bias: (16384,)
    
    Returns:
        output_hidden_states: (batch_size, 16384, seq_len)
        gate: (batch_size, 16384, seq_len)
    """
    batch_size, seq_len, _ = hidden_states.shape
    intermediate_size = 16384
    conv_kernel_size = 4
    
    # 1. Gated linear projection: (B, L, 8192) -> (B, L, 32768)
    projected = F.linear(hidden_states.reshape(batch_size * seq_len, -1),
                         in_proj_weight, in_proj_bias)
    projected = projected.reshape(batch_size, seq_len, 2 * intermediate_size)
    hidden_states_conv, gate = projected.transpose(1, 2).chunk(2, dim=1)
    
    # 4. Apply attention mask before convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    # 5. Causal 1D convolution with grouped convolution (depthwise)
    # Pad on the left for causal convolution
    w = conv1d_weight[:, 0, :]
    hidden_states_padded = F.pad(hidden_states_conv, (conv_kernel_size - 1, 0))
    hidden_states_conv = (
        hidden_states_padded[:, :, 0:seq_len] * w[:, 0][None, :, None]
        + hidden_states_padded[:, :, 1:seq_len + 1] * w[:, 1][None, :, None]
        + hidden_states_padded[:, :, 2:seq_len + 2] * w[:, 2][None, :, None]
        + hidden_states_padded[:, :, 3:seq_len + 3] * w[:, 3][None, :, None]
        + conv1d_bias[None, :, None]
    )
    
    # 6. Apply SiLU activation: silu(x) = x * sigmoid(x)
    hidden_states_conv = F.silu(hidden_states_conv)
    
    # 7. Apply attention mask after convolution
    hidden_states_conv = hidden_states_conv * attention_mask.unsqueeze(1)
    
    return hidden_states_conv, gate

@torch.inference_mode()
def run(hidden_states, attention_mask, in_proj_weight, in_proj_bias,
        conv1d_weight, conv1d_bias):
    return _compiled(hidden_states, attention_mask, in_proj_weight, in_proj_bias,
                     conv1d_weight, conv1d_bias)
