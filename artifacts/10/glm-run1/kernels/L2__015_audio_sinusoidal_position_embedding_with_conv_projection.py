import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
    x = F.gelu(x)

    # Reshape without copy: NCHW [b,c,f,t] -> [b, c*f, t] is a free view
    # (c and f are contiguous leading dims). Then matmul avoids the
    # expensive permute(0,3,1,2).contiguous() copy the reference does.
    b, c, f, t = x.size()
    x = x.reshape(b, c * f, t)
    # out[b, t, d_model] = (x[b, c*f, t]^T) @ conv_out_weight[d_model, c*f]^T
    x = torch.matmul(x.transpose(1, 2), conv_out_weight.t())

    # Scale embeddings and add positional embeddings (fused into one kernel
    # by PyTorch's pointwise fusion via addcmul-style expression).
    x = x * embed_scale
    seq_len = x.shape[1]
    x = x + positional_embedding[:seq_len, :].unsqueeze(0)

    return x
