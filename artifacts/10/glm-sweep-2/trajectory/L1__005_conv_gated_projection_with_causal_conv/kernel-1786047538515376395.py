import torch
import torch.nn.functional as F


@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    batch_size, seq_len, hidden_size = x.shape
    conv_kernel_size = conv_weight.shape[2]

    # Step 1: Triple linear projection -> [B, S, 3H] (contiguous)
    BCx = F.linear(x, in_proj_weight, in_proj_bias)

    # Split along last (hidden) dim -> each [B, S, H], contiguous views
    B, C, x_proj = BCx.chunk(3, dim=-1)

    # Step 2: Element-wise gating -> [B, S, H]
    Bx = B * x_proj

    # Step 3: Causal depthwise conv1d via unfold, staying in [B, S, H] layout.
    # Left-pad the sequence dim so the conv is causal (no future tokens).
    Bx_padded = F.pad(Bx, (0, 0, conv_kernel_size - 1, 0))  # [B, S+K-1, H]
    # Sliding windows of length K along the sequence axis: [B, S, H, K]
    windows = Bx_padded.unfold(1, conv_kernel_size, 1)
    # conv_weight: [H, 1, K] -> squeeze -> [H, K]; broadcast over [B, S, H, K]
    w = conv_weight.squeeze(1)  # [H, K]
    conv_out = (windows * w).sum(-1) + conv_bias  # [B, S, H]

    # Step 4: Output gating with C -> [B, S, H]
    y = C * conv_out

    # Step 5: Final output projection (natural contiguous [B, S, H] layout)
    output = F.linear(y, out_proj_weight, out_proj_bias)

    return output
