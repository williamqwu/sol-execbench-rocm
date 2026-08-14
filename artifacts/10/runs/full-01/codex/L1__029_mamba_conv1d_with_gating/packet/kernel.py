import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _causal_conv_silu(
    projected_ptr,
    mask_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Both projection and output storage are physically [time, channel].  The
    # Python wrapper returns a transposed view, so every GPU load/store here is
    # contiguous while the visible result still has shape [B, C, L].
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_b = tl.program_id(2)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    valid_t = t < seq_len
    acc = tl.zeros((BLOCK_T, BLOCK_C), tl.float32)

    # F.conv1d is cross-correlation: w[0] multiplies x[t-3], and w[3]
    # multiplies x[t].  Round the masked projection to bf16 before the dot,
    # exactly as the reference's first pointwise multiplication does.
    for k in tl.static_range(0, 4):
        source_t = t + k - 3
        source_valid = (source_t >= 0) & (source_t < seq_len)
        p_offsets = (
            (pid_b * seq_len + source_t[:, None]) * 32768
            + c[None, :]
        )
        x = tl.load(projected_ptr + p_offsets, mask=source_valid[:, None], other=0.0)
        m = tl.load(
            mask_ptr + pid_b * seq_len + source_t,
            mask=source_valid,
            other=0.0,
        )
        x = (x.to(tl.float32) * m[:, None].to(tl.float32)).to(tl.bfloat16)
        w = tl.load(weight_ptr + c * 4 + k)
        acc += x.to(tl.float32) * w[None, :].to(tl.float32)

    bias = tl.load(bias_ptr + c).to(tl.float32)
    conv = (acc + bias[None, :]).to(tl.bfloat16)

    # sigmoid itself and each following multiply materialize bf16 tensors in
    # the reference, so retain both intermediate rounding points.
    sig = tl.sigmoid(conv.to(tl.float32)).to(tl.bfloat16)
    activated = (
        conv.to(tl.float32) * sig.to(tl.float32)
    ).to(tl.bfloat16)
    final_mask = tl.load(
        mask_ptr + pid_b * seq_len + t,
        mask=valid_t,
        other=0.0,
    )
    result = (
        activated.to(tl.float32) * final_mask[:, None].to(tl.float32)
    ).to(tl.bfloat16)

    out_offsets = (
        (pid_b * seq_len + t[:, None]) * 16384
        + c[None, :]
    )
    tl.store(
        output_ptr + out_offsets,
        result,
        mask=valid_t[:, None],
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
):
    batch_size, seq_len, _ = hidden_states.shape

    # Keep the vendor GEMM: it is both faster than a generic Triton matmul for
    # this large fixed K/N and reproduces the reference's bf16 projection.
    projected = F.linear(hidden_states, in_proj_weight, in_proj_bias)
    output_storage = torch.empty(
        (batch_size, seq_len, 16384),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    block_t = 64
    block_c = 64
    grid = (triton.cdiv(16384, block_c), triton.cdiv(seq_len, block_t), batch_size)
    _causal_conv_silu[grid](
        projected,
        attention_mask,
        conv1d_weight,
        conv1d_bias,
        output_storage,
        seq_len=seq_len,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        num_warps=2,
    )

    output = output_storage.transpose(1, 2)
    gate = projected[:, :, 16384:].transpose(1, 2)
    return output, gate
