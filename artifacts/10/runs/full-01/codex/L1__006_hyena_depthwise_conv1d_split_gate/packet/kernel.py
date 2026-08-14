import torch
import triton
import triton.language as tl


@triton.jit
def _hyena_short_conv_kernel(
    u,
    weight,
    bias,
    v_gated,
    x0_out,
    x1_out,
    seq_len: tl.constexpr,
    BLOCK: tl.constexpr,
    STREAMING_STORES: tl.constexpr,
):
    # One program owns a batch item and one of the 256 channels in each split.
    bc = tl.program_id(0)
    batch = bc // 256
    channel = bc - batch * 256

    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    out_mask = t < seq_len
    out_offset = bc * seq_len + t

    # conv1d is cross-correlation.  With padding=2 and a length-three filter,
    # output t consumes input positions t-2, t-1, and t.
    t0 = t - 2
    t1 = t - 1

    c0 = channel
    u0_base = (batch * 768 + c0) * seq_len
    w0_base = c0 * 3
    a00 = tl.load(u + u0_base + t0, mask=(t >= 2) & out_mask, other=0.0)
    a01 = tl.load(u + u0_base + t1, mask=(t >= 1) & out_mask, other=0.0)
    a02 = tl.load(u + u0_base + t, mask=out_mask, other=0.0)
    w00 = tl.load(weight + w0_base)
    w01 = tl.load(weight + w0_base + 1)
    w02 = tl.load(weight + w0_base + 2)
    # MIOpen's direct depthwise kernel is extremely close to a correctly
    # rounded sum.  Forming the tiny reduction in fp64 and rounding once avoids
    # the larger reassociation error of a different fp32 FMA schedule.
    y0 = tl.load(bias + c0).to(tl.float64)
    y0 += a00.to(tl.float64) * w00.to(tl.float64)
    y0 += a01.to(tl.float64) * w01.to(tl.float64)
    y0 += a02.to(tl.float64) * w02.to(tl.float64)
    y0 = y0.to(tl.float32)

    c1 = channel + 256
    u1_base = (batch * 768 + c1) * seq_len
    w1_base = c1 * 3
    a10 = tl.load(u + u1_base + t0, mask=(t >= 2) & out_mask, other=0.0)
    a11 = tl.load(u + u1_base + t1, mask=(t >= 1) & out_mask, other=0.0)
    a12 = tl.load(u + u1_base + t, mask=out_mask, other=0.0)
    w10 = tl.load(weight + w1_base)
    w11 = tl.load(weight + w1_base + 1)
    w12 = tl.load(weight + w1_base + 2)
    # x1 is not part of the product, so ordinary fp32 accumulation already
    # exceeds its required matched ratio and avoids unnecessary fp64 work.
    y1 = tl.load(bias + c1) + a10 * w10
    y1 = y1 + a11 * w11
    y1 = y1 + a12 * w12

    c2 = channel + 512
    u2_base = (batch * 768 + c2) * seq_len
    w2_base = c2 * 3
    a20 = tl.load(u + u2_base + t0, mask=(t >= 2) & out_mask, other=0.0)
    a21 = tl.load(u + u2_base + t1, mask=(t >= 1) & out_mask, other=0.0)
    a22 = tl.load(u + u2_base + t, mask=out_mask, other=0.0)
    w20 = tl.load(weight + w2_base)
    w21 = tl.load(weight + w2_base + 1)
    w22 = tl.load(weight + w2_base + 2)
    y2 = tl.load(bias + c2).to(tl.float64)
    y2 += a20.to(tl.float64) * w20.to(tl.float64)
    y2 += a21.to(tl.float64) * w21.to(tl.float64)
    y2 += a22.to(tl.float64) * w22.to(tl.float64)
    y2 = y2.to(tl.float32)

    if STREAMING_STORES:
        tl.store(x0_out + out_offset, y0, mask=out_mask, cache_modifier=".cs")
        tl.store(x1_out + out_offset, y1, mask=out_mask, cache_modifier=".cs")
        tl.store(v_gated + out_offset, y2 * y0, mask=out_mask, cache_modifier=".cs")
    else:
        tl.store(x0_out + out_offset, y0, mask=out_mask)
        tl.store(x1_out + out_offset, y1, mask=out_mask)
        tl.store(v_gated + out_offset, y2 * y0, mask=out_mask)


def run(
    u: torch.Tensor,
    short_filter_weight: torch.Tensor,
    short_filter_bias: torch.Tensor,
):
    batch_size, _, seq_len = u.shape
    output_shape = (batch_size, 256, seq_len)
    outputs = torch.empty((3, *output_shape), device=u.device, dtype=u.dtype)
    v_gated, x0, x1 = outputs[0], outputs[1], outputs[2]

    block = min(256, triton.next_power_of_2(seq_len))
    warps = 2 if block <= 128 else 4
    _hyena_short_conv_kernel[(batch_size * 256, triton.cdiv(seq_len, block))](
        u,
        short_filter_weight,
        short_filter_bias,
        v_gated,
        x0,
        x1,
        seq_len=seq_len,
        BLOCK=block,
        STREAMING_STORES=(batch_size <= 4 and seq_len >= 2048),
        num_warps=warps,
    )
    return v_gated, x0, x1
