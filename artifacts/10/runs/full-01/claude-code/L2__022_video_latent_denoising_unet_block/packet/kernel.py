"""022_video_latent_denoising_unet_block -- MI355X.

The reference is extremely ill-conditioned: the "random" weights are unscaled
N(0,1), so attention logits reach ~1e4, softmax is effectively an argmax, and
the residual stream grows to ~1e4.  A relative perturbation of 1e-8 anywhere
early in the block flips enough argmaxes that only ~11% of output elements land
within tolerance (measured).  The allowed tolerance is floored at float32 eps.

Consequence: every arithmetic op must reproduce the reference *bit for bit*.
That rules out re-associating any reduction -- no fused Triton layernorm, no
flash attention, no torch.compile on the reductions.  Verified experimentally:
a Triton flash-attention pass, exact-erf GELU, and a Triton layernorm each
individually drop the match ratio to ~5-11%.

So the optimisation that is actually available is removing work that does not
change any value:

  * pure data movement -- the reference materialises several `.contiguous()`
    copies; the temporal branch does two full-tensor copies back to back where
    one suffices, and the spatial/cross branches copy where a strided linear
    input would do.
  * launch overhead -- ~40 kernel launches per call, several of them tiny.

Everything numeric below goes through the same torch op, with the same shapes
and strides, in the same order as the reference.
"""

import math

import torch
import torch.nn.functional as F

HID = 1024
NHEAD = 16
HD = HID // NHEAD
SCALE = 1.0 / math.sqrt(HD)
EPS = 1e-5


@torch.no_grad()
def run(
    video_latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    temporal_norm_weight: torch.Tensor,
    temporal_norm_bias: torch.Tensor,
    temporal_qkv_weight: torch.Tensor,
    temporal_qkv_bias: torch.Tensor,
    temporal_out_proj_weight: torch.Tensor,
    temporal_out_proj_bias: torch.Tensor,
    spatial_norm_weight: torch.Tensor,
    spatial_norm_bias: torch.Tensor,
    spatial_qkv_weight: torch.Tensor,
    spatial_qkv_bias: torch.Tensor,
    spatial_out_proj_weight: torch.Tensor,
    spatial_out_proj_bias: torch.Tensor,
    cross_attn_norm_weight: torch.Tensor,
    cross_attn_norm_bias: torch.Tensor,
    cross_attn_q_weight: torch.Tensor,
    cross_attn_q_bias: torch.Tensor,
    cross_attn_kv_weight: torch.Tensor,
    cross_attn_kv_bias: torch.Tensor,
    cross_attn_out_proj_weight: torch.Tensor,
    cross_attn_out_proj_bias: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    ffn_norm_bias: torch.Tensor,
    ffn_fc1_weight: torch.Tensor,
    ffn_fc1_bias: torch.Tensor,
    ffn_fc2_weight: torch.Tensor,
    ffn_fc2_bias: torch.Tensor,
    num_frames_scalar: int,
    num_spatial_tokens_scalar: int,
):
    B = video_latents.shape[0]
    Nv = video_latents.shape[1]
    T = text_embeddings.shape[1]
    Fr = int(num_frames_scalar)
    S = int(num_spatial_tokens_scalar)

    H3 = 3 * HID
    x = video_latents

    # SCALE is exactly 2**-3, so multiplying the Q half of a projection's
    # weight+bias by it is exact (no rounding), and every partial product in
    # the subsequent GEMM is exactly scaled -- so the logits come out bit
    # identical to computing `matmul(q, k^T) * SCALE`.  Verified with
    # torch.equal.  This deletes a read+write pass over the scores tensor,
    # which is the largest tensor in the block (B*NH*M*N, up to 2.7e8 elts).
    def _prescale_q(w, b, nq=HID):
        w2 = w.clone()
        w2[:nq] *= SCALE
        b2 = b.clone()
        b2[:nq] *= SCALE
        return w2, b2

    # ================= 1. Temporal self-attention =================
    residual = x
    x_norm = F.layer_norm(x, (HID,), temporal_norm_weight, temporal_norm_bias, EPS)
    tw, tb = _prescale_q(temporal_qkv_weight, temporal_qkv_bias)
    qkv = F.linear(x_norm, tw, tb)

    # (B, F, S, 3, H) -> per-head strided views (B*S, NH, F, HD).
    # The reference reaches the same tensors via permute+chunk+squeeze+view+
    # transpose; every one of those is a view, so this is the identical tensor
    # (asserted bit-exact in development), just spelled without the churn.
    bs = qkv.stride(0)  # = Fr * S * H3
    shape = (B * S, NHEAD, Fr, HD)
    strides = (H3, HD, S * H3, 1)

    def tview(offset):
        # batch index b*S+s -> b*bs + s*H3 ; head h -> h*HD ; frame f -> f*S*H3
        return qkv.as_strided(shape, strides, offset).view(B, S, NHEAD, Fr, HD) \
            if False else torch.as_strided(qkv, shape, strides, offset)

    # as_strided with a leading combined (B*S) axis needs a uniform stride, but
    # b advances by bs and s by H3, which are different. Build it as 5-D then
    # flatten the two leading axes (a view -- bs == S*H3 * Fr is a multiple).
    shape5 = (B, S, NHEAD, Fr, HD)
    strides5 = (bs, H3, HD, S * H3, 1)
    q = torch.as_strided(qkv, shape5, strides5, 0).flatten(0, 1)
    k = torch.as_strided(qkv, shape5, strides5, HID).flatten(0, 1)
    v = torch.as_strided(qkv, shape5, strides5, 2 * HID).flatten(0, 1)

    attn_scores = torch.matmul(q, k.transpose(-2, -1))  # SCALE folded into tw/tb
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)  # (B*S, NH, F, HD)

    # Reference: transpose(1,2).contiguous() -> view -> permute(0,2,1,3)
    # .contiguous() -> two full-size copies. One strided copy does it.
    attn_t = torch.empty(B * Nv, HID, device=x.device, dtype=x.dtype)
    attn_t.view(B, Fr, S, NHEAD, HD).permute(0, 2, 3, 1, 4).copy_(
        attn_output.view(B, S, NHEAD, Fr, HD)
    )

    output = F.linear(attn_t, temporal_out_proj_weight, temporal_out_proj_bias)
    x = output.view(B, Nv, HID) + residual

    # ================= 2. Spatial self-attention =================
    residual = x
    x_norm = F.layer_norm(x, (HID,), spatial_norm_weight, spatial_norm_bias, EPS)
    sw, sb = _prescale_q(spatial_qkv_weight, spatial_qkv_bias)
    qkv = F.linear(x_norm.view(B * Fr, S, HID), sw, sb)

    shape4 = (B * Fr, NHEAD, S, HD)
    strides4 = (S * H3, HD, H3, 1)
    q = torch.as_strided(qkv, shape4, strides4, 0)
    k = torch.as_strided(qkv, shape4, strides4, HID)
    v = torch.as_strided(qkv, shape4, strides4, 2 * HID)

    attn_scores = torch.matmul(q, k.transpose(-2, -1))  # SCALE folded into sw/sb
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)  # (B*F, NH, S, HD)

    attn_t = attn_output.transpose(1, 2).reshape(B * Nv, HID)
    output = F.linear(attn_t, spatial_out_proj_weight, spatial_out_proj_bias)
    x = output.view(B, Nv, HID) + residual

    # ================= 3. Cross-attention with text =================
    residual = x
    x_norm = F.layer_norm(x, (HID,), cross_attn_norm_weight, cross_attn_norm_bias, EPS)
    q = F.linear(x_norm, cross_attn_q_weight * SCALE, cross_attn_q_bias * SCALE)
    kv = F.linear(text_embeddings, cross_attn_kv_weight, cross_attn_kv_bias)

    H2 = 2 * HID
    q = q.view(B, Nv, NHEAD, HD).transpose(1, 2)
    kshape = (B, NHEAD, T, HD)
    kstr = (T * H2, HD, H2, 1)
    k = torch.as_strided(kv, kshape, kstr, 0)
    v = torch.as_strided(kv, kshape, kstr, HID)

    attn_scores = torch.matmul(q, k.transpose(-2, -1))  # SCALE folded into q proj
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_probs, v)  # (B, NH, Nv, HD)

    attn_t = attn_output.transpose(1, 2).reshape(B * Nv, HID)
    output = F.linear(attn_t, cross_attn_out_proj_weight, cross_attn_out_proj_bias)
    x = output.view(B, Nv, HID) + residual

    # ================= 4. Feedforward =================
    residual = x
    x_norm = F.layer_norm(x, (HID,), ffn_norm_weight, ffn_norm_bias, EPS)
    h = F.linear(x_norm, ffn_fc1_weight, ffn_fc1_bias)
    h = F.gelu(h)
    out = F.linear(h, ffn_fc2_weight, ffn_fc2_bias)
    out += residual
    return out
