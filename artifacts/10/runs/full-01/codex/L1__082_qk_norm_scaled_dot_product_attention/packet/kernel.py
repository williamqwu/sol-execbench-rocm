import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _normalize_and_pack_qkv(
    qkv,
    means,
    variances,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    packed,
    n_rows: tl.constexpr,
    seq_len: tl.constexpr,
    eps,
):
    # One wave owns one (batch, head, token) row.  In addition to applying
    # both normalizations, transpose V to the layout required by the BMMs.
    row = tl.program_id(0)
    batch = row // (24 * seq_len)
    rem = row - batch * (24 * seq_len)
    head = rem // seq_len
    token = rem - head * seq_len
    cols = tl.arange(0, 64)
    base = ((batch * seq_len + token) * 72 + head) * 64 + cols

    q = tl.load(qkv + base)
    k = tl.load(qkv + base + 24 * 64)
    v = tl.load(qkv + base + 48 * 64)
    moment_base = (batch * seq_len + token) * 48 + head
    q_mean = tl.load(means + moment_base)
    k_mean = tl.load(means + moment_base + 24)
    q_var = tl.load(variances + moment_base)
    k_var = tl.load(variances + moment_base + 24)
    qw = tl.load(q_weight + cols)
    qb = tl.load(q_bias + cols)
    kw = tl.load(k_weight + cols)
    kb = tl.load(k_bias + cols)

    # libdevice sqrt matches torch.sqrt; tl.sqrt uses the approximate native
    # instruction on AMD.  The explicit multiply instructions also prevent
    # LLVM from contracting the reference's separately-rounded mul and add.
    q = (q - q_mean) / tl.extra.libdevice.sqrt(q_var + eps)
    k = (k - k_mean) / tl.extra.libdevice.sqrt(k_var + eps)
    q = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [q, qw],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    k = tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [k, kw],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = (q + qb) * 0.125
    k = k + kb

    offset = row * 64 + cols
    tl.store(packed + offset, q)
    tl.store(packed + n_rows * 64 + offset, k)
    tl.store(packed + 2 * n_rows * 64 + offset, v)


@torch.no_grad()
def run(
    hidden_states,
    qkv_weight,
    qkv_bias,
    q_norm_weight,
    q_norm_bias,
    k_norm_weight,
    k_norm_bias,
    out_proj_weight,
    out_proj_bias,
    eps,
):
    batch_size, seq_len, dim = hidden_states.shape

    # linear folds the bias add into rocBLAS and is bit-identical to the two
    # reference operations for these float32 GEMMs.
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv_view = qkv.reshape(batch_size, seq_len, 3, 24, 64)
    qk = qkv_view[:, :, :2]

    # Keep ATen's exact reduction order.  Presenting rows in the projection's
    # physical token-major order improves memory locality; each 64-value row
    # still has exactly the reference's reduction result.
    means = qk.mean(dim=-1, keepdim=True)
    variances = qk.var(dim=-1, unbiased=False, keepdim=True)

    n_rows = batch_size * 24 * seq_len
    packed = torch.empty(
        (3, batch_size, 24, seq_len, 64),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    _normalize_and_pack_qkv[(n_rows,)](
        qkv,
        means,
        variances,
        q_norm_weight,
        q_norm_bias,
        k_norm_weight,
        k_norm_bias,
        packed,
        n_rows=n_rows,
        seq_len=seq_len,
        eps=eps,
        num_warps=1,
    )

    q = packed[0].reshape(batch_size * 24, seq_len, 64)
    k = packed[1].reshape(batch_size * 24, seq_len, 64)
    v = packed[2].reshape(batch_size * 24, seq_len, 64)
    scores = torch.bmm(q, k.transpose(1, 2))
    if seq_len >= 512:
        torch.softmax(scores, dim=-1, out=scores)
        probs = scores
    else:
        probs = F.softmax(scores, dim=-1)

    # Write the second BMM in S,B,H,D order.  It is a contiguous matrix for
    # the projection, so this avoids the reference's BHSD -> BSHD copy.  The
    # final permute is only a returned view.
    attention_s_b_h_d = torch.empty(
        (seq_len, batch_size, 24, 64),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    bmm_output = torch.as_strided(
        attention_s_b_h_d,
        (batch_size * 24, seq_len, 64),
        (64, batch_size * 24 * 64, 1),
    )
    torch.bmm(probs, v, out=bmm_output)
    projected = F.linear(
        attention_s_b_h_d.reshape(seq_len, batch_size, dim),
        out_proj_weight,
        out_proj_bias,
    )
    return projected.permute(1, 0, 2)
