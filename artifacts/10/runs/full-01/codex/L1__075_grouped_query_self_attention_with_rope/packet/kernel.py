import torch
import triton
import triton.language as tl


@triton.jit
def _ieee_mul(x, y):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _ieee_add(x, y):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _trig_kernel(position, inv_freq, cos, sin, n_elements: tl.constexpr,
                 SEQ: tl.constexpr, POS_STRIDE_B: tl.constexpr,
                 POS_STRIDE_S: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    pos = off % SEQ
    freq = (off // SEQ) % 32
    batch = off // (32 * SEQ)
    p = tl.load(
        position + batch * POS_STRIDE_B + pos * POS_STRIDE_S,
        mask=mask,
        other=0,
    ).to(tl.float32)
    inv = tl.load(inv_freq + freq, mask=mask, other=0.0)
    angle = _ieee_mul(inv, p)
    tl.store(cos + off, tl.cos(angle), mask=mask)
    tl.store(sin + off, tl.sin(angle), mask=mask)


@triton.jit
def _rope_kernel(q, k, cos, sin, qo, ko, nq: tl.constexpr,
                 SEQ: tl.constexpr, SRC_Q_STRIDE: tl.constexpr,
                 SRC_K_STRIDE: tl.constexpr, EXPAND_K: tl.constexpr,
                 BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    is_q = off < nq
    if EXPAND_K:
        is_k = (off >= nq) & (off < 2 * nq)
    else:
        is_k = (off >= nq) & (off < nq + nq // 4)
    qi = off
    ko_i = off - nq

    qd = qi & 63
    kd = ko_i & 63
    qtoken = qi // (16 * 64)
    qsi = qtoken * SRC_Q_STRIDE + qi % (16 * 64)
    qr_i = tl.where(qd < 32, qsi + 32, qsi - 32)
    qx = tl.load(q + qsi, mask=is_q, other=0.0)
    qrot = tl.load(q + qr_i, mask=is_q, other=0.0)
    qrot = tl.where(qd < 32, -qrot, qrot)

    if EXPAND_K:
        # ko is contiguous [B, 16, S, 64].
        kt = ko_i // 64
        kpos = kt % SEQ
        kh = (kt // SEQ) % 16
        kb = kt // (16 * SEQ)
        ktoken = kb * SEQ + kpos
        klocal = (kh // 4) * 64 + kd
    else:
        # ko preserves the packed physical [B, S, 4, 64] projection layout.
        ktoken = ko_i // (4 * 64)
        kpos = ktoken % SEQ
        kb = ktoken // SEQ
        klocal = ko_i % (4 * 64)
    ksi = ktoken * SRC_K_STRIDE + klocal
    kr_i = tl.where(kd < 32, ksi + 32, ksi - 32)
    kx = tl.load(k + ksi, mask=is_k, other=0.0)
    krot = tl.load(k + kr_i, mask=is_k, other=0.0)
    krot = tl.where(kd < 32, -krot, krot)

    # cos/sin preserve the transposed [B, 32, S] physical layout of freqs.
    qtrig = (qtoken // SEQ) * (32 * SEQ) + (qd & 31) * SEQ + qtoken % SEQ
    ktrig = kb * (32 * SEQ) + (kd & 31) * SEQ + kpos
    trig_i = tl.where(is_q, qtrig, ktrig)
    c = tl.load(cos + trig_i, mask=is_q | is_k, other=0.0)
    s = tl.load(sin + trig_i, mask=is_q | is_k, other=0.0)
    x = tl.where(is_q, qx, kx)
    rot = tl.where(is_q, qrot, krot)
    out = _ieee_add(_ieee_mul(x, c), _ieee_mul(rot, s))
    tl.store(qo + qi, out, mask=is_q)
    tl.store(ko + ko_i, out, mask=is_k)


@torch.no_grad()
def run(
    hidden_states,
    position_ids,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    o_proj_weight,
    inv_freq,
    is_causal,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_heads = 16
    num_key_value_heads = 4
    head_dim = 64

    token_count = batch_size * seq_len
    combined_projection = token_count >= 541
    if combined_projection:
        qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)
        qkv = torch.matmul(hidden_states, qkv_weight.t())
        query_states = qkv[..., :1024].view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = qkv[..., 1024:1280].view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        value_states = qkv[..., 1280:].view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        src_q_stride = 1536
        src_k_stride = 1536
    else:
        query_states = torch.matmul(hidden_states, q_proj_weight.t()).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        key_states = torch.matmul(hidden_states, k_proj_weight.t()).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        value_states = torch.matmul(hidden_states, v_proj_weight.t()).view(
            batch_size, seq_len, num_key_value_heads, head_dim
        ).transpose(1, 2)
        src_q_stride = 1024
        src_k_stride = 256

    # Fuse the reference's K=1 angle product, integer conversion, and both
    # transcendental operations.  The physical layout is [B, 32, S].
    trig_elements = token_count * 32
    cos = torch.empty(
        (batch_size, 32, seq_len), device=hidden_states.device, dtype=torch.float32
    )
    sin = torch.empty_like(cos)
    _trig_kernel[(triton.cdiv(trig_elements, 256),)](
        position_ids,
        inv_freq,
        cos,
        sin,
        n_elements=trig_elements,
        SEQ=seq_len,
        POS_STRIDE_B=position_ids.stride(0),
        POS_STRIDE_S=position_ids.stride(1),
        BLOCK=256,
    )
    rotated_query = torch.empty(
        (batch_size, seq_len, num_heads, head_dim),
        device=query_states.device,
        dtype=query_states.dtype,
    ).transpose(1, 2)
    expand_k = token_count < 600
    if expand_k:
        rotated_key = torch.empty(
            (batch_size, num_heads, seq_len, head_dim),
            device=key_states.device,
            dtype=key_states.dtype,
        )
    else:
        rotated_key = torch.empty(
            (batch_size, seq_len, num_key_value_heads, head_dim),
            device=key_states.device,
            dtype=key_states.dtype,
        ).transpose(1, 2)
    nq = token_count * num_heads * head_dim
    rope_elements = nq * 2 if expand_k else nq + nq // 4
    _rope_kernel[(triton.cdiv(rope_elements, 1024),)](
        query_states,
        key_states,
        cos,
        sin,
        rotated_query,
        rotated_key,
        nq=nq,
        SEQ=seq_len,
        SRC_Q_STRIDE=src_q_stride,
        SRC_K_STRIDE=src_k_stride,
        EXPAND_K=expand_k,
        BLOCK=1024,
    )
    query_states = rotated_query
    key_states = rotated_key

    if not expand_k:
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, 4, seq_len, head_dim
        ).reshape(batch_size, num_heads, seq_len, head_dim)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights = attn_weights.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)

    # A zero-stride V batch selects the same GEMM algorithm as the materialized
    # repeat, while avoiding a 4x copy of V.
    grouped_weights = attn_weights.view(
        batch_size, num_key_value_heads, 4, seq_len, seq_len
    )
    attn_output = torch.matmul(
        grouped_weights, value_states.unsqueeze(2)
    ).view(batch_size, num_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
    attn_output = torch.matmul(attn_output, o_proj_weight.t())
    return attn_output, attn_weights
