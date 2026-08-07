import torch
import triton
import triton.language as tl


@triton.jit
def _fmul_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fadd_rn(a, b):
    return tl.inline_asm_elementwise(
        "v_add_f32 $0, $1, $2",
        "=v,v,v",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pack_qkv(
    grad_query,
    grad_key,
    grad_value,
    cos,
    sin,
    packed,
    k_weight,
    v_weight,
    packed_kv_weight,
    seq_len: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < n_elements
    token = offs // 3072
    col = offs - token * 3072
    batch = token // seq_len
    seq = token - batch * seq_len

    # Q occupies columns [0, 2304).
    q_mask = valid & (col < 2304)
    q_col = col
    q_head = q_col // 96
    q_dim = q_col - q_head * 96
    q_pair = tl.where(q_dim < 48, q_dim + 48, q_dim - 48)
    q_base = ((batch * 24 + q_head) * seq_len + seq) * 96
    trig_base = token * 96
    q0 = tl.load(grad_query + q_base + q_dim, mask=q_mask, other=0.0)
    q1 = tl.load(grad_query + q_base + q_pair, mask=q_mask, other=0.0)
    qc = tl.load(cos + trig_base + q_dim, mask=q_mask, other=0.0)
    qs = tl.load(sin + trig_base + q_pair, mask=q_mask, other=0.0)
    q_prod0 = _fmul_rn(q0, qc)
    q_prod1 = _fmul_rn(q1, qs)
    q_out = _fadd_rn(q_prod0, tl.where(q_dim < 48, q_prod1, -q_prod1))

    # K occupies columns [2304, 2688).
    k_mask = valid & (col >= 2304) & (col < 2688)
    k_col = col - 2304
    k_head = k_col // 96
    k_dim = k_col - k_head * 96
    k_pair = tl.where(k_dim < 48, k_dim + 48, k_dim - 48)
    k_base = ((batch * 4 + k_head) * seq_len + seq) * 96
    k0 = tl.load(grad_key + k_base + k_dim, mask=k_mask, other=0.0)
    k1 = tl.load(grad_key + k_base + k_pair, mask=k_mask, other=0.0)
    kc = tl.load(cos + trig_base + k_dim, mask=k_mask, other=0.0)
    ks = tl.load(sin + trig_base + k_pair, mask=k_mask, other=0.0)
    k_prod0 = _fmul_rn(k0, kc)
    k_prod1 = _fmul_rn(k1, ks)
    k_out = _fadd_rn(k_prod0, tl.where(k_dim < 48, k_prod1, -k_prod1))

    # V occupies columns [2688, 3072).
    v_mask = valid & (col >= 2688)
    v_col = col - 2688
    v_head = v_col // 96
    v_dim = v_col - v_head * 96
    v_base = ((batch * 4 + v_head) * seq_len + seq) * 96
    v_out = tl.load(grad_value + v_base + v_dim, mask=v_mask, other=0.0)

    out = tl.where(col < 2304, q_out, tl.where(col < 2688, k_out, v_out))
    tl.store(packed + offs, out, mask=valid)

    # Pack K/V weights for a single strided-batched GEMM.  Doing this in the
    # layout kernel avoids a separate torch.stack launch.
    weight_plane = 384 * 1536
    wk_mask = offs < weight_plane
    wv_mask = (offs >= weight_plane) & (offs < 2 * weight_plane)
    wk = tl.load(k_weight + offs, mask=wk_mask, other=0.0)
    wv = tl.load(v_weight + offs - weight_plane, mask=wv_mask, other=0.0)
    tl.store(packed_kv_weight + offs, tl.where(offs < weight_plane, wk, wv), mask=wk_mask | wv_mask)


@triton.jit
def _sum_hidden_inplace(q_hidden, kv_hidden, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    q = tl.load(q_hidden + offs, mask=mask)
    k = tl.load(kv_hidden + offs, mask=mask)
    v = tl.load(kv_hidden + n_elements + offs, mask=mask)
    qk = _fadd_rn(q, k)
    out = _fadd_rn(qk, v)
    tl.store(q_hidden + offs, out, mask=mask)


@triton.jit
def _pack_qkv_segmented(
    grad_query,
    grad_key,
    grad_value,
    cos,
    sin,
    packed,
    seq_len: tl.constexpr,
    tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    q_slots: tl.constexpr = tl.cdiv(2304, BLOCK)
    kv_slots: tl.constexpr = tl.cdiv(384, BLOCK)
    slots: tl.constexpr = q_slots + 2 * kv_slots
    token = pid // slots
    slot = pid - token * slots
    active = token < tokens
    batch = token // seq_len
    seq = token - batch * seq_len
    trig_base = token * 96

    # Each branch is uniform for the whole workgroup, so only the arithmetic
    # for the selected Q/K/V segment is issued.
    if slot < q_slots:
        col = slot * BLOCK + lane
        mask = active & (col < 2304)
        head = col // 96
        dim = col - head * 96
        pair = tl.where(dim < 48, dim + 48, dim - 48)
        base = ((batch * 24 + head) * seq_len + seq) * 96
        x0 = tl.load(grad_query + base + dim, mask=mask, other=0.0)
        x1 = tl.load(grad_query + base + pair, mask=mask, other=0.0)
        c = tl.load(cos + trig_base + dim, mask=mask, other=0.0)
        s = tl.load(sin + trig_base + pair, mask=mask, other=0.0)
        p0 = _fmul_rn(x0, c)
        p1 = _fmul_rn(x1, s)
        out = _fadd_rn(p0, tl.where(dim < 48, p1, -p1))
        tl.store(packed + token * 3072 + col, out, mask=mask)
    elif slot < q_slots + kv_slots:
        col = (slot - q_slots) * BLOCK + lane
        mask = active & (col < 384)
        head = col // 96
        dim = col - head * 96
        pair = tl.where(dim < 48, dim + 48, dim - 48)
        base = ((batch * 4 + head) * seq_len + seq) * 96
        x0 = tl.load(grad_key + base + dim, mask=mask, other=0.0)
        x1 = tl.load(grad_key + base + pair, mask=mask, other=0.0)
        c = tl.load(cos + trig_base + dim, mask=mask, other=0.0)
        s = tl.load(sin + trig_base + pair, mask=mask, other=0.0)
        p0 = _fmul_rn(x0, c)
        p1 = _fmul_rn(x1, s)
        out = _fadd_rn(p0, tl.where(dim < 48, p1, -p1))
        tl.store(packed + token * 3072 + 2304 + col, out, mask=mask)
    else:
        col = (slot - q_slots - kv_slots) * BLOCK + lane
        mask = active & (col < 384)
        head = col // 96
        dim = col - head * 96
        base = ((batch * 4 + head) * seq_len + seq) * 96
        out = tl.load(grad_value + base + dim, mask=mask, other=0.0)
        tl.store(packed + token * 3072 + 2688 + col, out, mask=mask)



@torch.no_grad()
def run(
    grad_query,
    grad_key,
    grad_value,
    hidden_states,
    q_weight,
    k_weight,
    v_weight,
    cos,
    sin,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    tokens = batch_size * seq_len

    packed = torch.empty((tokens, 3072), device=hidden_states.device, dtype=torch.float32)
    pack_block = 512
    pack_slots = triton.cdiv(2304, pack_block) + 2 * triton.cdiv(384, pack_block)
    _pack_qkv_segmented[(tokens * pack_slots,)](
        grad_query,
        grad_key,
        grad_value,
        cos,
        sin,
        packed,
        seq_len=seq_len,
        tokens=tokens,
        BLOCK=pack_block,
        num_warps=2,
    )

    q_proj = packed[:, :2304]
    k_proj = packed[:, 2304:2688]
    v_proj = packed[:, 2688:]
    kv_proj = packed[:, 2304:].view(tokens, 2, 384).permute(1, 0, 2)

    grad_hidden = torch.mm(q_proj, q_weight)
    torch.addmm(grad_hidden, k_proj, k_weight, out=grad_hidden)
    torch.addmm(grad_hidden, v_proj, v_weight, out=grad_hidden)
    grad_hidden = grad_hidden.view(batch_size, seq_len, hidden_size)

    hidden_2d = hidden_states.view(tokens, hidden_size)
    if tokens < 4096:
        grad_weight = torch.mm(packed.t(), hidden_2d)
        grad_q_weight = grad_weight[:2304]
        grad_k_weight = grad_weight[2304:2688]
        grad_v_weight = grad_weight[2688:]
    else:
        grad_q_weight = torch.mm(q_proj.t(), hidden_2d)
        grad_kv_weight = torch.bmm(
            kv_proj.transpose(1, 2), hidden_2d.unsqueeze(0).expand(2, -1, -1)
        )
        grad_k_weight = grad_kv_weight[0]
        grad_v_weight = grad_kv_weight[1]

    return grad_hidden, grad_q_weight, grad_k_weight, grad_v_weight
