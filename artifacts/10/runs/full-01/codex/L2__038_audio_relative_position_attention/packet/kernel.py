import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scale_pad_q(q_ptr, scale_ptr, head_ptr, ac_ptr, time, padded_time,
                 qscale, n_elements: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    d = offs % 64
    t = (offs // 64) % padded_time
    h = (offs // (padded_time * 64)) % 8
    b = offs // (padded_time * 64 * 8)
    feat = h * 64 + d
    active = (offs < n_elements) & (t < time)
    src = b * time * 1536 + t * 1536 + feat
    x = tl.load(q_ptr + src, mask=active, other=0.0)
    s = tl.load(scale_ptr + d)
    x = x * qscale
    x = x * s
    live = offs < n_elements
    tl.store(head_ptr + offs, x, mask=live)
    block = t // 32
    w = t % 32
    ac_off = ((((b * (padded_time // 32) + block) * 8 + h) * 32 + w) * 64 + d)
    tl.store(ac_ptr + ac_off, x, mask=live)


@triton.jit
def _timing_signal(inv_ptr, out_ptr):
    offs = tl.program_id(0) * 256 + tl.arange(0, 256)
    row = offs // 512
    col = offs % 512
    freq = col % 256
    x = (127 - row).to(tl.float32) * tl.load(inv_ptr + freq)
    y = tl.where(col < 256, tl.extra.hip.libdevice.sin(x),
                 tl.extra.hip.libdevice.cos(x))
    tl.store(out_ptr + offs, y, mask=offs < 128 * 512)


@triton.jit
def _transpose_output(src_ptr, dst_ptr, time, nblocks,
                      n_elements: tl.constexpr):
    offs = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    feat = offs % 512
    d = feat % 64
    h = feat // 64
    t = (offs // 512) % time
    b = offs // (time * 512)
    block = t // 32
    w = t % 32
    src = ((((b * nblocks + block) * 8 + h) * 32 + w) * 64 + d)
    x = tl.load(src_ptr + src, mask=offs < n_elements)
    tl.store(dst_ptr + offs, x, mask=offs < n_elements)


@triton.jit
def _pack_context(qkv_ptr, mask_ptr, kt_ptr, vv_ptr, valid_ptr,
                  time, nblocks, BLOCK_C: tl.constexpr):
    # Materialize the two layouts consumed by rocBLAS directly.  The reference
    # first builds [B,U,C,H,D] and then permutes/copies both tensors.
    x = tl.program_id(0)
    tile = tl.program_id(1)
    h = x % 8
    block = (x // 8) % nblocks
    b = x // (nblocks * 8)
    c = tile * BLOCK_C + tl.arange(0, BLOCK_C)
    d = tl.arange(0, 64)
    t = block * 32 - 127 + c
    inside = (c < 160) & (t >= 0) & (t < time)
    src = b * time * 1536 + t[:, None] * 1536 + h * 64 + d[None, :]
    kval = tl.load(qkv_ptr + src + 512, mask=inside[:, None], other=0.0)
    vval = tl.load(qkv_ptr + src + 1024, mask=inside[:, None], other=0.0)

    # K layout [B,U,H,D,C], V layout [B,U,H,C,D].
    kt_base = ((b * nblocks + block) * 8 + h) * 64 * 160
    tl.store(kt_ptr + kt_base + d[None, :] * 160 + c[:, None], kval,
             mask=c[:, None] < 160)
    vv_base = ((b * nblocks + block) * 8 + h) * 160 * 64
    tl.store(vv_ptr + vv_base + c[:, None] * 64 + d[None, :], vval,
             mask=c[:, None] < 160)

    is_pad = tl.load(mask_ptr + b * time + t, mask=inside, other=True)
    tl.store(valid_ptr + (b * nblocks + block) * 160 + c,
             inside & (~is_pad), mask=(h == 0) & (c < 160))


@triton.jit
def _fuse_logits(ac_ptr, bd_ptr, valid_ptr, nblocks, softcap,
                 BLOCK_M: tl.constexpr):
    x = tl.program_id(0)
    tile = tl.program_id(1)
    h = x % 8
    block = (x // 8) % nblocks
    b = x // (nblocks * 8)
    w = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, 256)
    ac_offs = x * (32 * 160) + w[:, None] * 160 + c[None, :]
    rel = c[None, :] - w[:, None]
    bd_group = (b * 8 + h) * nblocks + block
    bd_offs = bd_group * (32 * 128) + w[:, None] * 128 + tl.maximum(rel, 0)
    lane = c[None, :] < 160
    a = tl.load(ac_ptr + ac_offs, mask=lane)
    d = tl.load(bd_ptr + bd_offs,
                mask=lane & (rel >= 0) & (rel < 128), other=0.0)
    z = a + d
    z = z / softcap
    z = tl.extra.hip.libdevice.tanh(z)
    z = z * softcap
    key_ok = tl.load(valid_ptr + (b * nblocks + block) * 160 + c,
                     mask=c < 160, other=False)
    causal = (c[None, :] >= w[:, None]) & (c[None, :] <= w[:, None] + 127)
    z = tl.where(key_ok[None, :] & causal, z, -3.4028234663852886e38)
    tl.store(ac_ptr + ac_offs, z, mask=lane)


@triton.jit
def _debug_ac(q_ptr, k_ptr, out_ptr, time, nblocks):
    pid = tl.program_id(0)
    block = pid % nblocks
    h = (pid // nblocks) % 8
    b = pid // (nblocks * 8)
    m = tl.arange(0, 32)
    n = tl.arange(0, 256)
    d = tl.arange(0, 64)
    qt = block * 32 + m
    kt = block * 32 - 127 + n
    q = tl.load(q_ptr + b * time * 512 + qt[:, None] * 512 + h * 64 + d[None, :],
                mask=qt[:, None] < time, other=0.0)
    k = tl.load(k_ptr + b * time * 512 + kt[None, :] * 512 + h * 64 + d[:, None],
                mask=(n[None, :] < 160) & (kt[None, :] >= 0) & (kt[None, :] < time), other=0.0)
    ac = tl.dot(q, k, input_precision="ieee")
    tl.store(out_ptr + pid * 32 * 160 + m[:, None] * 160 + n[None, :], ac,
             mask=n[None, :] < 160)


@triton.jit
def _debug_ac_tf32(q_ptr, k_ptr, out_ptr, time, nblocks):
    pid = tl.program_id(0)
    block = pid % nblocks
    h = (pid // nblocks) % 8
    b = pid // (nblocks * 8)
    m = tl.arange(0, 32); n = tl.arange(0, 256); d = tl.arange(0, 64)
    qt = block * 32 + m; kt = block * 32 - 127 + n
    q = tl.load(q_ptr + b*time*512 + qt[:,None]*512 + h*64 + d[None,:], mask=qt[:,None]<time, other=0.)
    k = tl.load(k_ptr + b*time*512 + kt[None,:]*512 + h*64 + d[:,None], mask=(n[None,:]<160)&(kt[None,:]>=0)&(kt[None,:]<time), other=0.)
    ac = tl.dot(q, k, input_precision="tf32")
    tl.store(out_ptr + pid*32*160 + m[:,None]*160 + n[None,:], ac, mask=n[None,:]<160)


@triton.jit
def _softmax_out(logits_ptr, v_ptr, out_ptr, time, nblocks):
    pid = tl.program_id(0)
    block = pid % nblocks
    h = (pid // nblocks) % 8
    b = pid // (nblocks * 8)
    m = tl.arange(0, 32); n = tl.arange(0, 256); d = tl.arange(0, 64)
    x = tl.load(logits_ptr + pid*32*160 + m[:,None]*160 + n[None,:], mask=n[None,:]<160, other=-float("inf"))
    mx = tl.max(x, axis=1)
    z = tl.extra.hip.libdevice.exp(x-mx[:,None])
    p = z / tl.sum(z, axis=1)[:,None]
    kt = block*32-127+n
    vv = tl.load(v_ptr + b*time*512 + kt[:,None]*512 + h*64 + d[None,:], mask=(n[:,None]<160)&(kt[:,None]>=0)&(kt[:,None]<time), other=0.)
    o = tl.dot(p, vv, input_precision="ieee")
    qt = block*32+m
    tl.store(out_ptr + b*time*512 + qt[:,None]*512 + h*64+d[None,:],o,mask=qt[:,None]<time)


@triton.jit
def _p_out(p_ptr, v_ptr, out_ptr, time, nblocks):
    pid=tl.program_id(0); block=pid%nblocks; h=(pid//nblocks)%8; b=pid//(nblocks*8)
    m=tl.arange(0,32); n=tl.arange(0,256); d=tl.arange(0,64)
    p=tl.load(p_ptr+pid*32*160+m[:,None]*160+n[None,:],mask=n[None,:]<160,other=0.)
    kt=block*32-127+n
    vv=tl.load(v_ptr+b*time*512+kt[:,None]*512+h*64+d[None,:],mask=(n[:,None]<160)&(kt[:,None]>=0)&(kt[:,None]<time),other=0.)
    o=tl.dot(p,vv,input_precision="ieee"); qt=block*32+m
    tl.store(out_ptr+b*time*512+qt[:,None]*512+h*64+d[None,:],o,mask=qt[:,None]<time)


@triton.jit
def _p_out160(p_ptr, v_ptr, out_ptr, time, nblocks):
    pid=tl.program_id(0); block=pid%nblocks; h=(pid//nblocks)%8; b=pid//(nblocks*8)
    m=tl.arange(0,32); n0=tl.arange(0,128); n1=tl.arange(0,32)+128; d=tl.arange(0,64)
    p0=tl.load(p_ptr+pid*32*160+m[:,None]*160+n0[None,:])
    p1=tl.load(p_ptr+pid*32*160+m[:,None]*160+n1[None,:])
    p=tl.cat(p0,p1,can_reorder=True)
    kt0=block*32-127+n0; kt1=block*32-127+n1
    v0=tl.load(v_ptr+b*time*512+kt0[None,:]*512+h*64+d[:,None],mask=(kt0[None,:]>=0)&(kt0[None,:]<time),other=0.)
    v1=tl.load(v_ptr+b*time*512+kt1[None,:]*512+h*64+d[:,None],mask=(kt1[None,:]>=0)&(kt1[None,:]<time),other=0.)
    vv=tl.trans(tl.cat(v0,v1,can_reorder=True))
    o=tl.dot(p,vv,input_precision="ieee"); qt=block*32+m
    tl.store(out_ptr+b*time*512+qt[:,None]*512+h*64+d[None,:],o,mask=qt[:,None]<time)


@triton.jit
def _p_out_split(p_ptr, v_ptr, out_ptr, time, nblocks):
    pid=tl.program_id(0); block=pid%nblocks; h=(pid//nblocks)%8; b=pid//(nblocks*8)
    m=tl.arange(0,32); n0=tl.arange(0,128); n1=tl.arange(0,32)+128; d=tl.arange(0,64)
    p0=tl.load(p_ptr+pid*32*160+m[:,None]*160+n0[None,:]); p1=tl.load(p_ptr+pid*32*160+m[:,None]*160+n1[None,:])
    kt0=block*32-127+n0; kt1=block*32-127+n1
    v0=tl.load(v_ptr+b*time*512+kt0[:,None]*512+h*64+d[None,:],mask=(kt0[:,None]>=0)&(kt0[:,None]<time),other=0.)
    v1=tl.load(v_ptr+b*time*512+kt1[:,None]*512+h*64+d[None,:],mask=(kt1[:,None]>=0)&(kt1[:,None]<time),other=0.)
    o=tl.dot(p0,v0,input_precision="ieee")
    o=tl.dot(p1,v1,acc=o,input_precision="ieee"); qt=block*32+m
    tl.store(out_ptr+b*time*512+qt[:,None]*512+h*64+d[None,:],o,mask=qt[:,None]<time)


@triton.jit
def _relative_attention(q_ptr, k_ptr, v_ptr, s_ptr, mask_ptr, out_ptr,
                        time, nblocks, softcap: tl.constexpr):
    # One program computes 32 adjacent queries for one batch/head.  Keeping the
    # 160-key block resident lets the logits, softmax and P@V share one launch.
    pid = tl.program_id(0)
    block = pid % nblocks
    h = (pid // nblocks) % 8
    b = pid // (nblocks * 8)

    m = tl.arange(0, 32)
    n = tl.arange(0, 256)
    d = tl.arange(0, 64)
    qt = block * 32 + m
    kt = block * 32 - 127 + n

    q = tl.load(q_ptr + b * time * 512 + qt[:, None] * 512 + h * 64 + d[None, :],
                mask=qt[:, None] < time, other=0.0)
    k = tl.load(k_ptr + b * time * 512 + kt[None, :] * 512 + h * 64 + d[:, None],
                mask=(n[None, :] < 160) & (kt[None, :] >= 0) & (kt[None, :] < time),
                other=0.0)
    # Extending the 128 relative vectors with zero columns gives bd the same
    # local shape as ac; gather then implements the reference relative shift.
    s = tl.load(s_ptr + n[None, :] * 512 + h * 64 + d[:, None],
                mask=n[None, :] < 128, other=0.0)
    ac = tl.dot(q, k, input_precision="ieee")
    bd_unshifted = tl.dot(q, s, input_precision="ieee")
    rel = n[None, :] - m[:, None]
    bd = tl.gather(bd_unshifted, tl.maximum(rel, 0), axis=1)
    logits = ac + bd
    logits = tl.extra.hip.libdevice.tanh(logits / softcap) * softcap

    key_in = (kt >= 0) & (kt < time)
    padded = tl.load(mask_ptr + b * time + kt, mask=key_in, other=True)
    causal = (n[None, :] >= m[:, None]) & (n[None, :] <= m[:, None] + 127)
    valid = (n[None, :] < 160) & causal & key_in[None, :] & (~padded[None, :])
    # Logical invalid entries are float32.min in the reference.  Columns used
    # only to make the tile a power of two are true -inf and never normalize.
    logits = tl.where(n[None, :] < 160,
                      tl.where(valid, logits, -3.4028234663852886e38),
                      -float("inf"))
    row_max = tl.max(logits, axis=1)
    numer = tl.extra.hip.libdevice.exp(logits - row_max[:, None])
    probs = numer / tl.sum(numer, axis=1)[:, None]

    vv = tl.load(v_ptr + b * time * 512 + kt[:, None] * 512 + h * 64 + d[None, :],
                 mask=(n[:, None] < 160) & (kt[:, None] >= 0) & (kt[:, None] < time),
                 other=0.0)
    out = tl.dot(probs, vv, input_precision="ieee")
    tl.store(out_ptr + b * time * 512 + qt[:, None] * 512 + h * 64 + d[None, :],
             out, mask=qt[:, None] < time)


@torch.no_grad()
def run(hidden_states, mask, q_proj_weight, k_proj_weight, v_proj_weight,
        pos_proj_weight, per_dim_scale, inv_timescales,
        attention_logits_soft_cap):
    B, T, _ = hidden_states.shape
    H, D = 8, 64

    # Keep the four GEMMs in PyTorch: these are the exact rocBLAS operations
    # used by the specification and dominate arithmetic throughput.
    qkv_weight = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)
    qkv = torch.matmul(hidden_states, qkv_weight.T)

    nb = (T + 31) // 32
    padded_t = nb * 32
    q = torch.empty((B, H, padded_t, D), device=qkv.device, dtype=qkv.dtype)
    q_ac = torch.empty((B, nb, H, 32, D), device=qkv.device, dtype=qkv.dtype)
    per_dim_sp = F.softplus(per_dim_scale)
    q_elements = B * padded_t * 512
    _scale_pad_q[(triton.cdiv(q_elements, 1024),)](
        qkv, per_dim_sp, q, q_ac, T, padded_t, 0.1803368777036667,
        n_elements=q_elements, num_warps=4,
    )

    timing = torch.empty((128, 512), device=q.device, dtype=q.dtype)
    _timing_signal[(128 * 2,)](inv_timescales, timing, num_warps=4)
    sin_emb = torch.matmul(timing, pos_proj_weight.T).reshape(128, H, D)

    kt = torch.empty((B, nb, H, D, 160), device=q.device, dtype=q.dtype)
    vv = torch.empty((B, nb, H, 160, D), device=q.device, dtype=q.dtype)
    valid = torch.empty((B, nb, 160), device=q.device, dtype=torch.bool)
    _pack_context[(B * H * nb, 10)](
        qkv, mask, kt, vv, valid, T, nb, BLOCK_C=16, num_warps=4,
    )

    ac = torch.matmul(q_ac, kt)
    bd0 = torch.matmul(q.reshape(B, H, nb * 32, D),
                       sin_emb.permute(1, 2, 0))
    bd0 = bd0.reshape(B, H, nb, 32, 128)

    _fuse_logits[(B * H * nb, 4)](
        ac, bd0, valid, nb, float(attention_logits_soft_cap),
        BLOCK_M=8, num_warps=8,
    )
    p = F.softmax(ac, dim=-1, dtype=torch.float32).to(dtype=vv.dtype)

    p3 = p.reshape(-1, 32, 160)
    bmm_out = torch.bmm(p3, vv.reshape(-1, 160, D))
    out = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
    out_elements = B * T * 512
    _transpose_output[(triton.cdiv(out_elements, 1024),)](
        bmm_out, out, T, nb, n_elements=out_elements, num_warps=4,
    )
    return out
