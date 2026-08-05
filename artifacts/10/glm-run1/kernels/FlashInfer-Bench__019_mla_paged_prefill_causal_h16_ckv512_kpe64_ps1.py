import torch
import math

torch.set_float32_matmul_precision("high")

_ln2 = math.log(2.0)


@torch.compile(mode="reduce-overhead", dynamic=True)
def _attend(qn, qp, Kc, Kp, sm_scale, prefix_len, q_len, kv_len):
    H = 16
    Dc = 512
    Dp = 64
    D = Dc + Dp
    Kcat = torch.cat([Kc, Kp], dim=-1)
    qcat = torch.cat([qn, qp], dim=-1)
    logits = (qcat.reshape(q_len * H, D) @ Kcat.t()).reshape(q_len, H, kv_len)
    logits = logits * sm_scale
    rows = torch.arange(q_len, device=qn.device, dtype=torch.long)
    allowed = (prefix_len + rows).unsqueeze(1)
    cols = torch.arange(kv_len, device=qn.device, dtype=torch.long).unsqueeze(0)
    mask = cols <= allowed
    logits = logits.masked_fill(~mask.unsqueeze(1), -float("inf"))
    lse_b = torch.logsumexp(logits, dim=-1) / _ln2
    attn = torch.softmax(logits, dim=-1)
    out = (attn.reshape(q_len * H, kv_len) @ Kc).reshape(q_len, H, Dc)
    return out, lse_b


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    device = q_nope.device

    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 1

    H = num_qo_heads
    Dc = head_dim_ckv

    output = torch.zeros((total_q, H, Dc), dtype=torch.float32, device=device)
    lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

    Kc_cache = ckv_cache[:, 0, :]
    Kp_cache = kpe_cache[:, 0, :]

    qo = qo_indptr.tolist()
    kv = kv_indptr.tolist()

    for b in range(batch_size):
        q_start = qo[b]
        q_end = qo[b + 1]
        page_beg = kv[b]
        page_end = kv[b + 1]

        if q_start >= q_end or page_beg >= page_end:
            continue

        kv_len = page_end - page_beg
        q_len = q_end - q_start
        prefix_len = kv_len - q_len

        tok_idx = kv_indices[page_beg:page_end].to(torch.long)
        Kc = Kc_cache.index_select(0, tok_idx).to(torch.float32)
        Kp = Kp_cache.index_select(0, tok_idx).to(torch.float32)
        qn = q_nope[q_start:q_end].to(torch.float32)
        qp = q_pe[q_start:q_end].to(torch.float32)

        out, lse_b = _attend(qn, qp, Kc, Kp, sm_scale, prefix_len, q_len, kv_len)
        output[q_start:q_end] = out
        lse[q_start:q_end] = lse_b

    return output.to(torch.bfloat16), lse
