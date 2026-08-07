import torch
import triton
import triton.language as tl


@triton.jit
def _rope_backward_products(
    grad_q_ptr,
    grad_k_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    q_product_ptr,
    k_product_ptr,
    seq_len,
    num_q_heads,
    num_k_heads,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    d_block = pid % (64 // BLOCK_D)
    token = pid // (64 // BLOCK_D)
    seq = token % seq_len
    batch = token // seq_len

    d0 = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d1 = d0 + 64
    embedding_base = (batch * seq_len + seq) * 128
    c0 = tl.load(cos_ptr + embedding_base + d0)
    c1 = tl.load(cos_ptr + embedding_base + d1)
    s0 = tl.load(sin_ptr + embedding_base + d0)
    s1 = tl.load(sin_ptr + embedding_base + d1)

    for head in tl.range(0, num_q_heads, num_stages=1):
        base = ((batch * num_q_heads + head) * seq_len + seq) * 128
        off0 = base + d0
        off1 = base + d1
        g0 = tl.load(grad_q_ptr + off0)
        g1 = tl.load(grad_q_ptr + off1)
        x0 = tl.load(q_ptr + off0)
        x1 = tl.load(q_ptr + off1)

        tl.store(out_q_ptr + off0, g0 * c0 + g1 * s0)
        tl.store(out_q_ptr + off1, g1 * c1 - g0 * s1)
        tl.store(q_product_ptr + off0 * 2, g0 * x0)
        tl.store(q_product_ptr + off1 * 2, g1 * x1)
        tl.store(q_product_ptr + off0 * 2 + 1, g0 * (-x1))
        tl.store(q_product_ptr + off1 * 2 + 1, g1 * x0)

    for head in tl.range(0, num_k_heads, num_stages=1):
        base = ((batch * num_k_heads + head) * seq_len + seq) * 128
        off0 = base + d0
        off1 = base + d1
        g0 = tl.load(grad_k_ptr + off0)
        g1 = tl.load(grad_k_ptr + off1)
        x0 = tl.load(k_ptr + off0)
        x1 = tl.load(k_ptr + off1)

        tl.store(out_k_ptr + off0, g0 * c0 + g1 * s0)
        tl.store(out_k_ptr + off1, g1 * c1 - g0 * s1)
        tl.store(k_product_ptr + off0 * 2, g0 * x0)
        tl.store(k_product_ptr + off1 * 2, g1 * x1)
        tl.store(k_product_ptr + off0 * 2 + 1, g0 * (-x1))
        tl.store(k_product_ptr + off1 * 2 + 1, g1 * x0)


@triton.jit
def _rope_backward_products_tiled(
    grad_q_ptr,
    grad_k_ptr,
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    q_product_ptr,
    k_product_ptr,
    seq_len,
    num_q_heads,
    num_k_heads,
    num_q_programs,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    dim = tl.arange(0, 64)[None, :]
    head_in_block = tl.arange(0, HEADS_PER_PROGRAM)[:, None]

    if pid < num_q_programs:
        head_blocks = num_q_heads // HEADS_PER_PROGRAM
        head_block = pid % head_blocks
        token = pid // head_blocks
        seq = token % seq_len
        batch = token // seq_len
        head = head_block * HEADS_PER_PROGRAM + head_in_block

        off0 = ((batch * num_q_heads + head) * seq_len + seq) * 128 + dim
        off1 = off0 + 64
        embedding_off0 = (batch * seq_len + seq) * 128 + dim
        embedding_off1 = embedding_off0 + 64
        c0 = tl.load(cos_ptr + embedding_off0)
        c1 = tl.load(cos_ptr + embedding_off1)
        s0 = tl.load(sin_ptr + embedding_off0)
        s1 = tl.load(sin_ptr + embedding_off1)
        g0 = tl.load(grad_q_ptr + off0)
        g1 = tl.load(grad_q_ptr + off1)
        x0 = tl.load(q_ptr + off0)
        x1 = tl.load(q_ptr + off1)

        tl.store(out_q_ptr + off0, g0 * c0 + g1 * s0)
        tl.store(out_q_ptr + off1, g1 * c1 - g0 * s1)
        tl.store(q_product_ptr + off0 * 2, g0 * x0)
        tl.store(q_product_ptr + off1 * 2, g1 * x1)
        tl.store(q_product_ptr + off0 * 2 + 1, g0 * (-x1))
        tl.store(q_product_ptr + off1 * 2 + 1, g1 * x0)
    else:
        k_pid = pid - num_q_programs
        head_blocks = num_k_heads // HEADS_PER_PROGRAM
        head_block = k_pid % head_blocks
        token = k_pid // head_blocks
        seq = token % seq_len
        batch = token // seq_len
        head = head_block * HEADS_PER_PROGRAM + head_in_block

        off0 = ((batch * num_k_heads + head) * seq_len + seq) * 128 + dim
        off1 = off0 + 64
        embedding_off0 = (batch * seq_len + seq) * 128 + dim
        embedding_off1 = embedding_off0 + 64
        c0 = tl.load(cos_ptr + embedding_off0)
        c1 = tl.load(cos_ptr + embedding_off1)
        s0 = tl.load(sin_ptr + embedding_off0)
        s1 = tl.load(sin_ptr + embedding_off1)
        g0 = tl.load(grad_k_ptr + off0)
        g1 = tl.load(grad_k_ptr + off1)
        x0 = tl.load(k_ptr + off0)
        x1 = tl.load(k_ptr + off1)

        tl.store(out_k_ptr + off0, g0 * c0 + g1 * s0)
        tl.store(out_k_ptr + off1, g1 * c1 - g0 * s1)
        tl.store(k_product_ptr + off0 * 2, g0 * x0)
        tl.store(k_product_ptr + off1 * 2, g1 * x1)
        tl.store(k_product_ptr + off0 * 2 + 1, g0 * (-x1))
        tl.store(k_product_ptr + off1 * 2 + 1, g1 * x0)


def run(grad_q_embed, grad_k_embed, q, k, cos, sin):
    batch, num_q_heads, seq_len, _ = grad_q_embed.shape
    num_k_heads = grad_k_embed.shape[1]

    grad_q = torch.empty_like(grad_q_embed)
    grad_k = torch.empty_like(grad_k_embed)
    if num_q_heads == num_k_heads:
        all_products = torch.empty(
            (2,) + tuple(grad_q_embed.shape) + (2,),
            dtype=grad_q_embed.dtype,
            device=grad_q_embed.device,
        )
        q_products = all_products[0]
        k_products = all_products[1]
    else:
        q_products = torch.empty(
            tuple(grad_q_embed.shape) + (2,),
            dtype=grad_q_embed.dtype,
            device=grad_q_embed.device,
        )
        k_products = torch.empty(
            tuple(grad_k_embed.shape) + (2,),
            dtype=grad_k_embed.dtype,
            device=grad_k_embed.device,
        )

    heads_per_program = 4
    num_q_programs = batch * seq_len * (num_q_heads // heads_per_program)
    num_k_programs = batch * seq_len * (num_k_heads // heads_per_program)
    grid = (num_q_programs + num_k_programs,)
    _rope_backward_products_tiled[grid](
        grad_q_embed,
        grad_k_embed,
        q,
        k,
        cos,
        sin,
        grad_q,
        grad_k,
        q_products,
        k_products,
        seq_len,
        num_q_heads,
        num_k_heads,
        num_q_programs,
        HEADS_PER_PROGRAM=heads_per_program,
        num_warps=4,
    )

    if num_q_heads == num_k_heads:
        reduced_products = all_products.sum(dim=2).sum(dim=0)
    else:
        reduced_products = q_products.sum(dim=1) + k_products.sum(dim=1)
    grad_cos = reduced_products[..., 0]
    grad_sin = reduced_products[..., 1]
    return grad_q, grad_k, grad_cos, grad_sin
