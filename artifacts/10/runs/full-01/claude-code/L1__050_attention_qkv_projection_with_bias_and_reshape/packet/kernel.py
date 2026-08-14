import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fused QKV projection (Gemma3-270m attention).
#
#   hidden   (B, S, 640)   bf16      -> M = B*S rows
#   q_weight (1024, 640)   bf16      row-major
#   k_weight ( 256, 640)   bf16
#   v_weight ( 256, 640)   bf16
#
# The three projections share the same activation matrix, so they are one
# GEMM with a block-diagonal-free, column-concatenated weight: N = 1024 + 256
# + 256 = 1536.  Rather than materialise the concatenated weight (which would
# cost an extra copy of 1536*640 elements every call), the kernel keeps the
# three weight tensors where they are and lets the N-tile index select which
# one to read.  1024 and 256 are both multiples of every BLOCK_N used, so a
# tile never straddles two weights.
#
# The three outputs are carved out of ONE allocation of (M, 1536): the store
# stride is then a compile-time constant and the three result views are free
# (no copy, correct shapes via .view()).  This also collapses three allocator
# calls into one.
#
# Numerics mirror the reference exactly:
#   fp32 MFMA accumulate -> round to bf16 (torch materialises a bf16 matmul
#   result) -> widen to fp32, add the bf16 bias in fp32 -> round to bf16.
# ---------------------------------------------------------------------------


@triton.jit
def _tile(
    x_ptr, w_ptr, b_ptr, o_ptr,
    pid_m, ln, gn, M,
    K: tl.constexpr, NW: tl.constexpr, NT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = ln + tl.arange(0, BLOCK_N)      # column inside this weight
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_M:
        offs_am = offs_m
    else:
        offs_am = tl.where(offs_m < M, offs_m, 0)

    a_ptrs = x_ptr + offs_am[:, None] * K + offs_k[None, :]
    # weight is (N, K) row-major -> load (BLOCK_N, BLOCK_K) so K is contiguous,
    # then transpose in registers for the dot.
    b_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, K // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # reference rounding: bf16 matmul result, then fp32 bias add, then bf16
    r = acc.to(tl.bfloat16).to(tl.float32)
    bias = tl.load(b_ptr + offs_n).to(tl.float32)
    out = (r + bias[None, :]).to(tl.bfloat16)

    offs_on = gn + tl.arange(0, BLOCK_N)     # column inside the (M,1536) buffer
    o_ptrs = o_ptr + offs_m[:, None] * NT + offs_on[None, :]
    if EVEN_M:
        tl.store(o_ptrs, out)
    else:
        tl.store(o_ptrs, out, mask=(offs_m < M)[:, None])


@triton.jit
def _qkv(
    x_ptr, qw_ptr, qb_ptr, kw_ptr, kb_ptr, vw_ptr, vb_ptr, o_ptr,
    M,
    K: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, EVEN_M: tl.constexpr,
):
    NT: tl.constexpr = NQ + 2 * NKV
    NQB: tl.constexpr = NQ // BLOCK_N
    NKB: tl.constexpr = NKV // BLOCK_N
    num_pid_n: tl.constexpr = NQB + 2 * NKB

    pid = tl.program_id(0)

    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_m = tl.cdiv(M, BLOCK_M)
        ng = GROUP_M * num_pid_n
        first_m = (pid // ng) * GROUP_M
        gs = min(num_pid_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % ng) % gs)
        pid_n = (pid % ng) // gs

    if pid_n < NQB:
        _tile(x_ptr, qw_ptr, qb_ptr, o_ptr, pid_m,
              pid_n * BLOCK_N, pid_n * BLOCK_N, M,
              K, NQ, NT, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M)
    elif pid_n < NQB + NKB:
        _tile(x_ptr, kw_ptr, kb_ptr, o_ptr, pid_m,
              (pid_n - NQB) * BLOCK_N, pid_n * BLOCK_N, M,
              K, NKV, NT, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M)
    else:
        _tile(x_ptr, vw_ptr, vb_ptr, o_ptr, pid_m,
              (pid_n - NQB - NKB) * BLOCK_N, pid_n * BLOCK_N, M,
              K, NKV, NT, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M)


# --------------------------------------------------------------------------
# Launch path.
#
# For most of these workloads the GEMM itself takes 6-35 us, and Triton's
# normal `kernel[grid](...)` dispatch costs ~20 us of pure Python on top of
# that.  So the kernel is compiled once per (config, EVEN_M) and afterwards
# launched through CompiledKernel.run directly, which is ~4 us.
# --------------------------------------------------------------------------

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
_CONFIGS = (
    (64, 64, 64, 1, 4, 3),      # small M
    (128, 128, 32, 8, 8, 3),    # medium M
    (128, 128, 64, 8, 8, 2),    # large M
)


def _config_for(M):
    if M <= 1024:
        return 0
    if M <= 4096:
        return 1
    return 2


_cache = {}
_stream = None
_dev = None


def _get(cfg_idx, even_m, K, NQ, NKV):
    key = (cfg_idx, even_m, K, NQ, NKV)
    got = _cache.get(key)
    if got is not None:
        return got

    BM, BN, BK, GM, nw, ns = _CONFIGS[cfg_idx]
    dev = torch.cuda.current_device()
    dummy_x = torch.empty((BM, K), device="cuda", dtype=torch.bfloat16)
    dummy_w = torch.empty((NQ, K), device="cuda", dtype=torch.bfloat16)
    dummy_wk = torch.empty((NKV, K), device="cuda", dtype=torch.bfloat16)
    dummy_b = torch.empty((NQ,), device="cuda", dtype=torch.bfloat16)
    dummy_bk = torch.empty((NKV,), device="cuda", dtype=torch.bfloat16)
    dummy_o = torch.empty((BM, NQ + 2 * NKV), device="cuda", dtype=torch.bfloat16)

    compiled = _qkv.warmup(
        dummy_x, dummy_w, dummy_b, dummy_wk, dummy_bk, dummy_wk, dummy_bk,
        dummy_o, BM,
        K=K, NQ=NQ, NKV=NKV,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, EVEN_M=even_m,
        num_warps=nw, num_stages=ns,
        grid=(1,),
    )
    compiled._init_handles()
    entry = (compiled.run, compiled.function, compiled.packed_metadata,
             BM, BN, BK, GM)
    _cache[key] = entry
    return entry


def _get_stream():
    global _stream, _dev
    if _stream is None:
        from triton.runtime import driver
        _dev = driver.active.get_current_device()
        _stream = driver.active.get_current_stream(_dev)
    return _stream


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    q_bias: torch.Tensor,
    k_weight: torch.Tensor,
    k_bias: torch.Tensor,
    v_weight: torch.Tensor,
    v_bias: torch.Tensor,
):
    batch_size, seq_len, K = hidden_states.shape
    M = batch_size * seq_len
    NQ = q_weight.shape[0]
    NKV = k_weight.shape[0]
    NT = NQ + 2 * NKV

    x = hidden_states
    if x.stride(2) != 1 or x.stride(1) != K:
        x = x.contiguous()
    if q_weight.stride(1) != 1:
        q_weight = q_weight.contiguous()
    if k_weight.stride(1) != 1:
        k_weight = k_weight.contiguous()
    if v_weight.stride(1) != 1:
        v_weight = v_weight.contiguous()

    ci = _config_for(M)
    BM = _CONFIGS[ci][0]
    even = (M % BM) == 0
    fn, func, pmeta, BM, BN, BK, GM = _get(ci, even, K, NQ, NKV)

    buf = torch.empty((M, NT), device=x.device, dtype=torch.bfloat16)

    grid = ((M + BM - 1) // BM) * (NT // BN)
    fn(grid, 1, 1, _get_stream(), func, pmeta, None, None, None,
       x, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, buf, M,
       K, NQ, NKV, BM, BN, BK, GM, even)

    head_dim = NKV
    q = buf[:, :NQ].view(batch_size, seq_len, NQ // head_dim, head_dim)
    k = buf[:, NQ:NQ + NKV].view(batch_size, seq_len, 1, head_dim)
    v = buf[:, NQ + NKV:].view(batch_size, seq_len, 1, head_dim)
    return q, k, v
