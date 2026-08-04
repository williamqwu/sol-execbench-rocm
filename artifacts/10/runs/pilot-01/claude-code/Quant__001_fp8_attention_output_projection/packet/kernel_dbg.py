import torch, triton, triton.language as tl
FP8MAX = tl.constexpr(448.0)
RECIP_FP8MAX = tl.constexpr(1.0 / 448.0)

@triton.jit
def _g(A, W, BIAS, C, M, N, K, stride_am, stride_wn, stride_cm,
       BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr,
       NUM_NBLK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM); num_pid_n = tl.cdiv(N, BN)
    g = GROUP_M * num_pid_n
    gid = pid // g; fm = gid * GROUP_M
    gs = min(num_pid_m - fm, GROUP_M)
    pid_m = fm + ((pid % g) % gs); pid_n = (pid % g) // gs
    offs_m = pid_m*BM + tl.arange(0,BM); offs_n = pid_n*BN + tl.arange(0,BN)
    offs_k = tl.arange(0,128)
    mask_m = offs_m < M
    am = tl.where(mask_m, offs_m, 0)
    a_ptrs = A + am[:,None]*stride_am + offs_k[None,:]
    w_ptrs = W + offs_n[:,None]*stride_wn + offs_k[None,:]
    acc = tl.zeros((BM,BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K,128)):
        a = tl.load(a_ptrs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)
        sa = tl.maximum(tl.max(tl.abs(a),axis=1)*RECIP_FP8MAX, 1e-12)
        qa = tl.clamp(a/sa[:,None], -FP8MAX, FP8MAX).to(tl.float8e4nv)
        if NUM_NBLK == 1:
            sw = tl.maximum(tl.max(tl.abs(w))*RECIP_FP8MAX, 1e-12)
            swb = tl.full((BN,), 1.0, tl.float32)*sw
        else:
            wr = tl.reshape(w, (NUM_NBLK,128,128))
            swv = tl.maximum(tl.max(tl.max(tl.abs(wr),axis=2),axis=1)*RECIP_FP8MAX, 1e-12)
            swb = tl.reshape(tl.broadcast_to(swv[:,None],(NUM_NBLK,128)),(BN,))
        qw = tl.clamp(w/swb[:,None], -FP8MAX, FP8MAX).to(tl.float8e4nv)
        if MODE == 0:
            d = tl.dot(qa, tl.trans(qw), out_dtype=tl.float32)
        elif MODE == 1:
            d = tl.dot(qa.to(tl.float32), tl.trans(qw).to(tl.float32), out_dtype=tl.float32)
        else:
            d = tl.dot(qa.to(tl.bfloat16), tl.trans(qw).to(tl.bfloat16), out_dtype=tl.float32)
        acc += d * (sa[:,None]*swb[None,:])
        a_ptrs += 128; w_ptrs += 128
    bias = tl.load(BIAS+offs_n).to(tl.float32)
    tl.store(C + offs_m[:,None]*stride_cm + offs_n[None,:], (acc+bias[None,:]).to(tl.bfloat16),
             mask=mask_m[:,None])

def run(a, w, bias, MODE=0, BM=128, BN=128):
    M,K = a.shape; N = w.shape[0]
    out = torch.empty((M,N), device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M,BM)*triton.cdiv(N,BN),)
    _g[grid](a,w,bias,out,M,N,K,a.stride(0),w.stride(0),out.stride(0),
             BM=BM,BN=BN,GROUP_M=1,NUM_NBLK=BN//128,MODE=MODE,num_warps=4,num_stages=2)
    return out
