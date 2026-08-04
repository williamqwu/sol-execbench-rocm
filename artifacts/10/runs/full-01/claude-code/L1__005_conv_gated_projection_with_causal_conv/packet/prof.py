import torch, torch.nn.functional as F, sys
sys.path.insert(0,'.')
import kernel, triton
DEV="cuda:0"; H=2048
torch.manual_seed(0)
def bench(fn,n=50):
    for _ in range(15): fn()
    torch.cuda.synchronize()
    ts=[]
    for _ in range(3):
        st=torch.cuda.Event(True);en=torch.cuda.Event(True);st.record()
        for _ in range(n): fn()
        en.record();torch.cuda.synchronize();ts.append(st.elapsed_time(en)/n*1000)
    return min(ts)

for (B,S) in [(1,256),(2,2048),(2,4096),(32,256),(1,8192)]:
    M=B*S
    x=torch.randn(B,S,H,dtype=torch.bfloat16,device=DEV)
    w1=torch.randn(3*H,H,dtype=torch.bfloat16,device=DEV)
    b1=torch.randn(3*H,dtype=torch.bfloat16,device=DEV)
    cw=torch.randn(H,1,4,dtype=torch.bfloat16,device=DEV)
    cb=torch.randn(H,dtype=torch.bfloat16,device=DEV)
    w2=torch.randn(H,H,dtype=torch.bfloat16,device=DEV)
    b2=torch.randn(H,dtype=torch.bfloat16,device=DEV)
    x2=x.reshape(M,H)
    bcx=F.linear(x2,w1,b1)
    bxp=torch.zeros((B,H,S+3),dtype=torch.bfloat16,device=DEV)
    BS,BH=64,64
    grid=(triton.cdiv(S,BS),triton.cdiv(H,BH),B)
    t_g1=bench(lambda: F.linear(x2,w1,b1))
    t_k1=bench(lambda: kernel._gate_pad_kernel[grid](bcx,bxp,S,H,M,bcx.stride(0),BS=BS,BH=BH,PAD=3,num_warps=4,num_stages=2))
    conv=F.conv1d(bxp,cw,cb,groups=H)
    t_cv=bench(lambda: F.conv1d(bxp,cw,cb,groups=H))
    y=torch.empty((M,H),dtype=torch.bfloat16,device=DEV)
    t_k2=bench(lambda: kernel._gate_out_kernel[grid](bcx,conv,y,S,H,M,bcx.stride(0),BS=BS,BH=BH,num_warps=4,num_stages=2))
    t_g2=bench(lambda: F.linear(y,w2,b2))
    tot=bench(lambda: kernel.run(x,w1,b1,cw,cb,w2,b2))
    # SOL-ish estimates
    fl1=2*M*H*3*H/1e12; fl2=2*M*H*H/1e12
    print(f"B={B:2d} S={S:5d} M={M:6d} | gemm1={t_g1:7.1f} k1={t_k1:6.1f} conv={t_cv:7.1f} k2={t_k2:6.1f} gemm2={t_g2:6.1f} | sum={t_g1+t_k1+t_cv+t_k2+t_g2:7.1f} tot={tot:7.1f}")
    print(f"        gemm1 TFLOPs={fl1/(t_g1*1e-6):7.1f}  gemm2 TFLOPs={fl2/(t_g2*1e-6):7.1f}  conv GB/s={(2*M*H*2)/(t_cv*1e-6)/1e9:7.1f}")
