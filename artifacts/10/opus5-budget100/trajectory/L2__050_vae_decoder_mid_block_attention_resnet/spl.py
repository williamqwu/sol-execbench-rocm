import torch, time
import torch.nn.functional as F
def bench(f,n=50):
    for _ in range(15): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3
torch.manual_seed(0)
C=512; scale=C**-0.5; z=torch.zeros(1,device='cuda')
for B,H,W in [(1,32,32),(16,32,32),(32,32,32),(2,64,64),(4,48,48)]:
    S=H*W
    h=torch.randn(B,S,C,device='cuda')
    w3=torch.randn(3*C,C,device='cuda')/22; b3=torch.randn(3*C,device='cuda')
    # interleave weights so q,k,v come out as contiguous BLOCKS along batch dim instead
    def base():
        qkv=F.linear(h,w3,b3); q,k,v=qkv.split(C,dim=-1)
        s=torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale)
        torch.softmax(s,dim=-1,out=s)
        return torch.matmul(s,v)
    # alternative: produce qkv as (3,B,S,C) contiguous via output-major weight layout
    def blk():
        qkv=F.linear(h,w3,b3).view(B,S,3,C).permute(2,0,1,3).contiguous()
        q,k,v=qkv[0],qkv[1],qkv[2]
        s=torch.baddbmm(z,q,k.transpose(-2,-1),beta=0.,alpha=scale)
        torch.softmax(s,dim=-1,out=s)
        return torch.matmul(s,v)
    r=base()
    print(f"B{B} {H}x{W}: base={bench(base):.4f} blkcontig={bench(blk):.4f} exact={(blk()==r).all().item()}")
