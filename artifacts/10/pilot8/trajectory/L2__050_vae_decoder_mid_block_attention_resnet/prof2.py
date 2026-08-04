import torch, time, torch.nn.functional as F
torch.manual_seed(0)
dev='cuda'
B,C,H,W=32,512,32,32
x=torch.randn(B,C,H,W,device=dev)
w=torch.randn(C,C,3,3,device=dev)
b=torch.randn(C,device=dev)
def bench(f,n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
print("base:", bench(lambda: F.conv2d(x,w,b,padding=1)))
torch.backends.cudnn.allow_tf32=True
torch.backends.cuda.matmul.allow_tf32=True
try:
    torch.backends.cuda.matmul.fp32_precision='tf32'
except Exception as e: print(e)
print("tf32:", bench(lambda: F.conv2d(x,w,b,padding=1)))
print("bf16:", bench(lambda: F.conv2d(x.bfloat16(),w.bfloat16(),b.bfloat16(),padding=1)))
print("fp16:", bench(lambda: F.conv2d(x.half(),w.half(),b.half(),padding=1)))
xc=x.contiguous(memory_format=torch.channels_last).bfloat16(); wc=w.contiguous(memory_format=torch.channels_last).bfloat16()
print("bf16 cl:", bench(lambda: F.conv2d(xc,wc,b.bfloat16(),padding=1)))
# matmul peak check
a=torch.randn(8192,8192,device=dev); bb=torch.randn(8192,8192,device=dev)
t=bench(lambda: a@bb); print("fp32 mm TFLOPs:", 2*8192**3/t*1e-9)
ah=a.bfloat16(); bh=bb.bfloat16()
t=bench(lambda: ah@bh); print("bf16 mm TFLOPs:", 2*8192**3/t*1e-9)
