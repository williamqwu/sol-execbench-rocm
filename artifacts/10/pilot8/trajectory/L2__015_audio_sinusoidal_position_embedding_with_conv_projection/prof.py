import torch, torch.nn.functional as F, math, time
import reference as R
dev = torch.device('cuda')
def bench(f, n=20):
    for _ in range(5): f()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1e3

for (B,T) in [(32,4328),(1,3000),(16,2048),(64,128)]:
    d = R.get_inputs({"batch_size":B,"time_dim":T}, dev)
    x0=d['input_features']; w1=d['conv2d1_weight'];b1=d['conv2d1_bias']
    w2=d['conv2d2_weight'];b2=d['conv2d2_bias'];w3=d['conv2d3_weight'];b3=d['conv2d3_bias']
    wo=d['conv_out_weight']; pe=d['positional_embedding']; es=d['embed_scale']
    x1=F.gelu(F.conv2d(x0,w1,b1,stride=2,padding=1))
    x2=F.gelu(F.conv2d(x1,w2,b2,stride=2,padding=1))
    x3=F.gelu(F.conv2d(x2,w3,b3,stride=2,padding=1))
    b,c,f_,t=x3.size(); xf=x3.permute(0,3,1,2).contiguous().view(b,t,c*f_)
    print(f"--- B={B} T={T}  shapes {tuple(x1.shape)} {tuple(x2.shape)} {tuple(x3.shape)}")
    print("  total ", bench(lambda: R.run(**d)))
    print("  c1    ", bench(lambda: F.conv2d(x0,w1,b1,stride=2,padding=1)))
    print("  c2    ", bench(lambda: F.conv2d(x1,w2,b2,stride=2,padding=1)))
    print("  c3    ", bench(lambda: F.conv2d(x2,w3,b3,stride=2,padding=1)))
    print("  perm  ", bench(lambda: x3.permute(0,3,1,2).contiguous()))
    print("  lin   ", bench(lambda: F.linear(xf,wo)))
    # channels_last variants
    x0c=x0.to(memory_format=torch.channels_last); w1c=w1.to(memory_format=torch.channels_last)
    x1c=x1.to(memory_format=torch.channels_last); w2c=w2.to(memory_format=torch.channels_last)
    x2c=x2.to(memory_format=torch.channels_last); w3c=w3.to(memory_format=torch.channels_last)
    print("  c1 CL ", bench(lambda: F.conv2d(x0c,w1c,b1,stride=2,padding=1)))
    print("  c2 CL ", bench(lambda: F.conv2d(x1c,w2c,b2,stride=2,padding=1)))
    print("  c3 CL ", bench(lambda: F.conv2d(x2c,w3c,b3,stride=2,padding=1)))
