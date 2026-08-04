import torch
dev='cuda:0'
M,N,K=256,7680,7680
a=torch.randn(M,K,device=dev).to(torch.float8_e4m3fn)
b=torch.randn(N,K,device=dev).to(torch.float8_e4m3fn)
sa=torch.rand(M,K//128,device=dev)*0.01+0.001
sb=torch.rand(N//128,K//128,device=dev)*0.01+0.001
for desc, args in [
  ("a_row_major_scales", (sa, sb)),
]:
    try:
        out = torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        print("OK", desc, out.shape, out.dtype)
    except Exception as e:
        print("FAIL", desc, type(e).__name__, str(e)[:600])
# try transposed-scale layouts
for name, (x,y) in {
  "sa_T": (sa.T.contiguous().T, sb),
  "sb_T": (sa, sb.T.contiguous().T),
  "both_T": (sa.T.contiguous().T, sb.T.contiguous().T),
}.items():
    try:
        out = torch._scaled_mm(a, b.T, scale_a=x, scale_b=y, out_dtype=torch.bfloat16)
        print("OK", name)
    except Exception as e:
        print("FAIL", name, str(e)[:300])
try:
    import aiter; print("aiter", aiter.__file__)
except Exception as e:
    print("no aiter", e)
