import sys, torch
sys.path.insert(0, "/var/tmp/solbench/agent/opus5-budget100/Quant__004_fp8_moe_expert_linear")
dev = 'cuda'
torch.manual_seed(0)
M, K, N = 512, 3584, 4096
a8 = (torch.randn(M, K, device=dev) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
b8 = (torch.randn(N, K, device=dev) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
af = a8.float()
bf = b8.float()
y32 = af @ bf.T
y64 = (af.double() @ bf.T.double())
d = (y32.double() - y64).abs()
print("fp32 matmul err vs fp64: max", d.max().item(), "mean", d.mean().item(),
      "| refmag", y64.abs().mean().item())
print("allow_tf32", torch.backends.cuda.matmul.allow_tf32)
try:
    print("fp32_precision", torch.backends.cuda.matmul.fp32_precision)
except Exception as e:
    print(e)
