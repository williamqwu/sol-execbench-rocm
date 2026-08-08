import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/var/tmp/solbench/te_m1')
import kernel_m
e = kernel_m._ext()

torch.manual_seed(0)
B,C,H,W = 1,512,32,32
x = torch.randn(B,C,H,W,device='cuda')
r = torch.randn(B,C,H,W,device='cuda')
g = torch.randn(C,device='cuda'); b = torch.randn(C,device='cuda')
eps=1e-6

ref_res = x + r
ref_out = F.silu(F.group_norm(ref_res, 32, g, b, eps))

res2 = torch.empty_like(x)
out = e.add_res_gn_silu(x, r, g, b, 32, eps, res2)
print("res mismatch:", (res2!=ref_res).sum().item(), "max", (res2-ref_res).abs().max().item())
print("out mismatch:", (out!=ref_out).sum().item(), "max", (out-ref_out).abs().max().item())

# also verify the simpler ops still good
print("gn_silu:", (e.gn_silu(x,g,b,32,eps)!=F.silu(F.group_norm(x,32,g,b,eps))).sum().item())
tb = torch.randn(B,C,device='cuda')
print("bias_add_gn_silu:", (e.bias_add_gn_silu(x,tb,g,b,32,eps)!=F.silu(F.group_norm(x+tb[:,:,None,None],32,g,b,eps))).sum().item())
print("add_res:", (e.add_res(x,r)!=ref_res).sum().item())
