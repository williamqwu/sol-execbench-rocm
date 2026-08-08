import torch, sys
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
from harness import make, bench, TOL
import reference

# breakdown: conv time vs everything else
for shape in TOL:
    B,H,W = shape
    args = make(B,H,W)
    x = args[0]; C=512
    w = args[4]; b = args[5]
    t_all = bench(lambda: reference.run(*args))
    t_conv = bench(lambda: F.conv2d(x,w,b,padding=1))*4
    g = args[2]; gb=args[3]
    t_gn = bench(lambda: F.group_norm(x,32,g,gb,1e-6))*5
    t_silu = bench(lambda: F.silu(x))*4
    t_add = bench(lambda: x+x)*5
    S=H*W
    hh = x.view(B,C,S).transpose(1,2).contiguous()
    W3 = torch.cat((args[14],args[16],args[18]),0); B3=torch.cat((args[15],args[17],args[19]),0)
    t_qkv = bench(lambda: F.linear(hh,W3,B3))
    q=hh; k=hh; v=hh
    t_attn = bench(lambda: torch.matmul(F.softmax(torch.matmul(q,k.transpose(-2,-1))*(C**-0.5),dim=-1),v))
    t_out = bench(lambda: F.linear(hh,args[20],args[21]))
    rest = t_all - (t_conv+t_gn+t_silu+t_add+t_qkv+t_attn+t_out)
    print(f"B{B} {H:2d}x{W:2d}: tot={t_all:7.4f} conv={t_conv:7.4f}({t_conv/t_all*100:4.1f}%) gn={t_gn:6.4f} silu={t_silu:6.4f} add={t_add:6.4f} qkv={t_qkv:6.4f} attn={t_attn:6.4f} out={t_out:6.4f} rest={rest:6.4f}")
