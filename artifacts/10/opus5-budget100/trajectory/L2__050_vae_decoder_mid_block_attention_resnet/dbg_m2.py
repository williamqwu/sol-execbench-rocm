import os, sys, torch
import torch.nn.functional as F
sys.path.insert(0,'/var/tmp/solbench/agent/opus5-budget100/L2__050_vae_decoder_mid_block_attention_resnet')
os.environ.setdefault('TORCH_EXTENSIONS_DIR','/var/tmp/solbench/te_m1')
from harness import make
import kernel_m
e = kernel_m._ext()

args = make(1,32,32)
(hidden_states, temb, r1n1w,r1n1b,r1c1w,r1c1b,r1tw,r1tb,r1n2w,r1n2b,r1c2w,r1c2b,
 agnw,agnb,qw,qb,kw,kb,vw,vb,ow,ob,
 r2n1w,r2n1b,r2c1w,r2c1b,r2tw,r2tb,r2n2w,r2n2b,r2c2w,r2c2b, eps) = args
B,C,H,W = hidden_states.shape
ng=32; S=H*W; scale=C**-0.5

# reference intermediates
h = F.conv2d(F.silu(F.group_norm(hidden_states,ng,r1n1w,r1n1b,eps)), r1c1w,r1c1b,padding=1)
tp1 = F.linear(F.silu(temb), r1tw, r1tb)
h = h + tp1[:,:,None,None]
h = F.conv2d(F.silu(F.group_norm(h,ng,r1n2w,r1n2b,eps)), r1c2w,r1c2b,padding=1)
R_hs = h + hidden_states
Rh = F.group_norm(R_hs,ng,agnw,agnb,eps).view(B,C,S).transpose(1,2)
q=F.linear(Rh,qw,qb); k=F.linear(Rh,kw,kb); v=F.linear(Rh,vw,vb)
p=F.softmax(torch.matmul(q,k.transpose(-2,-1))*scale,dim=-1)
Ro=F.linear(torch.matmul(p,v), ow, ob).transpose(1,2).reshape(B,C,H,W)
R_res2 = Ro + R_hs
R_gn2 = F.silu(F.group_norm(R_res2, ng, r2n1w, r2n1b, eps))

# mine
tp_all = F.linear(F.silu(temb), torch.cat((r1tw,r2tw),0), torch.cat((r1tb,r2tb),0))
m_tp1, m_tp2 = tp_all.split(C,dim=-1)
m_tp1=m_tp1.contiguous(); m_tp2=m_tp2.contiguous()
mh = e.gn_silu(hidden_states, r1n1w, r1n1b, ng, eps)
mh = F.conv2d(mh, r1c1w, r1c1b, padding=1)
mh = e.bias_add_gn_silu(mh, m_tp1, r1n2w, r1n2b, ng, eps)
mh = F.conv2d(mh, r1c2w, r1c2b, padding=1)
m_hs = e.add_res(mh, hidden_states)
print("hs match:", (m_hs!=R_hs).sum().item())
mg = e.gn_plain(m_hs, agnw, agnb, ng, eps)
mgt = mg.view(B,C,S).transpose(1,2)
qkv = F.linear(mgt, torch.cat((qw,kw,vw),0), torch.cat((qb,kb,vb),0))
mq,mk,mv = qkv.split(C,dim=-1)
mp_ = F.softmax(torch.matmul(mq,mk.transpose(-2,-1))*scale,dim=-1)
mo = F.linear(torch.matmul(mp_,mv), ow, ob).transpose(1,2).reshape(B,C,H,W)
print("attn out match:", (mo!=Ro).sum().item(), (mo-Ro).abs().max().item())
res2 = torch.empty_like(m_hs)
mgn2 = e.add_res_gn_silu(mo, m_hs, r2n1w, r2n1b, ng, eps, res2)
print("res2 match:", (res2!=R_res2).sum().item(), (res2-R_res2).abs().max().item())
print("gn2 match:", (mgn2!=R_gn2).sum().item(), (mgn2-R_gn2).abs().max().item())
print("mo contiguous:", mo.is_contiguous(), "m_hs contig:", m_hs.is_contiguous())
