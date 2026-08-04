import torch, triton, kernel_tri
bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).cuda()
ref = torch.tanh(bits / 30.0) * 30.0
mine = torch.empty_like(bits)
n = bits.numel()
kernel_tri._k_dbg[(triton.cdiv(n,1024),)](bits, mine, n, BLOCK=1024)
rb, mb = ref.view(torch.uint16), mine.view(torch.uint16)
fin = torch.isfinite(bits)
bad = (rb != mb) & fin
print("finite inputs:", int(fin.sum()), " mismatches:", int(bad.sum()))
for j in bad.nonzero()[:10].flatten().tolist():
    print(f"   x={bits[j].item():.6g} ref={ref[j].item():.8g} mine={mine[j].item():.8g}")
for nm,v in [("+inf",float('inf')),("-inf",float('-inf')),("nan",float('nan'))]:
    t=torch.tensor([v],dtype=torch.bfloat16,device='cuda'); o=torch.empty_like(t)
    kernel_tri._k_dbg[(1,)](t,o,1,BLOCK=1024)
    print(f"   {nm}: ref={(torch.tanh(t/30.0)*30.0).item()} mine={o.item()}")
