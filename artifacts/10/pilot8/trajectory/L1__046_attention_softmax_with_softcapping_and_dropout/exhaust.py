import torch, kernel_hip

# every one of the 65536 bf16 bit patterns
bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).cuda()
ref = (torch.tanh(bits / 30.0) * 30.0)
mine = kernel_hip._ext.softcap_dbg(bits)

rb, mb = ref.view(torch.uint16), mine.view(torch.uint16)
finite = torch.isfinite(bits)
bad = (rb != mb) & finite
print("finite bf16 inputs:", int(finite.sum()))
print("bit-exact mismatches over ALL finite bf16:", int(bad.sum()))
if bad.any():
    i = bad.nonzero()[:15].flatten()
    for j in i.tolist():
        print(f"  x={bits[j].item():.6g}  ref={ref[j].item():.6g}  mine={mine[j].item():.6g}")
# nan/inf behaviour
for name, val in [("+inf", float('inf')), ("-inf", float('-inf')), ("nan", float('nan'))]:
    t = torch.tensor([val], dtype=torch.bfloat16, device='cuda')
    print(f"  {name}: ref={(torch.tanh(t/30.0)*30.0).item()}  mine={kernel_hip._ext.softcap_dbg(t).item()}")
